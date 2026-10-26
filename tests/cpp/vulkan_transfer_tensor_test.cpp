#include "vulkan/operators/binary.h"
#include "vulkan/operators/comparison.h"
#include "vulkan/operators/convolution.h"
#include "vulkan/operators/linear.h"
#include "vulkan/operators/pooling.h"
#include "vulkan/operators/reduction.h"
#include "vulkan_allocator.h"
#include "vulkan_async_transfer.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_device_guard.h"
#include "vulkan_layout.h"
#include "vulkan_platform.h"
#include "vulkan_pinned_allocator.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

// Defined in this native target only. The library has one identical class
// definition in all TUs, and no Python/production definition of this overload.
VulkanCompletionId VulkanPlatform::enqueue_async_copy_for_testing(
    VulkanAsyncCopyRequest &&request,
    const std::function<void(VkCommandBuffer)> &record_override) const {
    return enqueue_async_copy_impl(std::move(request), record_override);
}

namespace pytorch_vulkan {
at::Tensor &formatter_presentation_copy(at::Tensor &destination,
                                        const at::Tensor &source);
}

namespace {

const c10::Device kDevice(c10::DeviceType::PrivateUse1, 0);

void expect(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_error(const std::function<void()> &operation, const char *text) {
    try {
        operation();
    } catch (const c10::Error &error) {
        const std::string message = error.what();
        expect(message.find(text) != std::string::npos,
               (std::string("expected '") + text + "' in Vulkan transfer error: " +
                message).c_str());
        return;
    } catch (const std::exception &error) {
        const std::string message = error.what();
        expect(message.find(text) != std::string::npos,
               (std::string("expected '") + text + "' in Vulkan transfer error: " +
                message).c_str());
        return;
    }
    throw std::runtime_error("Vulkan transfer accepted invalid input");
}

void test_async_copy_rejects_empty_request() {
    expect_error([&] { pytorch_vulkan::platform()->enqueue_async_copy({}); },
                 "Invalid async copy");
}

using Clock = std::chrono::steady_clock;
constexpr VkDeviceSize kAsyncBytes = 2 * sizeof(float);
constexpr auto kHostMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

// Native qualification must destroy the cached platform while validation-layer
// dispatch state is alive, not during libc atexit. Declared before main's try so
// all test locals unwind first on both normal and exception returns. Weak-only
// observation cannot itself delay destruction; expiry proves shutdown really
// destroyed the platform instead of leaving a strong owner for late atexit.
class NativePlatformShutdown {
  public:
    void observe(const std::shared_ptr<VulkanPlatform> &platform) {
        platform_ = platform;
        observed_ = true;
    }
    ~NativePlatformShutdown() {
        pytorch_vulkan::shutdown_platform();
        if (!platform_.expired()) {
            std::cerr << "STOP: native cached platform still strongly owned at exit\n";
            std::abort();
        }
        if (observed_)
            std::cout << "native-main-shutdown cached-platform-destroyed-before-exit\n";
    }
  private:
    std::weak_ptr<VulkanPlatform> platform_;
    bool observed_ = false;
};

// HS is synthetic completion observation on real hardware, not an unsignaled
// hardware fence. Every finalizer releases and asserts zero expiries, including
// exceptional exits. Never synchronize/read while held.
class AsyncHold {
  public:
    explicit AsyncHold(const VulkanPlatform &platform) : platform_(platform) {
        platform_.set_completion_hold_for_testing(true);
    }
    ~AsyncHold() { release(); }
    void release() {
        platform_.set_completion_hold_for_testing(false);
        if (platform_.completion_hold_expiry_count_for_testing() != 0) {
            std::cerr << "STOP: completion hold watchdog expired\n";
            std::abort();
        }
    }
  private:
    const VulkanPlatform &platform_;
};

struct AsyncPinned {
    VulkanPinnedHostAllocator allocator;
    c10::Storage storage;
    VulkanPinnedHostRange range;
    explicit AsyncPinned(const std::shared_ptr<VulkanPlatform> &platform,
                         VulkanPinnedHostTestOps ops = {})
        : allocator(platform, ops),
          storage(c10::Storage::use_byte_size_t(), 64, &allocator, false) {
        auto *interior = static_cast<char *>(storage.mutable_data()) + sizeof(float);
        expect(vulkan_pinned_host_range(storage, interior, kAsyncBytes, &range),
               "async pinned interior range failed");
        expect(range.offset == sizeof(float), "pinned interior offset lost");
    }
    void seed(float x = 2.0F, float y = 7.0F) {
        const float values[] = {x, y};
        std::memcpy(static_cast<char *>(storage.mutable_data()) + range.offset,
                    values, sizeof(values));
    }
    void expect_values(float x = 2.0F, float y = 7.0F) const {
        float actual[2]{};
        std::memcpy(actual, static_cast<const char *>(storage.data()) + range.offset,
                    sizeof(actual));
        expect(actual[0] == x && actual[1] == y, "pinned async result != CPU oracle");
    }
};

VulkanAsyncCopyRequest host_request(
    const VulkanPlatform &platform, VulkanAsyncCopyKind kind,
    const VulkanPinnedHostRange &host, const std::shared_ptr<VulkanBuffer> &device,
    VkDeviceSize device_offset = sizeof(float)) {
    VulkanAsyncCopyRequest request;
    request.kind = kind;
    request.source = kind == VulkanAsyncCopyKind::HostToDevice ? host.buffer->buffer() : device->buffer();
    request.destination = kind == VulkanAsyncCopyKind::HostToDevice ? device->buffer() : host.buffer->buffer();
    request.regions = kind == VulkanAsyncCopyKind::HostToDevice
        ? std::vector<VkBufferCopy>{{host.offset, device_offset, host.bytes}}
        : std::vector<VkBufferCopy>{{device_offset, host.offset, host.bytes}};
    request.reservation = platform.reserve_transfer_slot();
    request.owners = {host.lease, device};
    if (kind == VulkanAsyncCopyKind::HostToDevice)
        request.before_submit = [host] { host.buffer->flush_range(host.offset, host.bytes); };
    else
        request.publish_success = [host] { host.buffer->invalidate_range(host.offset, host.bytes); };
    return request;
}

VulkanAsyncCopyRequest device_request(const std::shared_ptr<VulkanBuffer> &source,
                                      const std::shared_ptr<VulkanBuffer> &destination) {
    VulkanAsyncCopyRequest request;
    request.kind = VulkanAsyncCopyKind::DeviceToDevice;
    request.source = source->buffer();
    request.destination = destination->buffer();
    request.regions = {{0, 0, kAsyncBytes}};
    request.owners = {source, destination};
    return request;
}

void test_async_pinned_offsets_and_publication() {
    const auto p = pytorch_vulkan::platform();
    p->synchronize_execution();
    AsyncPinned source(p), destination(p);
    source.seed();
    auto device = std::make_shared<VulkanBuffer>(*p, 64, kHostMemory);
    auto h2d = host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, device);
    const auto waits = p->execution_fence_wait_count(), transfer_waits = p->transfer_wait_count();
    AsyncHold hold(*p);
    const auto start = Clock::now();
    expect(p->enqueue_async_copy(std::move(h2d)) != 0, "host completion id is zero");
    expect(p->live_resource_snapshot().pending_transfers == 1,
           "H2D did not retain one pending transfer");
    expect(p->execution_fence_wait_count() == waits && p->transfer_wait_count() == transfer_waits,
           "H2D enqueue waited unexpectedly");
    expect(!p->query_execution(), "held H2D query reported complete");
    expect(Clock::now() - start < std::chrono::seconds(5), "H2D held transition exceeded 5s");
    hold.release();
    p->synchronize_execution();
    float actual[2]{};
    device->read(actual, sizeof(actual), sizeof(float));
    expect(actual[0] == 2.0F && actual[1] == 7.0F, "H2D offsets lost CPU seed");
    expect(p->live_resource_snapshot().pending_transfers == 0, "H2D did not retire");
    expect(p->query_execution(), "released H2D observation did not resume");

    unsigned published = 0;
    auto d2h = host_request(*p, VulkanAsyncCopyKind::DeviceToHost, destination.range, device);
    const auto invalidate = d2h.publish_success;
    d2h.publish_success = [&, invalidate] {
        expect(p->queue_mutex().try_lock(), "publication held queue lock");
        p->queue_mutex().unlock();
        invalidate();
        ++published;
        expect(!p->query_execution(), "query completed within publication");
    };
    AsyncHold read_hold(*p);
    p->enqueue_async_copy(std::move(d2h));
    expect(!p->query_execution() && published == 0, "D2H published while held");
    read_hold.release();
    p->synchronize_execution();
    expect(published == 1 && p->query_execution(), "D2H did not publish exactly once");
    destination.expect_values();
}

struct AsyncMaintenance {
    const VulkanPlatform *platform;
    unsigned flushes = 0, invalidates = 0;
    std::size_t copies_before = 0, submissions_before = 0;
    VkMappedMemoryRange flushed{}, invalidated{};
    static VkResult flush(void *cookie, VkDevice device, uint32_t n,
                          const VkMappedMemoryRange *ranges) {
        auto &c = *static_cast<AsyncMaintenance *>(cookie);
        expect(c.platform->queue_mutex().try_lock(), "flush ran under queue lock");
        c.platform->queue_mutex().unlock();
        expect(c.platform->copy_command_count() == c.copies_before &&
                   c.platform->compute_submitted_count() == c.submissions_before,
               "flush ran after recording/submission");
        ++c.flushes;
        expect(n == 1, "flush range count != 1");
        c.flushed = ranges[0];
        return vkFlushMappedMemoryRanges(device, n, ranges);
    }
    static VkResult invalidate(void *cookie, VkDevice device, uint32_t n,
                               const VkMappedMemoryRange *ranges) {
        auto &c = *static_cast<AsyncMaintenance *>(cookie);
        expect(c.platform->queue_mutex().try_lock(), "invalidate ran under queue lock");
        c.platform->queue_mutex().unlock();
        ++c.invalidates;
        expect(n == 1, "invalidate range count != 1");
        c.invalidated = ranges[0];
        return vkInvalidateMappedMemoryRanges(device, n, ranges);
    }
    VulkanPinnedHostTestOps ops() {
        VulkanPinnedHostTestOps ops;
        ops.map.cookie = this;
        ops.map.flush = flush;
        ops.map.invalidate = invalidate;
        ops.map.override_coherent_for_testing = true;
        ops.map.coherent_for_testing = false;
        return ops;
    }
    void checkpoint() {
        copies_before = platform->copy_command_count();
        submissions_before = platform->compute_submitted_count();
    }
    void check_range(const VkMappedMemoryRange &range, const VulkanPinnedHostRange &host) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(platform->physical_device(), &properties);
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(platform->device(), host.buffer->buffer(), &requirements);
        const auto atom = properties.limits.nonCoherentAtomSize;
        const auto begin = host.offset / atom * atom;
        const auto end = std::min(requirements.size,
            ((host.offset + host.bytes + atom - 1) / atom) * atom);
        // Accepted primitive uses VK_WHOLE_SIZE when the aligned cover reaches
        // allocation end; this is the exact same byte cover, not a larger range.
        const auto size = end == requirements.size ? VK_WHOLE_SIZE : end - begin;
        expect(range.offset == begin && range.size == size,
               "noncoherent range not exact atom-aligned cover");
    }
};

void test_async_synthetic_maintenance_real_queue() {
    const auto p = pytorch_vulkan::platform();
    p->synchronize_execution();
    AsyncMaintenance counts{p.get()};
    AsyncPinned source(p, counts.ops()), destination(p, counts.ops());
    source.seed();
    auto device = std::make_shared<VulkanBuffer>(*p, 64, kHostMemory);
    counts.checkpoint();
    auto request = host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, device);
    AsyncHold hold(*p);
    p->enqueue_async_copy(std::move(request));
    expect(counts.flushes == 1 && counts.invalidates == 0, "H2D maintenance counts wrong");
    expect(p->copy_command_count() == counts.copies_before + 1 &&
               p->compute_submitted_count() == counts.submissions_before + 1,
           "flush witness did not actually record/submit");
    expect(!p->query_execution(), "synthetic maintenance hold not active");
    counts.check_range(counts.flushed, source.range);
    hold.release();
    p->synchronize_execution();
    auto readback = host_request(*p, VulkanAsyncCopyKind::DeviceToHost, destination.range, device);
    AsyncHold second(*p);
    p->enqueue_async_copy(std::move(readback));
    expect(!p->query_execution() && counts.invalidates == 0, "invalidate ran before release");
    second.release();
    p->synchronize_execution();
    expect(counts.flushes == 1 && counts.invalidates == 1, "success maintenance not exactly once");
    counts.check_range(counts.invalidated, destination.range);
    destination.expect_values();
}

void test_async_maintenance_failures() {
    // Dedicated instance: a recorder cancellation intentionally leaves a sticky
    // execution error. No GPU commands are recorded by the throwing override.
    auto p = std::make_shared<VulkanPlatform>(true);
    AsyncMaintenance counts{p.get()};
    AsyncPinned source(p, counts.ops());
    source.seed();
    auto device = std::make_shared<VulkanBuffer>(*p, 64, kHostMemory);
    counts.checkpoint();
    {
        auto invalid = host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, device);
        invalid.regions.clear();
        expect_error([&] { p->enqueue_async_copy(std::move(invalid)); }, "Invalid async copy");
        expect(counts.flushes == 0 && counts.invalidates == 0, "validation failure ran maintenance");
    }
    for (unsigned bad = 0; bad < 7; ++bad) {
        auto invalid = host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, device);
        if (bad == 0) invalid.source = VK_NULL_HANDLE;
        if (bad == 1) invalid.regions[0].size = 0;
        if (bad == 2) invalid.regions[0].srcOffset = 1;
        if (bad == 3) invalid.regions[0].dstOffset = std::numeric_limits<VkDeviceSize>::max() - 3;
        if (bad == 4) invalid.owners.clear();
        if (bad == 5) invalid.reservation = {};
        if (bad == 6) invalid.kind = static_cast<VulkanAsyncCopyKind>(99);
        expect_error([&] { p->enqueue_async_copy(std::move(invalid)); }, "Invalid async copy");
        expect(counts.flushes == 0 && counts.invalidates == 0 && p->copy_command_count() == 0,
               "invalid request performed maintenance or recorded");
    }
    {
        auto snapshot = p->reserve_transfer(8);
        expect_error([&] { const float seed[] = {2, 7}; snapshot.staging->write(seed, 8, 4); }, "outside");
        expect(counts.flushes == 0, "snapshot failure ran pinned flush");
    }
    {
        auto pre = host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, device);
        pre.before_submit = [] { throw std::runtime_error("injected host maintenance failure"); };
        expect_error([&] { p->enqueue_async_copy(std::move(pre)); }, "host maintenance failure");
        expect(p->copy_command_count() == 0 && counts.invalidates == 0,
               "maintenance failure recorded/published");
    }
    auto cancel = host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, device);
    cancel.publish_success = [range = source.range] {
        range.buffer->invalidate_range(range.offset, range.bytes);
    };
    expect_error([&] {
        p->enqueue_async_copy_for_testing(std::move(cancel), [](VkCommandBuffer) {
            throw std::runtime_error("injected async recorder failure");
        });
    }, "injected async recorder failure");
    expect(counts.flushes == 1 && counts.invalidates == 0 &&
               p->copy_command_count() == 0 && p->compute_submitted_count() == 0,
           "cancel maintenance counts not flush=1 invalidate=0");
    counts.check_range(counts.flushed, source.range);
    expect(p->live_resource_snapshot().pending_transfers == 0, "cancel retained transfer slot");
}

void test_async_d2d_batching_and_slots() {
    const auto p = pytorch_vulkan::platform();
    p->synchronize_execution();
    auto a = std::make_shared<VulkanBuffer>(*p, kAsyncBytes, kHostMemory);
    auto b = std::make_shared<VulkanBuffer>(*p, kAsyncBytes, kHostMemory);
    auto c = std::make_shared<VulkanBuffer>(*p, kAsyncBytes, kHostMemory);
    const float seed[] = {2, 7};
    a->write(seed, sizeof(seed));
    const auto waits = p->execution_fence_wait_count(), tw = p->transfer_wait_count();
    AsyncHold hold(*p);
    expect(p->enqueue_async_copy(device_request(a, b)) == 0, "D2D allocated a transfer id");
    p->enqueue_async_copy(device_request(b, c));
    expect(!p->query_execution(), "held D2D reported completion");
    auto slot1 = p->reserve_transfer_slot(), slot2 = p->reserve_transfer_slot();
    expect(bool(slot1) && bool(slot2) && p->live_resource_snapshot().pending_transfers == 0,
           "D2D consumed host transfer slots");
    expect(p->execution_fence_wait_count() == waits && p->transfer_wait_count() == tw,
           "D2D enqueue/query waited");
    slot1 = {}; slot2 = {};
    hold.release();
    p->synchronize_execution();
    float actual[2]{}; c->read(actual, sizeof(actual));
    expect(actual[0] == 2 && actual[1] == 7, "chained D2D lost data");
    const auto submits = p->compute_submitted_count();
    AsyncHold batch(*p);
    for (unsigned i = 0; i < 17; ++i) p->enqueue_async_copy(device_request(a, b));
    expect(p->compute_submitted_count() >= submits + 1, "17 D2D exceeded shared flush bound");
    expect(!p->query_execution(), "held D2D batch completed");
    batch.release();
    p->synchronize_execution();
    b->read(actual, sizeof(actual));
    expect(actual[0] == 2 && actual[1] == 7, "D2D flush bound lost data");

    const auto cpu = at::tensor({2.0F, 7.0F});
    auto input = at::empty_like(cpu, cpu.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, cpu, false);
    const auto mixed_submits = p->compute_submitted_count();
    AsyncHold mixed(*p);
    for (unsigned i = 0; i < 15; ++i) p->enqueue_async_copy(device_request(a, b));
    auto output = at::add(input, 1.0F);
    expect(p->compute_submitted_count() == mixed_submits + 1,
           "compute and D2D did not share the SAME 16-dispatch bound");
    expect(!p->query_execution(), "mixed bound not held");
    mixed.release(); p->synchronize_execution();
    auto output_cpu = at::empty_like(cpu);
    pytorch_vulkan::copy_tensor(output_cpu, output, false);
    expect(output_cpu.equal(cpu + 1), "mixed batch lost compute output");
}

void test_async_d2d_shutdown_drains_unsubmitted_leases() {
    auto p = std::make_shared<VulkanPlatform>(true);
    auto a = std::make_shared<VulkanBuffer>(*p, kAsyncBytes, kHostMemory);
    auto b = std::make_shared<VulkanBuffer>(*p, kAsyncBytes, kHostMemory);
    const float seed[] = {2, 7}; a->write(seed, sizeof(seed));
    std::weak_ptr<VulkanBuffer> wa = a, wb = b;
    p->enqueue_async_copy(device_request(a, b));
    expect(p->execution_context().recording(), "D2D teardown witness not recording");
    a.reset(); b.reset();
    expect(!wa.expired() && !wb.expired(), "D2D teardown leases not attached");
    p.reset();
    expect(wa.expired() && wb.expired(), "D2D deferred callbacks outlived platform shutdown");
}

void test_async_compute_and_war_order() {
    const auto p = pytorch_vulkan::platform();
    p->synchronize_execution();
    const auto seed = at::tensor({2.0F, 7.0F});
    auto input = at::empty_like(seed, seed.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, seed, false);
    AsyncPinned destination(p), source(p);
    source.seed();
    auto device = std::make_shared<VulkanBuffer>(*p, 64, kHostMemory);
    const float overwrite[] = {11, 19};
    device->write(overwrite, sizeof(overwrite), sizeof(float));
    AsyncHold hold(*p);
    auto producer = at::add(input, 3.0F);
    VulkanAsyncCopyRequest read;
    read.kind = VulkanAsyncCopyKind::DeviceToHost;
    read.source = pytorch_vulkan::allocation_buffer(producer.storage().data_ptr()).buffer();
    read.destination = destination.range.buffer->buffer();
    read.regions = {{0, destination.range.offset, kAsyncBytes}};
    read.reservation = p->reserve_transfer_slot();
    read.owners = {pytorch_vulkan::allocation_lease(producer.storage().data_ptr()), destination.range.lease};
    const auto range = destination.range;
    read.publish_success = [range] { range.buffer->invalidate_range(range.offset, range.bytes); };
    p->enqueue_async_copy(std::move(read));
    auto consumer = at::add(producer, 4.0F);
    expect(!p->query_execution(), "compute/transfer chain not held");
    hold.release(); p->synchronize_execution();
    destination.expect_values(5, 10);
    auto actual = at::empty_like(seed);
    pytorch_vulkan::copy_tensor(actual, consumer, false);
    expect(actual.equal(seed + 7), "compute consumer != CPU oracle");

    auto first_device = std::make_shared<VulkanBuffer>(*p, 64, kHostMemory);
    auto h2d = host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, first_device);
    auto d2h = host_request(*p, VulkanAsyncCopyKind::DeviceToHost, source.range, device);
    AsyncHold war(*p);
    p->enqueue_async_copy(std::move(h2d));
    p->enqueue_async_copy(std::move(d2h));
    expect(!p->query_execution(), "WAR queue not held");
    war.release(); p->synchronize_execution();
    source.expect_values(11, 19);
    float first[2]{}; first_device->read(first, sizeof(first), sizeof(float));
    expect(first[0] == 2 && first[1] == 7, "H2D read observed later D2H overwrite (WAR)");
}

void test_async_backpressure() {
    const auto p = pytorch_vulkan::platform();
    p->synchronize_execution();
    AsyncPinned source(p); source.seed();
    std::vector<std::shared_ptr<VulkanBuffer>> devices;
    for (unsigned i = 0; i < 3; ++i) devices.push_back(std::make_shared<VulkanBuffer>(*p, 64, kHostMemory));
    const auto before = p->transfer_reservation_wait_count();
    AsyncHold hold(*p);
    p->enqueue_async_copy(host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, devices[0]));
    p->enqueue_async_copy(host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, devices[1]));
    expect(p->live_resource_snapshot().pending_transfers == 2, "two slots not occupied");
    std::atomic<bool> done{false}; std::exception_ptr failure;
    std::thread third([&] {
        try { p->enqueue_async_copy(host_request(*p, VulkanAsyncCopyKind::HostToDevice, source.range, devices[2])); }
        catch (...) { failure = std::current_exception(); }
        done.store(true, std::memory_order_release);
    });
    const auto poll_end = Clock::now() + std::chrono::seconds(5);
    while (p->transfer_reservation_wait_count() == before && Clock::now() < poll_end)
        std::this_thread::yield();
    const bool pressure = p->transfer_reservation_wait_count() == before + 1 && !done.load();
    hold.release();
    const auto join_end = Clock::now() + std::chrono::seconds(5);
    while (!done.load(std::memory_order_acquire) && Clock::now() < join_end) std::this_thread::yield();
    if (!done.load(std::memory_order_acquire)) {
        std::cerr << "STOP: async pressure thread missed 5s join bound\n";
        std::abort();
    }
    third.join();
    expect(pressure, "third request did not block at capacity transition within 5s");
    if (failure) std::rethrow_exception(failure);
    p->synchronize_execution();
    for (auto &device : devices) {
        float actual[2]{}; device->read(actual, sizeof(actual), sizeof(float));
        expect(actual[0] == 2 && actual[1] == 7, "pressure copy lost CPU seed");
    }
    expect(p->live_resource_snapshot().pending_transfers == 0, "pressure left slots pending");
}

void test_async_allocation_lifetimes() {
    const auto p = pytorch_vulkan::platform();
    p->synchronize_execution();
    const auto before = p->live_resource_snapshot();
    AsyncPinned pinned(p); pinned.seed();
    auto data = vulkan_allocator_instance()->allocate(64);
    VulkanAsyncCopyRequest request;
    request.kind = VulkanAsyncCopyKind::HostToDevice;
    request.source = pinned.range.buffer->buffer();
    request.destination = pytorch_vulkan::allocation_buffer(data).buffer();
    request.regions = {{pinned.range.offset, sizeof(float), kAsyncBytes}};
    request.reservation = p->reserve_transfer_slot();
    request.owners = {pinned.range.lease, pytorch_vulkan::allocation_lease(data)};
    request.before_submit = [range = pinned.range] {
        range.buffer->flush_range(range.offset, range.bytes);
    };
    const auto allocated = p->live_resource_snapshot();
    AsyncHold hold(*p);
    p->enqueue_async_copy(std::move(request));
    data.clear(); pinned.storage = {}; pinned.range = {};
    expect(p->live_resource_snapshot().allocations == allocated.allocations &&
               p->live_resource_snapshot().pinned_host_allocations == allocated.pinned_host_allocations,
           "pending copy did not retain dropped DataPtrs");
    expect(!p->query_execution(), "lifetime copy not held");
    hold.release(); p->synchronize_execution();
    expect(p->live_resource_snapshot().allocations == before.allocations,
           "device lease survived completion");
    const auto pinned_after = p->live_resource_snapshot().pinned_host_allocations;

    auto source = vulkan_allocator_instance()->allocate(kAsyncBytes);
    auto destination = vulkan_allocator_instance()->allocate(kAsyncBytes);
    VulkanAsyncCopyRequest d2d;
    d2d.kind = VulkanAsyncCopyKind::DeviceToDevice;
    d2d.source = pytorch_vulkan::allocation_buffer(source).buffer();
    d2d.destination = pytorch_vulkan::allocation_buffer(destination).buffer();
    d2d.regions = {{0, 0, kAsyncBytes}};
    d2d.owners = {pytorch_vulkan::allocation_lease(source), pytorch_vulkan::allocation_lease(destination)};
    // Initialize through the normal blocking host path before the held copy.
    auto seed_staging = p->reserve_transfer(kAsyncBytes);
    const float seed[] = {2, 7}; seed_staging.staging->write(seed, sizeof(seed));
    p->copy_buffer_sync(seed_staging.staging->buffer(), d2d.source, kAsyncBytes);
    seed_staging.reservation = {}; seed_staging.staging.reset();
    AsyncHold device_hold(*p);
    p->enqueue_async_copy(std::move(d2d));
    source.clear(); destination.clear();
    expect(p->live_resource_snapshot().allocations == before.allocations + 2,
           "D2D failed to lease dropped DataPtrs");
    expect(!p->query_execution(), "D2D lifetime not held");
    device_hold.release(); p->synchronize_execution();
    const auto device_after = p->live_resource_snapshot().allocations;
    expect(pinned_after == before.pinned_host_allocations && device_after == before.allocations,
           ("named requests retained leases after completion: pinned excess=" +
            std::to_string(pinned_after - before.pinned_host_allocations) +
            " device excess=" + std::to_string(device_after - before.allocations)).c_str());
}

void test_async_pageable_snapshot_and_modes() {
    const auto p = pytorch_vulkan::platform();
    p->synchronize_execution();
    auto device = std::make_shared<VulkanBuffer>(*p, kAsyncBytes, kHostMemory);
    auto staging = p->reserve_transfer(kAsyncBytes);
    float pageable[] = {2, 7}; staging.staging->write(pageable, sizeof(pageable));
    pageable[0] = 100; pageable[1] = 200;
    VulkanAsyncCopyRequest request;
    request.kind = VulkanAsyncCopyKind::HostToDevice;
    request.source = staging.staging->buffer(); request.destination = device->buffer();
    request.regions = {{0, 0, kAsyncBytes}};
    request.reservation = std::move(staging.reservation);
    request.owners = {staging.staging, device};
    AsyncHold hold(*p);
    p->enqueue_async_copy(std::move(request));
    expect(!p->query_execution(), "snapshot copy not held");
    hold.release(); p->synchronize_execution();
    float actual[2]{}; device->read(actual, sizeof(actual));
    expect(actual[0] == 2 && actual[1] == 7, "engine reread pageable source");

    // H: active training scope forces host completion before return, and D2D
    // stays in training recording until the existing end-training boundary.
    AsyncPinned pinned(p); pinned.seed();
    p->compute().begin_training_step();
    p->enqueue_async_copy(host_request(*p, VulkanAsyncCopyKind::HostToDevice, pinned.range, device, 0));
    expect(p->live_resource_snapshot().pending_transfers == 0 && p->query_execution(),
           "training host enqueue returned incomplete");
    auto other = std::make_shared<VulkanBuffer>(*p, kAsyncBytes, kHostMemory);
    p->enqueue_async_copy(device_request(device, other));
    expect(p->execution_context().recording(), "training D2D failed to join recording");
    p->compute().end_training_step(); p->synchronize_execution();
    other->read(actual, sizeof(actual));
    expect(actual[0] == 2 && actual[1] == 7, "training D2D changed values");

    const char *old = std::getenv("PYTORCH_VULKAN_ASYNC_EXECUTION");
    const std::string saved = old ? old : "";
    setenv("PYTORCH_VULKAN_ASYNC_EXECUTION", "0", 1);
    std::shared_ptr<VulkanPlatform> sync;
    try { sync = std::make_shared<VulkanPlatform>(true); }
    catch (...) { setenv("PYTORCH_VULKAN_ASYNC_EXECUTION", saved.c_str(), 1); throw; }
    setenv("PYTORCH_VULKAN_ASYNC_EXECUTION", saved.c_str(), 1);
    AsyncPinned sync_pinned(sync); sync_pinned.seed();
    auto a = std::make_shared<VulkanBuffer>(*sync, 64, kHostMemory);
    auto b = std::make_shared<VulkanBuffer>(*sync, kAsyncBytes, kHostMemory);
    sync->enqueue_async_copy(host_request(*sync, VulkanAsyncCopyKind::HostToDevice, sync_pinned.range, a, 0));
    expect(sync->live_resource_snapshot().pending_transfers == 0 && sync->query_execution(),
           "sync host enqueue returned incomplete");
    sync->enqueue_async_copy(device_request(a, b));
    expect(sync->query_execution(), "sync D2D returned incomplete");
    sync->synchronize_execution(); b->read(actual, sizeof(actual));
    expect(actual[0] == 2 && actual[1] == 7, "sync D2D changed values");
    std::cout << "H mode=sync env=0 per-instance (outer gate env=1 restored)\n";
}

void run_async_witness(const char *name, const char *evidence, void (*test)()) {
    const auto start = Clock::now();
    test();
    std::cout << "async-witness " << name << " evidence=" << evidence << " seconds="
              << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
}

void test_rnn_limit_validation_uses_supplied_limits() {
    constexpr uint32_t batch = 4;
    constexpr uint32_t sequence = 3;
    constexpr uint32_t input_dimension = 8;
    constexpr uint32_t hidden_dimension = 16;
    const VkDeviceSize input_bytes = batch * sequence * input_dimension * sizeof(float);
    const VkDeviceSize weight_bytes = input_dimension * hidden_dimension * sizeof(float);
    const VkDeviceSize recurrent_bytes = hidden_dimension * hidden_dimension * sizeof(float);
    const VkDeviceSize bias_bytes = hidden_dimension * sizeof(float);
    const VkDeviceSize output_bytes = batch * sequence * hidden_dimension * sizeof(float);
    const VkDeviceSize largest_range = std::max({input_bytes, weight_bytes, recurrent_bytes,
                                                  bias_bytes, output_bytes});
    const VkDeviceSize backward_range = std::max(
        {largest_range, batch * weight_bytes, batch * recurrent_bytes, batch * bias_bytes});

    VulkanCompute::validate_rnn_sequence_limits(
        batch, sequence, input_dimension, hidden_dimension, largest_range, batch);
    expect_error(
        [&] {
            VulkanCompute::validate_rnn_sequence_limits(
                batch, sequence, input_dimension, hidden_dimension, output_bytes - 1, batch);
        },
        "maxStorageBufferRange");
    expect_error(
        [&] {
            VulkanCompute::validate_rnn_sequence_limits(
                batch, sequence, input_dimension, hidden_dimension, largest_range, batch - 1);
        },
        "batch must be between 1 and 16");

    VulkanCompute::validate_rnn_sequence_backward_limits(
        batch, sequence, input_dimension, hidden_dimension, backward_range, batch);
    expect_error(
        [&] {
            VulkanCompute::validate_rnn_sequence_backward_limits(
                1, 1, 512, 256, std::numeric_limits<VkDeviceSize>::max(), 1);
        },
        "maxComputeWorkGroupCount[0]");
}

void test_copy_round_trip() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto device_tensor = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_tensor, source, false);
    auto result = at::empty({4}, source.options());
    pytorch_vulkan::copy_tensor(result, device_tensor, false);
    expect(result.equal(source), "Vulkan tensor round trip changed data");
}

void test_copy_returns_without_pending_transfer_resources() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto device_tensor = at::empty({4}, source.options().device(kDevice));
    const auto platform = pytorch_vulkan::platform();

    pytorch_vulkan::copy_tensor(device_tensor, source, false);

    expect(platform->pending_transfer_count() == 0,
           "synchronous tensor copy left pending transfer resources");
}

void test_contiguous_reshape_copy_is_bulk() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto device_source = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_source, source, false);
    const auto platform = pytorch_vulkan::platform();
    platform->reset_execution_counters();

    auto result = pytorch_vulkan::vulkan_contiguous_copy(device_source);

    expect(platform->vulkan_copy_count() == 1,
           "contiguous reshape transfer was not counted once");
    expect(platform->copy_command_count() == 1,
           "contiguous reshape did not record one Vulkan copy command");
    auto cpu_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(cpu_result, result, false);
    expect(cpu_result.equal(source), "contiguous reshape bulk copy changed data");
}

void test_execution_counters_reset_and_read_stably() {
    const auto platform = pytorch_vulkan::platform();
    platform->reset_execution_counters();
    expect(platform->compute_dispatch_count() == 0,
           "reset did not clear compute dispatch count");
    expect(platform->explicit_transfer_count() == 0,
           "reset did not clear explicit transfer count");

    auto source = at::tensor({1.0F, -2.0F, 3.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    const auto transfer_after_first = platform->explicit_transfer_count();
    expect(transfer_after_first == 1,
           "CPU-to-Vulkan copy did not increment explicit transfer count");
    auto first = at::neg(input);
    const auto dispatch_after_first = platform->compute_dispatch_count();
    expect(dispatch_after_first > 0,
           "unary operation did not increment dispatch count");

    auto second = at::neg(input);
    expect(platform->compute_dispatch_count() > dispatch_after_first,
           "second execution did not increment dispatch count");
    expect(platform->explicit_transfer_count() == transfer_after_first,
           "Vulkan-only execution changed explicit transfer count");
    auto result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(result, second, false);
    expect(platform->explicit_transfer_count() == transfer_after_first + 1,
           "Vulkan-to-CPU copy did not increment explicit transfer count");
    (void)first;
}

void test_vulkan_to_vulkan_copy_is_counted_once() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    auto view = at::as_strided(input, {2}, {1}, 1);
    auto output = at::empty({2}, source.options().device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    platform->reset_execution_counters();
    pytorch_vulkan::copy_tensor(output, view, false);
    expect(platform->vulkan_copy_count() == 1,
           "generic Vulkan-to-Vulkan copy was not counted exactly once");
    expect(platform->explicit_transfer_count() == 0,
           "Vulkan-to-Vulkan copy was counted as an explicit transfer");
    auto result = at::empty({2}, source.options());
    pytorch_vulkan::copy_tensor(result, output, false);
    expect(platform->vulkan_copy_count() == 1,
           "Vulkan-to-CPU presentation changed Vulkan copy count");
}

void test_formatter_presentation_copy_reads_exact_range_and_waits() {
    auto source = at::tensor({10.0F, 20.0F, 30.0F, 40.0F, 50.0F});
    auto device_tensor = at::empty({5}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_tensor, source, false);
    auto pending = at::add(device_tensor, at::Scalar(1.0F));
    auto view = at::as_strided(pending, {2}, {1}, 2);
    auto result = at::empty({2}, source.options());
    const auto platform = pytorch_vulkan::platform();

    pytorch_vulkan::formatter_presentation_copy(result, view);

    expect(result.equal(at::tensor({31.0F, 41.0F})),
           "formatter presentation copy read the wrong value range");
    expect(platform->pending_transfer_count() == 0,
           "formatter presentation copy left pending transfer resources");
    auto full_result = at::empty({5}, source.options());
    pytorch_vulkan::copy_tensor(full_result, pending, false);
    expect(full_result.equal(at::tensor({11.0F, 21.0F, 31.0F, 41.0F, 51.0F})),
           "formatter presentation copy invalidated the source buffer");
}

void test_formatter_presentation_copy_preserves_double_and_rejects_general_readback() {
    const auto platform = pytorch_vulkan::platform();
    if (!platform->supports_formatter_double())
        return;

    auto source = at::tensor({1.25F, -2.5F, 7.0F});
    auto input = at::empty({3}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    auto double_tensor = input.to(at::kDouble);
    auto result = at::empty({2}, at::TensorOptions().dtype(at::kDouble));
    auto view = at::as_strided(double_tensor, {2}, {1}, 1);
    pytorch_vulkan::formatter_presentation_copy(result, view);
    expect(
        result.equal(at::tensor({-2.5, 7.0}, at::TensorOptions().dtype(at::kDouble))),
        "formatter presentation copy changed Double values");
    expect_error(
        [&] {
            auto generic_view_result =
                at::empty({2}, at::TensorOptions().dtype(at::kDouble));
            pytorch_vulkan::copy_tensor(generic_view_result, view, false);
        },
        "readback to CPU is unsupported");
    expect_error(
        [&] {
            auto generic = at::empty({3}, at::TensorOptions().dtype(at::kDouble));
            pytorch_vulkan::copy_tensor(generic, double_tensor, false);
        },
        "readback to CPU is unsupported");
    expect(platform->pending_transfer_count() == 0,
           "Double formatter presentation copy left pending resources");
}

void test_formatter_presentation_copy_rejects_malformed_sources() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    auto input = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    auto result = at::empty({2}, source.options());

    auto noncontiguous = at::as_strided(input, {2, 2}, {1, 2});
    auto strided_result = at::empty({2, 2}, source.options());
    pytorch_vulkan::formatter_presentation_copy(strided_result, noncontiguous);
    expect(strided_result.equal(at::ones({2, 2}, source.options())),
           "strided presentation copy changed values");

    auto out_of_range = input;
    out_of_range.unsafeGetTensorImpl()->set_storage_offset(3);
    out_of_range.unsafeGetTensorImpl()->set_sizes_and_strides(std::vector<int64_t>{2},
                                                              std::vector<int64_t>{1});
    expect_error(
        [&] { pytorch_vulkan::formatter_presentation_copy(result, out_of_range); },
        "undersized");
    expect(pytorch_vulkan::platform()->pending_transfer_count() == 0,
           "malformed presentation source left pending resources");
}

void test_validation_boundaries() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    auto device_tensor = at::empty({4}, source.options().device(kDevice));
    expect_error([&] { pytorch_vulkan::copy_tensor(device_tensor, source, true); },
                 "non_blocking=True");
    expect_error([&] { pytorch_vulkan::copy_tensor(source, source, false); },
                 "exactly one CPU and one Vulkan device");
    auto wrong_type = at::ones({4}, at::TensorOptions().dtype(at::kDouble));
    expect_error([&] { pytorch_vulkan::copy_tensor(device_tensor, wrong_type, false); },
                 "only float32");
    auto wrong_size = at::ones({3}, at::TensorOptions().dtype(at::kFloat));
    expect_error([&] { pytorch_vulkan::copy_tensor(device_tensor, wrong_size, false); },
                 "expanded size");
}

void test_formatter_double_storage_and_conversion() {
    const auto platform = pytorch_vulkan::platform();
    auto source = at::tensor({1.25F, -2.5F, 0.0F, 7.0F});
    if (!platform->supports_formatter_double()) {
        expect_error(
            [&] {
                (void)at::empty({4},
                                at::TensorOptions().dtype(at::kDouble).device(kDevice));
            },
            "shaderFloat64");
        return;
    }

    auto double_tensor =
        at::empty({4}, at::TensorOptions().dtype(at::kDouble).device(kDevice));
    expect(double_tensor.nbytes() == 4 * sizeof(double),
           "Double Vulkan storage does not use native sizeof(double)");
    auto input = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    auto converted = input.to(at::kDouble);
    expect(converted.device() == kDevice && converted.scalar_type() == at::kDouble &&
               converted.is_contiguous() && converted.storage_offset() == 0,
           "F32-to-Double conversion did not preserve Vulkan formatter layout");
}

void test_compute_range_rejects_before_descriptor_setup() {
    auto tensor = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const VkBuffer buffer =
        pytorch_vulkan::allocation_buffer(tensor.storage().data_ptr()).buffer();
    const auto platform = pytorch_vulkan::platform();
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, "range test");
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            platform->compute().linear(buffer, buffer, buffer, buffer, layout, layout,
                                       layout, layout,
                                       std::numeric_limits<uint32_t>::max(), 1, 1);
        },
        "invalid range");
    expect(platform->compute_dispatch_count() == before,
           "maxStorageBufferRange rejection submitted a compute dispatch");
}

void test_linear_output_count_overflow_rejects_before_dispatch() {
    auto tensor = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const VkBuffer buffer =
        pytorch_vulkan::allocation_buffer(tensor.storage().data_ptr()).buffer();
    const auto platform = pytorch_vulkan::platform();
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, "range test");
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            platform->compute().linear(buffer, buffer, buffer, buffer, layout, layout,
                                       layout, layout,
                                       std::numeric_limits<uint32_t>::max(), 1, 2);
        },
        "linear output count overflow");
    expect(platform->compute_dispatch_count() == before,
           "linear output count overflow submitted a compute dispatch");
}

void test_convolution_rejects_undersized_allocation_before_dispatch() {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(kDevice);
    auto input = at::empty({2, 1, 8, 8}, options);
    auto weight = at::empty({4, 1, 3, 3}, options);
    auto bias = at::empty({4}, options);
    input.storage().set_data_ptr(vulkan_allocator_instance()->allocate(sizeof(float)));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            (void)pytorch_vulkan::convolution(input, weight, bias, {1, 1}, {1, 1},
                                              {1, 1}, false, {0, 0}, 1);
        },
        "outside its Vulkan allocation");
    expect(platform->compute_dispatch_count() == before,
           "undersized Conv2d allocation submitted a compute dispatch");
}

void test_pooling_rejects_undersized_allocation_before_dispatch() {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(kDevice);
    auto input = at::empty({2, 4, 3, 5}, options);
    input.storage().set_data_ptr(vulkan_allocator_instance()->allocate(sizeof(float)));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error([&] { (void)pytorch_vulkan::adaptive_avg_pool2d(input, {1, 1}); },
                 "outside its Vulkan allocation");
    expect(platform->compute_dispatch_count() == before,
           "undersized pooling allocation submitted a compute dispatch");
}

void test_pooling_rejects_dimension_product_overflow_before_dispatch() {
    auto tensor = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const VkBuffer buffer =
        pytorch_vulkan::allocation_buffer(tensor.storage().data_ptr()).buffer();
    const auto platform = pytorch_vulkan::platform();
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, "range test");
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            platform->compute().pooling(buffer, buffer, layout, layout,
                                        std::numeric_limits<uint32_t>::max(), 2, 2, 2);
        },
        "truncation");
    expect(platform->compute_dispatch_count() == before,
           "overflowing pooling metadata submitted a compute dispatch");
}

void test_nonzero_storage_offset_is_supported() {
    auto source = at::ones({4}, at::TensorOptions().dtype(at::kFloat));
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    auto destination_base = at::empty({5}, source.options().device(kDevice));
    auto destination = destination_base;
    destination.unsafeGetTensorImpl()->set_storage_offset(1);
    destination.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(destination.is_contiguous() && destination.storage_offset() != 0,
           "destination test tensor is not a contiguous offset view");
    pytorch_vulkan::copy_tensor(destination, source, false);
    auto destination_result = at::empty({4}, source.options());
    pytorch_vulkan::copy_tensor(destination_result, destination, false);
    expect(destination_result.equal(source), "offset destination copy changed values");

    auto source_base = at::empty({5}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(source_base, at::ones({5}, source.options()), false);
    auto source_view = source_base;
    source_view.unsafeGetTensorImpl()->set_storage_offset(1);
    source_view.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    auto result = at::empty({4}, source.options());
    expect(source_view.is_contiguous() && source_view.storage_offset() != 0,
           "source test tensor is not a contiguous offset view");
    pytorch_vulkan::copy_tensor(result, source_view, false);
    expect(result.equal(at::ones({4}, source.options())),
           "formatter presentation copy rejected a valid source range");
}

void test_zero_tensor_copy_is_noop() {
    auto source = at::empty({0}, at::TensorOptions().dtype(at::kFloat));
    auto device_tensor = at::empty({0}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_tensor, source, false);
    auto result = at::empty({0}, source.options());
    pytorch_vulkan::copy_tensor(result, device_tensor, false);
    expect(device_tensor.device() == kDevice, "empty Vulkan tensor has wrong device");
    expect(device_tensor.numel() == 0 && device_tensor.sizes() == source.sizes(),
           "empty Vulkan tensor has wrong metadata");
    expect(result.numel() == 0 && result.scalar_type() == at::kFloat,
           "empty Vulkan round trip has wrong metadata");

    for (int index = 0; index < 32; ++index) {
        auto repeated = at::empty({0}, source.options().device(kDevice));
        expect(repeated.numel() == 0, "repeated empty Vulkan tensor is non-empty");
    }
}

void test_strided_copy_reads_and_writes_logical_indices() {
    auto source = at::arange(12, at::TensorOptions().dtype(at::kFloat))
                      .reshape({3, 4})
                      .transpose(0, 1)
                      .narrow(1, 1, 2);
    auto destination =
        at::empty_strided({4, 2}, {1, 4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(destination, source, false);
    auto result = at::empty({4, 2}, source.options());
    pytorch_vulkan::copy_tensor(result, destination, false);
    expect(result.equal(source), "strided Vulkan copy changed logical values");
}

void test_copy_broadcasts_cpu_source_into_strided_vulkan_destination() {
    auto source = at::arange(3, at::TensorOptions().dtype(at::kFloat));
    auto base = at::full({3, 2}, -7.0F,
                        at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto destination = at::transpose(base, 0, 1);
    pytorch_vulkan::copy_tensor(destination, source, false);
    auto result = at::empty({2, 3}, source.options());
    pytorch_vulkan::copy_tensor(result, destination, false);
    expect(result.equal(source.expand({2, 3})),
           "broadcast copy into strided Vulkan destination changed values");
    auto base_result = at::empty({3, 2}, source.options());
    pytorch_vulkan::copy_tensor(base_result, base, false);
    expect(base_result.equal(source.expand({2, 3}).t()),
           "broadcast copy failed to update transposed destination storage");
}

void test_strided_copy_rejects_unsafe_overlap() {
    auto source = at::empty({2, 2}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto destination =
        at::empty_strided({2, 2}, {0, 1}, source.options().device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error([&] { pytorch_vulkan::copy_tensor(destination, source, false); },
                 "internal overlap");
    expect(platform->compute_dispatch_count() == before,
           "overlapping transfer submitted compute work");

    auto storage = at::empty({5}, source.options().device(kDevice));
    auto source_view = at::as_strided(storage, {3}, {1}, 0);
    auto destination_view = at::as_strided(storage, {3}, {1}, 1);
    expect_error(
        [&] { pytorch_vulkan::copy_tensor(destination_view, source_view, false); },
        "partially overlap");
}

void test_identical_vulkan_copy_is_a_noop() {
    auto tensor = at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->pending_transfer_count();
    pytorch_vulkan::copy_tensor(tensor, tensor, false);
    expect(platform->pending_transfer_count() == before,
           "identical Vulkan copy left pending transfer resources");
}

void test_strided_copy_zero_elements_is_noop() {
    auto source =
        at::empty_strided({0, 3}, {3, 1}, at::TensorOptions().dtype(at::kFloat));
    auto destination =
        at::empty_strided({0, 3}, {1, 1}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(destination, source, false);
    expect(destination.numel() == 0, "strided zero-element copy changed size");
}

void test_zero_allocator_payload() {
    auto data = vulkan_allocator_instance()->allocate(0);
    expect(data.device() == kDevice, "empty allocation has wrong device");
    expect(data.get() == data.get_context(), "empty allocation context mismatch");
    expect(data.get() != nullptr, "empty allocation payload is null");
}

void test_foreign_payload_rejected() {
    at::DataPtr foreign(nullptr, nullptr, nullptr, kDevice);
    expect_error([&] { (void)pytorch_vulkan::allocation_buffer(foreign); },
                 "not a Vulkan allocation");
}

void test_vulkan_layout_inspection() {
    const std::vector<int64_t> offset_sizes{4};
    const std::vector<int64_t> offset_strides{1};
    auto tensor =
        at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(tensor, "layout test");
    expect(layout.storage_offset == 0 && layout.numel == 6,
           "contiguous Vulkan layout has wrong element metadata");
    expect(layout.byte_offset == 0 && layout.byte_range == 6 * sizeof(float),
           "contiguous Vulkan layout has wrong byte range");
    expect(layout.allocation_bytes >= layout.byte_range,
           "contiguous Vulkan layout has too-small allocation");

    auto empty = at::empty({0, 3}, tensor.options());
    const auto empty_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(empty, "layout test");
    expect(empty_layout.numel == 0 && empty_layout.byte_range == 0 &&
               empty_layout.allocation_bytes == 0,
           "empty Vulkan layout requires a buffer or has a nonzero byte range");
    auto foreign_empty = empty;
    foreign_empty.storage().set_data_ptr(
        at::DataPtr(nullptr, nullptr, nullptr, kDevice));
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_tensor_layout(foreign_empty,
                                                               "layout test");
        },
        "invalid allocation payload");

    auto offset_base = at::empty({5}, tensor.options());
    auto offset = offset_base;
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes, offset_strides);
    const auto offset_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(offset, "layout test");
    expect(offset_layout.byte_offset == sizeof(float) &&
               offset_layout.byte_range == 4 * sizeof(float),
           "offset Vulkan layout has wrong byte range");

    auto foreign = tensor;
    foreign.storage().set_data_ptr(at::DataPtr(nullptr, nullptr, nullptr, kDevice));
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_tensor_layout(foreign, "layout test");
        },
        "invalid allocation payload");

    auto out_of_range = offset_base;
    out_of_range.unsafeGetTensorImpl()->set_storage_offset(2);
    out_of_range.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes,
                                                              offset_strides);
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_tensor_layout(out_of_range,
                                                               "layout test");
        },
        "outside its Vulkan allocation");

    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(offset_base, {2}, {-1}, 0,
                                                             "layout test");
        },
        "negative stride");
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(
                offset_base, {2}, {std::numeric_limits<int64_t>::max()}, 1,
                "layout test");
        },
        "overflow");
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(
                offset_base, {1}, {0}, std::numeric_limits<int64_t>::max(),
                "layout test");
        },
        "overflow");
}

void test_vulkan_layout_descriptor_and_index_mapping() {
    auto tensor =
        at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto transposed = at::as_strided(tensor, {3, 2}, {1, 3}, 0);
    const auto layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(transposed, "descriptor test");
    expect(layout.rank == 2 && layout.sizes == std::vector<int64_t>({3, 2}) &&
               layout.strides == std::vector<int64_t>({1, 3}) &&
               layout.element_bytes == sizeof(float) &&
               layout.internal_overlap == pytorch_vulkan::VulkanOverlap::No,
           "Vulkan layout descriptor lost shape, stride, or overlap metadata");
    expect(pytorch_vulkan::vulkan_storage_offset(layout, {0, 0}) == 0 &&
               pytorch_vulkan::vulkan_storage_offset(layout, {2, 1}) == 5 &&
               pytorch_vulkan::vulkan_storage_offset(layout, 5) == 5,
           "Vulkan layout descriptor mapped logical indices incorrectly");

    const auto broadcast = at::as_strided(tensor, {2, 3}, {0, 1}, 0);
    const auto broadcast_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(broadcast, "descriptor test");
    expect(broadcast_layout.internal_overlap == pytorch_vulkan::VulkanOverlap::Yes,
           "zero-stride Vulkan view was not classified as internally overlapping");
    expect(pytorch_vulkan::vulkan_storage_offset(broadcast_layout, 5) == 2,
           "zero-stride Vulkan view mapped its linear index incorrectly");

    expect_error([&] { (void)pytorch_vulkan::vulkan_storage_offset(layout, {3, 0}); },
                 "coordinate");
    expect_error([&] { (void)pytorch_vulkan::vulkan_storage_offset(layout, 6); },
                 "linear index");

    auto overlapping = at::empty_strided({3, 3}, {2, 4}, tensor.options());
    const auto overlapping_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(overlapping, "descriptor test");
    expect(overlapping_layout.internal_overlap != pytorch_vulkan::VulkanOverlap::No,
           "non-unit-stride repeated addresses were classified as non-overlapping");
}

void test_empty_reductions_stay_on_vulkan() {
    auto input =
        at::empty({0, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto sum = pytorch_vulkan::sum_tensor(input, 0, false, c10::nullopt);
    auto mean = pytorch_vulkan::mean_tensor(input, 0, false, c10::nullopt);
    expect(sum.device() == kDevice && mean.device() == kDevice,
           "empty reductions materialized their result on CPU");
    auto sum_cpu = at::empty_like(sum, sum.options().device(c10::kCPU));
    auto mean_cpu = at::empty_like(mean, mean.options().device(c10::kCPU));
    pytorch_vulkan::copy_tensor(sum_cpu, sum, false);
    pytorch_vulkan::copy_tensor(mean_cpu, mean, false);
    expect(sum_cpu.equal(at::zeros_like(sum_cpu)) &&
               at::isnan(mean_cpu).all().item<bool>(),
           "empty Vulkan reductions produced incorrect identity values");
}

void test_empty_reductions_validate_input_layout() {
    auto input = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto foreign = input;
    foreign.storage().set_data_ptr(at::DataPtr(nullptr, nullptr, nullptr, kDevice));
    expect_error(
        [&] { (void)pytorch_vulkan::sum_tensor(foreign, 0, false, c10::nullopt); }, "");

    auto out_of_range = input;
    out_of_range.unsafeGetTensorImpl()->set_storage_offset(1);
    out_of_range.unsafeGetTensorImpl()->set_sizes_and_strides(std::vector<int64_t>{0},
                                                              std::vector<int64_t>{1});
    expect_error(
        [&] {
            (void)pytorch_vulkan::mean_tensor(out_of_range, 0, false, c10::nullopt);
        },
        "");
}

void test_vulkan_layout_address_rejects_storage_offset_overflow() {
    const pytorch_vulkan::VulkanTensorLayout layout{
        1,
        {2},
        {1},
        static_cast<int>(at::kBool),
        1,
        std::numeric_limits<int64_t>::max(),
        2,
        static_cast<VkDeviceSize>(std::numeric_limits<int64_t>::max()),
        2,
        std::numeric_limits<VkDeviceSize>::max(),
        pytorch_vulkan::VulkanOverlap::No};
    expect_error([&] { (void)pytorch_vulkan::vulkan_storage_offset(layout, {1}); },
                 "address exceeds int64 range");
}

void test_vulkan_layout_rejects_shader_address_overflow() {
    auto input = at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const int64_t max_uint32 = std::numeric_limits<uint32_t>::max();

    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(input, {2}, {max_uint32},
                                                             1, "shader address");
        },
        "shader address");
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(
                input, {2, 2},
                {static_cast<int64_t>(uint32_t{0x80000000}),
                 static_cast<int64_t>(uint32_t{0x80000000})},
                0, "shader address");
        },
        "shader address");
}

void test_empty_vulkan_view_checks_allocation_boundary() {
    auto input = at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto valid = pytorch_vulkan::inspect_vulkan_view_layout(input, {0}, {1}, 4,
                                                                  "empty boundary");
    expect(valid.numel == 0 && valid.byte_range == 0 &&
               valid.byte_offset == valid.allocation_bytes,
           "empty Vulkan view at allocation boundary was rejected");
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(input, {0}, {1}, 5,
                                                             "empty boundary");
        },
        "empty boundary");
}

void test_empty_vulkan_view_does_not_use_shader_address_limit() {
    auto input = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    expect_error(
        [&] {
            (void)pytorch_vulkan::inspect_vulkan_view_layout(
                input, {0}, {1},
                static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1,
                "empty address");
        },
        "outside its Vulkan allocation");
}

void test_metadata_only_views() {
    auto source = at::tensor({0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
    auto input = at::empty({2, 3}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source.reshape({2, 3}), false);
    const auto source_sizes = input.sizes().vec();
    const auto source_strides = input.strides().vec();
    const auto platform = pytorch_vulkan::platform();

    const auto expect_view = [&](const at::Tensor &view, const char *message) {
        expect(view.data_ptr() == input.data_ptr() && view.is_contiguous(), message);
        expect(view.sizes().equals({6}) && view.strides().equals({1}),
               "Vulkan metadata-only view has wrong metadata");
        auto result = at::empty({6}, source.options());
        pytorch_vulkan::copy_tensor(result, view, false);
        expect(result.equal(source), "Vulkan metadata-only view changed values");
        expect(input.sizes().equals(source_sizes) &&
                   input.strides().equals(source_strides),
               "Vulkan metadata-only view changed its source metadata");
    };

    const std::size_t before = platform->compute_dispatch_count();
    expect_view(at::as_strided(input, {6}, {1}),
                "Vulkan as_strided did not preserve contiguous storage identity");
    expect_view(input.view({6}),
                "Vulkan view did not preserve contiguous storage identity");
    expect_view(at::_reshape_alias(input, {6}, {1}),
                "Vulkan _reshape_alias did not preserve contiguous storage identity");
    expect(platform->compute_dispatch_count() == before,
           "metadata-only view submitted compute work");

    const auto expect_metadata = [&](const at::Tensor &view, at::IntArrayRef sizes,
                                     at::IntArrayRef strides, int64_t offset) {
        expect(view.storage().data_ptr().get() == input.storage().data_ptr().get() &&
                   view.sizes().equals(sizes) && view.strides().equals(strides) &&
                   view.storage_offset() == offset,
               "Vulkan as_strided did not preserve general metadata");
    };
    expect_metadata(at::as_strided(input, {2}, {2}), {2}, {2}, 0);
    expect_metadata(at::as_strided(input, {2, 3}, {1, 2}), {2, 3}, {1, 2}, 0);
    expect_metadata(at::as_strided(input, {2, 3}, {0, 1}), {2, 3}, {0, 1}, 0);
    expect_metadata(at::as_strided(input, {2}, {1}, 1), {2}, {1}, 1);

    const auto expect_rejected = [&](const std::function<void()> &operation,
                                     const char *message) {
        const std::size_t dispatches = platform->compute_dispatch_count();
        (void)message;
        expect_error(operation, "");
        expect(platform->compute_dispatch_count() == dispatches,
               "invalid metadata-only view submitted compute work");
    };
    expect_rejected([&] { (void)at::as_strided(input, {6}, {1}, 1); },
                    "storage offset");
    expect_rejected([&] { (void)input.view({5}); }, "invalid for input");
    expect_rejected([&] { (void)at::as_strided(input, {2, 3}, {1, 3}); },
                    "Vulkan as_strided");
    expect_rejected([&] { (void)at::as_strided(input, {2, 3}, {-1, 1}); },
                    "Vulkan as_strided");
}

void test_bmm_out_rejects_input_alias_before_dispatch() {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(kDevice);
    auto lhs = at::empty({1, 2, 2}, options);
    auto rhs = at::empty({1, 2, 2}, options);
    const auto platform = pytorch_vulkan::platform();
    platform->reset_execution_counters();

    expect_error([&] { (void)pytorch_vulkan::bmm_out(lhs, rhs, lhs); },
                 "output may not alias an input");
    expect(platform->compute_dispatch_count() == 0,
           "aliased bmm output submitted compute work");
    expect(platform->vulkan_copy_count() == 0,
           "aliased bmm output submitted a copy");
    expect(platform->explicit_transfer_count() == 0,
           "aliased bmm output submitted a transfer");
    expect(platform->execution_counter_snapshot().fallbacks == 0,
           "aliased bmm output used a fallback");
}

void test_repeated_add_dispatch_and_retained_output() {
    auto lhs = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto rhs = at::tensor({10.0F, 20.0F, 30.0F, 40.0F});
    auto retained = pytorch_vulkan::add_tensor(
        at::empty_like(lhs, lhs.options().device(kDevice)),
        at::empty_like(rhs, rhs.options().device(kDevice)), 1.0F);

    pytorch_vulkan::copy_tensor(retained, lhs, false);
    auto device_rhs = at::empty_like(rhs, rhs.options().device(kDevice));
    pytorch_vulkan::copy_tensor(device_rhs, rhs, false);
    retained = pytorch_vulkan::add_tensor(retained, device_rhs, 1.0F);

    for (int iteration = 0; iteration < 16; ++iteration) {
        auto device_lhs = at::empty_like(lhs, lhs.options().device(kDevice));
        auto device_addend = at::empty_like(rhs, rhs.options().device(kDevice));
        pytorch_vulkan::copy_tensor(device_lhs, lhs, false);
        pytorch_vulkan::copy_tensor(device_addend, rhs, false);
        auto output = pytorch_vulkan::add_tensor(device_lhs, device_addend, 1.0F);
        auto result = at::empty_like(lhs);
        pytorch_vulkan::copy_tensor(result, output, false);
        expect(result.equal(at::tensor({11.0F, 22.0F, 33.0F, 44.0F})),
               "repeated Vulkan add changed data");
    }

    lhs = at::Tensor();
    rhs = at::Tensor();
    device_rhs = at::Tensor();
    auto retained_result = at::empty({4}, at::TensorOptions().dtype(at::kFloat));
    pytorch_vulkan::copy_tensor(retained_result, retained, false);
    expect(retained_result.equal(at::tensor({11.0F, 22.0F, 33.0F, 44.0F})),
           "retained Vulkan add output changed after input release");
}

void test_tensor_tensor_sub_and_mul_dispatch() {
    auto lhs_source = at::tensor({1.0F, -2.0F, 3.0F});
    auto rhs_source = at::tensor({4.0F, 5.0F, -6.0F});
    auto lhs = at::empty_like(lhs_source, lhs_source.options().device(kDevice));
    auto rhs = at::empty_like(rhs_source, rhs_source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::copy_tensor(rhs, rhs_source, false);

    auto sub = pytorch_vulkan::pointwise_tensor_operands(
        lhs, rhs, 1.0F, pytorch_vulkan::PointwiseOperation::Sub, "sub");
    auto mul = pytorch_vulkan::pointwise_tensor_operands(
        lhs, rhs, 1.0F, pytorch_vulkan::PointwiseOperation::Mul, "mul");
    auto sub_result = at::empty_like(lhs_source);
    auto mul_result = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(sub_result, sub, false);
    pytorch_vulkan::copy_tensor(mul_result, mul, false);

    expect(sub_result.equal(at::tensor({-3.0F, -7.0F, 9.0F})),
           "Vulkan tensor/tensor sub changed data");
    expect(mul_result.equal(at::tensor({4.0F, -10.0F, -18.0F})),
           "Vulkan tensor/tensor mul changed data");
    auto lhs_result = at::empty_like(lhs_source);
    auto rhs_result = at::empty_like(rhs_source);
    pytorch_vulkan::copy_tensor(lhs_result, lhs, false);
    pytorch_vulkan::copy_tensor(rhs_result, rhs, false);
    expect(lhs_result.equal(lhs_source) && rhs_result.equal(rhs_source),
           "Vulkan tensor/tensor arithmetic changed inputs");
    expect(sub.data_ptr() != lhs.data_ptr() && sub.data_ptr() != rhs.data_ptr() &&
               mul.data_ptr() != lhs.data_ptr() && mul.data_ptr() != rhs.data_ptr() &&
               sub.data_ptr() != mul.data_ptr(),
           "Vulkan tensor/tensor arithmetic aliased storage");
}

void test_zero_element_add_does_not_dispatch() {
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    auto lhs = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto rhs = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto output = pytorch_vulkan::add_tensor(lhs, rhs, 1.0F);
    expect(output.numel() == 0, "zero-element Vulkan add returned non-empty output");
    expect(platform->compute_dispatch_count() == before,
           "zero-element Vulkan add submitted a compute dispatch");
}

void test_scalar_add_dispatch_modes_and_lifecycle() {
    auto source = at::tensor({1.0F, -2.0F, 3.5F});
    auto tensor = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(tensor, source, false);
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();

    auto tensor_scalar = at::add(tensor, at::Scalar(2.5F));
    auto tensor_scalar_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(tensor_scalar_result, tensor_scalar, false);

    expect(tensor_scalar_result.equal(at::tensor({3.5F, 0.5F, 6.0F})),
           "tensor/scalar Vulkan add changed data");
    expect(platform->compute_dispatch_count() - before == 1,
           "scalar Vulkan add lost a dispatch count");
    expect(platform->pending_transfer_count() == 0,
           "scalar Vulkan add left pending transfer resources");
}

void test_zero_element_scalar_add_does_not_dispatch() {
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    auto tensor =
        at::empty({0, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto output = at::add(tensor, at::Scalar(1.25F));
    expect(output.numel() == 0 && output.sizes() == tensor.sizes(),
           "zero-element scalar Vulkan add returned wrong metadata");
    expect(platform->compute_dispatch_count() == before,
           "zero-element scalar Vulkan add submitted a compute dispatch");
    expect(platform->pending_transfer_count() == 0,
           "zero-element scalar Vulkan add left pending transfer resources");
}

void test_formatter_comparison_dispatch_is_counted() {
    auto source = at::tensor({-1.0F, 0.0F, 1.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    auto result = at::empty_like(input, input.options().dtype(at::kBool));
    pytorch_vulkan::isfinite_out(input, result);
    expect(result.sizes() == input.sizes(),
           "formatter comparison returned wrong metadata");
    expect(platform->compute_dispatch_count() - before == 1,
           "formatter comparison did not submit exactly one dispatch");
}

void test_scalar_pointwise_offset_is_supported() {
    auto source = at::tensor({2.0F, 3.0F, 4.0F, 5.0F});
    auto base = at::empty({5}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(base, at::tensor({0.0F, 2.0F, 3.0F, 4.0F, 5.0F}),
                                false);
    auto offset = base;
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(offset.is_contiguous() && offset.storage_offset() != 0,
           "scalar offset test tensor is not a contiguous offset view");

    for (const auto &[result, expected] :
         std::vector<std::pair<at::Tensor, at::Tensor>>{
             {at::add(offset, at::Scalar(1.0F)), at::tensor({3.0F, 4.0F, 5.0F, 6.0F})},
             {at::sub(offset, at::Scalar(1.0F)), at::tensor({1.0F, 2.0F, 3.0F, 4.0F})},
             {at::mul(offset, at::Scalar(2.0F)),
              at::tensor({4.0F, 6.0F, 8.0F, 10.0F})}}) {
        auto cpu_result = at::empty_like(expected);
        pytorch_vulkan::copy_tensor(cpu_result, result, false);
        expect(cpu_result.equal(expected), "scalar offset pointwise changed values");
    }
}

void test_concurrent_add_dispatches_are_serialized() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F, 4.0F});
    auto lhs = at::empty({4}, source.options().device(kDevice));
    auto rhs = at::empty({4}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(lhs, source, false);
    pytorch_vulkan::copy_tensor(rhs, source, false);

    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    constexpr int kThreads = 8;
    constexpr int kOperationsPerThread = 8;
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (int thread = 0; thread < kThreads; ++thread) {
        workers.emplace_back([&] {
            try {
                for (int operation = 0; operation < kOperationsPerThread; ++operation) {
                    auto output = pytorch_vulkan::add_tensor(lhs, rhs, 1.0F);
                    auto result = at::empty_like(source);
                    pytorch_vulkan::copy_tensor(result, output, false);
                    expect(result.equal(at::tensor({2.0F, 4.0F, 6.0F, 8.0F})),
                           "concurrent Vulkan add changed data");
                }
            } catch (...) {
                failed.store(true);
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    expect(!failed.load(), "concurrent Vulkan add failed");
    expect(platform->compute_dispatch_count() - before ==
               static_cast<std::size_t>(kThreads * kOperationsPerThread),
           "concurrent Vulkan add lost a dispatch count");
}

void test_add_invalid_input_cleans_up() {
    const std::vector<int64_t> rank_nine(9, 1);
    auto lhs =
        at::empty(rank_nine, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto rhs = at::empty_like(lhs);
    expect_error([&] { (void)pytorch_vulkan::add_tensor(lhs, rhs, 1.0F); },
                 "ranks up to 8");
    expect(pytorch_vulkan::platform()->pending_transfer_count() == 0,
           "invalid Vulkan add left pending transfer resources");
}

void test_repeated_unary_dispatch_and_input_readability() {
    auto source = at::tensor({-3.5F, 0.0F, 2.25F, -1.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);
    const auto expected = at::neg(source);

    for (int iteration = 0; iteration < 16; ++iteration) {
        auto output = at::neg(input);
        expect(output.data_ptr() != input.data_ptr(),
               "Vulkan unary output aliased its input");
        auto result = at::empty_like(source);
        pytorch_vulkan::copy_tensor(result, output, false);
        expect(result.equal(expected), "repeated Vulkan unary changed data");
    }

    auto input_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(input_result, input, false);
    expect(input_result.equal(source), "Vulkan unary changed its input");
}

void test_unary_nonzero_storage_offset_is_supported() {
    auto source = at::tensor({2.0F, 3.0F, 4.0F, 5.0F});
    auto base = at::empty({5}, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(base, at::tensor({0.0F, 2.0F, 3.0F, 4.0F, 5.0F}),
                                false);
    auto offset = base;
    const std::vector<int64_t> sizes{4};
    const std::vector<int64_t> strides{1};
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    expect(offset.is_contiguous() && offset.storage_offset() != 0,
           "unary offset test tensor is not a contiguous offset view");

    auto result = at::neg(offset);
    auto cpu_result = at::empty_like(source);
    pytorch_vulkan::copy_tensor(cpu_result, result, false);
    expect(cpu_result.equal(at::tensor({-2.0F, -3.0F, -4.0F, -5.0F})),
           "unary offset dispatch changed values");
}

void test_shared_out_dispatch_helpers() {
    auto source = at::tensor({-3.0F, 2.0F, 4.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);

    auto unary_out = at::empty({1}, source.options().device(kDevice));
    auto *unary_impl = unary_out.unsafeGetTensorImpl();
    auto &unary_result = pytorch_vulkan::dispatch_unary_out(
        input, unary_out, pytorch_vulkan::PointwiseOperation::Neg, "neg");
    expect(&unary_result == &unary_out && unary_out.unsafeGetTensorImpl() == unary_impl,
           "unary out did not preserve tensor identity");
    auto unary_cpu = at::empty_like(source);
    pytorch_vulkan::copy_tensor(unary_cpu, unary_out, false);
    expect(unary_cpu.equal(at::tensor({3.0F, -2.0F, -4.0F})),
           "unary out helper changed values");

    auto scalar_out = at::empty({1}, source.options().device(kDevice));
    auto &scalar_result = pytorch_vulkan::dispatch_tensor_scalar_out(
        input, at::Scalar(1.5F), at::Scalar(1.0F), scalar_out,
        pytorch_vulkan::PointwiseOperation::Add, "add");
    expect(&scalar_result == &scalar_out && scalar_out.sizes() == source.sizes(),
           "tensor scalar out helper did not resize in place");
    auto scalar_cpu = at::empty_like(source);
    pytorch_vulkan::copy_tensor(scalar_cpu, scalar_out, false);
    expect(scalar_cpu.equal(at::tensor({-1.5F, 3.5F, 5.5F})),
           "tensor scalar out helper changed values");

    auto rhs = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(rhs, at::tensor({1.0F, 3.0F, 5.0F}), false);
    auto binary_out = at::empty({1}, source.options().device(kDevice));
    auto &binary_result = pytorch_vulkan::dispatch_tensor_tensor_out(
        input, rhs, at::Scalar(1.0F), binary_out,
        pytorch_vulkan::PointwiseOperation::Sub, "sub");
    expect(&binary_result == &binary_out && binary_out.sizes() == source.sizes(),
           "tensor tensor out helper did not preserve identity or resize");
    auto binary_cpu = at::empty_like(source);
    pytorch_vulkan::copy_tensor(binary_cpu, binary_out, false);
    expect(binary_cpu.equal(at::tensor({-4.0F, -1.0F, -1.0F})),
           "tensor tensor out helper changed values");
}

void test_shared_out_empty_path_does_not_dispatch() {
    auto input =
        at::empty({0, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto out = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    pytorch_vulkan::dispatch_unary_out(input, out,
                                       pytorch_vulkan::PointwiseOperation::Neg, "neg");
    expect(out.sizes() == input.sizes() && platform->compute_dispatch_count() == before,
           "empty unary out path dispatched or kept the wrong shape");
}

void test_shared_out_rejects_offset_and_noncontiguous_output() {
    auto source = at::tensor({1.0F, 2.0F, 3.0F});
    auto input = at::empty_like(source, source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, source, false);

    auto offset = at::empty({4}, source.options().device(kDevice));
    offset.unsafeGetTensorImpl()->set_storage_offset(1);
    const std::vector<int64_t> offset_sizes{3};
    const std::vector<int64_t> offset_strides{1};
    offset.unsafeGetTensorImpl()->set_sizes_and_strides(offset_sizes, offset_strides);
    pytorch_vulkan::dispatch_unary_out(input, offset,
                                       pytorch_vulkan::PointwiseOperation::Neg, "neg");

    auto noncontiguous =
        at::empty_strided({3, 2}, {1, 3}, source.options().device(kDevice));
    auto matrix_input = at::empty({3, 2}, source.options().device(kDevice));
    pytorch_vulkan::dispatch_unary_out(matrix_input, noncontiguous,
                                       pytorch_vulkan::PointwiseOperation::Neg, "neg");

    auto alias = input;
    auto &alias_result = pytorch_vulkan::dispatch_unary_out(
        input, alias, pytorch_vulkan::PointwiseOperation::Neg, "neg");
    expect(&alias_result == &alias &&
               alias.unsafeGetTensorImpl() == input.unsafeGetTensorImpl(),
           "exact full-tensor output alias was not permitted");
}

void test_shared_out_rejects_partial_and_internal_overlap() {
    auto values = at::tensor({1.0F, 2.0F, 3.0F});
    auto input = at::empty_like(values, values.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, values, false);

    auto storage = at::empty({5}, values.options().device(kDevice));
    auto partial_input = storage.detach();
    partial_input.unsafeGetTensorImpl()->set_sizes_and_strides(std::vector<int64_t>{3},
                                                               std::vector<int64_t>{1});
    auto rhs = at::empty_like(values, values.options().device(kDevice));
    auto partial_output = storage;
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();
    expect_error(
        [&] {
            pytorch_vulkan::dispatch_tensor_tensor_out(
                partial_input, rhs, at::Scalar(1.0F), partial_output,
                pytorch_vulkan::PointwiseOperation::Add, "add");
        },
        "partially overlaps an input");
    expect(platform->compute_dispatch_count() == before,
           "partial overlap rejection submitted a compute dispatch");

    auto internal_output =
        at::empty_strided({3}, {0}, values.options().device(kDevice));
    expect_error(
        [&] {
            pytorch_vulkan::dispatch_tensor_tensor_out(
                input, rhs, at::Scalar(1.0F), internal_output,
                pytorch_vulkan::PointwiseOperation::Add, "add");
        },
        "internal overlap");
    expect(platform->compute_dispatch_count() == before,
           "internal overlap rejection submitted a compute dispatch");
}

void test_inplace_operations_and_rejections_preserve_contracts() {
    auto values = at::tensor({1.0F, -2.0F, 3.0F});
    auto input = at::empty_like(values, values.options().device(kDevice));
    auto other = at::empty_like(values, values.options().device(kDevice));
    pytorch_vulkan::copy_tensor(input, values, false);
    pytorch_vulkan::copy_tensor(other, values, false);
    const auto platform = pytorch_vulkan::platform();
    auto result = at::empty_like(values);

    const auto expect_mutation = [&](const std::function<void()> &operation,
                                     const at::Tensor &expected,
                                     const char *message) {
        pytorch_vulkan::copy_tensor(input, values, false);
        const std::size_t dispatches = platform->compute_dispatch_count();
        operation();
        expect(platform->compute_dispatch_count() == dispatches + 1,
               "supported in-place operation did not submit one compute dispatch");
        pytorch_vulkan::copy_tensor(result, input, false);
        expect(result.equal(expected), message);
    };

    expect_mutation([&] { input.add_(other); }, at::tensor({2.0F, -4.0F, 6.0F}),
                    "in-place add produced wrong values");
    expect_mutation([&] { input.add_(1.0F); }, at::tensor({2.0F, -1.0F, 4.0F}),
                    "in-place scalar add produced wrong values");
    expect_mutation([&] { input.sub_(other); }, at::tensor({0.0F, 0.0F, 0.0F}),
                    "in-place sub produced wrong values");
    expect_mutation([&] { input.sub_(1.0F); }, at::tensor({0.0F, -3.0F, 2.0F}),
                    "in-place scalar sub produced wrong values");
    expect_mutation([&] { input.mul_(other); }, at::tensor({1.0F, 4.0F, 9.0F}),
                    "in-place mul produced wrong values");
    expect_mutation([&] { input.mul_(2.0F); }, at::tensor({2.0F, -4.0F, 6.0F}),
                    "in-place scalar mul produced wrong values");

    const auto expect_unchanged = [&](const std::function<void()> &operation,
                                      const char *message) {
        pytorch_vulkan::copy_tensor(input, values, false);
        const std::size_t dispatches = platform->compute_dispatch_count();
        expect_error(operation, "in-place");
        expect(platform->compute_dispatch_count() == dispatches, message);
        pytorch_vulkan::copy_tensor(result, input, false);
        expect(result.equal(values), "in-place rejection mutated its input");
    };
    expect_unchanged([&] { input.neg_(); },
                     "in-place neg rejection submitted a compute dispatch");
    expect_unchanged([&] { input.abs_(); },
                     "in-place abs rejection submitted a compute dispatch");
    expect_unchanged([&] { input.relu_(); },
                     "in-place relu rejection submitted a compute dispatch");
}

void test_shared_out_exact_aliases_preserve_values_and_lifecycles() {
    auto lhs_source = at::tensor({1.0F, 2.0F, 3.0F});
    auto rhs_source = at::tensor({10.0F, 20.0F, 30.0F});
    auto lhs = at::empty_like(lhs_source, lhs_source.options().device(kDevice));
    auto rhs = at::empty_like(rhs_source, rhs_source.options().device(kDevice));
    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::copy_tensor(rhs, rhs_source, false);
    const auto platform = pytorch_vulkan::platform();
    const std::size_t before = platform->compute_dispatch_count();

    pytorch_vulkan::dispatch_tensor_tensor_out(lhs, rhs, at::Scalar(1.0F), lhs,
                                               pytorch_vulkan::PointwiseOperation::Add,
                                               "add");
    auto lhs_result = at::empty_like(lhs_source);
    auto rhs_after_lhs = at::empty_like(rhs_source);
    pytorch_vulkan::copy_tensor(lhs_result, lhs, false);
    pytorch_vulkan::copy_tensor(rhs_after_lhs, rhs, false);
    expect(lhs_result.equal(at::tensor({11.0F, 22.0F, 33.0F})) &&
               rhs_after_lhs.equal(rhs_source),
           "binary out=lhs alias produced wrong values or changed rhs");

    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::dispatch_tensor_tensor_out(lhs, rhs, at::Scalar(1.0F), rhs,
                                               pytorch_vulkan::PointwiseOperation::Sub,
                                               "sub");
    auto rhs_result = at::empty_like(rhs_source);
    auto lhs_after_rhs = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(rhs_result, rhs, false);
    pytorch_vulkan::copy_tensor(lhs_after_rhs, lhs, false);
    expect(rhs_result.equal(at::tensor({-9.0F, -18.0F, -27.0F})) &&
               lhs_after_rhs.equal(lhs_source),
           "binary out=rhs alias produced wrong values or changed lhs");

    pytorch_vulkan::dispatch_tensor_scalar_out(
        lhs, at::Scalar(2.0F), at::Scalar(1.0F), lhs,
        pytorch_vulkan::PointwiseOperation::Mul, "mul");
    auto scalar_result = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(scalar_result, lhs, false);
    expect(scalar_result.equal(at::tensor({2.0F, 4.0F, 6.0F})),
           "tensor/scalar exact alias produced wrong values");
    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::dispatch_tensor_scalar_out(
        lhs, at::Scalar(2.0F), at::Scalar(1.0F), lhs,
        pytorch_vulkan::PointwiseOperation::Add, "add", true);
    pytorch_vulkan::copy_tensor(scalar_result, lhs, false);
    expect(scalar_result.equal(at::tensor({3.0F, 4.0F, 5.0F})),
           "scalar-left add exact alias produced wrong values");

    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    pytorch_vulkan::dispatch_tensor_scalar_out(
        lhs, at::Scalar(2.0F), at::Scalar(1.0F), lhs,
        pytorch_vulkan::PointwiseOperation::Sub, "rsub", true);
    pytorch_vulkan::copy_tensor(scalar_result, lhs, false);
    expect(scalar_result.equal(at::tensor({1.0F, 0.0F, -1.0F})),
           "scalar-left rsub exact alias produced wrong values");
    expect(platform->compute_dispatch_count() - before == 5,
           "exact alias dispatches were not counted");
    expect(platform->pending_transfer_count() == 0,
           "exact alias dispatch left pending transfer resources");

    pytorch_vulkan::copy_tensor(lhs, lhs_source, false);
    platform->compute().unary_alias(
        pytorch_vulkan::allocation_buffer(lhs.storage().data_ptr()).buffer(),
        pytorch_vulkan::inspect_vulkan_tensor_layout(lhs, "direct unary input"),
        pytorch_vulkan::allocation_buffer(lhs.storage().data_ptr()).buffer(),
        pytorch_vulkan::inspect_vulkan_tensor_layout(lhs, "direct unary output"),
        static_cast<uint32_t>(pytorch_vulkan::PointwiseOperation::Neg));
    auto direct_alias_result = at::empty_like(lhs_source);
    pytorch_vulkan::copy_tensor(direct_alias_result, lhs, false);
    expect(direct_alias_result.equal(at::tensor({-1.0F, -2.0F, -3.0F})),
           "direct unary exact alias dispatch produced wrong values");
}

void test_shared_out_device_index_is_rejected_by_native_allocator() {
    expect_error(
        [&] {
            (void)at::empty({3},
                            at::TensorOptions()
                                .dtype(at::kFloat)
                                .device(c10::Device(c10::DeviceType::PrivateUse1, 1)));
        },
        "device index 0");
}

void test_platform_destruction_is_nothrow() {
    static_assert(std::is_nothrow_destructible<VulkanPlatform>::value,
                  "Vulkan cleanup must not throw during ownership quarantine");
}

void test_reduction_indexing_reject_malformed_metadata() {
    auto input =
        at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat).device(kDevice));
    auto output = at::empty({2, 1}, input.options());
    const auto input_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(input, "test input");
    const auto output_layout =
        pytorch_vulkan::inspect_vulkan_tensor_layout(output, "test output");
    auto input_buffer =
        pytorch_vulkan::allocation_buffer(input.storage().data_ptr()).buffer();
    auto output_buffer =
        pytorch_vulkan::allocation_buffer(output.storage().data_ptr()).buffer();
    auto platform = pytorch_vulkan::platform();

    auto missing_sizes = input_layout;
    missing_sizes.sizes.pop_back();
    expect_error(
        [&] {
            platform->compute().reduction(input_buffer, missing_sizes, output_buffer,
                                          output_layout, 2U, 3U, 2U, false);
        },
        "metadata lengths");

    auto rank_nine = input_layout;
    rank_nine.rank = 9;
    rank_nine.sizes.resize(9, 1);
    rank_nine.strides.resize(9, 1);
    expect_error(
        [&] {
            platform->compute().argmax(input_buffer, rank_nine, output_buffer,
                                       output_layout, 0U, 1U, 1U);
        },
        "ranks up to 8");

    auto missing_output = output_layout;
    missing_output.strides.pop_back();
    expect_error(
        [&] {
            platform->compute().broadcast(input_buffer, input_layout, output_buffer,
                                          missing_output, 2U, 1.0F);
        },
        "output layout metadata lengths");
}

} // namespace

int main(int argc, char **argv) {
    NativePlatformShutdown shutdown;
    try {
        shutdown.observe(pytorch_vulkan::platform());
        if (argc == 2 && std::string(argv[1]) == "--retention-only") {
            // RED/GREEN of the unchanged stronger lifetime body, but always
            // through the repaired pre-libc shutdown boundary above.
            run_async_witness("named-request-retention", "HS", test_async_allocation_lifetimes);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--ordinary-error-exit") {
            test_async_pinned_offsets_and_publication();
            const auto platform = pytorch_vulkan::platform();
            platform->synchronize_execution();
            expect(platform->query_execution() &&
                       platform->completion_hold_expiry_count_for_testing() == 0 &&
                       platform->live_resource_snapshot().pending_transfers == 0,
                   "ordinary error-exit fixture was not released and synchronized");
            throw std::runtime_error("intentional ordinary native error after hold release and synchronize");
        }
        expect(argc == 1, "unknown native transfer test arguments");
        test_async_copy_rejects_empty_request();
        run_async_witness("pinned-offsets-publication-hold-selftest", "HS", test_async_pinned_offsets_and_publication);
        run_async_witness("noncoherent-success-real-queue", "S+HS", test_async_synthetic_maintenance_real_queue);
        run_async_witness("maintenance-validation-snapshot-cancel", "S", test_async_maintenance_failures);
        run_async_witness("D2D-chain-slots-17-bound", "HS", test_async_d2d_batching_and_slots);
        run_async_witness("D2D-unsubmitted-shutdown-leases", "H", test_async_d2d_shutdown_drains_unsubmitted_leases);
        run_async_witness("compute-and-WAR-order", "HS", test_async_compute_and_war_order);
        run_async_witness("two-slots-backpressure-5s-poll-5s-join", "HS", test_async_backpressure);
        run_async_witness("pinned-device-D2D-lifetimes", "HS", test_async_allocation_lifetimes);
        run_async_witness("pageable-snapshot-training-sync-modes", "HS+H", test_async_pageable_snapshot_and_modes);
        test_copy_round_trip();
        test_copy_returns_without_pending_transfer_resources();
        test_contiguous_reshape_copy_is_bulk();
        test_execution_counters_reset_and_read_stably();
        test_vulkan_to_vulkan_copy_is_counted_once();
        test_formatter_presentation_copy_reads_exact_range_and_waits();
        test_formatter_presentation_copy_preserves_double_and_rejects_general_readback();
        test_formatter_presentation_copy_rejects_malformed_sources();
        test_validation_boundaries();
        test_formatter_double_storage_and_conversion();
        test_compute_range_rejects_before_descriptor_setup();
        test_linear_output_count_overflow_rejects_before_dispatch();
        test_convolution_rejects_undersized_allocation_before_dispatch();
        test_pooling_rejects_undersized_allocation_before_dispatch();
        test_pooling_rejects_dimension_product_overflow_before_dispatch();
        test_nonzero_storage_offset_is_supported();
        test_zero_tensor_copy_is_noop();
        test_strided_copy_reads_and_writes_logical_indices();
        test_copy_broadcasts_cpu_source_into_strided_vulkan_destination();
        test_strided_copy_rejects_unsafe_overlap();
        test_identical_vulkan_copy_is_a_noop();
        test_strided_copy_zero_elements_is_noop();
        test_zero_allocator_payload();
        test_foreign_payload_rejected();
        test_vulkan_layout_inspection();
        test_vulkan_layout_descriptor_and_index_mapping();
        test_empty_reductions_stay_on_vulkan();
        test_empty_reductions_validate_input_layout();
        test_vulkan_layout_address_rejects_storage_offset_overflow();
        test_vulkan_layout_rejects_shader_address_overflow();
        test_empty_vulkan_view_checks_allocation_boundary();
        test_empty_vulkan_view_does_not_use_shader_address_limit();
        test_metadata_only_views();
        test_bmm_out_rejects_input_alias_before_dispatch();
        test_repeated_add_dispatch_and_retained_output();
        test_tensor_tensor_sub_and_mul_dispatch();
        test_zero_element_add_does_not_dispatch();
        test_scalar_pointwise_offset_is_supported();
        test_scalar_add_dispatch_modes_and_lifecycle();
        test_zero_element_scalar_add_does_not_dispatch();
        test_formatter_comparison_dispatch_is_counted();
        test_concurrent_add_dispatches_are_serialized();
        test_add_invalid_input_cleans_up();
        test_repeated_unary_dispatch_and_input_readability();
        test_unary_nonzero_storage_offset_is_supported();
        test_shared_out_dispatch_helpers();
        test_shared_out_empty_path_does_not_dispatch();
        test_shared_out_rejects_offset_and_noncontiguous_output();
        test_shared_out_rejects_partial_and_internal_overlap();
        test_inplace_operations_and_rejections_preserve_contracts();
        test_shared_out_exact_aliases_preserve_values_and_lifecycles();
        test_shared_out_device_index_is_rejected_by_native_allocator();
        test_platform_destruction_is_nothrow();
        test_reduction_indexing_reject_malformed_metadata();
        test_rnn_limit_validation_uses_supplied_limits();
        std::cout << "Vulkan tensor transfer tests passed\n";
        return 0;
    } catch (const VulkanUnavailable &error) {
        std::cerr << error.what() << '\n';
        return 77;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
