#include "vulkan_buffer.h"
#include "vulkan_platform.h"
#include "vulkan_pinned_allocator.h"

#include <ATen/ATen.h>
#include <ATen/ops/_pin_memory.h>
#include <ATen/ops/pin_memory.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/Allocator.h>
#include <c10/core/alignment.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace pytorch_vulkan::testing {
void reset_pinned_memory_kernel_calls_for_testing() noexcept;
uint64_t pinned_memory_kernel_calls_for_testing() noexcept;
} // namespace pytorch_vulkan::testing

namespace {
struct PinnedFakeCalls {
    enum class FailAt { None, Create, Requirements, Allocate, Bind, Map } fail_at = FailAt::None;
    unsigned creates = 0, requirements = 0, allocations = 0, binds = 0, maps = 0;
    unsigned unmaps = 0, destroys = 0, frees = 0, safety_checks = 0;
    std::array<VkMemoryPropertyFlags, 2> available_flags{{
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0}};
    uint32_t memory_type_count = 1;
    uint32_t selected_memory_type = UINT32_MAX;
    std::vector<char> cleanup_order;
    void *mapped = nullptr;
    void *mapped_allocation = nullptr;
    enum class Guard { Clear, DeviceLost, Pending } guard = Guard::Clear;
};

VkResult fake_create(void *cookie, VkDevice, VkDeviceSize, VkBuffer *buffer) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    ++calls.creates;
    if (calls.fail_at == PinnedFakeCalls::FailAt::Create) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *buffer = reinterpret_cast<VkBuffer>(uintptr_t{1});
    return VK_SUCCESS;
}
void fake_requirements(void *cookie, VkDevice, VkBuffer, VkMemoryRequirements *requirements) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    ++calls.requirements;
    if (calls.fail_at == PinnedFakeCalls::FailAt::Requirements)
        throw std::runtime_error("injected requirements failure");
    requirements->size = 256;
    requirements->alignment = 64;
    requirements->memoryTypeBits = calls.memory_type_count == 2 ? 3U : 1U;
}
void fake_memory_properties(void *cookie, VkPhysicalDevice,
                            VkPhysicalDeviceMemoryProperties *properties) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    properties->memoryTypeCount = calls.memory_type_count;
    for (uint32_t i = 0; i < calls.memory_type_count; ++i)
        properties->memoryTypes[i].propertyFlags = calls.available_flags[i];
}
VkResult fake_allocate(void *cookie, VkDevice, const VkMemoryAllocateInfo *info,
                       VkDeviceMemory *memory) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    ++calls.allocations;
    if (calls.fail_at == PinnedFakeCalls::FailAt::Allocate) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    calls.selected_memory_type = info->memoryTypeIndex;
    *memory = reinterpret_cast<VkDeviceMemory>(uintptr_t{2});
    return VK_SUCCESS;
}
VkResult fake_bind(void *cookie, VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    ++calls.binds;
    return calls.fail_at == PinnedFakeCalls::FailAt::Bind ? VK_ERROR_MEMORY_MAP_FAILED : VK_SUCCESS;
}
VkResult fake_map(void *cookie, VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize,
                  VkMemoryMapFlags, void **mapped) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    ++calls.maps;
    if (calls.fail_at == PinnedFakeCalls::FailAt::Map) return VK_ERROR_MEMORY_MAP_FAILED;
    *mapped = calls.mapped;
    return VK_SUCCESS;
}
void fake_unmap(void *cookie, VkDevice, VkDeviceMemory) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    calls.cleanup_order.push_back('u');
    ++calls.unmaps;
    std::free(calls.mapped_allocation);
    calls.mapped_allocation = nullptr;
}
void fake_destroy(void *cookie, VkDevice, VkBuffer) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    ++calls.destroys;
    calls.cleanup_order.push_back('d');
}
void fake_free(void *cookie, VkDevice, VkDeviceMemory) {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    ++calls.frees;
    calls.cleanup_order.push_back('f');
    std::free(calls.mapped_allocation);
    calls.mapped_allocation = nullptr;
}
bool fake_should_leak(void *cookie, const VulkanPlatform &) noexcept {
    auto &calls = *static_cast<PinnedFakeCalls *>(cookie);
    ++calls.safety_checks;
    return calls.guard != PinnedFakeCalls::Guard::Clear;
}

VulkanPinnedHostTestOps fake_pinned_ops(PinnedFakeCalls &calls) {
    VulkanPinnedHostTestOps ops;
    ops.cookie = &calls;
    ops.buffer.cookie = &calls;
    ops.buffer.create_buffer = fake_create;
    ops.buffer.get_requirements = fake_requirements;
    ops.buffer.get_memory_properties = fake_memory_properties;
    ops.buffer.allocate_memory = fake_allocate;
    ops.buffer.bind_buffer = fake_bind;
    ops.buffer.destroy_buffer = fake_destroy;
    ops.buffer.free_memory = fake_free;
    ops.map.cookie = &calls;
    ops.map.map = fake_map;
    ops.map.unmap = fake_unmap;
    ops.should_leak_for_testing = fake_should_leak;
    return ops;
}

void require_failure_cleanup(PinnedFakeCalls::FailAt fail_at,
                             unsigned expected_unmaps, unsigned expected_destroys,
                             unsigned expected_frees) {
    PinnedFakeCalls calls;
    calls.fail_at = fail_at;
    if (fail_at == PinnedFakeCalls::FailAt::Bind || fail_at == PinnedFakeCalls::FailAt::Map) {
        calls.mapped_allocation = std::aligned_alloc(c10::gAlignment, 256);
        calls.mapped = calls.mapped_allocation;
    }
    VulkanPinnedHostAllocator allocator(pytorch_vulkan::platform(), fake_pinned_ops(calls));
    bool failed = false;
    try {
        (void)allocator.allocate(128);
    } catch (const std::exception &) {
        failed = true;
    }
    const unsigned stage = static_cast<unsigned>(fail_at);
    if (!failed || calls.creates != 1 || calls.requirements != (stage >= 2 ? 1U : 0U) ||
        calls.allocations != (stage >= 3 ? 1U : 0U) || calls.binds != (stage >= 4 ? 1U : 0U) ||
        calls.maps != (stage >= 5 ? 1U : 0U) || calls.unmaps != expected_unmaps ||
        calls.destroys != expected_destroys || calls.frees != expected_frees ||
        calls.mapped_allocation != nullptr ||
        calls.cleanup_order.size() != expected_unmaps + expected_destroys + expected_frees)
        throw std::runtime_error("injected allocator failure did not clean resources exactly once");
    if ((expected_unmaps != 0 && calls.cleanup_order.front() != 'u') ||
        (expected_destroys != 0 && calls.cleanup_order[expected_unmaps] != 'd') ||
        (expected_frees != 0 && calls.cleanup_order.back() != 'f'))
        throw std::runtime_error("injected allocator failure cleanup order was invalid");
}

void test_pinned_allocator_fake_lifecycle() {
    require_failure_cleanup(PinnedFakeCalls::FailAt::Create, 0, 0, 0);
    require_failure_cleanup(PinnedFakeCalls::FailAt::Requirements, 0, 1, 0);
    require_failure_cleanup(PinnedFakeCalls::FailAt::Allocate, 0, 1, 0);
    require_failure_cleanup(PinnedFakeCalls::FailAt::Bind, 0, 1, 1);
    require_failure_cleanup(PinnedFakeCalls::FailAt::Map, 0, 1, 1);

    for (const bool coherent : {true, false}) {
        PinnedFakeCalls calls;
        calls.available_flags[0] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            (coherent ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0U);
        calls.mapped_allocation = std::aligned_alloc(c10::gAlignment, 256);
        calls.mapped = calls.mapped_allocation;
        VulkanPinnedHostAllocator allocator(pytorch_vulkan::platform(), fake_pinned_ops(calls));
        auto data = allocator.allocate(128);
        const auto flags = vulkan_pinned_host_memory_properties_for_testing(data);
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0 ||
            ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) != coherent ||
            calls.creates != 1 || calls.requirements != 1 || calls.allocations != 1 ||
            calls.binds != 1 || calls.maps != 1)
            throw std::runtime_error("fake memory type selection or stage sequence was incorrect");
        if (calls.maps != 1)
            throw std::runtime_error("successful pinned allocation did not persistently map once");
        data.clear();
        if (calls.unmaps != 1 || calls.destroys != 1 || calls.frees != 1 ||
            calls.cleanup_order != std::vector<char>({'u', 'd', 'f'}))
            throw std::runtime_error("normal DataPtr destruction did not unmap/destroy/free once");
    }

    {
        PinnedFakeCalls calls;
        calls.memory_type_count = 2;
        calls.available_flags = {{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT}};
        calls.mapped_allocation = std::aligned_alloc(c10::gAlignment, 256);
        calls.mapped = calls.mapped_allocation;
        VulkanPinnedHostAllocator allocator(pytorch_vulkan::platform(), fake_pinned_ops(calls));
        auto data = allocator.allocate(128);
        if (calls.selected_memory_type != 1 ||
             vulkan_pinned_host_memory_properties_for_testing(data) != calls.available_flags[1])
            throw std::runtime_error("coherent memory was not preferred over visible fallback");
    }

    {
        PinnedFakeCalls calls;
        calls.mapped_allocation = std::aligned_alloc(c10::gAlignment, 256);
        calls.mapped = static_cast<unsigned char *>(calls.mapped_allocation) + 1;
        VulkanPinnedHostAllocator allocator(pytorch_vulkan::platform(), fake_pinned_ops(calls));
        bool misaligned = false;
        try { (void)allocator.allocate(128); }
        catch (const std::exception &) { misaligned = true; }
        if (!misaligned || calls.unmaps != 1 || calls.destroys != 1 || calls.frees != 1 ||
            calls.cleanup_order != std::vector<char>({'u', 'd', 'f'}) ||
            calls.mapped_allocation != nullptr)
            throw std::runtime_error("misaligned map was not rejected and cleaned exactly once");
    }

    for (const auto guard : {PinnedFakeCalls::Guard::DeviceLost,
                             PinnedFakeCalls::Guard::Pending}) {
        PinnedFakeCalls calls;
        calls.guard = guard;
        calls.mapped_allocation = std::aligned_alloc(c10::gAlignment, 256);
        calls.mapped = calls.mapped_allocation;
        VulkanPinnedHostAllocator allocator(pytorch_vulkan::platform(), fake_pinned_ops(calls));
        { auto data = allocator.allocate(128); }
        if (calls.unmaps != 0 || calls.destroys != 0 || calls.frees != 0 ||
            calls.mapped_allocation == nullptr || calls.safety_checks != 1)
            throw std::runtime_error("lost/pending guard freed a resource that must be leaked");
        // Intentional test-fixture leak models the safety rule under these states.
    }
}

void test_pinned_memory_operators() {
    auto source = at::arange(12, at::TensorOptions().dtype(at::kFloat))
                      .reshape({3, 4}).transpose(0, 1);
    const c10::Device vk(c10::DeviceType::PrivateUse1, 0);
    auto call_is_pinned = [](const at::Tensor &tensor,
                             c10::optional<c10::Device> device) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("aten::is_pinned", "")
            .typed<bool(const at::Tensor &, c10::optional<c10::Device>)>()
            .call(tensor, device);
    };
    auto public_pin_memory = [](const at::Tensor &tensor,
                                c10::optional<c10::Device> device) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("aten::pin_memory", "")
            .typed<at::Tensor(const at::Tensor &, c10::optional<c10::Device>)>()
            .call(tensor, device);
    };
    pytorch_vulkan::testing::reset_pinned_memory_kernel_calls_for_testing();
    auto pinned = public_pin_memory(source, vk);
    if (!pinned.device().is_cpu() || pinned.sizes() != source.sizes() ||
        pinned.strides() != source.strides() || pinned.storage_offset() != 0 ||
        pinned.storage().nbytes() != 12 * sizeof(float) ||
        pinned.storage().resizable() || !call_is_pinned(pinned, vk) ||
        pinned.data_ptr() == source.data_ptr() || !pinned.equal(source))
        throw std::runtime_error("PrivateUse1 pinning did not preserve CPU storage/layout/value contract");
    if (pytorch_vulkan::testing::pinned_memory_kernel_calls_for_testing() == 0)
        throw std::runtime_error("explicit CPU pinning did not reach the PrivateUse1 kernels");

    bool rejected_growth = false;
    try {
        pinned.resize_({32});
    } catch (const c10::Error &) {
        rejected_growth = true;
    }
    if (!rejected_growth)
        throw std::runtime_error("pinned storage allowed resize beyond its capacity");

    auto empty_view = pinned.narrow(0, 0, 0);
    if (!call_is_pinned(empty_view, vk))
        throw std::runtime_error("empty view lost nonempty pinned storage provenance");
    auto empty_source = at::empty({0, 4}, source.options());
    auto empty_pinned = public_pin_memory(empty_source, vk);
    if (empty_pinned.storage().nbytes() != 0 || call_is_pinned(empty_pinned, vk))
        throw std::runtime_error("zero-byte pinning did not preserve null-storage recognition");

    auto same_pinned = public_pin_memory(pinned, vk);
    if (same_pinned.unsafeGetTensorImpl() != pinned.unsafeGetTensorImpl())
        throw std::runtime_error("public pin_memory did not preserve already-pinned identity");

    auto cpu_roundtrip_source = at::arange(6, source.options()).reshape({2, 3});
    auto vk_source = cpu_roundtrip_source.to(vk);
    pytorch_vulkan::platform()->synchronize_execution();
    auto meta_source = at::empty({2}, at::TensorOptions().device(c10::DeviceType::Meta));
    std::string meta_error;
    try {
        (void)public_pin_memory(meta_source, c10::nullopt);
    } catch (const c10::Error &error) {
        meta_error = error.what();
    }
    if (meta_error.find("cannot pin 'MetaFloatType'") == std::string::npos ||
        meta_error.find("only dense CPU tensors can be pinned") == std::string::npos)
        throw std::runtime_error("meta reference did not reject pin_memory through BackendSelect");

    const auto calls_before_rejections =
        pytorch_vulkan::testing::pinned_memory_kernel_calls_for_testing();
    for (const auto device : {c10::optional<c10::Device>(), c10::optional<c10::Device>(vk),
                              c10::optional<c10::Device>(c10::Device(c10::DeviceType::CPU))}) {
        std::string vk_error;
        try {
            (void)public_pin_memory(vk_source, device);
        } catch (const c10::Error &error) {
            vk_error = error.what();
        }
        if (vk_error.empty() || vk_error.find("cannot pin 'privateuseoneFloatType'") ==
                                    std::string::npos ||
            vk_error.find("only dense CPU tensors can be pinned") == std::string::npos)
            throw std::runtime_error("Vulkan source pin_memory did not match BackendSelect guard");
        if (call_is_pinned(vk_source, device))
            throw std::runtime_error("Vulkan source is_pinned did not return false");
    }
    if (pytorch_vulkan::testing::pinned_memory_kernel_calls_for_testing() !=
        calls_before_rejections)
        throw std::runtime_error("BackendSelect Vulkan rejection unexpectedly reached pinning kernels");

    auto roundtrip_pinned = public_pin_memory(vk_source.cpu(), vk);
    if (!call_is_pinned(roundtrip_pinned, vk) ||
        !roundtrip_pinned.equal(cpu_roundtrip_source))
        throw std::runtime_error("CPU composition of Vulkan source did not pin/copy correctly");
}

struct Calls {
    unsigned maps = 0;
    unsigned unmaps = 0;
    unsigned flushes = 0;
    unsigned invalidates = 0;
    unsigned fail_maps = 1;
    unsigned fail_flushes = 0;
    unsigned fail_invalidates = 0;
    bool synthetic_cache = false;
    VkMappedMemoryRange flush_range{};
    VkMappedMemoryRange invalidate_range{};
};

VkResult map(void *cookie, VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
             VkDeviceSize size, VkMemoryMapFlags flags, void **mapped) {
    auto &calls = *static_cast<Calls *>(cookie);
    ++calls.maps;
    if (calls.fail_maps) {
        --calls.fail_maps;
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    return vkMapMemory(device, memory, offset, size, flags, mapped);
}
void unmap(void *cookie, VkDevice device, VkDeviceMemory memory) {
    ++static_cast<Calls *>(cookie)->unmaps;
    vkUnmapMemory(device, memory);
}
VkResult flush(void *cookie, VkDevice device, uint32_t count,
               const VkMappedMemoryRange *ranges) {
    auto &calls = *static_cast<Calls *>(cookie);
    ++calls.flushes;
    calls.flush_range = ranges[0];
    if (calls.fail_flushes) {
        --calls.fail_flushes;
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (calls.synthetic_cache) return VK_SUCCESS;
    return vkFlushMappedMemoryRanges(device, count, ranges);
}
VkResult invalidate(void *cookie, VkDevice device, uint32_t count,
                    const VkMappedMemoryRange *ranges) {
    auto &calls = *static_cast<Calls *>(cookie);
    ++calls.invalidates;
    calls.invalidate_range = ranges[0];
    if (calls.fail_invalidates) {
        --calls.fail_invalidates;
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (calls.synthetic_cache) return VK_SUCCESS;
    return vkInvalidateMappedMemoryRanges(device, count, ranges);
}
} // namespace

int main() {
    try {
        if (!VulkanPlatform::is_available()) {
            std::cout << "SKIP: Vulkan platform unavailable\n";
            return 77;
        }
        VulkanPlatform platform(false);
        test_pinned_allocator_fake_lifecycle();
        test_pinned_memory_operators();
        Calls calls;
        std::cout << "device=" << platform.device_info().name << '\n';
        {
            VulkanBuffer buffer(platform, 64, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            buffer.set_map_ops_for_testing({&calls, map, unmap, flush, invalidate});
            bool map_failed = false;
            try {
                (void)buffer.map_persistent();
            } catch (const std::runtime_error &) {
                map_failed = true;
            }
            if (!map_failed || buffer.mapped_base() != nullptr || calls.unmaps != 0)
                throw std::runtime_error("map failure changed state or unmapped unowned memory");
            void *base = buffer.map_persistent();
            if (base == nullptr || buffer.map_persistent() != base ||
                buffer.mapped_base() != base) {
                throw std::runtime_error("persistent map was not idempotent");
            }
            std::array<unsigned char, 8> input{{1, 2, 3, 4, 5, 6, 7, 8}};
            std::array<unsigned char, 8> output{};
            buffer.write(input.data(), input.size(), 4);
            buffer.read(output.data(), output.size(), 4);
            if (output != input) {
                throw std::runtime_error("persistent mapped read/write mismatch");
            }
            if (buffer.coherent() !=
                ((buffer.memory_properties() & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)) {
                throw std::runtime_error("coherent property does not match allocation");
            }
            std::cout << "actual_memory_properties=" << buffer.memory_properties()
                      << " actual_coherent=" << buffer.coherent() << '\n';
            const auto flushes_before_range = calls.flushes;
            const auto invalidates_before_range = calls.invalidates;
            buffer.flush_range(4, input.size());
            buffer.invalidate_range(4, output.size());
            const unsigned expected_range_calls = buffer.coherent() ? 0U : 1U;
            if (calls.flushes != flushes_before_range + expected_range_calls ||
                calls.invalidates != invalidates_before_range + expected_range_calls)
                throw std::runtime_error("cache maintenance did not follow coherency");
            if (!buffer.coherent()) {
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(platform.physical_device(), &properties);
                const auto atom = properties.limits.nonCoherentAtomSize;
                if (calls.flush_range.offset % atom != 0 ||
                    calls.invalidate_range.offset % atom != 0 ||
                    (calls.flush_range.size != VK_WHOLE_SIZE &&
                     calls.flush_range.size % atom != 0) ||
                    (calls.invalidate_range.size != VK_WHOLE_SIZE &&
                     calls.invalidate_range.size % atom != 0))
                    throw std::runtime_error("noncoherent range was not atom aligned");
            }
            std::cout << "actual_cache_calls={flush:" << calls.flushes
                      << ",invalidate:" << calls.invalidates << '}';
            if (calls.flushes != 0)
                std::cout << " actual_flush_range={offset:" << calls.flush_range.offset
                          << ",size:" << calls.flush_range.size << '}';
            if (calls.invalidates != 0)
                std::cout << " actual_invalidate_range={offset:"
                          << calls.invalidate_range.offset << ",size:"
                          << calls.invalidate_range.size << '}';
            std::cout << '\n';
            buffer.flush_range(0, 0);
            buffer.invalidate_range(0, 0);
            if (calls.flushes != flushes_before_range + expected_range_calls ||
                calls.invalidates != invalidates_before_range + expected_range_calls)
                throw std::runtime_error("zero-size cache operation was not a no-op");
            bool rejected = false;
            try {
                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(platform.device(), buffer.buffer(),
                                              &requirements);
                buffer.flush_range(requirements.size, 1);
            } catch (const std::out_of_range &) {
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("out-of-range flush was accepted");
            rejected = false;
            try {
                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(platform.device(), buffer.buffer(),
                                              &requirements);
                buffer.invalidate_range(requirements.size, 1);
            } catch (const std::out_of_range &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("out-of-range invalidate was accepted");
            rejected = false;
            try {
                buffer.write(input.data(), 1, buffer.size());
            } catch (const std::out_of_range &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("out-of-logical-range write was accepted");
            rejected = false;
            try {
                buffer.read(output.data(), 1, buffer.size());
            } catch (const std::out_of_range &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("out-of-logical-range read was accepted");

            // Synthetic test-only noncoherent path: record/cache-check ranges but
            // never call Vulkan cache maintenance against the actual allocation.
            calls.synthetic_cache = true;
            auto ops = VulkanBufferMapOps{&calls, map, unmap, flush, invalidate};
            ops.override_coherent_for_testing = true;
            ops.coherent_for_testing = true;
            buffer.set_map_ops_for_testing(ops);
            const auto synthetic_flushes = calls.flushes;
            const auto synthetic_invalidates = calls.invalidates;
            buffer.flush_range(3, 5);
            buffer.invalidate_range(3, 5);
            if (calls.flushes != synthetic_flushes ||
                calls.invalidates != synthetic_invalidates)
                throw std::runtime_error("synthetic coherent path issued cache calls");

            ops.coherent_for_testing = false;
            buffer.set_map_ops_for_testing(ops);
            buffer.flush_range(3, 5);
            buffer.invalidate_range(3, 5);
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(platform.physical_device(), &properties);
            const auto atom = properties.limits.nonCoherentAtomSize;
            if (calls.flushes != synthetic_flushes + 1 ||
                calls.invalidates != synthetic_invalidates + 1 ||
                calls.flush_range.offset % atom != 0 ||
                calls.invalidate_range.offset % atom != 0 ||
                (calls.flush_range.size != VK_WHOLE_SIZE &&
                 calls.flush_range.size % atom != 0) ||
                (calls.invalidate_range.size != VK_WHOLE_SIZE &&
                 calls.invalidate_range.size % atom != 0))
                throw std::runtime_error("synthetic noncoherent ranges/call counts invalid");
            std::cout << "synthetic_noncoherent_flush={offset:" << calls.flush_range.offset
                      << ",size:" << calls.flush_range.size << "} invalidate={offset:"
                      << calls.invalidate_range.offset << ",size:"
                      << calls.invalidate_range.size << "} atom=" << atom << '\n';

            calls.fail_flushes = 1;
            bool operation_failed = false;
            try {
                buffer.flush_range(3, 5);
            } catch (const std::runtime_error &) {
                operation_failed = true;
            }
            if (!operation_failed || buffer.mapped_base() != base || calls.unmaps != 0)
                throw std::runtime_error("persistent flush failure lost mapping ownership");
            calls.fail_invalidates = 1;
            operation_failed = false;
            try {
                buffer.invalidate_range(3, 5);
            } catch (const std::runtime_error &) {
                operation_failed = true;
            }
            if (!operation_failed || buffer.mapped_base() != base || calls.unmaps != 0)
                throw std::runtime_error("persistent invalidate failure lost mapping ownership");
        }
        if (calls.maps != 2 || calls.unmaps != 1)
            throw std::runtime_error("persistent mapping did not map/unmap exactly once");

        {
            VulkanPinnedHostAllocator allocator;
            auto empty = allocator.allocate(0);
            if (empty.get() != nullptr || empty.get_context() != nullptr ||
                empty.device() != c10::Device(c10::DeviceType::CPU))
                throw std::runtime_error("zero-byte pinned allocation was not null CPU storage");
            c10::Storage empty_storage(c10::Storage::use_byte_size_t(), 0, &allocator,
                                       false);
            if (is_vulkan_pinned_storage(empty_storage))
                throw std::runtime_error("standalone empty storage was recognized as pinned");
            if (allocator.raw_deleter() != nullptr)
                throw std::runtime_error("raw allocation must remain unsupported");

            const auto buffers_before =
                pytorch_vulkan::platform()->timing_snapshot().buffer_creations;
            auto data = allocator.allocate(256);
            const auto buffers_after =
                pytorch_vulkan::platform()->timing_snapshot().buffer_creations;
            if (buffers_after != buffers_before + 1)
                throw std::runtime_error("pinned allocation did not record one VulkanBuffer creation");
            if (data.get() == nullptr || data.get_context() == nullptr ||
                data.device() != c10::Device(c10::DeviceType::CPU) ||
                reinterpret_cast<std::uintptr_t>(data.get()) % c10::gAlignment != 0)
                throw std::runtime_error("pinned allocation has invalid pointer/device/alignment");
            std::cout << "allocator_actual_memory_properties="
                      << vulkan_pinned_host_memory_properties_for_testing(data)
                      << " allocator_actual_coherent="
                       << ((vulkan_pinned_host_memory_properties_for_testing(data) &
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) << '\n';
            std::array<unsigned char, 4> source{{7, 8, 9, 10}};
            allocator.copy_data(static_cast<unsigned char *>(data.get()), source.data(),
                                source.size());
            if (!std::equal(source.begin(), source.end(),
                            static_cast<unsigned char *>(data.get())))
                throw std::runtime_error("pinned allocator copy_data mismatch");

            c10::Storage storage(c10::Storage::use_byte_size_t(), 256, &allocator, false);
            if (!is_vulkan_pinned_storage(storage))
                throw std::runtime_error("allocator storage provenance was not recognized");
            auto view = at::empty({1}, at::TensorOptions().dtype(at::kByte));
            view.set_(storage, 3, {0}, {1});
            if (!is_vulkan_pinned_storage(view.storage()))
                throw std::runtime_error("empty interior storage view lost provenance");
            auto shared_view = at::empty({1}, at::TensorOptions().dtype(at::kByte));
            shared_view.set_(storage, 2, {1}, {1});
            if (!is_vulkan_pinned_storage(shared_view.storage()))
                throw std::runtime_error("set_-shared tensor lost pinned provenance");
            auto foreign = at::from_blob(data.get(), {256}, at::TensorOptions().dtype(at::kByte));
            if (is_vulkan_pinned_storage(foreign.storage()))
                throw std::runtime_error("foreign wrapper inherited pinned provenance");
            auto pageable = at::empty({8}, at::TensorOptions().dtype(at::kByte));
            if (is_vulkan_pinned_storage(pageable.storage()))
                throw std::runtime_error("ordinary CPU storage was recognized as pinned");
        }

        Calls temporary_calls;
        temporary_calls.fail_flushes = 1;
        temporary_calls.fail_invalidates = 1;
        temporary_calls.synthetic_cache = true;
        {
            VulkanBuffer buffer(platform, 64, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            auto ops = VulkanBufferMapOps{&temporary_calls, map, unmap, flush, invalidate};
            ops.override_coherent_for_testing = true;
            ops.coherent_for_testing = false;
            // Avoid the deliberate map-failure injection used by the first fixture.
            temporary_calls.fail_maps = 0;
            buffer.set_map_ops_for_testing(ops);
            std::array<unsigned char, 8> data{};
            bool failed = false;
            try {
                buffer.write(data.data(), data.size());
            } catch (const std::runtime_error &) {
                failed = true;
            }
            if (!failed || temporary_calls.maps != 1 || temporary_calls.unmaps != 1 ||
                buffer.mapped_base() != nullptr)
                throw std::runtime_error("temporary flush failure did not unmap cleanly");
            failed = false;
            try {
                buffer.read(data.data(), data.size());
            } catch (const std::runtime_error &) {
                failed = true;
            }
            if (!failed || temporary_calls.maps != 2 || temporary_calls.unmaps != 2 ||
                buffer.mapped_base() != nullptr)
                throw std::runtime_error("temporary invalidate failure did not unmap cleanly");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
