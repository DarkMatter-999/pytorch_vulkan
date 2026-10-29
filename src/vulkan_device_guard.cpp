#include "vulkan_device_guard.h"

#include "vulkan_platform.h"
#include "vulkan_buffer.h"
#include "vulkan_compute.h"
#include "vulkan_execution.h"
#include "vulkan_allocator.h"
#include "vulkan_pinned_allocator.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/util/Exception.h>

#include <cstdlib>
#include <algorithm>
#include <memory>
#include <mutex>
#include <pthread.h>

namespace pytorch_vulkan {
namespace {

VulkanLifecycleCoordinator* fork_coordinator = nullptr;

void abandon_singleton_child() noexcept {
    if (fork_coordinator) fork_coordinator->abandon_inherited_child();
}

void check_platform_late() noexcept {
    // Registered only after the holder exists. Check-only: never synchronize or
    // destroy Vulkan services in libc's late fallback.
    platform_lifecycle().check_late();
}

const c10::Device kDevice(c10::DeviceType::PrivateUse1, 0);
thread_local c10::Device current = kDevice;
thread_local c10::Stream stream(c10::Stream::DEFAULT, kDevice);

void check_device(c10::Device device) {
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
                "Vulkan guard received device of type ", device.type());
    TORCH_CHECK(device.index() == 0,
                "Vulkan backend supports only device index 0, got ", device.index());
}

class VulkanPrivateUse1Hooks final : public at::PrivateUse1HooksInterface {
  public:
    at::Allocator *getPinnedMemoryAllocator() const override {
        return vulkan_pinned_host_allocator_instance();
    }

    bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
        if (device_index != 0) {
            return false;
        }
        try {
            return platform() != nullptr;
        } catch (const __cxxabiv1::__forced_unwind&) { throw; }
        catch (...) {
            return false;
        }
    }

    void resizePrivateUse1Bytes(const c10::Storage &storage,
                                size_t newsize) const override {
        TORCH_CHECK(storage.resizable(),
                    "Trying to resize storage that is not resizable");
        TORCH_CHECK(storage.device_type() == c10::DeviceType::PrivateUse1 &&
                        storage.device().index() == 0,
                    "Vulkan storage resize requires PrivateUse1 device 0");
        const size_t oldsize = storage.nbytes();
        if (newsize <= oldsize) {
            storage.set_nbytes(newsize);
            return;
        }

        auto owner = pytorch_vulkan::platform();
        return with_vulkan_operation(owner->runtime_lifecycle(), [&] {
        owner->throw_if_device_lost();
        auto *storage_allocator = storage.allocator();
        TORCH_CHECK(storage_allocator != nullptr,
                    "Trying to resize storage that is not resizable");
        at::DataPtr replacement = storage_allocator->allocate(newsize);
        const at::DataPtr &old_ptr = storage.data_ptr();
        const size_t copy_bytes = std::min(oldsize, newsize);
        if (copy_bytes != 0) {
            pytorch_vulkan::validate_allocation(old_ptr, oldsize, "resize source");
            pytorch_vulkan::validate_allocation(replacement, newsize, "resize destination");
            owner->copy_buffer_sync(
                pytorch_vulkan::allocation_buffer(old_ptr).buffer(),
                pytorch_vulkan::allocation_buffer(replacement).buffer(),
                copy_bytes);
        }

        // Allocate the completion holder before replacing storage so host
        // allocation failure cannot strand an old DataPtr behind new metadata.
        auto retained = std::make_shared<at::DataPtr>();
        at::DataPtr previous = storage.set_data_ptr(std::move(replacement));
        storage.set_nbytes(newsize);
        if (previous.get_context() != nullptr &&
            pytorch_vulkan::is_vulkan_allocation(previous)) {
            *retained = std::move(previous);
            try {
                if (!owner->execution_context().retain_until_completion(
                        [retained] {}))
                    retained.reset();
            } catch (const __cxxabiv1::__forced_unwind&) { throw; }
            catch (...) {
                // DataPtr's existing Vulkan allocation deleter retains on
                // completion or intentionally leaks if pending-state bookkeeping
                // itself fails, so dropping this holder cannot free a live buffer.
                retained.reset();
                throw;
            }
        }
        });
    }
};

const bool hooks_registered = [] {
    at::RegisterPrivateUse1HooksInterface(new VulkanPrivateUse1Hooks());
    return true;
}();

} // namespace

std::shared_ptr<VulkanPlatform> platform() {
    ensure_process_local_vulkan();
    auto& coordinator = platform_lifecycle();
    auto owner = coordinator.acquire([&] {
        static std::once_flag registration;
        std::call_once(registration, [&] {
        register_fork_state_handler();
        fork_coordinator = &coordinator;
        if (pthread_atfork(nullptr, nullptr, &abandon_singleton_child) != 0)
            throw std::runtime_error("Could not register Vulkan lifecycle fork handler");
        // Register before construction so registration failure cannot unwind a
        // live device root. The holder's destructor is already registered.
        if (std::atexit(&check_platform_late) != 0)
            throw std::runtime_error("Could not register Vulkan lifecycle late check");
        });
        return make_vulkan_lifecycle_root(std::make_shared<VulkanPlatform>());
    });
    return std::static_pointer_cast<VulkanPlatform>(owner);
}

VulkanLifecycleCoordinator& platform_lifecycle() {
    static VulkanLifecycleCoordinator coordinator(vulkan_host_clock());
    return coordinator;
}

c10::DeviceType VulkanDeviceGuard::type() const { return c10::DeviceType::PrivateUse1; }

c10::Device VulkanDeviceGuard::exchangeDevice(c10::Device device) const {
    check_device(device);
    const c10::Device previous = current;
    current = device;
    return previous;
}

c10::Device VulkanDeviceGuard::getDevice() const { return current; }

void VulkanDeviceGuard::setDevice(c10::Device device) const {
    check_device(device);
    current = device;
}

void VulkanDeviceGuard::uncheckedSetDevice(c10::Device device) const noexcept {
    if (device.type() == c10::DeviceType::PrivateUse1 && device.index() == 0) {
        current = device;
    }
}

c10::Stream VulkanDeviceGuard::getStream(c10::Device device) const noexcept {
    TORCH_INTERNAL_ASSERT(device.type() == c10::DeviceType::PrivateUse1 &&
                              device.index() == 0,
                          "Vulkan backend supports only device index 0");
    return device == current ? stream : c10::Stream(c10::Stream::DEFAULT, device);
}

c10::Stream VulkanDeviceGuard::getDefaultStream(c10::Device device) const {
    check_device(device);
    return c10::Stream(c10::Stream::DEFAULT, device);
}

c10::Stream VulkanDeviceGuard::exchangeStream(c10::Stream next) const noexcept {
    TORCH_INTERNAL_ASSERT(next.device_type() == c10::DeviceType::PrivateUse1 &&
                              next.device_index() == 0,
                          "Vulkan backend supports only device index 0");
    const c10::Stream previous = stream;
    stream = next;
    return previous;
}

c10::DeviceIndex VulkanDeviceGuard::deviceCount() const noexcept {
    return VulkanPlatform::is_available() ? 1 : 0;
}

bool VulkanDeviceGuard::queryStream(const c10::Stream &stream) const {
    check_device(stream.device());
    const auto owner = platform();
    owner->throw_if_device_lost();
    return owner->query_execution();
}

void VulkanDeviceGuard::synchronizeStream(const c10::Stream &stream) const {
    check_device(stream.device());
    const auto owner = platform();
    owner->throw_if_device_lost();
    with_vulkan_operation(owner->runtime_lifecycle(), [&] { owner->synchronize_execution(); });
}

c10::Device current_device() { return current; }

void set_device(c10::DeviceIndex index) {
    VulkanDeviceGuard guard;
    guard.setDevice(c10::Device(c10::DeviceType::PrivateUse1, index));
}

void shutdown_platform() { platform_lifecycle().retire_cached(); }

} // namespace pytorch_vulkan

C10_REGISTER_GUARD_IMPL(PrivateUse1, pytorch_vulkan::VulkanDeviceGuard)
