#include "vulkan_allocator.h"
#include "vulkan_device_guard.h"
#include "vulkan_lifecycle.h"
#include "vulkan_platform.h"
#include "vulkan_terminal_selector.h"
#include "vulkan_transfer.h"

#include <ATen/ATen.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

const c10::Device kDevice(c10::DeviceType::PrivateUse1, 0);

void expect(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_opaque_storage_and_cpu_materialization() {
    auto source = at::tensor({1.0F, -2.5F, 3.25F, 0.0F});
    auto tensor = at::empty({4}, source.options().device(kDevice));
    const auto &data = tensor.storage().data_ptr();

    expect(pytorch_vulkan::is_vulkan_allocation(data),
           "Vulkan tensor did not use the Vulkan allocator");
    expect(data.get() != nullptr && data.get() == data.get_context(),
           "Vulkan DataPtr lost its opaque allocation context");

    pytorch_vulkan::copy_tensor(tensor, source, false);
    auto restored = at::empty_like(source);
    pytorch_vulkan::copy_tensor(restored, tensor, false);
    expect(restored.equal(source),
           "synchronous Vulkan-to-CPU materialization changed data");
}

void test_terminal_smoke() {
    auto source = at::tensor({1.0F, -2.0F, 3.5F, 4.0F});
    auto tensor = at::empty({4}, source.options().device(kDevice));
    expect(pytorch_vulkan::is_vulkan_allocation(tensor.storage().data_ptr()),
           "terminal smoke tensor did not use the Vulkan allocator");
    pytorch_vulkan::copy_tensor(tensor, source, false);
    pytorch_vulkan::platform()->synchronize_execution();
    auto restored = at::empty_like(source);
    pytorch_vulkan::copy_tensor(restored, tensor, false);
    expect(restored.equal(source), "terminal smoke Vulkan round trip changed data");
}

} // namespace

int main(int argc, char **argv) {
    pytorch_vulkan::NativeVulkanTerminalGuard terminal(pytorch_vulkan::platform_lifecycle());
    try {
        const auto selector = vulkan_terminal_selector::parse(argc, argv);
        if (selector != vulkan_terminal_selector::Selector::Default) {
            test_terminal_smoke();
            if (selector == vulkan_terminal_selector::Selector::OrdinaryError)
                throw std::runtime_error("intentional ordinary native error after terminal smoke use");
            std::cout << "Vulkan serialization probe terminal smoke passed\n";
            return 0;
        }
        test_opaque_storage_and_cpu_materialization();
        std::cout << "Vulkan serialization probe passed\n";
        return 0;
    } catch (const VulkanUnavailable &error) {
        std::cerr << error.what() << '\n';
        return 77;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
