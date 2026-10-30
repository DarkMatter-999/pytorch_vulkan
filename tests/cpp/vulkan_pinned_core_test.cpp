#include "vulkan_pinned_core_test.h"

int main(int argc, char **argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--case")
            throw std::runtime_error("requires --case release|retention|policy|reporter-failure");
        const std::string name(argv[2]);
        if (name == "release") vulkan_pinned_core_testing::release();
        else if (name == "retention") vulkan_pinned_core_testing::retention();
        else if (name == "alias-leases") vulkan_pinned_core_testing::alias_lease_counts();
        else if (name == "policy") vulkan_pinned_core_testing::release(true);
        else if (name == "reporter-failure") vulkan_pinned_core_testing::reporter_failure();
        else throw std::runtime_error("unknown pinned core CPU case");
        std::cout << "PASS pinned core CPU " << name << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
