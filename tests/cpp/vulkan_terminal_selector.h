#pragma once

#include <string>

namespace vulkan_terminal_selector {

enum class Selector { Default, Smoke, OrdinaryError };

// Only an exact sole argument selects an opt-in terminal case. Every other
// argument list keeps the default body, as the native mains did when they ignored argv.
inline Selector parse(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--terminal-smoke")
        return Selector::Smoke;
    if (argc == 2 && std::string(argv[1]) == "--terminal-ordinary-error")
        return Selector::OrdinaryError;
    return Selector::Default;
}

} // namespace vulkan_terminal_selector
