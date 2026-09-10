cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dlprim_backend LANGUAGES CXX)

# The legacy OpenCL implementation is retained as reference material only.
# Vulkan and CPU integration will add active targets in later phases.
message(STATUS "GPU backend disabled; configuring CPU-only project shell")

option(BUILD_VULKAN_PROBE "Build the Vulkan device discovery probe" OFF)

if(BUILD_VULKAN_PROBE)
    find_package(Vulkan 1.1 REQUIRED)

    add_executable(vulkan_device_probe src/vulkan_device_probe.cpp)
    target_compile_features(vulkan_device_probe PRIVATE cxx_std_17)
    target_link_libraries(vulkan_device_probe PRIVATE Vulkan::Vulkan)
endif()
