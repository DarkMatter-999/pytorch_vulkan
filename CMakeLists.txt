cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dlprim_backend LANGUAGES CXX)

# The legacy OpenCL implementation is retained as reference material only.
# Vulkan and CPU integration will add active targets in later phases.
message(STATUS "GPU backend disabled; configuring CPU-only project shell")

option(BUILD_VULKAN_PROBE "Build the Vulkan device discovery probe" OFF)

if(BUILD_VULKAN_PROBE)
    find_package(Vulkan 1.1 REQUIRED)

    add_library(vulkan_platform STATIC src/vulkan_platform.cpp)
    target_compile_features(vulkan_platform PUBLIC cxx_std_17)
    target_include_directories(vulkan_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vulkan_platform PUBLIC Vulkan::Vulkan)

    add_library(vulkan_buffer STATIC src/vulkan_buffer.cpp)
    target_compile_features(vulkan_buffer PUBLIC cxx_std_17)
    target_include_directories(vulkan_buffer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vulkan_buffer PUBLIC vulkan_platform)

    add_executable(vulkan_device_probe src/vulkan_device_probe.cpp)
    target_compile_features(vulkan_device_probe PRIVATE cxx_std_17)
    target_link_libraries(vulkan_device_probe PRIVATE vulkan_buffer)
endif()
