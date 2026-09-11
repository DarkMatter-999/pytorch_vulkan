cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dlprim_backend LANGUAGES CXX)
include(CTest)

# The legacy OpenCL implementation is retained as reference material only.
# Vulkan and CPU integration will add active targets in later phases.
message(STATUS "GPU backend disabled; configuring CPU-only project shell")

option(BUILD_VULKAN_PROBE "Build the Vulkan device discovery probe" OFF)
option(BUILD_PYTHON_EXTENSION "Build the minimal Python Vulkan extension" OFF)

if(BUILD_VULKAN_PROBE OR BUILD_PYTHON_EXTENSION)
    find_package(Vulkan 1.1 REQUIRED)
endif()

if(BUILD_VULKAN_PROBE OR BUILD_PYTHON_EXTENSION)
  if(NOT TARGET vulkan_platform)
    add_library(vulkan_platform STATIC src/vulkan_platform.cpp)
    target_compile_features(vulkan_platform PUBLIC cxx_std_17)
    set_target_properties(vulkan_platform PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(vulkan_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vulkan_platform PUBLIC Vulkan::Vulkan)
  endif()
endif()

if(BUILD_VULKAN_PROBE)

    add_library(vulkan_buffer STATIC src/vulkan_buffer.cpp)
    target_compile_features(vulkan_buffer PUBLIC cxx_std_17)
    target_include_directories(vulkan_buffer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vulkan_buffer PUBLIC vulkan_platform)

    add_executable(vulkan_device_probe src/vulkan_device_probe.cpp)
    target_compile_features(vulkan_device_probe PRIVATE cxx_std_17)
    target_link_libraries(vulkan_device_probe PRIVATE vulkan_buffer)
    if(BUILD_TESTING)
        add_test(NAME vulkan_device_probe COMMAND vulkan_device_probe)
        set_tests_properties(vulkan_device_probe PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(BUILD_PYTHON_EXTENSION)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Torch REQUIRED)

     pybind11_add_module(
         pytorch_vulkan_python MODULE src/python_module.cpp src/vulkan_device_guard.cpp)
    target_compile_features(pytorch_vulkan_python PRIVATE cxx_std_17)
    target_include_directories(pytorch_vulkan_python PRIVATE ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(pytorch_vulkan_python PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(pytorch_vulkan_python PRIVATE vulkan_platform)
    target_link_libraries(pytorch_vulkan_python PRIVATE ${TORCH_LIBRARIES})
    if(PYTORCH_VULKAN_TORCH_COMPILE_PROBE)
        target_sources(
            pytorch_vulkan_python
            PRIVATE "${PYTORCH_VULKAN_TORCH_COMPILE_PROBE}")
    endif()
    set_target_properties(pytorch_vulkan_python PROPERTIES OUTPUT_NAME "_C")
    set_target_properties(
        pytorch_vulkan_python
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_vulkan")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_vulkan/__init__.py"
        "${CMAKE_BINARY_DIR}/pytorch_vulkan/__init__.py"
        COPYONLY)
endif()
