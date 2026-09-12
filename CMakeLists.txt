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
    add_library(vulkan_platform STATIC src/vulkan_platform.cpp
                  src/vulkan_compute.cpp src/vulkan/shaders/generated/pointwise_spv.h
                  src/vulkan/shaders/glsl/pointwise.comp)
    target_compile_features(vulkan_platform PUBLIC cxx_std_17)
    set_target_properties(vulkan_platform PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(vulkan_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vulkan_platform PUBLIC Vulkan::Vulkan)
  endif()
endif()

if(BUILD_VULKAN_PROBE OR BUILD_PYTHON_EXTENSION)

    if(NOT TARGET vulkan_buffer)
        add_library(vulkan_buffer STATIC src/vulkan_buffer.cpp)
        target_compile_features(vulkan_buffer PUBLIC cxx_std_17)
        set_target_properties(vulkan_buffer PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_include_directories(vulkan_buffer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(vulkan_buffer PUBLIC vulkan_platform)
    endif()

endif()

if(BUILD_VULKAN_PROBE)
    add_executable(vulkan_device_probe src/vulkan_device_probe.cpp)
    target_compile_features(vulkan_device_probe PRIVATE cxx_std_17)
    target_link_libraries(vulkan_device_probe PRIVATE vulkan_buffer)
    if(BUILD_TESTING)
        add_test(NAME vulkan_device_probe COMMAND vulkan_device_probe)
        set_tests_properties(vulkan_device_probe PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(BUILD_VULKAN_PROBE OR BUILD_PYTHON_EXTENSION)
    add_executable(vulkan_transfer_test tests/cpp/vulkan_transfer_test.cpp)
    target_compile_features(vulkan_transfer_test PRIVATE cxx_std_17)
    target_link_libraries(vulkan_transfer_test PRIVATE vulkan_buffer)
    if(BUILD_TESTING)
        add_test(NAME vulkan_transfer_test COMMAND vulkan_transfer_test)
        set_tests_properties(vulkan_transfer_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(BUILD_PYTHON_EXTENSION)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Torch REQUIRED)

    if(TORCH_CXX_FLAGS)
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(vulkan_platform PRIVATE ${TORCH_CXX_FLAGS_LIST})
        target_compile_options(vulkan_buffer PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()

    add_library(vulkan_device_guard STATIC src/vulkan_device_guard.cpp)
    target_compile_features(vulkan_device_guard PUBLIC cxx_std_17)
    set_target_properties(vulkan_device_guard PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(vulkan_device_guard PUBLIC
                               ${CMAKE_CURRENT_SOURCE_DIR}/src ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(vulkan_device_guard PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(vulkan_device_guard PUBLIC vulkan_platform ${TORCH_LIBRARIES})

    add_library(vulkan_allocator STATIC src/vulkan_allocator.cpp src/vulkan_transfer.cpp)
    target_compile_features(vulkan_allocator PUBLIC cxx_std_17)
    set_target_properties(vulkan_allocator PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(vulkan_allocator PUBLIC
                               ${CMAKE_CURRENT_SOURCE_DIR}/src ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(vulkan_allocator PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(vulkan_allocator PUBLIC vulkan_buffer vulkan_device_guard
                          vulkan_platform
                          ${TORCH_LIBRARIES})

     pybind11_add_module(
          pytorch_vulkan_python MODULE src/python_module.cpp src/vulkan/operators/add.cpp
           src/vulkan/operators/sub.cpp src/vulkan/operators/mul.cpp
           src/vulkan/operators/view.cpp src/vulkan/operators/abs.cpp
           src/vulkan/operators/add.h src/vulkan/operators/abs.h)
    target_compile_features(pytorch_vulkan_python PRIVATE cxx_std_17)
    target_include_directories(pytorch_vulkan_python PRIVATE ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(pytorch_vulkan_python PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(pytorch_vulkan_python PRIVATE vulkan_allocator
                          vulkan_device_guard vulkan_buffer vulkan_platform)
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

    add_executable(vulkan_allocator_test tests/cpp/vulkan_allocator_test.cpp)
    target_compile_features(vulkan_allocator_test PRIVATE cxx_std_17)
    target_include_directories(vulkan_allocator_test PRIVATE ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        target_compile_options(vulkan_allocator_test PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(vulkan_allocator_test PRIVATE vulkan_allocator
                          vulkan_device_guard vulkan_buffer vulkan_platform
                          ${TORCH_LIBRARIES})
    if(BUILD_TESTING)
        add_test(NAME vulkan_allocator_test COMMAND vulkan_allocator_test)
        set_tests_properties(vulkan_allocator_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()

     add_executable(vulkan_transfer_tensor_test tests/cpp/vulkan_transfer_tensor_test.cpp
                    src/vulkan/operators/add.cpp src/vulkan/operators/view.cpp
                    src/vulkan/operators/abs.cpp)
    target_compile_features(vulkan_transfer_tensor_test PRIVATE cxx_std_17)
    target_include_directories(vulkan_transfer_tensor_test PRIVATE ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        target_compile_options(vulkan_transfer_tensor_test PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(vulkan_transfer_tensor_test PRIVATE vulkan_allocator
                          vulkan_device_guard vulkan_buffer vulkan_platform
                          ${TORCH_LIBRARIES})
    if(BUILD_TESTING)
        add_test(NAME vulkan_transfer_tensor_test COMMAND vulkan_transfer_tensor_test)
        set_tests_properties(vulkan_transfer_tensor_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
