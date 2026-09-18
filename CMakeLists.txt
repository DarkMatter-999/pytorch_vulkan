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
    add_library(vulkan_platform STATIC src/vulkan_platform.cpp src/vulkan_execution.cpp
                   src/vulkan_compute.cpp src/vulkan/shaders/generated/pointwise_spv.h
                   src/vulkan/shaders/generated/reduction_indexing_spv.h
                   src/vulkan/shaders/generated/reduction_indexing_spv.sha256
                   src/vulkan/shaders/generated/model_spv.h
                   src/vulkan/shaders/generated/model_spv.sha256
                   src/vulkan/shaders/generated/masked_select_spv.h
                   src/vulkan/shaders/generated/masked_select_spv.sha256
                     src/vulkan/shaders/generated/convolution_spv.h
                     src/vulkan/shaders/generated/convolution_spv.sha256
                     src/vulkan/shaders/generated/pooling_spv.h
                     src/vulkan/shaders/generated/pooling_spv.sha256
                    src/vulkan/shaders/generated/f32_to_double_spv.h
                    src/vulkan/shaders/generated/linear_relu_backward_spv.h
                    src/vulkan/shaders/generated/linear_relu_backward_spv.sha256
                     src/vulkan/shaders/generated/formatter_double_spv.h
                     src/vulkan/shaders/generated/formatter_double_spv.sha256
                   src/vulkan/shaders/glsl/pointwise.comp
                  src/vulkan/shaders/glsl/reduction.comp src/vulkan/shaders/glsl/indexing.comp
                    src/vulkan/shaders/glsl/broadcast.comp
                    src/vulkan/shaders/glsl/masked_select.comp
                     src/vulkan/shaders/glsl/convolution.comp
                       src/vulkan/shaders/glsl/pooling.comp)
        add_custom_target(vulkan_linear_relu_backward_shader_integrity
                          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_linear_relu_backward_spv.py
                          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_linear_relu_backward_spv.py
                                  ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_linear_relu_backward_spv.py
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/linear_relu_backward_spv.h
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/linear_relu_backward_spv.sha256
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/linear_relu_backward.comp)
        add_custom_command(
          OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/linear_relu_backward_spv.h
                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/linear_relu_backward_spv.sha256
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_linear_relu_backward_spv.py
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_linear_relu_backward_spv.py
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/linear_relu_backward.comp
          VERBATIM)
        add_custom_target(vulkan_linear_relu_backward_shader_generate DEPENDS
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/linear_relu_backward_spv.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/linear_relu_backward_spv.sha256)
        add_dependencies(vulkan_platform vulkan_linear_relu_backward_shader_generate)
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

    add_executable(vulkan_execution_test tests/cpp/vulkan_execution_test.cpp)
    target_compile_features(vulkan_execution_test PRIVATE cxx_std_17)
    target_link_libraries(vulkan_execution_test PRIVATE vulkan_buffer vulkan_platform)
    if(BUILD_TESTING)
        add_test(NAME vulkan_execution_test COMMAND vulkan_execution_test)
        set_tests_properties(vulkan_execution_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(BUILD_VULKAN_PROBE OR BUILD_PYTHON_EXTENSION)
    add_executable(vulkan_multi_output_backward_test
                   tests/cpp/vulkan_multi_output_backward_test.cpp)
    target_compile_features(vulkan_multi_output_backward_test PRIVATE cxx_std_17)
    target_include_directories(vulkan_multi_output_backward_test PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(vulkan_multi_output_backward_test PRIVATE vulkan_buffer
                          vulkan_platform)
    if(BUILD_TESTING)
        add_test(NAME vulkan_multi_output_backward_test
                 COMMAND vulkan_multi_output_backward_test)
        set_tests_properties(vulkan_multi_output_backward_test PROPERTIES
                             SKIP_RETURN_CODE 77)
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

    add_library(vulkan_allocator STATIC src/vulkan_allocator.cpp src/vulkan_layout.cpp
                src/vulkan_transfer.cpp src/vulkan/operators/capability.cpp)
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
        pytorch_vulkan_python MODULE src/python_module.cpp src/vulkan/operators/autograd.cpp
        src/vulkan/operators/add.cpp
        src/vulkan/operators/sub.cpp src/vulkan/operators/mul.cpp
         src/vulkan/operators/autograd.h src/vulkan/operators/binary.h
         src/vulkan/operators/out.cpp src/vulkan/operators/optimizer.cpp
          src/vulkan/operators/out.h src/vulkan/operators/capability.h
          src/vulkan/operators/unary.cpp
         src/vulkan/operators/unary.h src/vulkan/operators/view.cpp
          src/vulkan/operators/view.h src/vulkan/operators/reduction.cpp
          src/vulkan/operators/reduction.h src/vulkan/operators/indexing.cpp
           src/vulkan/operators/indexing.h src/vulkan/operators/linear.cpp
           src/vulkan/operators/linear.h src/vulkan/operators/convolution.cpp
           src/vulkan/operators/convolution.h src/vulkan/operators/pooling.cpp
             src/vulkan/operators/pooling.h src/vulkan/operators/comparison.cpp
              src/vulkan/operators/comparison.h src/vulkan/operators/optimizer.h
               src/vulkan/operators/f32_to_double.cpp
              src/vulkan/operators/formatter_double.cpp src/vulkan/operators/formatter_double.h)
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
    add_custom_target(vulkan_reduction_shader_integrity
                      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_reduction_indexing_spv.py
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_reduction_indexing_spv.py
                              ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_reduction_indexing_spv.py
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/reduction_indexing_spv.h
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/reduction_indexing_spv.sha256
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/reduction.comp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/indexing.comp
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/broadcast.comp)
      add_custom_target(vulkan_model_shader_integrity
                       COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_model_spv.py
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_model_spv.py
                               ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_model_spv.py
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/model_spv.h
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/model_spv.sha256
                                ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/model.comp)
        add_custom_target(vulkan_masked_select_shader_integrity
                         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_masked_select_spv.py
                         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_masked_select_spv.py
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/masked_select_spv.h
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/masked_select_spv.sha256
                                ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/masked_select.comp)
       add_custom_target(vulkan_convolution_shader_integrity
                         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_convolution_spv.py
                         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_convolution_spv.py
                                 ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_convolution_spv.py
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/convolution_spv.h
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/convolution_spv.sha256
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/convolution.comp)
       add_custom_target(vulkan_pooling_shader_integrity
                          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_pooling_spv.py
                          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_pooling_spv.py
                          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_pooling_spv.py
                                  ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_pooling_spv.py
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/pooling_spv.h
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/pooling_spv.sha256
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/pooling.comp)
       add_custom_target(vulkan_formatter_double_shader_integrity
                         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_formatter_double_spv.py
                         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_formatter_double_spv.py
                                 ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_formatter_double_spv.py
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/formatter_double_spv.h
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/formatter_double_spv.sha256
                                  ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/formatter_double.comp)
       add_custom_target(vulkan_pointwise_shader_integrity
                         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_pointwise_spv.py
                         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_pointwise_spv.py
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/pointwise_spv.h
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/generated/pointwise_spv.sha256
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders/glsl/pointwise.comp)
      if(BUILD_TESTING)
          add_test(NAME vulkan_reduction_shader_integrity
                   COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_reduction_indexing_spv.py)
          add_test(NAME vulkan_model_shader_integrity
                   COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_model_spv.py)
            add_test(NAME vulkan_masked_select_shader_integrity
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_masked_select_spv.py)
            add_test(NAME vulkan_convolution_shader_integrity
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_convolution_spv.py)
           add_test(NAME vulkan_pooling_shader_integrity
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_pooling_spv.py)
           add_test(NAME vulkan_formatter_double_shader_integrity
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_formatter_double_spv.py)
            add_test(NAME vulkan_pointwise_shader_integrity
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_pointwise_spv.py)
            add_test(NAME vulkan_linear_relu_backward_shader_integrity
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_linear_relu_backward_spv.py)
      endif()
    configure_file(
         "${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_vulkan/serialization.py"
         "${CMAKE_BINARY_DIR}/pytorch_vulkan/serialization.py"
          COPYONLY)
     configure_file(
         "${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_vulkan/compiler.py"
         "${CMAKE_BINARY_DIR}/pytorch_vulkan/compiler.py" COPYONLY)
     configure_file(
          "${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_vulkan/compiler_metadata.py"
          "${CMAKE_BINARY_DIR}/pytorch_vulkan/compiler_metadata.py" COPYONLY)

     if(BUILD_TESTING)
         add_test(NAME vulkan_python_suite
                  COMMAND ${Python3_EXECUTABLE} -m pytest -q -rs
                          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)
         set_tests_properties(vulkan_python_suite PROPERTIES
                              WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                              ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}")
     endif()

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
                    src/vulkan/operators/add.cpp src/vulkan/operators/sub.cpp
                    src/vulkan/operators/mul.cpp src/vulkan/operators/out.cpp
                    src/vulkan/operators/unary.cpp src/vulkan/operators/autograd.cpp
                     src/vulkan/operators/view.cpp src/vulkan/operators/reduction.cpp
                      src/vulkan/operators/indexing.cpp src/vulkan/operators/comparison.cpp
                       src/vulkan/operators/linear.cpp src/vulkan/operators/convolution.cpp
                        src/vulkan/operators/f32_to_double.cpp
                        src/vulkan/operators/formatter_double.cpp
                       src/vulkan/operators/pooling.cpp)
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

    add_executable(vulkan_dtype_test tests/cpp/vulkan_dtype_test.cpp)
    target_compile_features(vulkan_dtype_test PRIVATE cxx_std_17)
    target_include_directories(vulkan_dtype_test PRIVATE ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        target_compile_options(vulkan_dtype_test PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(vulkan_dtype_test PRIVATE vulkan_allocator
                          vulkan_device_guard vulkan_buffer vulkan_platform
                          ${TORCH_LIBRARIES})
    if(BUILD_TESTING)
        add_test(NAME vulkan_dtype_test COMMAND vulkan_dtype_test)
        set_tests_properties(vulkan_dtype_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(vulkan_serialization_probe tests/cpp/vulkan_serialization_probe.cpp)
    target_compile_features(vulkan_serialization_probe PRIVATE cxx_std_17)
    target_include_directories(vulkan_serialization_probe PRIVATE ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        target_compile_options(vulkan_serialization_probe PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(vulkan_serialization_probe PRIVATE vulkan_allocator
                          vulkan_device_guard vulkan_buffer vulkan_platform
                          ${TORCH_LIBRARIES})
    if(BUILD_TESTING)
        add_test(NAME vulkan_serialization_probe COMMAND vulkan_serialization_probe)
        set_tests_properties(vulkan_serialization_probe PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
