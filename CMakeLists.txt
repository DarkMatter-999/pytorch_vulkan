cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dlprim_backend LANGUAGES CXX)

# The legacy OpenCL implementation is retained as reference material only.
# Vulkan and CPU integration will add active targets in later phases.
message(STATUS "GPU backend disabled; configuring CPU-only project shell")
