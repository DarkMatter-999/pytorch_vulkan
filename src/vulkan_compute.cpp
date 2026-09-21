#include "vulkan_compute.h"

#include "vulkan/shaders/generated/convolution_spv.h"
#include "vulkan/shaders/generated/f32_to_double_spv.h"
#include "vulkan/shaders/generated/formatter_double_spv.h"
#include "vulkan/shaders/generated/gemm_spv.h"
#include "vulkan/shaders/generated/linear_relu_backward_spv.h"
#include "vulkan/shaders/generated/loss_spv.h"
#include "vulkan/shaders/generated/masked_select_spv.h"
#include "vulkan/shaders/generated/model_spv.h"
#include "vulkan/shaders/generated/operator_spv.h"
#include "vulkan/shaders/generated/pointwise_spv.h"
#include "vulkan/shaders/generated/pooling_spv.h"
#include "vulkan/shaders/generated/reduction_indexing_spv.h"
#include "vulkan_buffer.h"
#include "vulkan_execution.h"
#include "vulkan/shader_registry.h"
#include "vulkan/pipeline_cache.h"
#include "vulkan/descriptor_arena.h"
#include "vulkan_platform.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

struct OperatorParams {
    uint32_t mode;
    uint32_t batch;
    uint32_t channels;
    uint32_t spatial;
    uint32_t classes;
    int32_t ignore_index;
    uint32_t padding0;
    uint32_t padding1;
    float momentum;
    float eps;
};

void check_result(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

std::runtime_error contextual_error(const char *operation,
                                    const std::exception &error) {
    return std::runtime_error(std::string("Vulkan compute ") + operation + ": " +
                              error.what());
}

struct TensorMetadata {
    uint32_t rank;
    uint32_t sizes[8];
    uint32_t strides[8];
    uint32_t storage_offset;
    uint32_t padding[2];
};
struct Params {
    float scalar;
    uint32_t element_count;
    uint32_t operation;
    uint32_t padding;
};
struct PointwiseMetadata {
    uint32_t data[80];
};

struct ReductionParams {
    uint32_t rank, output_numel, reduce_mask, reduce_numel;
    uint32_t sizes[8];
    uint32_t strides[8];
    uint32_t storage_offset;
    uint32_t operation;
    uint32_t reduce_dim;
};
struct ReductionBackwardParams {
    uint32_t rank, input_numel, reduce_numel, reduce_mask, reduce_dim, operation,
        keepdim;
    uint32_t sizes[8];
    uint32_t strides[8];
    uint32_t storage_offset;
};
struct LossParams {
    uint32_t element_count, reduction, backward, padding;
};

struct IndexingParams {
    uint32_t rank, output_numel, reduce_dim, reduce_size;
    uint32_t sizes[8];
    uint32_t strides[8];
    uint32_t storage_offset;
};

struct BroadcastParams {
    uint32_t rank, output_numel;
    float scale;
};
struct BroadcastMetadata {
    uint32_t input_sizes[8];
    uint32_t output_sizes[8];
    uint32_t sizes[8];
    uint32_t strides[8];
    uint32_t storage_offset;
};

struct LinearParams {
    uint32_t rows, features, outputs, transposed_weight, has_bias, operation;
};
struct MultiOutputParams {
    uint32_t rows, input_features, output_features;
    uint32_t input_region_offset, input_region_count;
    uint32_t weight_region_offset, weight_region_count;
    uint32_t bias_region_offset, bias_region_count;
};
struct ModelMetadata {
    TensorMetadata tensors[4];
};
struct MultiOutputMetadata {
    TensorMetadata tensors[7];
};
struct PoolingMetadata {
    TensorMetadata tensors[2];
};
struct ConvolutionParams {
    uint32_t batch, input_channels, input_height, input_width, output_channels,
        output_height, output_width, kernel_height, kernel_width, operation;
};
struct MaskedParams {
    uint32_t element_count;
};
struct PoolingParams {
    uint32_t batch, channels, height, width, operation;
};
struct GemmParams {
    uint32_t m, n, k;
    uint32_t a_row_stride, a_col_stride;
    uint32_t b_row_stride, b_col_stride;
    uint32_t c_row_stride, c_col_stride;
    uint32_t d_row_stride, d_col_stride;
    uint32_t bias_stride;
    float alpha, beta;
    uint32_t has_bias, batch_count;
    uint32_t batch_stride_a, batch_stride_b, batch_stride_c, batch_stride_d;
};
static_assert(sizeof(GemmParams) == 80, "GEMM push-constant ABI size mismatch");
static_assert(offsetof(GemmParams, m) == 0, "GEMM ABI m offset mismatch");
static_assert(offsetof(GemmParams, n) == 4, "GEMM ABI n offset mismatch");
static_assert(offsetof(GemmParams, k) == 8, "GEMM ABI k offset mismatch");
static_assert(offsetof(GemmParams, a_row_stride) == 12, "GEMM ABI A row offset mismatch");
static_assert(offsetof(GemmParams, a_col_stride) == 16, "GEMM ABI A col offset mismatch");
static_assert(offsetof(GemmParams, b_row_stride) == 20, "GEMM ABI B row offset mismatch");
static_assert(offsetof(GemmParams, b_col_stride) == 24, "GEMM ABI B col offset mismatch");
static_assert(offsetof(GemmParams, c_row_stride) == 28, "GEMM ABI C row offset mismatch");
static_assert(offsetof(GemmParams, c_col_stride) == 32, "GEMM ABI C col offset mismatch");
static_assert(offsetof(GemmParams, d_row_stride) == 36, "GEMM ABI D row offset mismatch");
static_assert(offsetof(GemmParams, d_col_stride) == 40, "GEMM ABI D col offset mismatch");
static_assert(offsetof(GemmParams, bias_stride) == 44, "GEMM ABI bias offset mismatch");
static_assert(offsetof(GemmParams, alpha) == 48, "GEMM ABI alpha offset mismatch");
static_assert(offsetof(GemmParams, beta) == 52, "GEMM ABI beta offset mismatch");
static_assert(offsetof(GemmParams, has_bias) == 56, "GEMM ABI has_bias offset mismatch");
static_assert(offsetof(GemmParams, batch_count) == 60, "GEMM ABI batch offset mismatch");
struct F32ToDoubleParams {
    uint32_t element_count;
};
struct FormatterDoubleParams {
    double scalar;
    uint32_t element_count;
    uint32_t operation;
};

constexpr uint32_t kAdd = 0;
constexpr uint32_t kWorkgroupSize = 256;
constexpr uint32_t kMaxPointwiseRank = 8;
constexpr uint32_t kGemmDescriptorPoolCapacity = 4096;
// VulkanTensorLayout stores the ATen ScalarType as its stable integer value.
constexpr int kFloatScalarType = 6;

bool ranges_overlap(VkBuffer lhs_buffer, const VulkanTensorLayout &lhs,
                    VkBuffer rhs_buffer, const VulkanTensorLayout &rhs) {
    if (lhs_buffer != rhs_buffer || lhs.byte_range == 0 || rhs.byte_range == 0)
        return false;
    const uint64_t lhs_end = static_cast<uint64_t>(lhs.byte_offset) + lhs.byte_range;
    const uint64_t rhs_end = static_cast<uint64_t>(rhs.byte_offset) + rhs.byte_range;
    return static_cast<uint64_t>(lhs.byte_offset) < rhs_end &&
           static_cast<uint64_t>(rhs.byte_offset) < lhs_end;
}

template <typename Params>
void fill_layout_metadata(Params &params, const VulkanTensorLayout &layout,
                          const char *operation) {
    if (layout.rank < 0 || layout.rank > 8)
        throw std::invalid_argument(std::string("Vulkan compute ") + operation +
                                    " supports ranks up to 8");
    if (layout.sizes.size() != static_cast<size_t>(layout.rank) ||
        layout.strides.size() != static_cast<size_t>(layout.rank))
        throw std::invalid_argument(std::string("Vulkan compute ") + operation +
                                    " layout metadata lengths do not match rank");
    if (layout.storage_offset < 0 || static_cast<uint64_t>(layout.storage_offset) >
                                         std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(std::string("Vulkan compute ") + operation +
                                    " storage offset overflows");
    params.rank = static_cast<uint32_t>(layout.rank);
    params.storage_offset = static_cast<uint32_t>(layout.storage_offset);
    for (uint32_t i = 0; i < params.rank; ++i) {
        if (layout.sizes[i] < 0 || layout.strides[i] < 0 ||
            static_cast<uint64_t>(layout.sizes[i]) >
                std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(layout.strides[i]) >
                std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument(std::string("Vulkan compute ") + operation +
                                        " layout metadata overflows");
        params.sizes[i] = static_cast<uint32_t>(layout.sizes[i]);
        params.strides[i] = static_cast<uint32_t>(layout.strides[i]);
    }
}

TensorMetadata pointwise_metadata(const VulkanTensorLayout &layout) {
    if (layout.rank < 0 || layout.rank > kMaxPointwiseRank)
        throw std::invalid_argument("Vulkan compute pointwise supports ranks up to 8");
    TensorMetadata params{static_cast<uint32_t>(layout.rank),
                          {},
                          {},
                          static_cast<uint32_t>(layout.storage_offset),
                          {}};
    for (uint32_t dim = 0; dim < params.rank; ++dim) {
        if (layout.sizes[dim] > std::numeric_limits<uint32_t>::max() ||
            layout.strides[dim] > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument(
                "Vulkan compute pointwise layout metadata overflows");
        params.sizes[dim] = static_cast<uint32_t>(layout.sizes[dim]);
        params.strides[dim] = static_cast<uint32_t>(layout.strides[dim]);
    }
    if (layout.storage_offset < 0 || static_cast<uint64_t>(layout.storage_offset) >
                                         std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(
            "Vulkan compute pointwise storage offset overflows");
    return params;
}

uint64_t checked_product(uint64_t lhs, uint64_t rhs, const char *name) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        throw std::invalid_argument(std::string("Vulkan pooling ") + name +
                                    " product overflow");
    return lhs * rhs;
}

uint64_t checked_dispatch_groups(uint32_t output_numel) {
    constexpr uint64_t rounding = static_cast<uint64_t>(kWorkgroupSize) - 1;
    if (static_cast<uint64_t>(output_numel) >
        std::numeric_limits<uint64_t>::max() - rounding)
        throw std::invalid_argument(
            "Vulkan compute dispatch group count overflows uint64");
    return (static_cast<uint64_t>(output_numel) + rounding) / kWorkgroupSize;
}

VkDeviceSize checked_bytes(uint64_t elements, const char *name) {
    const uint64_t bytes = checked_product(elements, sizeof(float), name);
    if (bytes > std::numeric_limits<VkDeviceSize>::max())
        throw std::invalid_argument(std::string("Vulkan pooling ") + name +
                                    " byte range overflow");
    return static_cast<VkDeviceSize>(bytes);
}

} // namespace

VulkanCompute::VulkanCompute(const VulkanPlatform &platform)
    : platform_(platform), device_(platform.device()), queue_(platform.compute_queue()),
      command_pool_(platform.command_pool()),
      descriptor_arena_(std::make_unique<DescriptorArena>(
          device_, platform.execution_context())) {
    try {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(platform.physical_device(), &properties);
        max_storage_buffer_range_ = properties.limits.maxStorageBufferRange;
        max_compute_workgroup_count_x_ = properties.limits.maxComputeWorkGroupCount[0];
        max_compute_workgroup_count_y_ = properties.limits.maxComputeWorkGroupCount[1];
        max_compute_workgroup_count_z_ = properties.limits.maxComputeWorkGroupCount[2];
        max_compute_shared_memory_size_ = properties.limits.maxComputeSharedMemorySize;
        max_push_constants_size_ = properties.limits.maxPushConstantsSize;
        if (properties.limits.maxPushConstantsSize < sizeof(Params))
            throw std::runtime_error(
                "device maxPushConstantsSize is smaller than pointwise ABI");
        if (properties.limits.maxPushConstantsSize < sizeof(LinearParams))
            throw std::runtime_error(
                "device maxPushConstantsSize is smaller than linear ABI");
        if (properties.limits.maxPushConstantsSize < sizeof(MultiOutputParams))
            throw std::runtime_error(
                "device maxPushConstantsSize is smaller than backward ABI");

        const VkDescriptorSetLayoutBinding lhs_binding = {
            0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr};
        const VkDescriptorSetLayoutBinding rhs_binding = {
            1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr};
        const VkDescriptorSetLayoutBinding output_binding = {
            2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr};
        const VkDescriptorSetLayoutBinding metadata_binding = {
            3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr};
        const VkDescriptorSetLayoutBinding tensor_tensor_bindings[] = {
            lhs_binding, rhs_binding, output_binding, metadata_binding};
        const VkDescriptorSetLayoutBinding tensor_scalar_bindings[] = {
            lhs_binding, output_binding, metadata_binding};
        const VkDescriptorSetLayoutBinding scalar_tensor_bindings[] = {
            lhs_binding, output_binding, metadata_binding};
        const VkDescriptorSetLayoutBinding unary_bindings[] = {
            lhs_binding, output_binding, metadata_binding};

        const VkShaderModuleCreateInfo tensor_tensor_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kTensorTensorCodeSize,
            vulkan_pointwise_shader::kTensorTensorCode};
        const VkShaderModuleCreateInfo tensor_scalar_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kTensorScalarCodeSize,
            vulkan_pointwise_shader::kTensorScalarCode};
        const VkShaderModuleCreateInfo scalar_tensor_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kScalarTensorCodeSize,
            vulkan_pointwise_shader::kScalarTensorCode};
        const VkShaderModuleCreateInfo unary_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kUnaryCodeSize,
            vulkan_pointwise_shader::kUnaryCode};
        const VkShaderModuleCreateInfo bool_tensor_tensor_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kBoolTensorTensorCodeSize,
            vulkan_pointwise_shader::kBoolTensorTensorCode};
        const VkShaderModuleCreateInfo bool_output_tensor_scalar_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kBoolOutputTensorScalarCodeSize,
            vulkan_pointwise_shader::kBoolOutputTensorScalarCode};
        const VkShaderModuleCreateInfo bool_output_unary_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kBoolOutputUnaryCodeSize,
            vulkan_pointwise_shader::kBoolOutputUnaryCode};
        const VkShaderModuleCreateInfo bool_output_tensor_tensor_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kBoolOutputTensorTensorCodeSize,
            vulkan_pointwise_shader::kBoolOutputTensorTensorCode};
        const VkDescriptorSetLayoutBinding compound_bindings[] = {
            lhs_binding,
            rhs_binding,
            output_binding,
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr}};
        const VkShaderModuleCreateInfo compound_mul_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kCompoundMulCodeSize,
            vulkan_pointwise_shader::kCompoundMulCode};
        const VkShaderModuleCreateInfo compound_div_shader{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_pointwise_shader::kCompoundDivCodeSize,
            vulkan_pointwise_shader::kCompoundDivCode};

        const auto create_mode = [&](uint32_t mode, const char *name,
                                     const VkDescriptorSetLayoutBinding *bindings,
                                     uint32_t binding_count,
                                     const VkShaderModuleCreateInfo &shader_info) {
            VkDescriptorSetLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout.bindingCount = binding_count;
            layout.pBindings = bindings;
            check_result(vkCreateDescriptorSetLayout(device_, &layout, nullptr,
                                                     &descriptor_set_layouts_[mode]),
                         "could not create pointwise descriptor-set layout");
            shader_modules_[mode] = platform_.shader_registry().get_or_create(
                {name, vulkan_shader_code_hash(shader_info.pCode,
                                               shader_info.codeSize / sizeof(uint32_t))},
                shader_info.pCode, shader_info.codeSize / sizeof(uint32_t));
        };
        create_mode(0, "pointwise_tensor_tensor", tensor_tensor_bindings, 4,
                    tensor_tensor_shader);
        create_mode(1, "pointwise_tensor_scalar", tensor_tensor_bindings, 4,
                    tensor_scalar_shader);
        create_mode(2, "pointwise_scalar_tensor", tensor_tensor_bindings, 4,
                    scalar_tensor_shader);
        create_mode(3, "pointwise_unary", tensor_tensor_bindings, 4, unary_shader);
        if (platform.supports_bool_pointwise()) {
            create_mode(4, "pointwise_bool_tensor_tensor", tensor_tensor_bindings, 4,
                        bool_tensor_tensor_shader);
            create_mode(5, "pointwise_bool_output_tensor_scalar", tensor_tensor_bindings,
                        4, bool_output_tensor_scalar_shader);
            create_mode(6, "pointwise_bool_output_unary", tensor_tensor_bindings, 4,
                        bool_output_unary_shader);
            create_mode(7, "pointwise_bool_output_tensor_tensor", tensor_tensor_bindings,
                        4, bool_output_tensor_tensor_shader);
        }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
        const auto create_pipeline = [&](uint32_t mode, const char *name,
                                         const VkShaderModuleCreateInfo &shader_info) {
            VkPipelineLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout.setLayoutCount = 1;
            layout.pSetLayouts = &descriptor_set_layouts_[mode];
            layout.pushConstantRangeCount = 1;
            layout.pPushConstantRanges = &push;
            pipeline_layouts_[mode] = platform_.pipeline_cache().get_or_create_layout(
                {reinterpret_cast<uint64_t>(descriptor_set_layouts_[mode]),
                 VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)},
                layout);
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader_modules_[mode];
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline.stage = stage;
            pipeline.layout = pipeline_layouts_[mode];
            pipelines_[mode] = platform_.pipeline_cache().get_or_create(
                {name, reinterpret_cast<uint64_t>(descriptor_set_layouts_[mode]),
                 vulkan_shader_code_hash(shader_info.pCode,
                                         shader_info.codeSize / sizeof(uint32_t)),
                 {}, 0},
                pipeline_layouts_[mode], shader_modules_[mode], pipeline);
        };
        create_pipeline(0, "pointwise_tensor_tensor", tensor_tensor_shader);
        create_pipeline(1, "pointwise_tensor_scalar", tensor_scalar_shader);
        create_pipeline(2, "pointwise_scalar_tensor", scalar_tensor_shader);
        create_pipeline(3, "pointwise_unary", unary_shader);
        if (platform.supports_bool_pointwise()) {
            create_pipeline(4, "pointwise_bool_tensor_tensor", bool_tensor_tensor_shader);
            create_pipeline(5, "pointwise_bool_output_tensor_scalar",
                            bool_output_tensor_scalar_shader);
            create_pipeline(6, "pointwise_bool_output_unary", bool_output_unary_shader);
            create_pipeline(7, "pointwise_bool_output_tensor_tensor",
                            bool_output_tensor_tensor_shader);
        }
        const auto create_compound = [&](uint32_t index, const char *name,
                                         const VkShaderModuleCreateInfo &shader_info) {
            VkDescriptorSetLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout.bindingCount = 5;
            layout.pBindings = compound_bindings;
            check_result(
                vkCreateDescriptorSetLayout(device_, &layout, nullptr,
                                            &compound_descriptor_layouts_[index]),
                "could not create compound descriptor-set layout");
            compound_shader_modules_[index] = platform_.shader_registry().get_or_create(
                {name, vulkan_shader_code_hash(shader_info.pCode,
                                               shader_info.codeSize / sizeof(uint32_t))},
                shader_info.pCode, shader_info.codeSize / sizeof(uint32_t));
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
            VkPipelineLayoutCreateInfo pipeline_layout{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipeline_layout.setLayoutCount = 1;
            pipeline_layout.pSetLayouts = &compound_descriptor_layouts_[index];
            pipeline_layout.pushConstantRangeCount = 1;
            pipeline_layout.pPushConstantRanges = &push;
            compound_pipeline_layouts_[index] =
                platform_.pipeline_cache().get_or_create_layout(
                    {reinterpret_cast<uint64_t>(compound_descriptor_layouts_[index]),
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)},
                    pipeline_layout);
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = compound_shader_modules_[index];
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline.stage = stage;
            pipeline.layout = compound_pipeline_layouts_[index];
            compound_pipelines_[index] = platform_.pipeline_cache().get_or_create(
                {name, reinterpret_cast<uint64_t>(compound_descriptor_layouts_[index]),
                 vulkan_shader_code_hash(shader_info.pCode,
                                         shader_info.codeSize / sizeof(uint32_t)),
                 {}, 0},
                compound_pipeline_layouts_[index], compound_shader_modules_[index], pipeline);
        };
        create_compound(0, "pointwise_compound_mul", compound_mul_shader);
        create_compound(1, "pointwise_compound_div", compound_div_shader);
        const VkDescriptorSetLayoutBinding reduction_bindings[] = {
            lhs_binding,
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr}};
        const auto create_extra = [&](const char *name,
                                      VkDescriptorSetLayout &descriptor_layout,
                                      VkShaderModule &module,
                                      VkPipelineLayout &pipeline_layout,
                                      VkPipeline &pipeline, const uint32_t *code,
                                      std::size_t code_size,
                                      uint32_t push_size = sizeof(ReductionParams),
                                      uint32_t descriptor_count = 2) {
            VkDescriptorSetLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout.bindingCount = descriptor_count;
            layout.pBindings = reduction_bindings;
            check_result(vkCreateDescriptorSetLayout(device_, &layout, nullptr,
                                                     &descriptor_layout),
                         "could not create reduction descriptor-set layout");
            module = platform_.shader_registry().get_or_create(
                {name, vulkan_shader_code_hash(code, code_size / sizeof(uint32_t))}, code,
                code_size / sizeof(uint32_t));
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
            VkPipelineLayoutCreateInfo pipeline_layout_info{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipeline_layout_info.setLayoutCount = 1;
            pipeline_layout_info.pSetLayouts = &descriptor_layout;
            pipeline_layout_info.pushConstantRangeCount = 1;
            pipeline_layout_info.pPushConstantRanges = &push;
            pipeline_layout = platform_.pipeline_cache().get_or_create_layout(
                {reinterpret_cast<uint64_t>(descriptor_layout), VK_SHADER_STAGE_COMPUTE_BIT,
                 0, push_size},
                pipeline_layout_info);
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = module;
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline_info{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline_layout;
            pipeline = platform_.pipeline_cache().get_or_create(
                {name, reinterpret_cast<uint64_t>(descriptor_layout),
                 vulkan_shader_code_hash(code, code_size / sizeof(uint32_t)), {}, 0},
                pipeline_layout, module, pipeline_info);
        };
        create_extra("reduction", reduction_descriptor_layout_, reduction_shader_,
                     reduction_pipeline_layout_, reduction_pipeline_,
                     vulkan_reduction_shader::kReductionCode,
                     vulkan_reduction_shader::kReductionCodeSize);
        create_extra("reduction_backward", reduction_backward_descriptor_layout_,
                     reduction_backward_shader_,
                     reduction_backward_pipeline_layout_, reduction_backward_pipeline_,
                     vulkan_reduction_shader::kReductionBackwardCode,
                     vulkan_reduction_shader::kReductionBackwardCodeSize,
                     sizeof(ReductionBackwardParams), 4);
        create_extra("loss", loss_descriptor_layout_, loss_shader_, loss_pipeline_layout_,
                     loss_pipeline_, vulkan_loss_shader::kCode,
                     vulkan_loss_shader::kCodeSize, sizeof(LossParams), 4);
        create_extra("indexing", indexing_descriptor_layout_, indexing_shader_,
                     indexing_pipeline_layout_, indexing_pipeline_,
                     vulkan_reduction_shader::kIndexingCode,
                     vulkan_reduction_shader::kIndexingCodeSize);
        create_extra(
            "broadcast",
            broadcast_descriptor_layout_, broadcast_shader_, broadcast_pipeline_layout_,
            broadcast_pipeline_, vulkan_reduction_shader::kBroadcastCode,
            vulkan_reduction_shader::kBroadcastCodeSize, sizeof(BroadcastParams), 3);
        create_extra("pooling", pooling_descriptor_layout_, pooling_shader_,
                     pooling_pipeline_layout_, pooling_pipeline_,
                     vulkan_pooling_shader::kCode, vulkan_pooling_shader::kCodeSize,
                     sizeof(PoolingParams), 3);
        create_extra("normalization", normalization_descriptor_layout_, normalization_shader_,
                     normalization_pipeline_layout_, normalization_pipeline_,
                     vulkan_normalization_shader::kCode,
                     vulkan_normalization_shader::kCodeSize, sizeof(OperatorParams),
                     10);
        create_extra("classification", classification_descriptor_layout_, classification_shader_,
                     classification_pipeline_layout_, classification_pipeline_,
                     vulkan_classification_shader::kCode,
                     vulkan_classification_shader::kCodeSize, sizeof(OperatorParams),
                     6);
        if (platform.supports_formatter_double()) {
            create_extra("f32_to_double", f32_to_double_descriptor_layout_,
                         f32_to_double_shader_,
                         f32_to_double_pipeline_layout_, f32_to_double_pipeline_,
                         vulkan_f32_to_double_shader::kCode,
                         vulkan_f32_to_double_shader::kCodeSize,
                         sizeof(F32ToDoubleParams));
            const VkDescriptorSetLayoutBinding formatter_bindings[] = {
                lhs_binding,
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                 nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                 nullptr},
                {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                 nullptr}};
            VkDescriptorSetLayoutCreateInfo formatter_layout{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            formatter_layout.bindingCount = 4;
            formatter_layout.pBindings = formatter_bindings;
            check_result(
                vkCreateDescriptorSetLayout(device_, &formatter_layout, nullptr,
                                            &formatter_double_descriptor_layout_),
                "could not create formatter descriptor layout");
            VkShaderModuleCreateInfo formatter_shader{
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                vulkan_formatter_double_shader::kCodeSize,
                vulkan_formatter_double_shader::kCode};
            check_result(vkCreateShaderModule(device_, &formatter_shader, nullptr,
                                              &formatter_double_shader_),
                         "could not create formatter shader module");
            VkPushConstantRange formatter_push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                               sizeof(FormatterDoubleParams)};
            VkPipelineLayoutCreateInfo formatter_pipeline_layout{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            formatter_pipeline_layout.setLayoutCount = 1;
            formatter_pipeline_layout.pSetLayouts =
                &formatter_double_descriptor_layout_;
            formatter_pipeline_layout.pushConstantRangeCount = 1;
            formatter_pipeline_layout.pPushConstantRanges = &formatter_push;
            check_result(vkCreatePipelineLayout(device_, &formatter_pipeline_layout,
                                                nullptr,
                                                &formatter_double_pipeline_layout_),
                         "could not create formatter pipeline layout");
            VkPipelineShaderStageCreateInfo formatter_stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            formatter_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            formatter_stage.module = formatter_double_shader_;
            formatter_stage.pName = "main";
            VkComputePipelineCreateInfo formatter_pipeline{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            formatter_pipeline.stage = formatter_stage;
            formatter_pipeline.layout = formatter_double_pipeline_layout_;
            check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                                  &formatter_pipeline, nullptr,
                                                  &formatter_double_pipeline_),
                         "could not create formatter pipeline");
        }
        const VkDescriptorSetLayoutBinding model_bindings[] = {
            lhs_binding,
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr}};
        VkDescriptorSetLayoutCreateInfo model_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        model_layout.bindingCount = 5;
        model_layout.pBindings = model_bindings;
        check_result(vkCreateDescriptorSetLayout(device_, &model_layout, nullptr,
                                                 &model_descriptor_layout_),
                     "could not create model descriptor-set layout");
        VkShaderModuleCreateInfo model_shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_model_shader::kLinearCodeSize, vulkan_model_shader::kLinearCode};
        check_result(
            vkCreateShaderModule(device_, &model_shader_info, nullptr, &model_shader_),
            "could not create model shader module");
        VkPushConstantRange model_push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(LinearParams)};
        VkPipelineLayoutCreateInfo model_pipeline_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        model_pipeline_layout_info.setLayoutCount = 1;
        model_pipeline_layout_info.pSetLayouts = &model_descriptor_layout_;
        model_pipeline_layout_info.pushConstantRangeCount = 1;
        model_pipeline_layout_info.pPushConstantRanges = &model_push;
        check_result(vkCreatePipelineLayout(device_, &model_pipeline_layout_info,
                                            nullptr, &model_pipeline_layout_),
                     "could not create model pipeline layout");
        VkPipelineShaderStageCreateInfo model_stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        model_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        model_stage.module = model_shader_;
        model_stage.pName = "main";
        VkComputePipelineCreateInfo model_pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        model_pipeline_info.stage = model_stage;
        model_pipeline_info.layout = model_pipeline_layout_;
        check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                              &model_pipeline_info, nullptr,
                                              &model_pipeline_),
                     "could not create model pipeline");
        // This layout is deliberately independent of model_descriptor_layout_.
        VkDescriptorSetLayoutBinding backward_bindings[8]{};
        for (uint32_t binding = 0; binding < 8; ++binding) {
            backward_bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                          VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo backward_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        backward_layout.bindingCount = 8;
        backward_layout.pBindings = backward_bindings;
        check_result(vkCreateDescriptorSetLayout(device_, &backward_layout, nullptr,
                                                 &backward_descriptor_layout_),
                     "could not create backward descriptor-set layout");
        VkShaderModuleCreateInfo backward_shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_backward_shader::kCodeSize, vulkan_backward_shader::kCode};
        check_result(vkCreateShaderModule(device_, &backward_shader_info, nullptr,
                                          &backward_shader_),
                     "could not create backward shader module");
        VkPushConstantRange backward_push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                          sizeof(MultiOutputParams)};
        VkPipelineLayoutCreateInfo backward_pipeline_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        backward_pipeline_layout_info.setLayoutCount = 1;
        backward_pipeline_layout_info.pSetLayouts = &backward_descriptor_layout_;
        backward_pipeline_layout_info.pushConstantRangeCount = 1;
        backward_pipeline_layout_info.pPushConstantRanges = &backward_push;
        check_result(vkCreatePipelineLayout(device_, &backward_pipeline_layout_info,
                                            nullptr, &backward_pipeline_layout_),
                     "could not create backward pipeline layout");
        VkPipelineShaderStageCreateInfo backward_stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        backward_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        backward_stage.module = backward_shader_;
        backward_stage.pName = "main";
        VkComputePipelineCreateInfo backward_pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        backward_pipeline_info.stage = backward_stage;
        backward_pipeline_info.layout = backward_pipeline_layout_;
        check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                              &backward_pipeline_info, nullptr,
                                              &backward_pipeline_),
                     "could not create backward compute pipeline");
        VkDescriptorSetLayoutCreateInfo convolution_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        convolution_layout.bindingCount = 5;
        convolution_layout.pBindings = model_bindings;
        check_result(vkCreateDescriptorSetLayout(device_, &convolution_layout, nullptr,
                                                 &convolution_descriptor_layout_),
                     "could not create convolution descriptor-set layout");
        VkShaderModuleCreateInfo convolution_shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
            vulkan_convolution_shader::kCodeSize, vulkan_convolution_shader::kCode};
        check_result(vkCreateShaderModule(device_, &convolution_shader_info, nullptr,
                                          &convolution_shader_),
                     "could not create convolution shader module");
        VkPushConstantRange convolution_push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                             sizeof(ConvolutionParams)};
        VkPipelineLayoutCreateInfo convolution_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        convolution_layout_info.setLayoutCount = 1;
        convolution_layout_info.pSetLayouts = &convolution_descriptor_layout_;
        convolution_layout_info.pushConstantRangeCount = 1;
        convolution_layout_info.pPushConstantRanges = &convolution_push;
        check_result(vkCreatePipelineLayout(device_, &convolution_layout_info, nullptr,
                                            &convolution_pipeline_layout_),
                     "could not create convolution pipeline layout");
        VkPipelineShaderStageCreateInfo convolution_stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        convolution_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        convolution_stage.module = convolution_shader_;
        convolution_stage.pName = "main";
        VkComputePipelineCreateInfo convolution_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        convolution_info.stage = convolution_stage;
        convolution_info.layout = convolution_pipeline_layout_;
        check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                              &convolution_info, nullptr,
                                              &convolution_pipeline_),
                     "could not create convolution pipeline");
        const VkDescriptorSetLayoutBinding masked_bindings[] = {
            lhs_binding,
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
             nullptr}};
        const auto create_masked = [&](VkDescriptorSetLayout &descriptor,
                                       VkShaderModule &module,
                                       VkPipelineLayout &layout_handle,
                                       VkPipeline &pipeline, const uint32_t *code,
                                       std::size_t code_size, uint32_t binding_count) {
            VkDescriptorSetLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout.bindingCount = binding_count;
            layout.pBindings = masked_bindings;
            check_result(
                vkCreateDescriptorSetLayout(device_, &layout, nullptr, &descriptor),
                "could not create masked-select descriptor layout");
            VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            nullptr, 0, code_size, code};
            check_result(vkCreateShaderModule(device_, &shader, nullptr, &module),
                         "could not create masked-select shader module");
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                     sizeof(MaskedParams)};
            VkPipelineLayoutCreateInfo pipeline_layout_info{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipeline_layout_info.setLayoutCount = 1;
            pipeline_layout_info.pSetLayouts = &descriptor;
            pipeline_layout_info.pushConstantRangeCount = 1;
            pipeline_layout_info.pPushConstantRanges = &push;
            check_result(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr,
                                                &layout_handle),
                         "could not create masked-select pipeline layout");
            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = module;
            stage.pName = "main";
            VkComputePipelineCreateInfo info{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = layout_handle;
            check_result(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info,
                                                  nullptr, &pipeline),
                         "could not create masked-select pipeline");
        };
        create_masked(masked_count_descriptor_layout_, masked_count_shader_,
                      masked_count_pipeline_layout_, masked_count_pipeline_,
                      vulkan_masked_select_shader::kCountCode,
                      vulkan_masked_select_shader::kCountCodeSize, 3);
        create_masked(masked_compact_descriptor_layout_, masked_compact_shader_,
                      masked_compact_pipeline_layout_, masked_compact_pipeline_,
                      vulkan_masked_select_shader::kCompactCode,
                      vulkan_masked_select_shader::kCompactCodeSize, 4);

        VkDescriptorSetLayoutBinding gemm_bindings[5]{};
        for (uint32_t binding = 0; binding < 5; ++binding) {
            gemm_bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo gemm_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        gemm_layout.bindingCount = 5;
        gemm_layout.pBindings = gemm_bindings;
        check_result(vkCreateDescriptorSetLayout(device_, &gemm_layout, nullptr,
                                                 &gemm_descriptor_layout_),
                     "could not create GEMM descriptor-set layout");
        gemm_shader_ = platform_.shader_registry().get_or_create(
            {"gemm", vulkan_shader_code_hash(
                         vulkan_gemm_shader::kCode,
                         vulkan_gemm_shader::kCodeSize / sizeof(uint32_t))},
            vulkan_gemm_shader::kCode, vulkan_gemm_shader::kCodeSize / sizeof(uint32_t));
        VkPushConstantRange gemm_push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                      sizeof(GemmParams)};
        VkPipelineLayoutCreateInfo gemm_pipeline_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        gemm_pipeline_layout_info.setLayoutCount = 1;
        gemm_pipeline_layout_info.pSetLayouts = &gemm_descriptor_layout_;
        gemm_pipeline_layout_info.pushConstantRangeCount = 1;
        gemm_pipeline_layout_info.pPushConstantRanges = &gemm_push;
        gemm_pipeline_layout_ = platform_.pipeline_cache().get_or_create_layout(
            {reinterpret_cast<uint64_t>(gemm_descriptor_layout_), VK_SHADER_STAGE_COMPUTE_BIT,
             0, sizeof(GemmParams)},
            gemm_pipeline_layout_info);
        VkPipelineShaderStageCreateInfo gemm_stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        gemm_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        gemm_stage.module = gemm_shader_;
        gemm_stage.pName = "main";
        VkComputePipelineCreateInfo gemm_pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        gemm_pipeline_info.stage = gemm_stage;
        gemm_pipeline_info.layout = gemm_pipeline_layout_;
        gemm_pipeline_ = platform_.pipeline_cache().get_or_create(
            {"gemm", reinterpret_cast<uint64_t>(gemm_descriptor_layout_),
             vulkan_shader_code_hash(vulkan_gemm_shader::kCode,
                                     vulkan_gemm_shader::kCodeSize / sizeof(uint32_t)),
             {}, 0},
            gemm_pipeline_layout_, gemm_shader_, gemm_pipeline_info);
    } catch (const std::exception &error) {
        for (auto &pipeline : pipelines_)
            pipeline = VK_NULL_HANDLE;
        compound_pipelines_[0] = VK_NULL_HANDLE;
        compound_pipelines_[1] = VK_NULL_HANDLE;
        reduction_pipeline_ = VK_NULL_HANDLE;
        reduction_backward_pipeline_ = VK_NULL_HANDLE;
        loss_pipeline_ = VK_NULL_HANDLE;
        indexing_pipeline_ = VK_NULL_HANDLE;
        broadcast_pipeline_ = VK_NULL_HANDLE;
        pooling_pipeline_ = VK_NULL_HANDLE;
        normalization_pipeline_ = VK_NULL_HANDLE;
        classification_pipeline_ = VK_NULL_HANDLE;
        f32_to_double_pipeline_ = VK_NULL_HANDLE;
        gemm_pipeline_ = VK_NULL_HANDLE;
        for (auto &layout : pipeline_layouts_)
            layout = VK_NULL_HANDLE;
        for (auto &layout : compound_pipeline_layouts_)
            layout = VK_NULL_HANDLE;
        reduction_pipeline_layout_ = VK_NULL_HANDLE;
        reduction_backward_pipeline_layout_ = VK_NULL_HANDLE;
        loss_pipeline_layout_ = VK_NULL_HANDLE;
        indexing_pipeline_layout_ = VK_NULL_HANDLE;
        broadcast_pipeline_layout_ = VK_NULL_HANDLE;
        pooling_pipeline_layout_ = VK_NULL_HANDLE;
        normalization_pipeline_layout_ = VK_NULL_HANDLE;
        classification_pipeline_layout_ = VK_NULL_HANDLE;
        f32_to_double_pipeline_layout_ = VK_NULL_HANDLE;
        gemm_pipeline_layout_ = VK_NULL_HANDLE;
        platform_.pipeline_cache().destroy_all();
        for (uint32_t mode = 0; mode < 8; ++mode)
            shader_modules_[mode] = VK_NULL_HANDLE;
        compound_shader_modules_[0] = VK_NULL_HANDLE;
        compound_shader_modules_[1] = VK_NULL_HANDLE;
        reduction_shader_ = VK_NULL_HANDLE;
        reduction_backward_shader_ = VK_NULL_HANDLE;
        loss_shader_ = VK_NULL_HANDLE;
        indexing_shader_ = VK_NULL_HANDLE;
        broadcast_shader_ = VK_NULL_HANDLE;
        pooling_shader_ = VK_NULL_HANDLE;
        normalization_shader_ = VK_NULL_HANDLE;
        classification_shader_ = VK_NULL_HANDLE;
        f32_to_double_shader_ = VK_NULL_HANDLE;
        for (auto pipeline : pipelines_) {
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline, nullptr);
            }
        }
        for (auto pipeline : compound_pipelines_) {
            if (pipeline != VK_NULL_HANDLE)
                vkDestroyPipeline(device_, pipeline, nullptr);
        }
        if (reduction_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, reduction_pipeline_, nullptr);
        if (reduction_backward_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, reduction_backward_pipeline_, nullptr);
        if (loss_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, loss_pipeline_, nullptr);
        if (indexing_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, indexing_pipeline_, nullptr);
        if (broadcast_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, broadcast_pipeline_, nullptr);
        if (model_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, model_pipeline_, nullptr);
        if (backward_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, backward_pipeline_, nullptr);
        if (convolution_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, convolution_pipeline_, nullptr);
        if (pooling_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, pooling_pipeline_, nullptr);
        if (masked_count_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, masked_count_pipeline_, nullptr);
        if (masked_compact_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, masked_compact_pipeline_, nullptr);
        if (gemm_pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, gemm_pipeline_, nullptr);
        for (auto module : shader_modules_) {
            if (module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, module, nullptr);
            }
        }
        for (auto module : compound_shader_modules_) {
            if (module != VK_NULL_HANDLE)
                vkDestroyShaderModule(device_, module, nullptr);
        }
        if (reduction_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, reduction_shader_, nullptr);
        if (reduction_backward_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, reduction_backward_shader_, nullptr);
        if (loss_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, loss_shader_, nullptr);
        if (indexing_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, indexing_shader_, nullptr);
        if (broadcast_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, broadcast_shader_, nullptr);
        if (model_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, model_shader_, nullptr);
        if (backward_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, backward_shader_, nullptr);
        if (convolution_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, convolution_shader_, nullptr);
        if (pooling_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, pooling_shader_, nullptr);
        if (masked_count_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, masked_count_shader_, nullptr);
        if (masked_compact_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, masked_compact_shader_, nullptr);
        if (normalization_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, normalization_shader_, nullptr);
        if (classification_shader_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, classification_shader_, nullptr);
        for (auto layout : pipeline_layouts_) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, layout, nullptr);
            }
        }
        for (auto layout : compound_pipeline_layouts_) {
            if (layout != VK_NULL_HANDLE)
                vkDestroyPipelineLayout(device_, layout, nullptr);
        }
        if (reduction_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, reduction_pipeline_layout_, nullptr);
        if (reduction_backward_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, reduction_backward_pipeline_layout_,
                                    nullptr);
        if (loss_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, loss_pipeline_layout_, nullptr);
        if (indexing_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, indexing_pipeline_layout_, nullptr);
        if (broadcast_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, broadcast_pipeline_layout_, nullptr);
        if (model_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, model_pipeline_layout_, nullptr);
        if (backward_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, backward_pipeline_layout_, nullptr);
        if (convolution_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, convolution_pipeline_layout_, nullptr);
        if (pooling_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, pooling_pipeline_layout_, nullptr);
        if (masked_count_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, masked_count_pipeline_layout_, nullptr);
        if (masked_compact_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, masked_compact_pipeline_layout_, nullptr);
        if (normalization_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, normalization_pipeline_layout_, nullptr);
        if (classification_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, classification_pipeline_layout_, nullptr);
        if (gemm_pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, gemm_pipeline_layout_, nullptr);
        for (auto layout : descriptor_set_layouts_) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, layout, nullptr);
            }
        }
        for (auto layout : compound_descriptor_layouts_) {
            if (layout != VK_NULL_HANDLE)
                vkDestroyDescriptorSetLayout(device_, layout, nullptr);
        }
        if (reduction_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, reduction_descriptor_layout_,
                                         nullptr);
        if (reduction_backward_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, reduction_backward_descriptor_layout_,
                                         nullptr);
        if (loss_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, loss_descriptor_layout_, nullptr);
        if (indexing_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, indexing_descriptor_layout_, nullptr);
        if (broadcast_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, broadcast_descriptor_layout_,
                                         nullptr);
        if (model_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, model_descriptor_layout_, nullptr);
        if (backward_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, backward_descriptor_layout_, nullptr);
        if (convolution_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, convolution_descriptor_layout_,
                                         nullptr);
        if (pooling_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, pooling_descriptor_layout_, nullptr);
        if (masked_count_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, masked_count_descriptor_layout_,
                                         nullptr);
        if (masked_compact_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, masked_compact_descriptor_layout_,
                                         nullptr);
        if (normalization_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, normalization_descriptor_layout_,
                                         nullptr);
        if (classification_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, classification_descriptor_layout_,
                                         nullptr);
        if (gemm_descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, gemm_descriptor_layout_, nullptr);
        throw contextual_error("initialization failed", error);
    }
}

VulkanCompute::~VulkanCompute() {
    for (auto &pipeline : pipelines_)
        pipeline = VK_NULL_HANDLE;
    compound_pipelines_[0] = VK_NULL_HANDLE;
    compound_pipelines_[1] = VK_NULL_HANDLE;
    reduction_pipeline_ = VK_NULL_HANDLE;
    reduction_backward_pipeline_ = VK_NULL_HANDLE;
    loss_pipeline_ = VK_NULL_HANDLE;
    indexing_pipeline_ = VK_NULL_HANDLE;
    broadcast_pipeline_ = VK_NULL_HANDLE;
    pooling_pipeline_ = VK_NULL_HANDLE;
    normalization_pipeline_ = VK_NULL_HANDLE;
    classification_pipeline_ = VK_NULL_HANDLE;
    f32_to_double_pipeline_ = VK_NULL_HANDLE;
    gemm_pipeline_ = VK_NULL_HANDLE;
    for (auto &layout : pipeline_layouts_)
        layout = VK_NULL_HANDLE;
    for (auto &layout : compound_pipeline_layouts_)
        layout = VK_NULL_HANDLE;
    reduction_pipeline_layout_ = VK_NULL_HANDLE;
    reduction_backward_pipeline_layout_ = VK_NULL_HANDLE;
    loss_pipeline_layout_ = VK_NULL_HANDLE;
    indexing_pipeline_layout_ = VK_NULL_HANDLE;
    broadcast_pipeline_layout_ = VK_NULL_HANDLE;
    pooling_pipeline_layout_ = VK_NULL_HANDLE;
    normalization_pipeline_layout_ = VK_NULL_HANDLE;
    classification_pipeline_layout_ = VK_NULL_HANDLE;
    f32_to_double_pipeline_layout_ = VK_NULL_HANDLE;
    gemm_pipeline_layout_ = VK_NULL_HANDLE;
    platform_.pipeline_cache().destroy_all();
    for (auto pipeline : pipelines_) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline, nullptr);
        }
    }
    for (auto pipeline : compound_pipelines_) {
        if (pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, pipeline, nullptr);
    }
    if (reduction_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, reduction_pipeline_, nullptr);
    if (reduction_backward_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, reduction_backward_pipeline_, nullptr);
    if (loss_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, loss_pipeline_, nullptr);
    if (indexing_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, indexing_pipeline_, nullptr);
    if (broadcast_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, broadcast_pipeline_, nullptr);
    if (model_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, model_pipeline_, nullptr);
    if (backward_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, backward_pipeline_, nullptr);
    if (convolution_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, convolution_pipeline_, nullptr);
    if (pooling_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, pooling_pipeline_, nullptr);
    if (masked_count_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, masked_count_pipeline_, nullptr);
    if (masked_compact_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, masked_compact_pipeline_, nullptr);
    if (normalization_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, normalization_pipeline_, nullptr);
    if (classification_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, classification_pipeline_, nullptr);
    if (f32_to_double_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, f32_to_double_pipeline_, nullptr);
    if (formatter_double_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, formatter_double_pipeline_, nullptr);
    if (gemm_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, gemm_pipeline_, nullptr);
    for (uint32_t mode = 0; mode < 8; ++mode)
        shader_modules_[mode] = VK_NULL_HANDLE;
    compound_shader_modules_[0] = VK_NULL_HANDLE;
    compound_shader_modules_[1] = VK_NULL_HANDLE;
    reduction_shader_ = VK_NULL_HANDLE;
    reduction_backward_shader_ = VK_NULL_HANDLE;
    loss_shader_ = VK_NULL_HANDLE;
    indexing_shader_ = VK_NULL_HANDLE;
    broadcast_shader_ = VK_NULL_HANDLE;
    pooling_shader_ = VK_NULL_HANDLE;
    normalization_shader_ = VK_NULL_HANDLE;
    classification_shader_ = VK_NULL_HANDLE;
    f32_to_double_shader_ = VK_NULL_HANDLE;
    for (auto module : shader_modules_) {
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, module, nullptr);
        }
    }
    for (auto module : compound_shader_modules_) {
        if (module != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, module, nullptr);
    }
    if (reduction_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, reduction_shader_, nullptr);
    if (reduction_backward_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, reduction_backward_shader_, nullptr);
    if (loss_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, loss_shader_, nullptr);
    if (indexing_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, indexing_shader_, nullptr);
    if (broadcast_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, broadcast_shader_, nullptr);
    if (model_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, model_shader_, nullptr);
    if (backward_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, backward_shader_, nullptr);
    if (convolution_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, convolution_shader_, nullptr);
    if (pooling_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, pooling_shader_, nullptr);
    if (masked_count_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, masked_count_shader_, nullptr);
    if (masked_compact_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, masked_compact_shader_, nullptr);
    if (normalization_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, normalization_shader_, nullptr);
    if (classification_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, classification_shader_, nullptr);
    if (f32_to_double_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, f32_to_double_shader_, nullptr);
    if (formatter_double_shader_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(device_, formatter_double_shader_, nullptr);
    gemm_shader_ = VK_NULL_HANDLE;
    descriptor_arena_.reset();
    for (auto layout : pipeline_layouts_) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, layout, nullptr);
        }
    }
    for (auto layout : compound_pipeline_layouts_) {
        if (layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, layout, nullptr);
    }
    if (reduction_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, reduction_pipeline_layout_, nullptr);
    if (reduction_backward_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, reduction_backward_pipeline_layout_, nullptr);
    if (loss_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, loss_pipeline_layout_, nullptr);
    if (indexing_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, indexing_pipeline_layout_, nullptr);
    if (broadcast_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, broadcast_pipeline_layout_, nullptr);
    if (model_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, model_pipeline_layout_, nullptr);
    if (backward_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, backward_pipeline_layout_, nullptr);
    if (convolution_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, convolution_pipeline_layout_, nullptr);
    if (pooling_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, pooling_pipeline_layout_, nullptr);
    if (masked_count_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, masked_count_pipeline_layout_, nullptr);
    if (masked_compact_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, masked_compact_pipeline_layout_, nullptr);
    if (normalization_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, normalization_pipeline_layout_, nullptr);
    if (classification_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, classification_pipeline_layout_, nullptr);
    if (f32_to_double_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, f32_to_double_pipeline_layout_, nullptr);
    if (formatter_double_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, formatter_double_pipeline_layout_, nullptr);
    if (gemm_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, gemm_pipeline_layout_, nullptr);
    for (auto layout : descriptor_set_layouts_) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, layout, nullptr);
        }
    }
    for (auto layout : compound_descriptor_layouts_) {
        if (layout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, layout, nullptr);
    }
    if (reduction_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, reduction_descriptor_layout_, nullptr);
    if (loss_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, loss_descriptor_layout_, nullptr);
    if (reduction_backward_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, reduction_backward_descriptor_layout_,
                                     nullptr);
    if (indexing_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, indexing_descriptor_layout_, nullptr);
    if (broadcast_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, broadcast_descriptor_layout_, nullptr);
    if (model_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, model_descriptor_layout_, nullptr);
    if (backward_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, backward_descriptor_layout_, nullptr);
    if (convolution_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, convolution_descriptor_layout_, nullptr);
    if (pooling_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, pooling_descriptor_layout_, nullptr);
    if (masked_count_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, masked_count_descriptor_layout_, nullptr);
    if (masked_compact_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, masked_compact_descriptor_layout_,
                                     nullptr);
    if (normalization_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, normalization_descriptor_layout_,
                                     nullptr);
    if (classification_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, classification_descriptor_layout_,
                                     nullptr);
    if (f32_to_double_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, f32_to_double_descriptor_layout_,
                                     nullptr);
    if (formatter_double_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, formatter_double_descriptor_layout_,
                                     nullptr);
    if (gemm_descriptor_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, gemm_descriptor_layout_, nullptr);
}

void VulkanCompute::add(VkBuffer lhs, const VulkanTensorLayout &lhs_layout,
                        VkBuffer rhs, const VulkanTensorLayout &rhs_layout,
                        VkBuffer output,
                        const VulkanTensorLayout &output_layout) const {
    tensor_tensor(lhs, lhs_layout, rhs, rhs_layout, output, output_layout, kAdd);
}

void VulkanCompute::tensor_tensor(VkBuffer lhs, const VulkanTensorLayout &lhs_layout,
                                  VkBuffer rhs, const VulkanTensorLayout &rhs_layout,
                                  VkBuffer output,
                                  const VulkanTensorLayout &output_layout,
                                  uint32_t operation, bool bool_dtype,
                                  float alpha) const {
    dispatch(0, lhs, &lhs_layout, rhs, &rhs_layout, output, output_layout, alpha,
             operation, false, bool_dtype);
}

void VulkanCompute::tensor_tensor_alias(
    VkBuffer lhs, const VulkanTensorLayout &lhs_layout, VkBuffer rhs,
    const VulkanTensorLayout &rhs_layout, VkBuffer output,
    const VulkanTensorLayout &output_layout, uint32_t operation, bool bool_dtype,
    float alpha) const {
    dispatch(0, lhs, &lhs_layout, rhs, &rhs_layout, output, output_layout, alpha,
             operation, true, bool_dtype);
}

void VulkanCompute::tensor_scalar(VkBuffer tensor,
                                  const VulkanTensorLayout &tensor_layout,
                                  VkBuffer output,
                                  const VulkanTensorLayout &output_layout, float scalar,
                                  uint32_t operation, bool bool_dtype) const {
    dispatch(1, tensor, &tensor_layout, VK_NULL_HANDLE, nullptr, output, output_layout,
             scalar, operation, false, bool_dtype);
}

void VulkanCompute::tensor_scalar_alias(VkBuffer tensor,
                                        const VulkanTensorLayout &tensor_layout,
                                        VkBuffer output,
                                        const VulkanTensorLayout &output_layout,
                                        float scalar, uint32_t operation,
                                        bool bool_dtype) const {
    dispatch(1, tensor, &tensor_layout, VK_NULL_HANDLE, nullptr, output, output_layout,
             scalar, operation, true, bool_dtype);
}

void VulkanCompute::compound_tensor_tensor_alias(
    VkBuffer self, const VulkanTensorLayout &self_layout, VkBuffer tensor1,
    const VulkanTensorLayout &tensor1_layout, VkBuffer tensor2,
    const VulkanTensorLayout &tensor2_layout, VkBuffer output,
    const VulkanTensorLayout &output_layout, float value, uint32_t operation) const {
    dispatch_compound(self, self_layout, tensor1, tensor1_layout, tensor2,
                      tensor2_layout, output, output_layout, value, operation);
}

void VulkanCompute::scalar_tensor(float scalar, VkBuffer tensor,
                                  const VulkanTensorLayout &tensor_layout,
                                  VkBuffer output,
                                  const VulkanTensorLayout &output_layout,
                                  uint32_t operation, bool bool_dtype) const {
    dispatch(2, VK_NULL_HANDLE, nullptr, tensor, &tensor_layout, output, output_layout,
             scalar, operation, false, bool_dtype);
}

void VulkanCompute::scalar_tensor_alias(float scalar, VkBuffer tensor,
                                        const VulkanTensorLayout &tensor_layout,
                                        VkBuffer output,
                                        const VulkanTensorLayout &output_layout,
                                        uint32_t operation, bool bool_dtype) const {
    dispatch(2, VK_NULL_HANDLE, nullptr, tensor, &tensor_layout, output, output_layout,
             scalar, operation, true, bool_dtype);
}

void VulkanCompute::unary(VkBuffer input, const VulkanTensorLayout &input_layout,
                          VkBuffer output, const VulkanTensorLayout &output_layout,
                          uint32_t operation, bool bool_dtype) const {
    dispatch(3, input, &input_layout, VK_NULL_HANDLE, nullptr, output, output_layout,
             0.0F, operation, false, bool_dtype);
}

void VulkanCompute::unary_alias(VkBuffer input, const VulkanTensorLayout &input_layout,
                                VkBuffer output,
                                const VulkanTensorLayout &output_layout,
                                uint32_t operation, bool bool_dtype) const {
    dispatch(3, input, &input_layout, VK_NULL_HANDLE, nullptr, output, output_layout,
             0.0F, operation, true, bool_dtype);
}

void VulkanCompute::fill_alias(VkBuffer input, const VulkanTensorLayout &input_layout,
                               VkBuffer output, const VulkanTensorLayout &output_layout,
                               float scalar) const {
    dispatch(3, input, &input_layout, VK_NULL_HANDLE, nullptr, output, output_layout,
             scalar, 11, true, false);
}

void VulkanCompute::comparison_scalar(VkBuffer input,
                                      const VulkanTensorLayout &input_layout,
                                      VkBuffer output,
                                      const VulkanTensorLayout &output_layout,
                                      float scalar) const {
    dispatch(1, input, &input_layout, VK_NULL_HANDLE, nullptr, output, output_layout,
             scalar, 8, false, false, true);
}

void VulkanCompute::comparison_tensor(
    VkBuffer lhs, const VulkanTensorLayout &lhs_layout, VkBuffer rhs,
    const VulkanTensorLayout &rhs_layout, VkBuffer output,
    const VulkanTensorLayout &output_layout, uint32_t operation) const {
    dispatch(0, lhs, &lhs_layout, rhs, &rhs_layout, output, output_layout, 0.0F,
             operation, false, false, true);
}

void VulkanCompute::isfinite(VkBuffer input, const VulkanTensorLayout &input_layout,
                             VkBuffer output,
                             const VulkanTensorLayout &output_layout) const {
    dispatch(3, input, &input_layout, VK_NULL_HANDLE, nullptr, output, output_layout,
             0.0F, 9, false, false, true);
}

void VulkanCompute::masked_select_count(VkBuffer input, VkBuffer mask, VkBuffer counter,
                                        uint32_t element_count) const {
    dispatch_masked(input, mask, VK_NULL_HANDLE, counter, element_count,
                    sizeof(uint32_t), masked_count_pipeline_,
                    masked_count_pipeline_layout_, masked_count_descriptor_layout_, 3);
}

void VulkanCompute::masked_select_compact(VkBuffer input, VkBuffer mask,
                                          VkBuffer output, VkBuffer counter,
                                          uint32_t element_count,
                                          uint32_t output_count) const {
    dispatch_masked(input, mask, output, counter, element_count,
                    static_cast<VkDeviceSize>(output_count) * sizeof(float),
                    masked_compact_pipeline_, masked_compact_pipeline_layout_,
                    masked_compact_descriptor_layout_, 4);
}

void VulkanCompute::reduction(VkBuffer input, const VulkanTensorLayout &input_layout,
                              VkBuffer output, const VulkanTensorLayout &output_layout,
                              uint32_t reduce_mask, uint32_t reduce_numel,
                              uint32_t output_numel, uint32_t operation,
                              uint32_t reduce_dim) const {
    ReductionParams params{};
    fill_layout_metadata(params, input_layout, "reduction");
    params.output_numel = output_numel;
    params.reduce_mask = reduce_mask;
    params.reduce_numel = reduce_numel;
    params.operation = operation;
    params.reduce_dim = reduce_dim;
    dispatch_extra(input, output, input_layout.allocation_bytes,
                   output_layout.allocation_bytes, reduction_pipeline_,
                   reduction_pipeline_layout_, reduction_descriptor_layout_, &params,
                   sizeof(params), output_numel);
}

void VulkanCompute::reduction_backward(
    const VulkanBuffer *input, const VulkanTensorLayout &input_layout,
    const VulkanBuffer *forward, const VulkanTensorLayout &forward_layout,
    const VulkanBuffer *grad_output, const VulkanTensorLayout &grad_output_layout,
    const VulkanBuffer *grad_input, const VulkanTensorLayout &grad_input_layout,
    uint32_t reduce_mask, uint32_t reduce_numel, uint32_t reduce_dim,
    uint32_t operation, bool keepdim) const {
    ReductionBackwardParams params{};
    fill_layout_metadata(params, input_layout, "reduction backward");
    params.input_numel = static_cast<uint32_t>(input_layout.numel);
    params.reduce_numel = reduce_numel;
    params.reduce_mask = reduce_mask;
    params.reduce_dim = reduce_dim;
    params.operation = operation;
    params.keepdim = keepdim ? 1U : 0U;
    const VulkanBuffer *inputs[] = {input, forward, grad_output};
    const VulkanTensorLayout *layouts[] = {&input_layout, &forward_layout,
                                           &grad_output_layout};
    const VulkanBuffer *outputs[] = {grad_input};
    const VulkanTensorLayout *output_layouts[] = {&grad_input_layout};
    dispatch_multi_output(
        inputs, layouts, outputs, output_layouts, &params, sizeof(params), nullptr, 0,
        reduction_backward_pipeline_, reduction_backward_pipeline_layout_,
        reduction_backward_descriptor_layout_, params.input_numel, 3, 1);
}

void VulkanCompute::mse_loss(
    const VulkanBuffer *input, const VulkanTensorLayout &input_layout,
    const VulkanBuffer *target, const VulkanTensorLayout &target_layout,
    const VulkanBuffer *aux, const VulkanTensorLayout &aux_layout,
    const VulkanBuffer *output, const VulkanTensorLayout &output_layout,
    uint32_t element_count, uint32_t reduction, bool backward) const {
    LossParams params{element_count, reduction, backward ? 1U : 0U, 0U};
    const VulkanBuffer *inputs[] = {input, target, aux};
    const VulkanTensorLayout *layouts[] = {&input_layout, &target_layout, &aux_layout};
    const VulkanBuffer *outputs[] = {output};
    const VulkanTensorLayout *output_layouts[] = {&output_layout};
    dispatch_multi_output(
        inputs, layouts, outputs, output_layouts, &params, sizeof(params), nullptr, 0,
        loss_pipeline_, loss_pipeline_layout_, loss_descriptor_layout_,
        backward ? element_count : (reduction == 0 ? element_count : 1U), 3, 1);
}

void VulkanCompute::argmax(VkBuffer input, const VulkanTensorLayout &input_layout,
                           VkBuffer output, const VulkanTensorLayout &output_layout,
                           uint32_t dim, uint32_t reduce_size,
                           uint32_t output_numel) const {
    IndexingParams params{};
    fill_layout_metadata(params, input_layout, "argmax");
    params.output_numel = output_numel;
    params.reduce_dim = dim;
    params.reduce_size = reduce_size;
    dispatch_extra(input, output, input_layout.allocation_bytes,
                   output_layout.allocation_bytes, indexing_pipeline_,
                   indexing_pipeline_layout_, indexing_descriptor_layout_, &params,
                   sizeof(params), output_numel);
}

void VulkanCompute::broadcast(VkBuffer input, const VulkanTensorLayout &input_layout,
                              VkBuffer output, const VulkanTensorLayout &output_layout,
                              uint32_t output_numel, float scale) const {
    BroadcastParams params{};
    BroadcastMetadata metadata{};
    ReductionParams input_metadata{};
    fill_layout_metadata(input_metadata, input_layout, "broadcast");
    params.rank = input_metadata.rank;
    metadata.storage_offset = input_metadata.storage_offset;
    for (uint32_t i = 0; i < params.rank; ++i) {
        metadata.sizes[i] = input_metadata.sizes[i];
        metadata.strides[i] = input_metadata.strides[i];
    }
    if (output_layout.rank != input_layout.rank ||
        output_layout.sizes.size() != static_cast<size_t>(output_layout.rank) ||
        output_layout.strides.size() != static_cast<size_t>(output_layout.rank))
        throw std::invalid_argument(
            "Vulkan compute broadcast output metadata lengths do not match rank");
    if (output_layout.rank > 8)
        throw std::invalid_argument("Vulkan compute broadcast supports ranks up to 8");
    params.output_numel = output_numel;
    params.scale = scale;
    for (uint32_t i = 0; i < params.rank; ++i) {
        metadata.input_sizes[i] = static_cast<uint32_t>(input_layout.sizes[i]);
        metadata.output_sizes[i] = static_cast<uint32_t>(output_layout.sizes[i]);
    }
    dispatch_extra(input, output, input_layout.allocation_bytes,
                   output_layout.allocation_bytes, broadcast_pipeline_,
                   broadcast_pipeline_layout_, broadcast_descriptor_layout_, &params,
                   sizeof(params), output_numel, &metadata, sizeof(metadata));
}

void VulkanCompute::linear(VkBuffer input, VkBuffer weight, VkBuffer bias,
                           VkBuffer output, const VulkanTensorLayout &input_layout,
                           const VulkanTensorLayout &weight_layout,
                           const VulkanTensorLayout &bias_layout,
                           const VulkanTensorLayout &output_layout, uint32_t rows,
                           uint32_t features, uint32_t outputs, bool transposed_weight,
                           bool has_bias, uint32_t operation) const {
    const uint64_t rows64 = rows;
    const uint64_t outputs64 = outputs;
    if (outputs64 != 0 && rows64 > std::numeric_limits<uint64_t>::max() / outputs64)
        throw std::invalid_argument(
            "Vulkan linear output count multiplication overflows uint64");
    const uint64_t output_numel64 = operation == 5
                                        ? static_cast<uint64_t>(features) * outputs64
                                    : operation == 6 ? outputs64
                                                     : rows64 * outputs64;
    if (output_numel64 > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Vulkan linear output count overflow");
    const uint64_t input_numel64 = rows64 * static_cast<uint64_t>(features);
    const uint64_t weight_numel64 = static_cast<uint64_t>(features) * outputs64;
    const uint64_t bias_numel64 = has_bias ? outputs64 : 1;
    const uint64_t expected_input_numel =
        operation == 2   ? static_cast<uint64_t>(features) * rows64
        : operation == 4 ? rows64 * static_cast<uint64_t>(features)
        : operation == 5 ? rows64 * outputs64
                         : input_numel64;
    const uint64_t expected_weight_numel =
        operation == 5 || operation == 6 ? rows64 * static_cast<uint64_t>(features)
                                         : weight_numel64;
    const uint64_t expected_bias_numel =
        operation == 4                     ? rows64 * static_cast<uint64_t>(features)
        : operation == 5 || operation == 6 ? rows64 * outputs64
                                           : bias_numel64;
    if (input_layout.numel != static_cast<int64_t>(expected_input_numel) ||
        weight_layout.numel != static_cast<int64_t>(expected_weight_numel) ||
        bias_layout.numel != static_cast<int64_t>(expected_bias_numel) ||
        output_layout.numel != static_cast<int64_t>(output_numel64))
        throw std::invalid_argument(
            "Vulkan linear metadata does not match dimensions; invalid range");
    ModelMetadata metadata{};
    fill_layout_metadata(metadata.tensors[0], input_layout, "linear input");
    fill_layout_metadata(metadata.tensors[1], weight_layout, "linear weight");
    fill_layout_metadata(metadata.tensors[2], bias_layout, "linear bias");
    fill_layout_metadata(metadata.tensors[3], output_layout, "linear output");
    LinearParams params{rows,     features, outputs, transposed_weight,
                        has_bias, operation};
    dispatch_model(input, weight, bias, output, input_layout.allocation_bytes,
                   weight_layout.allocation_bytes, bias_layout.allocation_bytes,
                   output_layout.allocation_bytes, &params, sizeof(params),
                   static_cast<uint32_t>(output_numel64), VK_NULL_HANDLE,
                   VK_NULL_HANDLE, VK_NULL_HANDLE, &metadata, sizeof(metadata));
}

void VulkanCompute::gemm(VkBuffer a, const VulkanTensorLayout &a_layout, VkBuffer b,
                         const VulkanTensorLayout &b_layout, VkBuffer c,
                         const VulkanTensorLayout &c_layout, VkBuffer output,
                         const VulkanTensorLayout &output_layout, VkBuffer bias,
                         const VulkanTensorLayout &bias_layout, uint32_t m, uint32_t n,
                         uint32_t k, float alpha, float beta, bool has_bias,
                         uint32_t batch_count, uint32_t batch_stride_a,
                         uint32_t batch_stride_b, uint32_t batch_stride_c,
                         uint32_t batch_stride_d) const {
    const bool batched = batch_count != 0;
    const uint32_t dispatch_batches = batched ? batch_count : 1U;
    const uint32_t matrix_rank = batched ? 3U : 2U;
    const uint32_t matrix_base = batched ? 1U : 0U;
    const auto validate_layout = [&](VkBuffer buffer, const VulkanTensorLayout &layout,
                                     uint32_t rank, uint32_t rows, uint32_t columns,
                                     const char *name) {
        const uint32_t expected_rank = rank == 2 ? matrix_rank : rank;
        if (buffer == VK_NULL_HANDLE || layout.rank != expected_rank ||
            layout.scalar_type != kFloatScalarType ||
            layout.element_bytes != sizeof(float) || layout.sizes.size() != expected_rank ||
            layout.strides.size() != expected_rank || layout.storage_offset < 0 ||
            layout.byte_range == 0 || layout.allocation_bytes == 0 ||
            layout.byte_offset > layout.allocation_bytes ||
            layout.byte_range > layout.allocation_bytes - layout.byte_offset ||
            layout.internal_overlap != pytorch_vulkan::VulkanOverlap::No)
            throw std::invalid_argument(std::string("Vulkan GEMM ") + name +
                                        " has an invalid layout");
        if (layout.sizes[matrix_base] != rows || layout.sizes[matrix_base + 1] != columns ||
            layout.strides[matrix_base] < 0 || layout.strides[matrix_base + 1] < 0 ||
            (!batched && (layout.strides[matrix_base + 1] != 1 ||
                          layout.strides[matrix_base] != columns)) ||
            (batched && (layout.strides[matrix_base] > std::numeric_limits<uint32_t>::max() ||
                         layout.strides[matrix_base + 1] > std::numeric_limits<uint32_t>::max())) ||
            static_cast<uint64_t>(layout.storage_offset) >
                std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(layout.strides[0]) >
                std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(layout.strides[1]) >
                std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument(std::string("Vulkan GEMM ") + name +
                                        " is not a representable contiguous matrix");
        const uint64_t row_span =
            static_cast<uint64_t>(rows - 1) * static_cast<uint64_t>(layout.strides[matrix_base]);
        const uint64_t column_span = static_cast<uint64_t>(columns - 1) *
                                     static_cast<uint64_t>(layout.strides[matrix_base + 1]);
        if (row_span > std::numeric_limits<uint64_t>::max() - column_span ||
            row_span + column_span == std::numeric_limits<uint64_t>::max())
            throw std::invalid_argument(std::string("Vulkan GEMM ") + name +
                                        " footprint overflows");
        const uint64_t required_elements = row_span + column_span + 1;
        if (required_elements >
                std::numeric_limits<uint64_t>::max() / layout.element_bytes ||
            layout.byte_range < required_elements * layout.element_bytes)
            throw std::invalid_argument(std::string("Vulkan GEMM ") + name +
                                        " byte range is smaller than its footprint");
        if (layout.byte_offset % sizeof(float) != 0 ||
            layout.byte_range > max_storage_buffer_range_ ||
            layout.byte_offset > max_storage_buffer_range_ - layout.byte_range)
            throw std::invalid_argument(std::string("Vulkan GEMM ") + name +
                                        " exceeds the storage-buffer limit");
    };
    const auto validate_bias = [&] {
        if (bias == VK_NULL_HANDLE || bias_layout.rank != 1 ||
            bias_layout.scalar_type != kFloatScalarType ||
            bias_layout.element_bytes != sizeof(float) ||
            bias_layout.sizes.size() != 1 || bias_layout.strides.size() != 1 ||
            bias_layout.sizes[0] != n || bias_layout.strides[0] != 1 ||
            bias_layout.storage_offset < 0 || bias_layout.byte_range == 0 ||
            bias_layout.allocation_bytes == 0 ||
            bias_layout.byte_offset > bias_layout.allocation_bytes ||
            bias_layout.byte_range >
                bias_layout.allocation_bytes - bias_layout.byte_offset ||
            bias_layout.internal_overlap != pytorch_vulkan::VulkanOverlap::No ||
            bias_layout.byte_offset % sizeof(float) != 0 ||
            bias_layout.byte_range > max_storage_buffer_range_ ||
            bias_layout.byte_offset >
                max_storage_buffer_range_ - bias_layout.byte_range)
            throw std::invalid_argument("Vulkan GEMM bias has an invalid layout");
        const uint64_t required_elements =
            static_cast<uint64_t>(bias_layout.sizes[0] - 1) *
                static_cast<uint64_t>(bias_layout.strides[0]) +
            1;
        if (required_elements >
                std::numeric_limits<uint64_t>::max() / bias_layout.element_bytes ||
            bias_layout.byte_range < required_elements * bias_layout.element_bytes)
            throw std::invalid_argument(
                "Vulkan GEMM bias byte range is smaller than its footprint");
    };
    const uint64_t m_groups = (static_cast<uint64_t>(m) + 15) / 16;
    const uint64_t n_groups = (static_cast<uint64_t>(n) + 15) / 16;
        if (m == 0 || n == 0 || k == 0 || sizeof(GemmParams) > max_push_constants_size_ ||
        m_groups > max_compute_workgroup_count_y_ ||
        n_groups > max_compute_workgroup_count_x_ ||
        (batched && dispatch_batches > max_compute_workgroup_count_z_) ||
        2U * 16U * 16U * sizeof(float) > max_compute_shared_memory_size_)
        throw std::invalid_argument("Vulkan GEMM exceeds device limits");
    validate_layout(a, a_layout, 2, m, k, "A");
    validate_layout(b, b_layout, 2, k, n, "B");
    validate_layout(output, output_layout, 2, m, n, "output");
    if (c != VK_NULL_HANDLE)
        validate_layout(c, c_layout, 2, m, n, "C");
    else if (beta != 0.0F)
        throw std::invalid_argument("Vulkan GEMM requires C when beta is nonzero");
    if (bias != VK_NULL_HANDLE)
        validate_bias();
    else if (has_bias)
        throw std::invalid_argument("Vulkan GEMM requires bias when enabled");
    if (ranges_overlap(output, output_layout, a, a_layout) ||
        ranges_overlap(output, output_layout, b, b_layout) ||
        (c != VK_NULL_HANDLE && ranges_overlap(output, output_layout, c, c_layout)) ||
        (bias != VK_NULL_HANDLE &&
         ranges_overlap(output, output_layout, bias, bias_layout)))
        throw std::invalid_argument("Vulkan GEMM output overlaps an input");

    const VkBuffer ignored_c = c == VK_NULL_HANDLE ? output : c;
    const VkBuffer ignored_bias = bias == VK_NULL_HANDLE ? output : bias;
    const VulkanTensorLayout &effective_c =
        c == VK_NULL_HANDLE ? output_layout : c_layout;
    const VulkanTensorLayout &effective_bias =
        bias == VK_NULL_HANDLE ? output_layout : bias_layout;
    std::scoped_lock lock(platform_.queue_mutex());
    try {
        record_dispatch("gemm");
        VkCommandBuffer cmd = platform_.execution_context().command_buffer();
        const VkDescriptorSet set =
            acquire_descriptor_set(gemm_descriptor_layout_, 5,
                                   kGemmDescriptorPoolCapacity);
        const VkDescriptorBufferInfo buffers[] = {
            {a, a_layout.byte_offset, a_layout.byte_range},
            {b, b_layout.byte_offset, b_layout.byte_range},
            {ignored_c, effective_c.byte_offset, effective_c.byte_range},
            {output, output_layout.byte_offset, output_layout.byte_range},
            {ignored_bias, effective_bias.byte_offset, effective_bias.byte_range}};
        VkWriteDescriptorSet writes[5]{};
        for (uint32_t binding = 0; binding < 5; ++binding) {
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = set;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &buffers[binding];
        }
        vkUpdateDescriptorSets(device_, 5, writes, 0, nullptr);
        const GemmParams params{m,
                                n,
                                k,
                                static_cast<uint32_t>(a_layout.strides[matrix_base]),
                                static_cast<uint32_t>(a_layout.strides[matrix_base + 1]),
                                static_cast<uint32_t>(b_layout.strides[matrix_base]),
                                static_cast<uint32_t>(b_layout.strides[matrix_base + 1]),
                                static_cast<uint32_t>(effective_c.strides[matrix_base]),
                                static_cast<uint32_t>(effective_c.strides[matrix_base + 1]),
                                static_cast<uint32_t>(output_layout.strides[matrix_base]),
                                static_cast<uint32_t>(output_layout.strides[matrix_base + 1]),
                                1,
                                alpha,
                                beta,
                                has_bias ? 1U : 0U,
                                batch_count,
                                batch_stride_a,
                                batch_stride_b,
                                batch_stride_c,
                                batch_stride_d};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gemm_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                gemm_pipeline_layout_, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, gemm_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(params), &params);
        vkCmdDispatch(cmd, static_cast<uint32_t>(n_groups),
                      static_cast<uint32_t>(m_groups), dispatch_batches);
        finish_dispatch();
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        cancel_recording();
        throw contextual_error("GEMM dispatch failed", error);
    }
}

void VulkanCompute::linear_relu_backward_input(
    VkBuffer grad_output, VkBuffer weight, VkBuffer activation, VkBuffer output,
    const VulkanTensorLayout &go, const VulkanTensorLayout &w,
    const VulkanTensorLayout &a, const VulkanTensorLayout &out, uint32_t rows,
    uint32_t output_features, uint32_t input_features) const {
    linear(grad_output, weight, activation, output, go, w, a, out, rows,
           output_features, input_features, false, false, 4);
}

void VulkanCompute::linear_relu_backward_weight(
    VkBuffer grad_output, VkBuffer input, VkBuffer activation, VkBuffer output,
    const VulkanTensorLayout &go, const VulkanTensorLayout &x,
    const VulkanTensorLayout &a, const VulkanTensorLayout &out, uint32_t rows,
    uint32_t features, uint32_t outputs) const {
    linear(grad_output, input, activation, output, go, x, a, out, rows, features,
           outputs, false, false, 5);
}

void VulkanCompute::linear_relu_backward_bias(VkBuffer grad_output, VkBuffer activation,
                                              VkBuffer output,
                                              const VulkanTensorLayout &go,
                                              const VulkanTensorLayout &a,
                                              const VulkanTensorLayout &out,
                                              uint32_t rows, uint32_t outputs) const {
    linear(grad_output, activation, activation, output, go, a, a, out, rows, outputs,
           outputs, false, false, 6);
}

void VulkanCompute::linear_relu_backward(
    const VulkanBuffer *grad_output, const VulkanTensorLayout &grad_output_layout,
    const VulkanBuffer *input, const VulkanTensorLayout &input_layout,
    const VulkanBuffer *weight, const VulkanTensorLayout &weight_layout,
    const VulkanBuffer *activation, const VulkanTensorLayout &activation_layout,
    const VulkanBuffer *d_input, const VulkanTensorLayout &d_input_layout,
    const VulkanBuffer *d_weight, const VulkanTensorLayout &d_weight_layout,
    const VulkanBuffer *d_bias, const VulkanTensorLayout &d_bias_layout, uint32_t rows,
    uint32_t features, uint32_t outputs) const {
    const uint64_t input_numel = checked_product(rows, features, "backward input");
    const uint64_t weight_numel = checked_product(features, outputs, "backward weight");
    const uint64_t gradient_numel = checked_product(rows, outputs, "backward gradient");
    const uint64_t bias_offset = input_numel + weight_numel;
    const uint64_t invocation_count = bias_offset + outputs;
    if (input_numel > std::numeric_limits<uint32_t>::max() ||
        weight_numel > std::numeric_limits<uint32_t>::max() ||
        bias_offset > std::numeric_limits<uint32_t>::max() ||
        invocation_count > std::numeric_limits<uint32_t>::max() ||
        gradient_numel > std::numeric_limits<int64_t>::max() || outputs == 0 ||
        grad_output_layout.numel != static_cast<int64_t>(gradient_numel) ||
        input_layout.numel != static_cast<int64_t>(input_numel) ||
        weight_layout.numel != static_cast<int64_t>(weight_numel) ||
        activation_layout.numel != static_cast<int64_t>(gradient_numel) ||
        d_input_layout.numel != static_cast<int64_t>(input_numel) ||
        d_weight_layout.numel != static_cast<int64_t>(weight_numel) ||
        d_bias_layout.numel != static_cast<int64_t>(outputs))
        throw std::invalid_argument(
            "Vulkan multi-output backward metadata does not match dimensions");
    const auto has_shape = [](const VulkanTensorLayout &layout, uint32_t rank,
                              std::initializer_list<uint32_t> shape) {
        if (layout.rank != static_cast<int64_t>(rank) || layout.sizes.size() != rank ||
            layout.strides.size() != rank)
            return false;
        uint32_t index = 0;
        for (uint32_t size : shape)
            if (layout.sizes[index++] != static_cast<int64_t>(size))
                return false;
        return true;
    };
    if (!has_shape(grad_output_layout, 2, {rows, outputs}) ||
        !has_shape(input_layout, 2, {rows, features}) ||
        !has_shape(weight_layout, 2, {outputs, features}) ||
        !has_shape(activation_layout, 2, {rows, outputs}) ||
        !has_shape(d_input_layout, 2, {rows, features}) ||
        !has_shape(d_weight_layout, 2, {outputs, features}) ||
        !has_shape(d_bias_layout, 1, {outputs}))
        throw std::invalid_argument(
            "Vulkan multi-output backward tensor shapes do not match dimensions");

    MultiOutputMetadata metadata{};
    const VulkanTensorLayout *layouts[] = {
        &grad_output_layout, &input_layout,    &weight_layout, &activation_layout,
        &d_input_layout,     &d_weight_layout, &d_bias_layout};
    for (uint32_t i = 0; i < 7; ++i)
        fill_layout_metadata(metadata.tensors[i], *layouts[i], "backward tensor");
    MultiOutputParams params{
        rows,
        features,
        outputs,
        0,
        static_cast<uint32_t>(input_numel),
        static_cast<uint32_t>(input_numel),
        static_cast<uint32_t>(weight_numel),
        static_cast<uint32_t>(bias_offset),
        outputs,
    };
    const VulkanBuffer *inputs[] = {grad_output, input, weight, activation};
    const VulkanTensorLayout *input_layouts[] = {&grad_output_layout, &input_layout,
                                                 &weight_layout, &activation_layout};
    const VulkanBuffer *outputs_buffers[] = {d_input, d_weight, d_bias};
    const VulkanTensorLayout *output_layouts[] = {&d_input_layout, &d_weight_layout,
                                                  &d_bias_layout};
    dispatch_multi_output(
        inputs, input_layouts, outputs_buffers, output_layouts, &params, sizeof(params),
        &metadata, sizeof(metadata), backward_pipeline_, backward_pipeline_layout_,
        backward_descriptor_layout_, static_cast<uint32_t>(invocation_count));
}

void VulkanCompute::convolution(VkBuffer input, VkBuffer weight, VkBuffer bias,
                                VkBuffer output, const VulkanTensorLayout &input_layout,
                                const VulkanTensorLayout &weight_layout,
                                const VulkanTensorLayout &bias_layout,
                                const VulkanTensorLayout &output_layout,
                                uint32_t operation) const {
    ModelMetadata metadata{};
    fill_layout_metadata(metadata.tensors[0], input_layout, "convolution input");
    fill_layout_metadata(metadata.tensors[1], weight_layout, "convolution weight");
    fill_layout_metadata(metadata.tensors[2], bias_layout, "convolution bias");
    fill_layout_metadata(metadata.tensors[3], output_layout, "convolution output");
    auto dimension = [](const VulkanTensorLayout &layout, size_t index) {
        return static_cast<uint32_t>(layout.sizes.at(index));
    };
    uint32_t batch = dimension(input_layout, 0);
    uint32_t input_channels = operation == 0 || operation == 3
                                  ? dimension(input_layout, 1)
                                  : dimension(weight_layout, 1);
    uint32_t input_height = operation == 2 ? dimension(weight_layout, 2)
                                           : dimension(input_layout, 2);
    uint32_t input_width = operation == 2 ? dimension(weight_layout, 3)
                                          : dimension(input_layout, 3);
    uint32_t output_channels = operation == 0 ? dimension(output_layout, 1)
                                              : dimension(input_layout, 1);
    uint32_t output_height = operation == 0 ? dimension(output_layout, 2)
                                            : dimension(input_layout, 2);
    uint32_t output_width = operation == 0 ? dimension(output_layout, 3)
                                           : dimension(input_layout, 3);
    uint32_t kernel_height = operation == 2 ? dimension(output_layout, 2) : 3;
    uint32_t kernel_width = operation == 2 ? dimension(output_layout, 3) : 3;
    ConvolutionParams params{batch, input_channels, input_height, input_width,
                             output_channels, output_height, output_width,
                             kernel_height, kernel_width, operation};
    const uint32_t output_numel = static_cast<uint32_t>(output_layout.numel);
    dispatch_model(input, weight, bias, output, input_layout.allocation_bytes,
                   weight_layout.allocation_bytes, bias_layout.allocation_bytes,
                   output_layout.allocation_bytes, &params, sizeof(params),
                   output_numel, convolution_pipeline_, convolution_pipeline_layout_,
                   convolution_descriptor_layout_, &metadata, sizeof(metadata));
}

void VulkanCompute::pooling(VkBuffer input, VkBuffer output,
                            const VulkanTensorLayout &input_layout,
                            const VulkanTensorLayout &output_layout, uint32_t batch,
                            uint32_t channels, uint32_t height, uint32_t width,
                            uint32_t operation) const {
    if (operation > 1)
        throw std::invalid_argument("Vulkan pooling operation is unsupported");
    const uint64_t spatial = checked_product(height, width, "spatial");
    const uint64_t batch_channels = checked_product(batch, channels, "batch-channel");
    const uint64_t input_numel = checked_product(batch_channels, spatial, "input");
    const uint64_t output_numel64 = operation == 0 ? batch_channels : input_numel;
    if (output_numel64 > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Vulkan pooling output count truncation");
    PoolingMetadata metadata{};
    fill_layout_metadata(metadata.tensors[0], input_layout, "pooling input");
    fill_layout_metadata(metadata.tensors[1], output_layout, "pooling output");
    PoolingParams params{batch, channels, height, width, operation};
    const uint32_t output_numel = static_cast<uint32_t>(output_numel64);
    dispatch_extra(input, output, input_layout.allocation_bytes,
                   output_layout.allocation_bytes, pooling_pipeline_,
                   pooling_pipeline_layout_, pooling_descriptor_layout_, &params,
                   sizeof(params), output_numel, &metadata, sizeof(metadata));
}

void VulkanCompute::f32_to_double(VkBuffer input, VkBuffer output,
                                  VkDeviceSize input_bytes, VkDeviceSize output_bytes,
                                  uint32_t element_count) const {
    if (!platform_.supports_formatter_double())
        throw std::invalid_argument("Vulkan formatter Double support is unavailable");
    F32ToDoubleParams params{element_count};
    dispatch_extra(input, output, input_bytes, output_bytes, f32_to_double_pipeline_,
                   f32_to_double_pipeline_layout_, f32_to_double_descriptor_layout_,
                   &params, sizeof(params), element_count);
}

void VulkanCompute::formatter_double(VkBuffer input, VkBuffer rhs, VkBuffer output,
                                     VkDeviceSize input_bytes, VkDeviceSize rhs_bytes,
                                     VkDeviceSize output_bytes, uint32_t element_count,
                                     uint32_t operation, double scalar,
                                     uint32_t output_numel, bool bool_output) const {
    if (!platform_.supports_formatter_double())
        throw std::invalid_argument("Vulkan formatter Double support is unavailable");
    FormatterDoubleParams params{scalar, element_count, operation};
    dispatch_formatter(input, rhs, output, input_bytes, rhs_bytes, output_bytes,
                       &params, sizeof(params), output_numel, bool_output);
}

void VulkanCompute::dispatch_model(
    VkBuffer input, VkBuffer weight, VkBuffer bias, VkBuffer output,
    VkDeviceSize input_bytes, VkDeviceSize weight_bytes, VkDeviceSize bias_bytes,
    VkDeviceSize output_bytes, const void *params, uint32_t params_size,
    uint32_t output_numel, VkPipeline pipeline, VkPipelineLayout pipeline_layout,
    VkDescriptorSetLayout descriptor_layout, const void *metadata,
    VkDeviceSize metadata_size) const {
    if (pipeline == VK_NULL_HANDLE) {
        pipeline = model_pipeline_;
        pipeline_layout = model_pipeline_layout_;
        descriptor_layout = model_descriptor_layout_;
    }
    const uint64_t dispatch_groups =
        (static_cast<uint64_t>(output_numel) + kWorkgroupSize - 1) / kWorkgroupSize;
    if (input == VK_NULL_HANDLE || weight == VK_NULL_HANDLE || bias == VK_NULL_HANDLE ||
        output == VK_NULL_HANDLE || !input_bytes || !weight_bytes || !bias_bytes ||
        !output_bytes || !output_numel || input_bytes > max_storage_buffer_range_ ||
        weight_bytes > max_storage_buffer_range_ ||
        bias_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ ||
        dispatch_groups > max_compute_workgroup_count_x_)
        throw std::invalid_argument("Vulkan model compute has an invalid range");
    if (metadata == nullptr || metadata_size == 0 ||
        metadata_size > max_storage_buffer_range_)
        throw std::invalid_argument("Vulkan model compute has invalid metadata");
    std::scoped_lock lock(platform_.queue_mutex());
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    std::shared_ptr<VulkanBuffer> metadata_holder;
    try {
        record_dispatch();
        cmd = platform_.execution_context().command_buffer();
        if (metadata) {
            metadata_holder = std::make_shared<VulkanBuffer>(
                platform_, metadata_size,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        const uint32_t descriptor_count = 5;
        const VkDescriptorSet set =
            acquire_descriptor_set(descriptor_layout, descriptor_count);
        if (metadata)
            metadata_holder->write(metadata, metadata_size);
        VkDescriptorBufferInfo infos[] = {
            {input, 0, input_bytes},
            {weight, 0, weight_bytes},
            {bias, 0, bias_bytes},
            {output, 0, output_bytes},
            {metadata_holder->buffer(), 0, metadata_size}};
        VkWriteDescriptorSet writes[5]{};
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, descriptor_count, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0,
                                1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           params_size, params);
        vkCmdDispatch(cmd, static_cast<uint32_t>(dispatch_groups), 1, 1);
        platform_.execution_context().defer_destruction([metadata_holder] {});
        finish_dispatch();
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        cancel_recording();
        throw contextual_error("model dispatch failed", error);
    }
}

void VulkanCompute::dispatch_multi_output(
    const VulkanBuffer *const *inputs, const VulkanTensorLayout *const *input_layouts,
    const VulkanBuffer *const *outputs, const VulkanTensorLayout *const *output_layouts,
    const void *params, uint32_t params_size, const void *metadata,
    VkDeviceSize metadata_size, VkPipeline pipeline, VkPipelineLayout pipeline_layout,
    VkDescriptorSetLayout descriptor_layout, uint32_t invocation_count,
    uint32_t input_count, uint32_t output_count, uint32_t descriptor_capacity) const {
    if (inputs == nullptr || input_layouts == nullptr || outputs == nullptr ||
        output_layouts == nullptr || params == nullptr || params_size == 0 ||
        params_size > max_push_constants_size_ ||
        (metadata == nullptr && metadata_size != 0) ||
        (metadata != nullptr && metadata_size != sizeof(MultiOutputMetadata)) ||
        metadata_size > max_storage_buffer_range_ || pipeline == VK_NULL_HANDLE ||
        pipeline_layout == VK_NULL_HANDLE || descriptor_layout == VK_NULL_HANDLE ||
        invocation_count == 0)
        throw std::invalid_argument(
            "Vulkan multi-output backward has invalid metadata");

    const auto validate = [&](const VulkanBuffer *buffer,
                              const VulkanTensorLayout *layout) {
        if (buffer == nullptr || layout == nullptr ||
            buffer->platform() != &platform_ || buffer->buffer() == VK_NULL_HANDLE ||
            layout->allocation_bytes == 0 ||
            layout->allocation_bytes > max_storage_buffer_range_ ||
            layout->byte_range == 0 || layout->byte_offset > layout->allocation_bytes ||
            layout->byte_range > layout->allocation_bytes - layout->byte_offset ||
            layout->internal_overlap != pytorch_vulkan::VulkanOverlap::No)
            throw std::invalid_argument(
                "Vulkan multi-output backward has an invalid range");
    };
    if (input_count == 0 || input_count > 10 || output_count == 0 || output_count > 3)
        throw std::invalid_argument("Vulkan multi-output backward has invalid counts");
    for (uint32_t i = 0; i < input_count; ++i)
        validate(inputs[i], input_layouts[i]);
    for (uint32_t i = 0; i < output_count; ++i)
        validate(outputs[i], output_layouts[i]);
    for (uint32_t output = 0; output < output_count; ++output) {
        for (uint32_t other = output + 1; other < output_count; ++other) {
            if (ranges_overlap(outputs[output]->buffer(), *output_layouts[output],
                               outputs[other]->buffer(), *output_layouts[other]))
                throw std::invalid_argument(
                    "Vulkan multi-output backward outputs overlap");
        }
        for (uint32_t input = 0; input < input_count; ++input) {
            if (ranges_overlap(outputs[output]->buffer(), *output_layouts[output],
                               inputs[input]->buffer(), *input_layouts[input]))
                throw std::invalid_argument(
                    "Vulkan multi-output backward output aliases input");
        }
    }
    const uint64_t dispatch_groups = checked_dispatch_groups(invocation_count);
    if (dispatch_groups == 0 || dispatch_groups > max_compute_workgroup_count_x_)
        throw std::invalid_argument(
            "Vulkan multi-output backward dispatch count overflows");

    std::scoped_lock lock(platform_.queue_mutex());
    std::shared_ptr<VulkanBuffer> metadata_holder;
    try {
        record_dispatch();
        VkCommandBuffer cmd = platform_.execution_context().command_buffer();
        if (metadata) {
            metadata_holder = std::make_shared<VulkanBuffer>(
                platform_, metadata_size,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        const uint32_t descriptor_count =
            input_count + output_count + (metadata ? 1u : 0u);
        const VkDescriptorSet set = acquire_descriptor_set(
            descriptor_layout, std::max(descriptor_count, descriptor_capacity));
        if (metadata)
            metadata_holder->write(metadata, metadata_size);
        VkDescriptorBufferInfo infos[descriptor_count]{};
        for (uint32_t i = 0; i < input_count; ++i)
            infos[i] = {inputs[i]->buffer(), 0, input_layouts[i]->allocation_bytes};
        for (uint32_t i = 0; i < output_count; ++i)
            infos[input_count + i] = {outputs[i]->buffer(), 0,
                                      output_layouts[i]->allocation_bytes};
        if (metadata)
            infos[input_count + output_count] = {metadata_holder->buffer(), 0,
                                                 metadata_size};
        VkWriteDescriptorSet writes[descriptor_count]{};
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, descriptor_count, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0,
                                1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           params_size, params);
        vkCmdDispatch(cmd, static_cast<uint32_t>(dispatch_groups), 1, 1);
        platform_.execution_context().defer_destruction([metadata_holder] {});
        finish_dispatch();
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        cancel_recording();
        throw contextual_error("multi-output dispatch failed", error);
    }
}

void VulkanCompute::compute_multi_output(
    const VulkanBuffer *const *inputs, const VulkanTensorLayout *const *input_layouts,
    const VulkanBuffer *const *outputs, const VulkanTensorLayout *const *output_layouts,
    uint32_t input_count, uint32_t output_count, uint32_t invocation_count,
    const void *params, uint32_t params_size, bool classification) const {
    const auto *pipeline =
        classification ? &classification_pipeline_ : &normalization_pipeline_;
    const auto *pipeline_layout = classification ? &classification_pipeline_layout_
                                                 : &normalization_pipeline_layout_;
    const auto *descriptor_layout = classification ? &classification_descriptor_layout_
                                                   : &normalization_descriptor_layout_;
    dispatch_multi_output(inputs, input_layouts, outputs, output_layouts, params,
                          params_size, nullptr, 0, *pipeline, *pipeline_layout,
                          *descriptor_layout, invocation_count, input_count,
                          output_count, classification ? 6 : 10);
}

void VulkanCompute::dispatch_extra(
    VkBuffer input, VkBuffer output, VkDeviceSize input_bytes,
    VkDeviceSize output_bytes, VkPipeline pipeline, VkPipelineLayout pipeline_layout,
    VkDescriptorSetLayout descriptor_layout, const void *params, uint32_t params_size,
    uint32_t output_numel, const void *metadata, VkDeviceSize metadata_size) const {
    const uint64_t max_elements =
        checked_product(static_cast<uint64_t>(max_compute_workgroup_count_x_),
                        kWorkgroupSize, "dispatch workgroup limit");
    const uint64_t dispatch_groups = checked_dispatch_groups(output_numel);
    if (input == VK_NULL_HANDLE || output == VK_NULL_HANDLE || output_numel == 0 ||
        input_bytes == 0 || input_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ ||
        static_cast<uint64_t>(output_numel) > max_elements ||
        dispatch_groups > static_cast<uint64_t>(max_compute_workgroup_count_x_) ||
        dispatch_groups > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Vulkan compute reduction has an invalid range");
    if ((metadata == nullptr) != (metadata_size == 0) ||
        (metadata_size && metadata_size > max_storage_buffer_range_))
        throw std::invalid_argument("Vulkan compute metadata is invalid");
    std::scoped_lock lock(platform_.queue_mutex());
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    std::shared_ptr<VulkanBuffer> metadata_holder;
    try {
        record_dispatch();
        cmd = platform_.execution_context().command_buffer();
        if (metadata) {
            metadata_holder = std::make_shared<VulkanBuffer>(
                platform_, metadata_size,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        const uint32_t descriptor_count = metadata ? 3 : 2;
        const VkDescriptorSet set =
            acquire_descriptor_set(descriptor_layout, descriptor_count);
        if (metadata)
            metadata_holder->write(metadata, metadata_size);
        VkDescriptorBufferInfo buffers[] = {
            {input, 0, input_bytes},
            {output, 0, output_bytes},
            {metadata_holder ? metadata_holder->buffer() : VK_NULL_HANDLE, 0,
             metadata_size}};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device_, descriptor_count, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0,
                                1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           params_size, params);
        vkCmdDispatch(cmd, static_cast<uint32_t>(dispatch_groups), 1, 1);
        platform_.execution_context().defer_destruction([metadata_holder] {});
        finish_dispatch();
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        cancel_recording();
        throw contextual_error("reduction dispatch failed", error);
    }
}

void VulkanCompute::dispatch_formatter(VkBuffer input, VkBuffer rhs, VkBuffer output,
                                       VkDeviceSize input_bytes, VkDeviceSize rhs_bytes,
                                       VkDeviceSize output_bytes, const void *params,
                                       uint32_t params_size, uint32_t output_numel,
                                       bool bool_output) const {
    if (training_step_)
        throw std::logic_error(
            "Vulkan formatter dispatch is unsupported in a training step");
    const uint64_t max_elements =
        static_cast<uint64_t>(max_compute_workgroup_count_x_) *
        static_cast<uint64_t>(kWorkgroupSize);
    if (input == VK_NULL_HANDLE || rhs == VK_NULL_HANDLE || output == VK_NULL_HANDLE ||
        output_numel == 0 || input_bytes == 0 || rhs_bytes == 0 || output_bytes == 0 ||
        input_bytes > max_storage_buffer_range_ ||
        rhs_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ || output_numel > max_elements)
        throw std::invalid_argument(
            "Vulkan formatter Double compute has an invalid range");
    std::scoped_lock lock(platform_.queue_mutex());
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    const auto cleanup = [&] {
        if (pool)
            vkDestroyDescriptorPool(device_, pool, nullptr);
    };
    try {
        platform_.execution_context().begin();
        VkCommandBuffer cmd = platform_.execution_context().command_buffer();
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool),
                     "could not create formatter descriptor pool");
        VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &formatter_double_descriptor_layout_;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set),
                     "could not allocate formatter descriptor set");
        const VkDeviceSize bool_bytes = output_bytes;
        VkDescriptorBufferInfo buffers[] = {{input, 0, input_bytes},
                                            {rhs, 0, rhs_bytes},
                                            {output, 0, output_bytes},
                                            {output, 0, bool_bytes}};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device_, 4, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          formatter_double_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                formatter_double_pipeline_layout_, 0, 1, &set, 0,
                                nullptr);
        vkCmdPushConstants(cmd, formatter_double_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params);
        vkCmdDispatch(cmd, (output_numel + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        platform_.execution_context().retain(pool);
        pool = VK_NULL_HANDLE;
        platform_.execution_context().submit();
        platform_.execution_context().wait();
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        cancel_recording();
        cleanup();
        throw contextual_error("formatter dispatch failed", error);
    }
}

void VulkanCompute::dispatch_masked(VkBuffer input, VkBuffer mask, VkBuffer output,
                                    VkBuffer counter, uint32_t element_count,
                                    VkDeviceSize output_bytes, VkPipeline pipeline,
                                    VkPipelineLayout pipeline_layout,
                                    VkDescriptorSetLayout descriptor_layout,
                                    uint32_t descriptor_count) const {
    if (training_step_)
        throw std::logic_error(
            "Vulkan masked-select dispatch is unsupported in a training step");
    const uint64_t max_elements =
        static_cast<uint64_t>(max_compute_workgroup_count_x_) *
        static_cast<uint64_t>(kWorkgroupSize);
    const VkDeviceSize input_bytes =
        static_cast<VkDeviceSize>(element_count) * sizeof(float);
    const VkDeviceSize mask_bytes = element_count;
    if (input == VK_NULL_HANDLE || mask == VK_NULL_HANDLE ||
        counter == VK_NULL_HANDLE ||
        (descriptor_count == 4 && output == VK_NULL_HANDLE) || element_count == 0 ||
        input_bytes > max_storage_buffer_range_ ||
        mask_bytes > max_storage_buffer_range_ ||
        output_bytes > max_storage_buffer_range_ || output_bytes == 0 ||
        static_cast<uint64_t>(element_count) > max_elements)
        throw std::invalid_argument("Vulkan masked-select has an invalid range");
    std::scoped_lock lock(platform_.queue_mutex());
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    const auto cleanup = [&] {
        if (pool)
            vkDestroyDescriptorPool(device_, pool, nullptr);
    };
    try {
        platform_.execution_context().begin();
        VkCommandBuffer cmd = platform_.execution_context().command_buffer();
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       descriptor_count};
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check_result(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool),
                     "could not create masked-select descriptor pool");
        VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &descriptor_layout;
        check_result(vkAllocateDescriptorSets(device_, &set_info, &set),
                     "could not allocate masked-select descriptor set");
        VkDescriptorBufferInfo infos[] = {{input, 0, input_bytes},
                                          {mask, 0, mask_bytes},
                                          {output, 0, output_bytes},
                                          {counter, 0, sizeof(uint32_t)}};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < descriptor_count; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i == 3 ? 3 : i];
        }
        if (descriptor_count == 3)
            writes[2].pBufferInfo = &infos[3];
        vkUpdateDescriptorSets(device_, descriptor_count, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0,
                                1, &set, 0, nullptr);
        MaskedParams params{element_count};
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(params), &params);
        vkCmdDispatch(cmd, 1, 1, 1);
        platform_.execution_context().retain(pool);
        pool = VK_NULL_HANDLE;
        platform_.execution_context().submit();
        platform_.execution_context().wait();
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        cancel_recording();
        cleanup();
        throw contextual_error("masked-select dispatch failed", error);
    }
}

void VulkanCompute::dispatch(uint32_t mode, VkBuffer lhs,
                             const VulkanTensorLayout *lhs_layout, VkBuffer rhs,
                             const VulkanTensorLayout *rhs_layout, VkBuffer output,
                             const VulkanTensorLayout &output_layout, float scalar,
                             uint32_t operation, bool exact_alias, bool bool_dtype,
                             bool bool_output, VkDeviceSize, VkDeviceSize) const {
    if (mode > 3 || output == VK_NULL_HANDLE ||
        ((mode != 2) && lhs_layout == nullptr) ||
        ((mode == 2) && rhs_layout == nullptr) ||
        (mode == 0 &&
         (lhs == VK_NULL_HANDLE || rhs == VK_NULL_HANDLE || rhs_layout == nullptr)) ||
        (mode != 0 && ((lhs == VK_NULL_HANDLE) == (rhs == VK_NULL_HANDLE)))) {
        throw std::invalid_argument("Vulkan compute pointwise requires valid buffers");
    }
    if ((bool_dtype || bool_output) && !platform_.supports_bool_pointwise()) {
        throw std::invalid_argument("Vulkan compute bool pointwise is unavailable");
    }
    if (exact_alias && ((mode == 0 && output != lhs && output != rhs) ||
                        (mode == 1 && output != lhs) || (mode == 2 && output != rhs) ||
                        (mode == 3 && output != lhs))) {
        throw std::invalid_argument(
            "Vulkan compute exact alias does not match an input buffer");
    }
    const uint64_t element_count = static_cast<uint64_t>(output_layout.numel);
    if (element_count > std::numeric_limits<uint32_t>::max() ||
        output_layout.rank > static_cast<int64_t>(kMaxPointwiseRank) ||
        (lhs_layout && lhs_layout->numel != output_layout.numel) ||
        (rhs_layout && rhs_layout->numel != output_layout.numel))
        throw std::invalid_argument(
            "Vulkan compute pointwise has invalid layout metadata");
    PointwiseMetadata metadata{};
    const auto write_metadata = [&](uint32_t tensor, const VulkanTensorLayout &layout) {
        const TensorMetadata values = pointwise_metadata(layout);
        const uint32_t base = tensor * 20;
        metadata.data[base] = values.rank;
        for (uint32_t dim = 0; dim < 8; ++dim) {
            metadata.data[base + 1 + dim] = values.sizes[dim];
            metadata.data[base + 9 + dim] = values.strides[dim];
        }
        metadata.data[base + 17] = values.storage_offset;
    };
    if (lhs_layout)
        write_metadata(0, *lhs_layout);
    if (rhs_layout)
        write_metadata(1, *rhs_layout);
    write_metadata(2, output_layout);
    const std::size_t element_bytes = bool_dtype ? sizeof(bool) : sizeof(float);
    const VkDeviceSize output_bytes =
        element_count * (bool_output ? sizeof(bool) : element_bytes);
    const VkDeviceSize input_bytes = element_count * element_bytes;
    if (!input_bytes || !output_bytes || output_bytes > max_storage_buffer_range_ ||
        (lhs_layout && lhs_layout->allocation_bytes > max_storage_buffer_range_) ||
        (rhs_layout && rhs_layout->allocation_bytes > max_storage_buffer_range_) ||
        output_layout.allocation_bytes > max_storage_buffer_range_)
        throw std::invalid_argument(
            "Vulkan compute pointwise has an invalid byte range");
    const uint32_t elements = static_cast<uint32_t>(element_count);
    const uint32_t groups = (elements + kWorkgroupSize - 1) / kWorkgroupSize;
    if (groups > max_compute_workgroup_count_x_)
        throw std::invalid_argument("Vulkan compute pointwise exceeds workgroup limit");

    std::scoped_lock lock(platform_.queue_mutex());
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    std::shared_ptr<VulkanBuffer> metadata_holder;
    try {
        record_dispatch();
        cmd = platform_.execution_context().command_buffer();
        metadata_holder = std::make_shared<VulkanBuffer>(
            platform_, sizeof(metadata),
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        metadata_holder->write(&metadata, sizeof(metadata));
        const uint32_t pipeline_mode = bool_output
                                           ? (mode == 0 ? 7U : (mode == 1 ? 5U : 6U))
                                           : mode + (bool_dtype ? 4 : 0);
        const uint32_t descriptor_count = 4;
        const VkDescriptorSet set =
            acquire_descriptor_set(descriptor_set_layouts_[pipeline_mode],
                                   descriptor_count);
        VkDescriptorBufferInfo buffers[] = {
            {lhs, 0, lhs_layout ? lhs_layout->allocation_bytes : 0},
            {rhs, 0, rhs_layout ? rhs_layout->allocation_bytes : 0},
            {output, 0, output_layout.allocation_bytes},
            {metadata_holder->buffer(), 0, sizeof(metadata)}};
        VkWriteDescriptorSet writes[4]{};
        const uint32_t bindings[] = {0, 1, 2, 3};
        const uint32_t write_count = 4;
        for (uint32_t i = 0; i < write_count; ++i) {
            const uint32_t source =
                mode == 0 ? i : (i < 2 ? (mode == 2 ? 1U : 0U) : (i == 2 ? 2U : 3U));
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = bindings[i];
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[source];
        }
        vkUpdateDescriptorSets(device_, write_count, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelines_[pipeline_mode]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layouts_[pipeline_mode], 0, 1, &set, 0,
                                nullptr);
        Params params{scalar, elements, operation, 0};
        vkCmdPushConstants(cmd, pipeline_layouts_[pipeline_mode],
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
        vkCmdDispatch(cmd, groups, 1, 1);
        platform_.execution_context().defer_destruction([metadata_holder] {});
        finish_dispatch();
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        cancel_recording();
        throw contextual_error("dispatch failed", error);
    }
}

void VulkanCompute::dispatch_compound(
    VkBuffer self, const VulkanTensorLayout &self_layout, VkBuffer tensor1,
    const VulkanTensorLayout &tensor1_layout, VkBuffer tensor2,
    const VulkanTensorLayout &tensor2_layout, VkBuffer output,
    const VulkanTensorLayout &output_layout, float value, uint32_t operation) const {
    if ((operation > 1) || self == VK_NULL_HANDLE || tensor1 == VK_NULL_HANDLE ||
        tensor2 == VK_NULL_HANDLE || output == VK_NULL_HANDLE)
        throw std::invalid_argument("Vulkan compute compound requires valid buffers");
    if (output != self)
        throw std::invalid_argument("Vulkan compute compound requires an exact alias");
    const uint64_t element_count = static_cast<uint64_t>(output_layout.numel);
    if (element_count == 0 || element_count > std::numeric_limits<uint32_t>::max() ||
        output_layout.rank > kMaxPointwiseRank ||
        self_layout.numel != output_layout.numel ||
        tensor1_layout.numel != output_layout.numel ||
        tensor2_layout.numel != output_layout.numel)
        throw std::invalid_argument(
            "Vulkan compute compound has invalid layout metadata");
    PointwiseMetadata metadata{};
    const auto write_metadata = [&](uint32_t tensor, const VulkanTensorLayout &layout) {
        const TensorMetadata values = pointwise_metadata(layout);
        const uint32_t base = tensor * 20;
        metadata.data[base] = values.rank;
        for (uint32_t dim = 0; dim < 8; ++dim) {
            metadata.data[base + 1 + dim] = values.sizes[dim];
            metadata.data[base + 9 + dim] = values.strides[dim];
        }
        metadata.data[base + 17] = values.storage_offset;
    };
    write_metadata(0, self_layout);
    write_metadata(1, tensor1_layout);
    write_metadata(2, tensor2_layout);
    write_metadata(3, output_layout);
    const VkDeviceSize bytes = checked_bytes(element_count, "compound");
    if (bytes > max_storage_buffer_range_ ||
        self_layout.allocation_bytes > max_storage_buffer_range_ ||
        tensor1_layout.allocation_bytes > max_storage_buffer_range_ ||
        tensor2_layout.allocation_bytes > max_storage_buffer_range_ ||
        output_layout.allocation_bytes > max_storage_buffer_range_)
        throw std::invalid_argument(
            "Vulkan compute compound has an invalid byte range");
    const uint32_t elements = static_cast<uint32_t>(element_count);
    const uint32_t groups = (elements + kWorkgroupSize - 1) / kWorkgroupSize;
    if (groups > max_compute_workgroup_count_x_)
        throw std::invalid_argument("Vulkan compute compound exceeds workgroup limit");

    std::scoped_lock lock(platform_.queue_mutex());
    auto metadata_holder = std::make_shared<VulkanBuffer>(
        platform_, sizeof(metadata),
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    metadata_holder->write(&metadata, sizeof(metadata));
    try {
        record_dispatch();
        VkCommandBuffer cmd = platform_.execution_context().command_buffer();
        const VkDescriptorSet set =
            acquire_descriptor_set(compound_descriptor_layouts_[operation], 5);
        VkDescriptorBufferInfo buffers[] = {
            {self, 0, self_layout.allocation_bytes},
            {tensor1, 0, tensor1_layout.allocation_bytes},
            {tensor2, 0, tensor2_layout.allocation_bytes},
            {output, 0, output_layout.allocation_bytes},
            {metadata_holder->buffer(), 0, sizeof(metadata)}};
        VkWriteDescriptorSet writes[5]{};
        for (uint32_t i = 0; i < 5; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vkUpdateDescriptorSets(device_, 5, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          compound_pipelines_[operation]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compound_pipeline_layouts_[operation], 0, 1, &set, 0,
                                nullptr);
        Params params{value, elements, operation, 0};
        vkCmdPushConstants(cmd, compound_pipeline_layouts_[operation],
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
        vkCmdDispatch(cmd, groups, 1, 1);
        platform_.execution_context().defer_destruction([metadata_holder] {});
        finish_dispatch();
        dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
        cancel_recording();
        throw contextual_error("compound dispatch failed", error);
    }
}

void VulkanCompute::begin_training_step() const {
    std::scoped_lock lock(platform_.queue_mutex());
    if (training_step_)
        throw std::logic_error("Vulkan training step is already recording");
    platform_.execution_context().begin("training");
    const VkMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(platform_.execution_context().command_buffer(),
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
    training_step_ = true;
}

void VulkanCompute::end_training_step() const {
    std::scoped_lock lock(platform_.queue_mutex());
    if (!training_step_)
        throw std::logic_error("Vulkan training step is not recording");
    if (!platform_.execution_context().recording()) {
        training_step_ = false;
        return;
    }
    try {
        const VkMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(platform_.execution_context().command_buffer(),
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                             nullptr, 0, nullptr);
        platform_.execution_context().submit();
        platform_.execution_context().wait();
        submission_count_.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
        training_step_ = false;
        cancel_recording();
        throw;
    }
    training_step_ = false;
}

void VulkanCompute::cancel_training_step() const {
    std::scoped_lock lock(platform_.queue_mutex());
    if (!training_step_)
        return;
    training_step_ = false;
    cancel_recording();
}

bool VulkanCompute::training_step_active() const { return training_step_; }

void VulkanCompute::record_dispatch(const char *scope) const {
    VulkanExecutionContext &context = platform_.execution_context();
    platform_.throw_if_device_lost();
    if (!context.recording()) {
        context.begin(scope);
        return;
    }
    const VkMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(context.command_buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
}

void VulkanCompute::cancel_recording() const {
    VulkanExecutionContext &context = platform_.execution_context();
    if (context.recording())
        context.cancel();
}

void VulkanCompute::finish_dispatch() const {
    if (training_step_)
        return;
    VulkanExecutionContext &context = platform_.execution_context();
    context.submit();
    context.wait();
    submission_count_.fetch_add(1, std::memory_order_relaxed);
}

VkDescriptorSet VulkanCompute::acquire_descriptor_set(
    VkDescriptorSetLayout descriptor_layout, uint32_t descriptor_count,
    uint32_t pool_capacity) const {
    const VkDescriptorSet set = descriptor_arena_->acquire(
        descriptor_layout, submission_count_.load(std::memory_order_relaxed), descriptor_count,
        pool_capacity);
    descriptor_arena_->release_after_completion(set);
    return set;
}

std::size_t VulkanCompute::dispatch_count() const {
    return dispatch_count_.load(std::memory_order_relaxed);
}

void VulkanCompute::reset_dispatch_count() const {
    dispatch_count_.store(0, std::memory_order_relaxed);
}

std::size_t VulkanCompute::submission_count() const {
    return submission_count_.load(std::memory_order_relaxed);
}

void VulkanCompute::reset_submission_count() const {
    submission_count_.store(0, std::memory_order_relaxed);
}

std::size_t VulkanCompute::descriptor_pool_creation_count() const {
    const auto snapshot = descriptor_arena_->snapshot();
    return snapshot.pool_creations >= descriptor_pool_baseline_
               ? snapshot.pool_creations - descriptor_pool_baseline_
               : 0;
}

std::size_t VulkanCompute::descriptor_set_allocation_count() const {
    const auto snapshot = descriptor_arena_->snapshot();
    return snapshot.allocations >= descriptor_allocation_baseline_
               ? snapshot.allocations - descriptor_allocation_baseline_
               : 0;
}

std::size_t VulkanCompute::descriptor_set_reuse_count() const {
    const auto snapshot = descriptor_arena_->snapshot();
    return snapshot.reuses >= descriptor_reuse_baseline_
               ? snapshot.reuses - descriptor_reuse_baseline_
               : 0;
}

std::size_t VulkanCompute::live_descriptor_pool_count() const {
    return descriptor_arena_->snapshot().pool_count;
}

std::size_t VulkanCompute::live_descriptor_set_count() const {
    return descriptor_arena_->snapshot().live_sets;
}

void VulkanCompute::invalidate_device_loss() const {
    descriptor_arena_->invalidate_device_loss();
}

DescriptorArenaSnapshot VulkanCompute::descriptor_arena_snapshot() const {
    return descriptor_arena_->snapshot();
}

std::size_t VulkanCompute::pipeline_count() const {
    std::size_t count = platform_.pipeline_cache().snapshot().pipeline_count;
    // Cached migrated handles and pending deferred destructions are represented
    // by the cache snapshot above.
    // Only facade-owned pipelines are added here.
    count += model_pipeline_ != VK_NULL_HANDLE;
    count += backward_pipeline_ != VK_NULL_HANDLE;
    count += convolution_pipeline_ != VK_NULL_HANDLE;
    count += masked_count_pipeline_ != VK_NULL_HANDLE;
    count += masked_compact_pipeline_ != VK_NULL_HANDLE;
    count += formatter_double_pipeline_ != VK_NULL_HANDLE;
    return count;
}

std::size_t VulkanCompute::shader_module_count() const {
    std::size_t count = platform_.shader_registry().snapshot().module_count;
    // Cached migrated handles are represented by the registry count above.
    // Only facade-owned shader modules are added here.
    count += model_shader_ != VK_NULL_HANDLE;
    count += backward_shader_ != VK_NULL_HANDLE;
    count += convolution_shader_ != VK_NULL_HANDLE;
    count += masked_count_shader_ != VK_NULL_HANDLE;
    count += masked_compact_shader_ != VK_NULL_HANDLE;
    count += formatter_double_shader_ != VK_NULL_HANDLE;
    return count;
}

void VulkanCompute::reset_descriptor_resource_counters() const {
    const auto snapshot = descriptor_arena_->snapshot();
    descriptor_pool_baseline_ = snapshot.pool_creations;
    descriptor_allocation_baseline_ = snapshot.allocations;
    descriptor_reuse_baseline_ = snapshot.reuses;
}
