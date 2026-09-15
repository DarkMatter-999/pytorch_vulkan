#pragma once

#include <ATen/core/TensorBody.h>

namespace pytorch_vulkan {

at::Tensor &addcmul_inplace(at::Tensor &self, const at::Tensor &tensor1,
                            const at::Tensor &tensor2, const at::Scalar &value);
at::Tensor &addcdiv_inplace(at::Tensor &self, const at::Tensor &tensor1,
                            const at::Tensor &tensor2, const at::Scalar &value);

} // namespace pytorch_vulkan
