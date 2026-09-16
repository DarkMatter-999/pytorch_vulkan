#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/DispatchKey.h>

namespace pytorch_vulkan {

inline bool is_fake_tensor(const at::Tensor &tensor) {
    const auto &data = tensor.storage().data_ptr();
    return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Python) &&
        data.get() == nullptr && data.get_context() == nullptr;
}

} // namespace pytorch_vulkan
