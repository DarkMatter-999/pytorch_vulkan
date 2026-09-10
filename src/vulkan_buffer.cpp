#include "vulkan_buffer.h"

#include <cstring>
#include <stdexcept>

namespace {

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        const bool type_matches = (type_filter & (1U << index)) != 0;
        const bool properties_match =
            (memory_properties.memoryTypes[index].propertyFlags & properties) ==
            properties;
        if (type_matches && properties_match) {
            return index;
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type is available");
}

} // namespace

VulkanBuffer::VulkanBuffer(const VulkanPlatform &platform, VkDeviceSize size,
                           VkMemoryPropertyFlags memory_properties)
    : device_(platform.device()), size_(size), memory_properties_(memory_properties) {
    if (size == 0) {
        throw std::invalid_argument("Vulkan buffer size must be greater than zero");
    }

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_) != VK_SUCCESS) {
        throw std::runtime_error("Could not create Vulkan buffer");
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer_, &requirements);
    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = find_memory_type(
        platform.physical_device(), requirements.memoryTypeBits, memory_properties_);
    if (vkAllocateMemory(device_, &allocation_info, nullptr, &memory_) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("Could not allocate Vulkan buffer memory");
    }
    if (vkBindBufferMemory(device_, buffer_, memory_, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
        buffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("Could not bind Vulkan buffer memory");
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory_, nullptr);
    }
}

VkBuffer VulkanBuffer::buffer() const { return buffer_; }

VkDeviceMemory VulkanBuffer::memory() const { return memory_; }

VkDeviceSize VulkanBuffer::size() const { return size_; }

void VulkanBuffer::write(const void *data, VkDeviceSize size, VkDeviceSize offset) {
    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        throw std::runtime_error("Vulkan buffer memory is not host visible");
    }
    if (data == nullptr || offset > size_ || size > size_ - offset) {
        throw std::out_of_range("Vulkan buffer write is outside the buffer");
    }

    void *mapped = nullptr;
    if (vkMapMemory(device_, memory_, offset, size, 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("Could not map Vulkan buffer memory for writing");
    }
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(device_, memory_);
}

void VulkanBuffer::read(void *data, VkDeviceSize size, VkDeviceSize offset) const {
    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        throw std::runtime_error("Vulkan buffer memory is not host visible");
    }
    if (data == nullptr || offset > size_ || size > size_ - offset) {
        throw std::out_of_range("Vulkan buffer read is outside the buffer");
    }

    void *mapped = nullptr;
    if (vkMapMemory(device_, memory_, offset, size, 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("Could not map Vulkan buffer memory for reading");
    }
    std::memcpy(data, mapped, static_cast<size_t>(size));
    vkUnmapMemory(device_, memory_);
}
