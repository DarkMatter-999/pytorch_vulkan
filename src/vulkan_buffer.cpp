#include "vulkan_buffer.h"

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

VulkanBuffer::VulkanBuffer(const VulkanPlatform &platform, VkDeviceSize size)
    : device_(platform.device()), size_(size) {
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
    allocation_info.memoryTypeIndex =
        find_memory_type(platform.physical_device(), requirements.memoryTypeBits,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device_, &allocation_info, nullptr, &memory_) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("Could not allocate Vulkan buffer memory");
    }
    if (vkBindBufferMemory(device_, buffer_, memory_, 0) != VK_SUCCESS) {
        vkFreeMemory(device_, memory_, nullptr);
        vkDestroyBuffer(device_, buffer_, nullptr);
        memory_ = VK_NULL_HANDLE;
        buffer_ = VK_NULL_HANDLE;
        throw std::runtime_error("Could not bind Vulkan buffer memory");
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory_, nullptr);
    }
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer_, nullptr);
    }
}

VkBuffer VulkanBuffer::buffer() const { return buffer_; }

VkDeviceMemory VulkanBuffer::memory() const { return memory_; }

VkDeviceSize VulkanBuffer::size() const { return size_; }
