#include "vulkan_buffer.h"

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace {

void get_memory_properties(const VulkanBufferResourceOps &ops, VkPhysicalDevice physical,
                           VkPhysicalDeviceMemoryProperties *properties) {
    if (ops.get_memory_properties)
        ops.get_memory_properties(ops.cookie, physical, properties);
    else
        vkGetPhysicalDeviceMemoryProperties(physical, properties);
}

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter,
                          VkMemoryPropertyFlags properties,
                          const VulkanBufferResourceOps &ops) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    get_memory_properties(ops, physical_device, &memory_properties);
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        const bool type_matches = (type_filter & (1U << index)) != 0;
        const bool properties_match =
            (memory_properties.memoryTypes[index].propertyFlags & properties) ==
            properties;
        if (type_matches && properties_match) {
            return index;
        }
    }
    if (properties ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
            const bool type_matches = (type_filter & (1U << index)) != 0;
            const bool host_visible =
                (memory_properties.memoryTypes[index].propertyFlags &
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            if (type_matches && host_visible) {
                return index;
            }
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type is available");
}

} // namespace

VulkanBuffer::VulkanBuffer(const VulkanPlatform &platform, VkDeviceSize size,
                           VkMemoryPropertyFlags memory_properties,
                           const VulkanBufferResourceOps *resource_ops_for_testing)
    : platform_(&platform), size_(size), memory_properties_(memory_properties),
      resource_ops_for_testing_(resource_ops_for_testing ? *resource_ops_for_testing
                                                        : VulkanBufferResourceOps{}) {
    const auto &ops = resource_ops_for_testing_;
    const bool any = ops.create_buffer || ops.get_requirements || ops.get_memory_properties ||
                     ops.allocate_memory || ops.bind_buffer || ops.destroy_buffer || ops.free_memory;
    const bool all = ops.create_buffer && ops.get_requirements && ops.get_memory_properties &&
                     ops.allocate_memory && ops.bind_buffer && ops.destroy_buffer && ops.free_memory;
    if (any && !all)
        throw std::invalid_argument("VulkanBuffer resource ops must be all real or all injected");
    device_ = platform.device();
    if (size == 0) {
        throw std::invalid_argument("Vulkan buffer size must be greater than zero");
    }
    pytorch_vulkan::with_vulkan_operation(platform.runtime_lifecycle(), [&] {
    const auto create_start = std::chrono::steady_clock::now();

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    const auto create = ops.create_buffer
        ? ops.create_buffer(ops.cookie, device_, size, &buffer_)
        : vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_);
    if (create != VK_SUCCESS) {
        throw std::runtime_error("Could not create Vulkan buffer");
    }

    try {
        VkMemoryRequirements requirements{};
        if (ops.get_requirements)
            ops.get_requirements(ops.cookie, device_, buffer_, &requirements);
        else
            vkGetBufferMemoryRequirements(device_, buffer_, &requirements);
        allocation_size_ = requirements.size;
        VkMemoryAllocateInfo allocation_info{};
        allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation_info.allocationSize = requirements.size;
        allocation_info.memoryTypeIndex =
            find_memory_type(platform.physical_device(), requirements.memoryTypeBits,
                             memory_properties_, ops);
        VkPhysicalDeviceMemoryProperties available_memory{};
        get_memory_properties(ops, platform.physical_device(), &available_memory);
        memory_properties_ =
            available_memory.memoryTypes[allocation_info.memoryTypeIndex].propertyFlags;
        const auto allocate = ops.allocate_memory
            ? ops.allocate_memory(ops.cookie, device_, &allocation_info, &memory_)
            : vkAllocateMemory(device_, &allocation_info, nullptr, &memory_);
        if (allocate !=
            VK_SUCCESS) {
            throw std::runtime_error("Could not allocate Vulkan buffer memory");
        }
        const auto bind = ops.bind_buffer
            ? ops.bind_buffer(ops.cookie, device_, buffer_, memory_, 0)
            : vkBindBufferMemory(device_, buffer_, memory_, 0);
        if (bind != VK_SUCCESS) {
            throw std::runtime_error("Could not bind Vulkan buffer memory");
        }
    } catch (const __cxxabiv1::__forced_unwind&) { throw; }
    catch (...) {
        if (buffer_ != VK_NULL_HANDLE) {
            if (ops.destroy_buffer) ops.destroy_buffer(ops.cookie, device_, buffer_);
            else vkDestroyBuffer(device_, buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
        }
        if (memory_ != VK_NULL_HANDLE) {
            if (ops.free_memory) ops.free_memory(ops.cookie, device_, memory_);
            else vkFreeMemory(device_, memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
        throw;
    }

    platform_->record_buffer_created(size);
    platform_->record_timing(
        VulkanTimingCategory::BufferCreate,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - create_start)
            .count());
    });
}

VulkanBuffer::~VulkanBuffer() {
    if (pytorch_vulkan::inherited_fork_state()) {
        // Parent-owned Vulkan handles are invalid for destruction in a forked child.
        return;
    }
    if (buffer_ == VK_NULL_HANDLE && memory_ == VK_NULL_HANDLE)
        return;
    pytorch_vulkan::VulkanCleanupScope cleanup(platform_->runtime_lifecycle());
    const auto destroy_start = std::chrono::steady_clock::now();
    struct DestroyTimer {
        const VulkanPlatform *platform;
        std::chrono::steady_clock::time_point start;
        ~DestroyTimer() {
            platform->record_timing(
                VulkanTimingCategory::BufferDestroy,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                    .count());
        }
    } destroy_timer{platform_, destroy_start};
    if (mapped_base_ != nullptr) {
        unmap_memory();
        mapped_base_ = nullptr;
    }
    if (buffer_ != VK_NULL_HANDLE) {
        if (resource_ops_for_testing_.destroy_buffer)
            resource_ops_for_testing_.destroy_buffer(resource_ops_for_testing_.cookie, device_, buffer_);
        else
            vkDestroyBuffer(device_, buffer_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
        if (resource_ops_for_testing_.free_memory)
            resource_ops_for_testing_.free_memory(resource_ops_for_testing_.cookie, device_, memory_);
        else
            vkFreeMemory(device_, memory_, nullptr);
    }
}

VkBuffer VulkanBuffer::buffer() const { return buffer_; }

const VulkanPlatform *VulkanBuffer::platform() const { return platform_; }

VkDeviceMemory VulkanBuffer::memory() const { return memory_; }

VkDeviceSize VulkanBuffer::size() const { return size_; }

VulkanMappedMemoryRange VulkanBuffer::mapped_range(
    VkDeviceSize offset, VkDeviceSize size, VkDeviceSize allocation_size,
    VkDeviceSize atom_size) {
    if (atom_size == 0 || offset > allocation_size ||
        size > allocation_size - offset)
        throw std::out_of_range("Invalid Vulkan mapped memory range");
    if (size == 0)
        return {offset, 0};
    const VkDeviceSize aligned_offset = (offset / atom_size) * atom_size;
    const VkDeviceSize requested_end = offset + size;
    const VkDeviceSize remainder = requested_end % atom_size;
    VkDeviceSize aligned_end = requested_end;
    if (remainder != 0) {
        const VkDeviceSize increment = atom_size - remainder;
        aligned_end = increment > allocation_size - requested_end
                          ? allocation_size
                          : requested_end + increment;
    }
    return {aligned_offset, aligned_end == allocation_size
                                ? VK_WHOLE_SIZE
                                : aligned_end - aligned_offset};
}

VkMemoryPropertyFlags VulkanBuffer::memory_properties() const {
    return memory_properties_;
}

bool VulkanBuffer::coherent() const {
    if (map_ops_for_testing_.override_coherent_for_testing)
        return map_ops_for_testing_.coherent_for_testing;
    return (memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

VkResult VulkanBuffer::map_memory(void **mapped) const {
    if (map_ops_for_testing_.map)
        return map_ops_for_testing_.map(map_ops_for_testing_.cookie, device_, memory_, 0,
                                        VK_WHOLE_SIZE, 0, mapped);
    return vkMapMemory(device_, memory_, 0, VK_WHOLE_SIZE, 0, mapped);
}

void VulkanBuffer::unmap_memory() const {
    if (map_ops_for_testing_.unmap)
        map_ops_for_testing_.unmap(map_ops_for_testing_.cookie, device_, memory_);
    else
        vkUnmapMemory(device_, memory_);
}

VkResult VulkanBuffer::flush_memory(const VkMappedMemoryRange *range) const {
    if (map_ops_for_testing_.flush)
        return map_ops_for_testing_.flush(map_ops_for_testing_.cookie, device_, 1, range);
    return vkFlushMappedMemoryRanges(device_, 1, range);
}

VkResult VulkanBuffer::invalidate_memory(const VkMappedMemoryRange *range) const {
    if (map_ops_for_testing_.invalidate)
        return map_ops_for_testing_.invalidate(map_ops_for_testing_.cookie, device_, 1,
                                              range);
    return vkInvalidateMappedMemoryRanges(device_, 1, range);
}

VkMappedMemoryRange VulkanBuffer::vk_range(VkDeviceSize offset,
                                           VkDeviceSize size) const {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(platform_->physical_device(), &properties);
    const auto aligned = mapped_range(offset, size, allocation_size_,
                                      properties.limits.nonCoherentAtomSize);
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory_;
    range.offset = aligned.offset;
    range.size = aligned.size;
    return range;
}

void *VulkanBuffer::map_persistent() {
    return pytorch_vulkan::with_vulkan_operation(platform_->runtime_lifecycle(), [&]() -> void* {
    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        throw std::runtime_error("Vulkan buffer memory is not host visible");
    if (mapped_base_ != nullptr) return mapped_base_;
    if (before_map_for_testing_) before_map_for_testing_();
    void *mapped = nullptr;
    if (map_memory(&mapped) != VK_SUCCESS)
        throw std::runtime_error("Could not persistently map Vulkan buffer memory");
    if (mapped == nullptr) {
        unmap_memory();
        throw std::runtime_error("Vulkan persistent map returned a null pointer");
    }
    mapped_base_ = mapped;
    return mapped_base_;
    });
}

void *VulkanBuffer::mapped_base() const { return mapped_base_; }

void VulkanBuffer::flush_range(VkDeviceSize offset, VkDeviceSize size) {
    maintain_range(platform_->runtime_lifecycle(), size,
                   [&] { return vk_range(offset, size); }, [&] { return coherent(); },
                   [&](const VkMappedMemoryRange *range) { return flush_memory(range); },
                   "Could not flush Vulkan buffer memory");
}

void VulkanBuffer::invalidate_range(VkDeviceSize offset, VkDeviceSize size) {
    maintain_range(platform_->runtime_lifecycle(), size,
                   [&] { return vk_range(offset, size); }, [&] { return coherent(); },
                   [&](const VkMappedMemoryRange *range) { return invalidate_memory(range); },
                   "Could not invalidate Vulkan buffer memory");
}

void VulkanBuffer::write(const void *data, VkDeviceSize size, VkDeviceSize offset) {
    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        throw std::runtime_error("Vulkan buffer memory is not host visible");
    }
    if (data == nullptr || offset > size_ || size > size_ - offset) {
        throw std::out_of_range("Vulkan buffer write is outside the buffer");
    }
    if (size == 0)
        return;

    return pytorch_vulkan::with_vulkan_operation(platform_->runtime_lifecycle(), [&] {
    const auto map_start = std::chrono::steady_clock::now();
    void *mapped = mapped_base_;
    const bool temporary_map = mapped == nullptr;
    if (temporary_map) {
        if (before_map_for_testing_) before_map_for_testing_();
        if (map_memory(&mapped) != VK_SUCCESS)
            throw std::runtime_error("Could not map Vulkan buffer memory for writing");
    }
    auto *bytes = static_cast<unsigned char *>(mapped) + offset;
    std::memcpy(bytes, data, static_cast<size_t>(size));
    if (!coherent()) {
        const auto range = vk_range(offset, size);
        if (flush_memory(&range) != VK_SUCCESS) {
            if (temporary_map) unmap_memory();
            throw std::runtime_error("Could not flush Vulkan buffer memory");
        }
    }
    if (temporary_map) unmap_memory();
    platform_->record_timing(
        VulkanTimingCategory::BufferMapWrite,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - map_start)
            .count());
    });
}

void VulkanBuffer::read(void *data, VkDeviceSize size, VkDeviceSize offset) const {
    if ((memory_properties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        throw std::runtime_error("Vulkan buffer memory is not host visible");
    }
    if (data == nullptr || offset > size_ || size > size_ - offset) {
        throw std::out_of_range("Vulkan buffer read is outside the buffer");
    }
    if (size == 0)
        return;

    return pytorch_vulkan::with_vulkan_operation(platform_->runtime_lifecycle(), [&] {
    const auto map_start = std::chrono::steady_clock::now();
    void *mapped = mapped_base_;
    const bool temporary_map = mapped == nullptr;
    if (temporary_map) {
        if (before_map_for_testing_) before_map_for_testing_();
        if (map_memory(&mapped) != VK_SUCCESS)
            throw std::runtime_error("Could not map Vulkan buffer memory for reading");
    }
    if (!coherent()) {
        const auto range = vk_range(offset, size);
        if (invalidate_memory(&range) != VK_SUCCESS) {
            if (temporary_map) unmap_memory();
            throw std::runtime_error("Could not invalidate Vulkan buffer memory");
        }
    }
    const auto *bytes = static_cast<const unsigned char *>(mapped) + offset;
    std::memcpy(data, bytes, static_cast<size_t>(size));
    if (temporary_map) unmap_memory();
    platform_->record_timing(
        VulkanTimingCategory::BufferMapRead,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - map_start)
            .count());
    });
}
