#pragma once

#include <mutex>
#include <thread>

namespace vulkan_test {
// std::mutex::try_lock() on a mutex the calling thread already owns is
// undefined, which is exactly the failure a "lock is not held" witness must
// survive. Probe from a separate thread; join() bounds it because try_lock()
// never blocks. `observer` optionally reports the probing thread's id.
inline bool other_thread_can_lock(std::mutex &mutex, std::thread::id *observer = nullptr) {
    bool acquired = false;
    std::thread probe([&] {
        if (observer) *observer = std::this_thread::get_id();
        acquired = mutex.try_lock();
        if (acquired) mutex.unlock();
    });
    probe.join();
    return acquired;
}
} // namespace vulkan_test
