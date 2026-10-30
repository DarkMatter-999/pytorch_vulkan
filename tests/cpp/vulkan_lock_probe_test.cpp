#include "vulkan_lock_probe.h"

#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

namespace {
void expect(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::_Exit(1);
    }
}
} // namespace

int main() {
    std::mutex mutex;
    for (int i = 0; i != 100; ++i) {
        std::thread::id observer;
        expect(vulkan_test::other_thread_can_lock(mutex, &observer), "unheld mutex reported held");
        expect(observer != std::this_thread::get_id(), "probe ran on the calling thread");
        // The probe must not leave the mutex locked.
        expect(mutex.try_lock(), "probe left an unheld mutex locked");
        mutex.unlock();
    }
    {
        std::lock_guard<std::mutex> guard(mutex);
        std::thread::id observer;
        expect(!vulkan_test::other_thread_can_lock(mutex, &observer), "mutex held by caller reported free");
        expect(observer != std::this_thread::get_id(), "held probe ran on the calling thread");
        expect(!vulkan_test::other_thread_can_lock(mutex), "probe released a mutex it did not own");
    }
    {
        std::mutex gate_mutex;
        std::condition_variable gate;
        bool locked = false, release = false;
        std::thread holder([&] {
            std::lock_guard<std::mutex> guard(mutex);
            { std::lock_guard<std::mutex> g(gate_mutex); locked = true; }
            gate.notify_all();
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate.wait(lock, [&] { return release; });
        });
        {
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate.wait(lock, [&] { return locked; });
        }
        expect(!vulkan_test::other_thread_can_lock(mutex), "mutex held by a third thread reported free");
        { std::lock_guard<std::mutex> g(gate_mutex); release = true; }
        gate.notify_all();
        holder.join();
        expect(vulkan_test::other_thread_can_lock(mutex), "released mutex reported held");
    }
    std::cout << "PASS vulkan lock probe\n";
    return 0;
}
