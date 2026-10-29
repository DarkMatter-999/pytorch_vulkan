#pragma once

#include "vulkan_buffer.h"
#include "vulkan_lifecycle_seam_test.h"
#include <array>
#include <cstring>

// The peer supplies only the native range query/cache calls. Admission, range
// ordering, coherent/zero handling and driver-error propagation are production.
struct VulkanBufferMaintenanceTestPeer {
    template<class Range, class Coherent, class Cache>
    static void run(pytorch_vulkan::VulkanRuntimeLifecycle &runtime,
                    VkDeviceSize size, Range range, Coherent coherent, Cache cache) {
        VulkanBuffer::maintain_range(runtime, size, range, coherent, cache,
                                     "Could not invalidate Vulkan buffer memory");
    }
};

namespace buffer_maintenance_test {
using namespace pytorch_vulkan;
using lifecycle_seam_test::require;

struct Cache {
    VulkanRuntimeLifecycle &runtime;
    unsigned ranges = 0, flushes = 0, invalidates = 0;
    bool fail = false;
    std::function<void()> on_cache;
    void run(bool coherent, bool flush = false, VkDeviceSize size = 5) {
        VulkanBufferMaintenanceTestPeer::run(runtime, size, [&] {
            // Models the physical-device query AND range construction: neither
            // may run before protection, even when coherent/empty.
            const auto activity = runtime.host_activity();
            require(runtime.operation_active_on_this_thread() ||
                    runtime.cleanup_active_on_this_thread(), "range query was unprotected");
            if (vulkan_callback_observation_active())
                require(runtime.cleanup_active_on_this_thread() &&
                        (runtime.operation_active_on_this_thread() || activity.cleanup_epilogues != 0) &&
                        activity.audited_native_cleanup_epilogues == 0,
                        "callback cache maintenance must be General cleanup");
            ++ranges;
            const auto aligned = VulkanBuffer::mapped_range(3, size, 64, 8);
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = lifecycle_seam_test::handle<VkDeviceMemory>(9);
            range.offset = aligned.offset; range.size = aligned.size;
            return range;
        }, [=] { return coherent; }, [&](const VkMappedMemoryRange *range) {
            require(range->offset == 0 && range->size == 8 &&
                    range->memory == lifecycle_seam_test::handle<VkDeviceMemory>(9),
                    "cache operation lost its aligned native range");
            if (flush) ++flushes; else ++invalidates;
            if (on_cache) on_cache();
            return fail ? VK_ERROR_MEMORY_MAP_FAILED : VK_SUCCESS;
        });
    }
};

inline void publication(bool coherent, bool fail = false) {
    lifecycle_seam_test::Root root;
    Cache cache{root.runtime}; cache.fail = fail;
    unsigned destroyed = 0;
    unsigned publications = 0;
    const std::array<unsigned char, 5> expected{{5, 9, 13, 17, 21}};
    std::array<unsigned char, 5> witness{};
    std::weak_ptr<std::array<unsigned char, 5>> weak_mapped;
    std::weak_ptr<int> weak_capture;
    bool callback_work_rejected = false, original = false;
    try {
        with_vulkan_operation(root.runtime, [&] {
            auto slot = root.context.reserve_transfer();
            auto owner = std::shared_ptr<int>(new int, [&](int *p) { ++destroyed; delete p; });
            auto mapped_owner = std::make_shared<std::array<unsigned char, 5>>(expected);
            const auto *mapped = mapped_owner->data();
            weak_mapped = mapped_owner;
            auto capture = std::make_shared<int>(7);
            weak_capture = capture;
            cache.on_cache = [&] {
                require(witness == std::array<unsigned char, 5>{} && publications == 0 &&
                        !weak_mapped.expired() && !weak_capture.expired(),
                        "invalidation must precede byte publication with live owners");
            };
            std::vector<std::shared_ptr<void>> owners{owner}; owner.reset();
            owners.push_back(mapped_owner);
            root.context.begin("callback-cache-maintenance");
            root.context.defer_transfer(std::move(slot), std::move(owners), [&, mapped_owner, mapped, capture] {
                require(vulkan_callback_observation_active(), "publication lost pump identity");
                try { with_vulkan_operation(root.runtime, [] {}); }
                catch (const VulkanAdmissionClosed &) { callback_work_rejected = true; }
                cache.run(coherent);
                require(cache.ranges == 1 && cache.invalidates == (coherent ? 0U : 1U),
                        "byte publication preceded exact-range maintenance");
                std::memcpy(witness.data(), mapped, witness.size());
                ++publications;
            });
            mapped_owner.reset(); capture.reset();
            root.context.submit();
            require(!root.context.query_complete() && publications == 0 && cache.ranges == 0 &&
                    witness == std::array<unsigned char, 5>{} && !weak_mapped.expired() &&
                    !weak_capture.expired() && root.context.lifecycle_snapshot().reservations == 1,
                    "unsignaled fence published bytes or released transfer ownership/slot");
            root.context.synchronize();
        });
    } catch (const std::runtime_error &e) {
        original = std::string(e.what()) == "Could not invalidate Vulkan buffer memory";
        if (!fail || !original) throw;
    }
    require(callback_work_rejected && destroyed == 1, "publication lost rejection or owner cleanup");
    require(weak_mapped.expired() && weak_capture.expired() &&
            root.context.lifecycle_snapshot().reservations == 0 &&
            publications == (fail ? 0U : 1U) &&
            witness == (fail ? std::array<unsigned char, 5>{} : expected),
            "mapped publication lost bytes, success-only counting, capture or slot retirement");
    require(cache.ranges == 1 && cache.invalidates == (coherent ? 0U : 1U),
            "publication skipped cache maintenance or invalidated coherent memory");
    require(root.runtime.outer_permits() == 0 && root.runtime.cleanup_epilogues() == 0,
            "publication stranded host accounting");
    const auto snapshot = root.context.lifecycle_snapshot();
    if (!fail) require(snapshot.healthy_empty() && root.context.query_complete(),
                       "successful publication is not empty/complete");
    else {
        require(original && !snapshot.healthy_empty() && root.context.pending_transfer_count() == 0,
                "failed cache publication must release payload but retain sticky failure");
        bool sticky = false;
        try { root.context.synchronize(); }
        catch (const std::runtime_error &e) {
            sticky = std::string(e.what()) == "Could not invalidate Vulkan buffer memory";
        }
        require(sticky, "cache driver error was cleared by later synchronization");
    }
}

inline void boundaries() {
    VulkanRuntimeLifecycle runtime(vulkan_host_clock()), foreign(vulkan_host_clock());
    Cache cache{runtime};
    // Preserve the ordinary public boundary, including closed admission.
    cache.run(false, true); cache.run(false);
    require(cache.flushes == 1 && cache.invalidates == 1, "ordinary cache path did not run");
    runtime.close_admission();
    bool closed = false;
    try { cache.run(true); } catch (const VulkanAdmissionClosed &) { closed = true; }
    runtime.reopen_admission();
    require(closed && cache.ranges == 2, "closed ordinary cache path touched native state");
    for (unsigned context = 0; context < 3; ++context) {
        // None, foreign operation, foreign cleanup: none covers this runtime.
        std::unique_ptr<VulkanOperationScope> operation;
        std::unique_ptr<VulkanCleanupScope> cleanup;
        if (context == 1) operation = std::make_unique<VulkanOperationScope>(foreign);
        if (context == 2) cleanup = std::make_unique<VulkanCleanupScope>(foreign);
        VulkanCallbackObservationScope callback;
        bool rejected = false;
        try { cache.run(true); } catch (const VulkanAdmissionClosed &) { rejected = true; }
        require(rejected && cache.ranges == 2, "uncovered callback touched native cache state");
    }
    {
        VulkanCleanupScope enclosing(runtime);
        VulkanCallbackObservationScope callback;
        cache.run(false, true); cache.run(false); cache.run(true); cache.run(false, false, 0);
        bool rejected = false;
        try { with_vulkan_operation(runtime, [] {}); }
        catch (const VulkanAdmissionClosed &) { rejected = true; }
        require(rejected && cache.flushes == 2 && cache.invalidates == 2 && cache.ranges == 6,
                "bounded cache cleanup admitted work or skipped required maintenance");
    }
    require(runtime.cleanup_epilogues() == 0, "cache cleanup accounting leaked");
}

inline void closing_wait() {
    VulkanRuntimeLifecycle runtime(vulkan_host_clock());
    Cache cache{runtime};
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    cache.on_cache = [&] { entered.set_value(); released.wait(); };
    std::exception_ptr error;
    auto worker = std::async(std::launch::async, [&] {
        try {
            VulkanCleanupScope enclosing(runtime);
            VulkanCallbackObservationScope callback;
            cache.run(false);
        } catch (...) { error = std::current_exception(); entered.set_value(); }
    });
    entered.get_future().wait();
    runtime.close_admission();
    const auto active = runtime.host_activity();
    auto closer = std::async(std::launch::async, [&] {
        runtime.wait_idle_until(std::chrono::steady_clock::now() + std::chrono::seconds(2),
                               VulkanIdleWait::AllEpilogues);
    });
    const bool premature = closer.wait_for(std::chrono::milliseconds(20)) == std::future_status::ready;
    release.set_value(); worker.get(); closer.get();
    if (error) std::rethrow_exception(error);
    require(!premature && active.cleanup_epilogues == 1 &&
            active.audited_native_cleanup_epilogues == 0 && runtime.cleanup_epilogues() == 0,
            "Closing did not wait through bounded General cache epilogue");
}

// Matches the production root adapter's synchronize_execution_until boundary:
// no new operation admission, and no completion at the PythonNative boundary.
struct CompletionRoot : lifecycle_seam_test::Root {
    static void complete(void *p, VulkanHostTime deadline) {
        auto &root = *static_cast<CompletionRoot *>(p);
        VulkanCleanupScope cleanup(root.runtime);
        ++root.completions;
        root.context.synchronize_with_flush_until([&] {
            if (root.context.recording()) root.context.submit_locked();
        }, deadline, vulkan_host_clock());
    }
    VulkanLifecycleRoot adapter(std::shared_ptr<CompletionRoot> owner) {
        auto result = lifecycle_seam_test::Root::adapter(owner);
        result.complete = complete;
        return result;
    }
};

inline void pending_publication(CompletionRoot &root, Cache &cache, bool &rejected) {
    with_vulkan_operation(root.runtime, [&] {
        auto slot = root.context.reserve_transfer();
        auto mapped_owner = std::make_shared<std::array<unsigned char, 5>>(
            std::array<unsigned char, 5>{{5, 9, 13, 17, 21}});
        const auto *mapped = mapped_owner->data();
        std::vector<std::shared_ptr<void>> owners{mapped_owner};
        root.context.begin("coordinator-cache-publication");
        root.context.defer_transfer(std::move(slot), std::move(owners), [&, mapped_owner, mapped] {
            try { with_vulkan_operation(root.runtime, [] {}); }
            catch (const VulkanAdmissionClosed &) { rejected = true; }
            cache.run(false);
            std::array<unsigned char, 5> witness{};
            require(cache.invalidates == 1, "General completion copied before invalidation");
            std::memcpy(witness.data(), mapped, witness.size());
            require(witness == std::array<unsigned char, 5>{{5, 9, 13, 17, 21}},
                    "retained-root General completion lost mapped bytes");
        });
        root.context.submit();
    });
    require(cache.ranges == 0, "pending publication ran before coordinator completion");
}

inline void completion() {
    // Preflight gives a caught ordinary RED when the internal completion entry
    // lacks coverage, rather than entering a terminal coordinator fault child.
    {
        CompletionRoot root;
        Cache cache{root.runtime}; bool rejected = false;
        pending_publication(root, cache, rejected);
        CompletionRoot::complete(&root, std::chrono::steady_clock::now() + std::chrono::seconds(2));
        require(rejected && cache.invalidates == 1 && root.context.lifecycle_snapshot().healthy_empty(),
                "internal completion did not cover native cache publication");
    }
    for (unsigned route = 0; route < 3; ++route) {
        VulkanLifecycleCoordinator coordinator(vulkan_host_clock());
        auto root = std::static_pointer_cast<CompletionRoot>(coordinator.acquire([&] {
            auto r = std::make_shared<CompletionRoot>(); return r->adapter(r);
        }));
        std::weak_ptr<CompletionRoot> lifetime = root;
        Cache cache{root->runtime}; bool rejected = false;
        pending_publication(*root, cache, rejected);
        root.reset();
        if (route == 1) coordinator.retire_cached();
        if (route == 2) {
            const auto deadline = coordinator.prepare_python_finalizing_sync();
            require(cache.ranges == 0, "prepare phase must not publish");
            coordinator.finish_python_finalizing_sync(deadline);
        }
        coordinator.close(VulkanCloseBoundary::NativeEarly);
        require(lifetime.expired() && rejected && cache.ranges == 1 && cache.invalidates == 1,
                "coordinator route did not publish under retained-root General cleanup");
    }
}
} // namespace buffer_maintenance_test
