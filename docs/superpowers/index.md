# Superpowers Documentation Index

This index covers the current project documentation under `docs/superpowers/` and the legacy `superpowers/` tree. The legacy documents are retained for historical reference.

## Current Documentation: `docs/superpowers/`

### Specifications and Roadmaps

- [Vulkan Phase 4 Capability Matrix](../vulkan_operator_capability_matrix.md) - Versioned PyTorch 2.4 operator/dtype inventory and fixed CNN/MLP workload contract.
- [Vulkan Serialization And Multiprocessing Design](specs/2026-09-12-vulkan-serialization-multiprocessing-design.md) - Explicit CPU-owned Vulkan serialization and spawn-based multiprocessing contract.
- [Vulkan Autograd Integration Design](specs/2026-09-12-vulkan-autograd-design.md) - Hybrid first-order autograd support for shipped operators and paired forward/backward rollout policy.
- [Vulkan View And Reshape Design](specs/2026-09-12-vulkan-view-reshape-design.md) - Staged design for safe metadata-only views and later reshape support.
- [Vulkan Pointwise Scalar Support Design](specs/2026-09-12-vulkan-pointwise-scalars-design.md) - Contract and architecture for scalar pointwise Vulkan operations.
- [Vulkan Subphase 3A.1 Design](specs/2026-09-11-vulkan-transfer-3a1-design.md) - Design for synchronous contiguous float32 host-device transfers.
- [PrivateUse1 Vulkan Allocator Design](specs/2026-09-11-privateuse1-vulkan-allocator-design.md) - Native allocator and device-guard integration design for Vulkan tensors.
- [Vulkan Subphase 3B.1 Design](specs/2026-09-11-vulkan-add-3b1-design.md) - Narrow Vulkan `add` operator contract and compute-dispatch design.
- [Vulkan Phase 3 Roadmap Design](specs/2026-09-11-vulkan-phase-3-roadmap-design.md) - Capability roadmap for transfers, pointwise operators, views, and advanced operations.

### Implementation Plans

- [Vulkan Serialization And Multiprocessing Implementation Plan](plans/2026-09-12-vulkan-serialization-multiprocessing.md) - Completed feasibility-gated explicit serialization and CPU-boundary multiprocessing plan.
- [Vulkan Autograd Integration Implementation Plan](plans/2026-09-12-vulkan-autograd.md) - First-order autograd for shipped operators with paired forward/backward rollout.
- [Vulkan Autograd Task 4 Brief](../../.superpowers/sdd/2026-09-12-vulkan-autograd/task-4-brief.md) - Final edge-case, documentation, and complete-verification gate.
- [Vulkan Metadata View Foundation Implementation Plan](plans/2026-09-12-vulkan-metadata-view-foundation.md) - Test-first plan for contiguous zero-offset metadata-only views and checked layout validation.
- [Vulkan Formatter-Compatible View Implementation Plan](plans/2026-09-12-vulkan-print-view.md) - Steps to support the metadata-only view needed by tensor formatting.
- [Vulkan Pointwise `out=` Task 4 Implementation Plan](plans/2026-09-12-vulkan-pointwise-out-task-4.md) - Registration, test, and verification tasks for pointwise `out=` support.
- [Vulkan Synchronous Transfer 3A.1 Implementation Plan](plans/2026-09-11-vulkan-transfer-3a1.md) - Test-first implementation plan for safe synchronous tensor transfers.
- [Vulkan Pointwise Scalar Support Implementation Plan](plans/2026-09-12-vulkan-pointwise-scalars.md) - Test-first rollout for scalar add, subtract, and multiply operations.
- [Vulkan `add` 3B.1 Implementation Plan](plans/2026-09-11-vulkan-add-3b1.md) - Implementation tasks for the initial Vulkan binary add operator.
- [Vulkan Unary Operators Implementation Plan](plans/2026-09-12-vulkan-unary-operators.md) - Implementation and verification plan for Vulkan unary operations.
- [Vulkan `abs.out` Implementation Plan](plans/2026-09-12-vulkan-abs-out.md) - Focused plan for the Vulkan `abs.out` registration and coverage.
- [PrivateUse1 Vulkan Allocator Implementation Plan](plans/2026-09-11-privateuse1-vulkan-allocator.md) - Tasks to integrate and validate the PrivateUse1 Vulkan allocator.

## Legacy Documentation: `superpowers/`

These documents remain in the legacy tree and may overlap with the current documentation above.

### Specifications and Roadmaps

- [Vulkan Backend Migration Design](../../superpowers/specs/2026-09-10-vulkan-backend-migration-design.md) - Overall phased plan for migrating the Vulkan backend.
- [Vulkan Subphase 3A.1 Design](../../superpowers/specs/2026-09-11-vulkan-transfer-3a1-design.md) - Design for synchronous contiguous float32 host-device transfers.
- [Vulkan Subphase 3B.1 Design](../../superpowers/specs/2026-09-11-vulkan-add-3b1-design.md) - Narrow Vulkan `add` operator contract and compute-dispatch design.
- [Vulkan Data-Type Expansion Design Notes](../../superpowers/specs/2026-09-12-vulkan-dtype-expansion-design.md) - Scope and constraints for expanding supported Vulkan tensor dtypes.
- [Vulkan Pointwise `out=` Design](../../superpowers/specs/2026-09-12-vulkan-pointwise-out-design.md) - Operator API and validation design for pointwise output tensors.
- [Vulkan Unary Operators Design](../../superpowers/specs/2026-09-12-vulkan-unary-operators-design.md) - Compute and registration design for Vulkan unary operators.
- [Vulkan Phase 3 Roadmap Design](../../superpowers/specs/2026-09-11-vulkan-phase-3-roadmap-design.md) - Capability roadmap for transfers, pointwise operators, views, and advanced operations.
- [Vulkan Pointwise Scalar Support Design](../../superpowers/specs/2026-09-12-vulkan-pointwise-scalars-design.md) - Contract and architecture for scalar pointwise Vulkan operations.
- [PrivateUse1 Vulkan Allocator Design](../../superpowers/specs/2026-09-11-privateuse1-vulkan-allocator-design.md) - Native allocator and device-guard integration design for Vulkan tensors.

### Implementation Plans

- [Vulkan Unary Operators Implementation Plan](../../superpowers/plans/2026-09-12-vulkan-unary-operators.md) - Implementation and verification plan for Vulkan unary operations.
- [PrivateUse1 Vulkan Allocator Implementation Plan](../../superpowers/plans/2026-09-11-privateuse1-vulkan-allocator.md) - Tasks to integrate and validate the PrivateUse1 Vulkan allocator.
- [Vulkan Pointwise `out=` Implementation Plan](../../superpowers/plans/2026-09-12-vulkan-pointwise-out.md) - Test-first rollout plan for pointwise output-tensor support.
- [Vulkan Pointwise Scalar Support Implementation Plan](../../superpowers/plans/2026-09-12-vulkan-pointwise-scalars.md) - Test-first rollout for scalar add, subtract, and multiply operations.
- [Vulkan `add` 3B.1 Implementation Plan](../../superpowers/plans/2026-09-11-vulkan-add-3b1.md) - Implementation tasks for the initial Vulkan binary add operator.
- [Vulkan `abs.out` Implementation Plan](../../superpowers/plans/2026-09-12-vulkan-abs-out.md) - Focused plan for the Vulkan `abs.out` registration and coverage.
- [Vulkan Synchronous Transfer 3A.1 Implementation Plan](../../superpowers/plans/2026-09-11-vulkan-transfer-3a1.md) - Test-first implementation plan for safe synchronous tensor transfers.
- [Vulkan Formatter-Compatible View Implementation Plan](../../superpowers/plans/2026-09-12-vulkan-print-view.md) - Steps to support the metadata-only view needed by tensor formatting.

### Reports and Project Tracking

- [Vulkan Migration Ledger](../../superpowers/migration-ledger.md) - Status ledger for migration phases, delivered work, and remaining gaps.
- [Epic: Establish the Vulkan Build and Test Boundary](../../superpowers/github/epic-phase1-vulkan-transition.md) - GitHub epic defining the initial Vulkan build and test transition.
- [PR Description](../../superpowers/github/pr-description.md) - Pull-request summary, testing evidence, and review notes for the migration work.
