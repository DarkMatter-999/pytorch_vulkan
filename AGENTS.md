# Repository development standard

Read [docs/backend-development-standard.md](docs/backend-development-standard.md)
before designing or modifying backend behavior. This is the standing project
direction, including work outside convolution.

- Goal: CPU/CUDA-comparable functionality through the ordinary PyTorch API.
- Reference the supported PyTorch version's schemas, native/composite
  implementations, dispatch registrations and autograd rules before designing
  Vulkan equivalents. Prefer existing PyTorch generated autograd/composites with
  Vulkan leaf kernels over duplicate backend-specific semantic machinery.
- Prove reference behavior with small CPU probes and qualify Vulkan execution.
  Treat a missing dependency as implementation work, not a reason to redefine
  expected PyTorch behavior or silently detach a requested gradient graph.
- Record unsupported contracts explicitly. Higher-order, forward-mode and
  transform support require their own evidence; do not infer them from first
  backward or operator registration.
- Keep model/training acceptance tests on normal PyTorch modules/APIs. Do not
  substitute backend-private APIs to bypass a compatibility failure.
- The human owns commits and pushes. Agents must not run git commit or git push.
- Never weaken or skip tests to hide failures; deliberate contract expansions
  replace old rejection tests with positive reference-based coverage.
- Run GPU gates one at a time. Use CPU-seeded tiny tensors, transfer to Vulkan,
  and synchronize before readback. Recent hardware evidence is Renoir iGPU only.
- Preserve excluded docs/superpowers and .superpowers records. Those records
  carry decisions and evidence; durable project policy lives in tracked docs.
- User prefers subagents to conserve context and GPT-6.1 Sol for heavy work.
  Give workers bounded tasks and reference evidence; verify their conclusions.
