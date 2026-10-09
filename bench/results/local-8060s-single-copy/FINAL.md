# Shared-RAM expert storage: final local validation

## Outcome and boundaries

Patched checkout: Strata d5ea7133741e67743c0e886bb426c0ce8d69cf6c plus local diff. No commit/PR was made. Model: existing IQ3_S shards. SDK: private `/home/shi/Strata-sdk/rocm-7.14.1`; system ROCm, kernel, driver and BIOS were not changed.

Active saved config: `/home/shi/Strata/strata-iq3_s.json`, `STRATA_UMA_SINGLE_COPY=1`, context capacity 262144, prefill chunk **16384**, parallel 1, loopback only. The **128** in the measurements is output length, not prefill chunk size. Server is stopped at handoff.

The mode is opt-in and gated to Linux HIP, an integrated GPU, one GPU, a nonempty profile, an enabled cache, no remote/peer expert cache, no shared expert arena and no explicitly selected resident-expert mode. Shared storage was validated on gfx1151 only. Discrete-GPU, Windows and multi-GPU storage selection remains the original path; no real hardware tests for those machines were available. The context-boundary repair applies to CLI/server across backends.

## Ownership repair

- Avoid loading the full CPU expert arena.
- Allocate the GPU expert cache once with HIP fine-grained device memory. CPU views point into that same physical allocation.
- Keep only non-GPU experts in the existing resident CPU complement; full GPU residency means **0 GiB** of fixed CPU expert copies.
- Bind CPU aliases only when every expert is GPU resident. No prefill borrowing/overwriting of the only copy; allocate separate prefill workspace.
- Reuse aliases for blob/copy_blob/pinned/device_alias, so full-resident prefill does not reread expert weights from files.
- Use existing O_DIRECT startup loading to avoid a second expert-sized buffered file cache. Other dense weights/PLE/library/file caches are not eliminated or claimed to be zero.
- Release 0.63 GiB of no-longer-needed native startup assembly buffers after full alias binding (when asynchronous I/O prefetch is off).
- Clamp MTP and lookup windows to remaining context positions. Before the repair, the 64K CLI boundary exited with position65533 despite valid remaining output positions; after the repair all 128 output tokens completed.

## Measured memory reduction

Final same-cache control, GPU cache 7846 slots / 14.88 GiB in both arms:

| Quantity | CPU arena control | Single-copy complement |
| --- | ---: | ---: |
| CPU expert bytes | 46.84 GiB | 31.96 GiB |
| MemAvailable at ready | 17.106 GiB | 25.769 GiB |
| Engine VmSwap | 0 | 0 |

**Net measured RAM saving: 8.66 GiB**, not 46.84 GiB at this matched partial-cache size. Expert duplication removed is 14.88 GiB. A separate prefill workspace adds approximately 5 GiB (GTT increased 21.394→26.396 GiB); native assembly/runtime staging and different host working sets account for further overhead. CPU RSS and GPU GTT are not summed because mappings may describe the same physical pages.

Full-cache case: GPU expert allocation46.84GiB, CPU expert complement0GiB. Max-context ready GTT about61.7GiB includes expert cache, prefill workspace, dense weights, KV/state, MTP and other GPU buffers; it is not 61.7GiB of extra independent VRAM. The final startup assembly release logged0.63GiB; observed CPU anonymous RSS fell1.017→0.578GiB across the two ready snapshots. These cross-start snapshots do not isolate allocator retention from other working-set variation. Separate workspace is necessary for this implementation's immutable shared weights; no claim of globally minimal workspace.

## Long-context validation

GGUF metadata: `qwen4exp.context_length=262144`.

| Surface / version | Config capacity | Actual input | Generated | PP / TG tok/s | Swap out |
| --- | ---: | ---: | ---: | --- | ---: |
| CLI, boundary fix | 65536 | 65408 | 128 | 1026.28 / 43.85 | 0 pages |
| CLI, boundary fix | 131072 | 130944 | 128 | 981.01 / 40.07 | 0 pages |
| CLI, final staging release | 262144 | 262016 | 128 | 890.80 / 40.06 | 0 pages |
| HTTP, adopted final config | 262144 | 262008 | 128 | see engine log; 299.98s wall | 0 pages |

HTTP intentionally reserves8 positions. A request with262016 input+128 output was correctly rejected by its existing admission check, rather than silently truncated. Accepted final HTTP run: count_tokens262008, usage input262008/output128, cache_read_input_tokens0, stop_reason max_tokens; lowest MemAvailable22.068GiB; engine major faults0; swap-in38 pages; swap usage3888592→3888436KiB and peak never exceeded starting usage. Existing swapped pages were not forcibly swapped in or cleared.

Earlier HTTP runs before staging release succeeded at the same accepted boundary but had384/256 pages of swap-out while total/peak swap usage did not grow and engine VmSwap remained0. Earlier prototype failures and their safety-triggered exits are retained. Do not describe every startup/benchmark as zero swapping: final fresh8K run3 had1385 pages of system swap-out.

## Published benchmark reproduction status

**Not reproduced.** Exact maintainer IQ3_S input/continuation token files and the earlier comparison engine branch were not found in the checkout. `docs/STRIX_HALO.md` publishes switches, capacities, throughput formula and3 fresh-process medians, but not these inputs. Repository search found the table in docs and unrelated community benchmarks, not a runnable matching IQ3_S recipe. The community gfx1151 Q2_0 synthetic benchmark is a different workload. Next input needed for exact reproduction: maintainer prompt IDs, continuation IDs/sampling and compared engine revision.

Final independent synthetic8K fresh-process runs: input8064, output128, same prefill16384 and official fast switches. PP1111.42/1105.69/1120.83 (median1111.42), TG44.19/43.07/44.24 (median44.19). MTP accepted78/103,78/102,77/102 (75.5–76.5%). Official IQ3_S8K table is roughly1242–1265 PP /59–60 TG; the different corpus means the gap is not a kernel-only comparison. Final output128 token IDs exactly match the original mmap-path reference for the same synthetic input. The reference PP/TG was1014.25/41.90, one run, not an interleaved confidence interval.

## Runnable checks and artifacts

- Build: `cmake --build build-halo-714 --target strata hip_expert_cache_staging file_expert_source_test -j 8`.
- Private SDK LD_LIBRARY_PATH is required for binaries/ctest.
- `ctest --test-dir build-halo-714 -R '^(hip_expert_cache_staging|file_expert_source_test)$' --output-on-failure`: **2 passed**. Shared-cache test exercises packed slot offsets, GPU→CPU bytes, CPU→GPU updates, invalid slots, close/reopen; original staging checks remain.
- `tools/hip/check_shared_expert_memory.cpp`: real gfx1151 HIP CPU/GPU read/write parity for mapped-host and fine-grained-device allocations.
- `.venv/bin/python bench/check_single_copy_memory.py`: final matched-cache snapshots, >=8GiB net savings and engine VmSwap0. It prints system swap change (+6.66MiB across the two separate partial-cache starts); it does not claim zero startup swapping.
- `.venv/bin/python bench/single_copy_context.py 262144 UNIQUE_LABEL strata-iq3_s.json`: fresh CLI boundary test, 128 output, paging/PSI/process memory/GTT samples and safety floor.
- Start with `./run-iq3_s.sh`, then `.venv/bin/python bench/server_context_boundary.py 262144 UNIQUE_LABEL`: server-counted accepted boundary, zero prefix reuse, output128 and peak swap-usage gate.
- `adopted-8k-{1,2,3}.*`, `staging-released-context-256k.*`, `adopted-http-boundary-256k-*`, `adopted-api-answer.json`, `adopted-engine.log`, `adopted-config.json`, final memory snapshots and `final-ctest-success.out` are concrete evidence.
- `git diff --check` and Python py_compile passed.

Results scripts are local additions, not shipped upstream. Earlier fixed-name memory-pressure artifacts were accidentally overwritten during prototype reuse; the earlier Activity/MEMORY-PRESSURE.md retain summaries, not the original raw sample JSON. Labelled output was added afterward. No claim of full original raw-history preservation.

## Completion audit

| Requirement | Evidence / status |
| --- | --- |
| Remove duplicate expert ownership | fine-grained cache + CPU alias; full cache complement0; equal-cache control14.88GiB expert reduction |
| Verify actual memory saving | final memory snapshots and runnable check8.66GiB net |
| Investigate limited savings and fix avoidable leftovers | separate prefill workspace identified; additional0.63GiB startup staging released and final256K rerun |
| Decode/MTP/prefill correctness | owner HIP tests, final arithmetic42, 128-token reference parity, CLI/HTTP long completions |
| Exact published benchmark | blocked on missing source inputs/engine branch; local3-run rates/acceptance and mismatch documented, not claimed reproduced |
| Largest model-supported context | metadata262144; CLI262016+128, HTTP262008+128+8 reservation |
| Swap and memory at long context | final CLI/HTTP swap-out0, HTTP peak swap usage unchanged, sampled PSI/faults/GTT and available RAM |
| Preserve platform/model/system boundaries | opt-in HIP/Linux/integrated/single-GPU selection; existing discrete/Windows/multi-GPU storage paths; no system/model replacement |
| Reproducible records / Activity | this report, scripts, raw logs/samples, follow-up shin902 Activity |
| No runaway work | CLI processes exited; server explicitly TERM-stopped, process absence checked |

Completion is for the ownership repair and measured validation, with the exact-benchmark branch explicitly blocked; it is not a claim to have reached official speeds or mathematically identical answers for every possible prompt.
