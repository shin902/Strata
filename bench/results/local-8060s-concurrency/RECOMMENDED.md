# Local Strix Halo single-stream setup

System: Ryzen AI Max+ 395 / gfx1151, Linux 7.2.6 Arch, OS-visible RAM 91 GiB. Strata engine 0.1.40.3.

Active config: `/home/shi/Strata/strata-iq3_s.json`.
Engine: `/home/shi/Strata/build-halo-714/strata`.
SDK: `/home/shi/Strata-sdk/rocm-7.14.1`, archive SHA256 verified against docs/STRIX_HALO.md.
System `/opt/rocm` was not changed.

Settings follow docs/STRIX_HALO.md's fast configuration: hipBLASLt gfx1151/100401 table, exact gfx1151 defaults plus its eight opt-in environment flags; prefill 16384; spec 4; lookup-chain 3; mtp-q4 all; int8 KV. Context 32768, parallel 1. No image input or external listening.

Local memory adaptation: `--mmap-experts` maps the existing GGUF instead of allocating an additional 46.84 GiB host expert arena. `--expert-cache auto --vram-reserve-mib 8192` then loads all 24576 experts into 46.84 GiB of GPU-addressable shared RAM. No extra experts.bin or model download. Around 30 GiB of MemAvailable remains after validation. Swap around 3.7 GiB remained from earlier pressure, without material growth during these runs. Opt-in fast kernels change numerical rounding and need not produce bit-identical answers to default kernels.

Validation: loaded health, concurrency.serving=1, hipBLASLt table accepted, all experts resident, six successful long-input completions with 512 output tokens (three at ~8K and three at ~16K), and Japanese arithmetic response 42. The previous ROCm 7.2 / auto-prefill 16K-input failure is retained in failed-16k.log; its cause was not isolated to the ROCm version.

Three-repeat medians, temperature 0, thinking off, unique request prefixes, zero cached prompt tokens:

| Active fast configuration workload | PP tok/s (engine) | TG tok/s (engine) |
| --- | ---: | ---: |
| ~8266-token synthetic code input, 512 output | 935.7 | 37.2 |
| ~16460-token synthetic code input, 512 output | 915.5 | 37.9 |
| ~75-token essay instruction, 512 output | — | 40.6 |

These are NOT a reproduction of the official fixed prompt/token benchmark; OS, memory setup, corpus, continuation, and MTP acceptance differ. Official ~1300 PP / ~60 TG have not been reached. 32K-capacity boundary input was not exercised.

Re-run: `cd /home/shi/Strata && .venv/bin/python bench/local_concurrency.py NAME 8192 solo-only long-generation`.
Start: `cd /home/shi/Strata && ./run-iq3_s.sh`.
Raw responses/timings: recommended-fast-8k.json and recommended-fast-16k.json. Current config and logs: recommended-config.json, recommended-engine.log, recommended-status.json. Earlier config backup: pre-recommended-config.json.
