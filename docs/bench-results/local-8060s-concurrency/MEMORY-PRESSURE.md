# Memory-pressure check

Command: `.venv/bin/python bench/memory_pressure.py 2954485` on the active recommended single-stream configuration.

147.2 seconds including warm-up and three repeats each of ~16K synthetic code input + 512 output and short input + 512 output. Zero prompt cache reuse. Sampled /proc/meminfo, /proc/vmstat, /proc/pressure/memory and the actual engine PID's /proc I/O and fault counters every 0.2 s.

- Lowest MemAvailable: 29.779 GiB; lowest MemFree: 1.866 GiB.
- Swap out: **0 pages**. Swap in: 3205 pages (12.52 MiB on this 4 KiB page machine).
- Swap used fell from 3.696 to 3.464 GiB; it never exceeded the initial value.
- Direct reclaim scans: **0**.
- Memory PSI some: 2520 us total; full: 2511 us, about 0.0017% of the measured duration.
- Actual engine storage reads: 344657920 bytes (328.7 MiB) over the whole run.
- Engine major page faults: 52629. These are real file paging, not zero: absence of swap-out/direct reclaim/PSI pressure does not prove mmap is free or all I/O is negligible. The expert-only log reports 0 major faults for its measured expert fetches; the process-wide counter includes other file mappings.
- Median engine PP: 926.9 tok/s. Short-input TG: 40.1 tok/s. TG after long input: 39.1 tok/s.

Conclusion: these slow runs did not exhibit RAM-capacity pressure, swap-out, or direct-reclaim stalls. The current ~927 PP / ~40 TG gap cannot reasonably be attributed to active swapping or reclaiming due to insufficient RAM in this workload. mmap/file-I/O overhead and the remaining gap to the official benchmark have not been causally isolated by this measurement.

Raw evidence: memory-pressure-summary.json, memory-pressure-samples.json, memory-pressure-16k.json, memory-pressure-16k-summary.json; stdout at /home/shi/Strata/bench-memory-pressure.out.
