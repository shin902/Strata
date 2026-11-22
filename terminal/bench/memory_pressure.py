#!/usr/bin/env python3
"""Record actual memory stalls, paging and disk reads during the local inference benchmark."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'bench/results/local-8060s-concurrency'
pid = int(sys.argv[1])
label = sys.argv[2] if len(sys.argv) > 2 else 'memory-pressure'


def read_counters(path):
    return {line.split()[0].rstrip(':'): int(line.split()[1])
            for line in Path(path).read_text().splitlines() if len(line.split()) >= 2}


def sample():
    memory = read_counters('/proc/meminfo')
    vm = read_counters('/proc/vmstat')
    io = read_counters(f'/proc/{pid}/io')
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    pressure = {}
    for line in Path('/proc/pressure/memory').read_text().splitlines():
        parts = line.split()
        pressure[parts[0]] = int(parts[-1].split('=')[1])
    return {'time': time.time(), 'available_kib': memory['MemAvailable'], 'free_kib': memory['MemFree'],
            'swap_used_kib': memory['SwapTotal'] - memory['SwapFree'],
            'pswpin_pages': vm['pswpin'], 'pswpout_pages': vm['pswpout'],
            'global_major_faults': vm['pgmajfault'], 'engine_major_faults': int(fields[9]),
            'engine_read_bytes': io['read_bytes'], 'memory_some_us': pressure['some'],
            'memory_full_us': pressure['full'],
            'direct_scans': sum(value for key, value in vm.items() if key.startswith('pgscan_direct'))}


before = sample()
rows = [before]
with (ROOT / f'bench-{label}.out').open('w') as log:
    benchmark = subprocess.Popen([sys.executable, str(ROOT / 'bench/local_concurrency.py'),
                                  label + '-16k', '16384', 'solo-only', 'long-generation'],
                                 stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    while benchmark.poll() is None:
        time.sleep(0.2)
        rows.append(sample())
after = sample()
rows.append(after)
seconds = after['time'] - before['time']
deltas = {key: after[key] - before[key] for key in before if key not in
          ('time', 'available_kib', 'free_kib', 'swap_used_kib')}
summary = {'exit_code': benchmark.returncode, 'duration_s': seconds,
           'min_available_gib': min(row['available_kib'] for row in rows) / 1048576,
           'min_free_gib': min(row['free_kib'] for row in rows) / 1048576,
           'swap_before_gib': before['swap_used_kib'] / 1048576,
           'swap_after_gib': after['swap_used_kib'] / 1048576,
           'peak_swap_gib': max(row['swap_used_kib'] for row in rows) / 1048576,
           'memory_some_stall_pct': deltas['memory_some_us'] / (seconds * 10000),
           'memory_full_stall_pct': deltas['memory_full_us'] / (seconds * 10000), 'deltas': deltas}
(OUT / f'{label}-samples.json').write_text(json.dumps(rows))
(OUT / f'{label}-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
assert benchmark.returncode == 0, 'benchmark failed; inspect bench-memory-pressure.out'
