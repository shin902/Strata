"""Capture actual engine/system memory without counting shared CPU/GPU mappings twice."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
label = sys.argv[1]
pid = int(subprocess.check_output(['pgrep', '-f', '^/home/shi/Strata/build-halo-714/strata --serve']).split()[0])
meminfo = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
engine = {line.split(':')[0]: int(line.split()[1]) for line in Path(f'/proc/{pid}/status').read_text().splitlines()
          if line.startswith(('VmRSS:', 'RssAnon:', 'RssFile:', 'RssShmem:', 'VmSwap:'))}
record = {'time': time.time(), 'pid': pid, 'meminfo_kib': meminfo, 'engine_kib': engine,
          'gtt_bytes': int(Path('/sys/class/drm/card1/device/mem_info_gtt_used').read_text()),
          'vmstat': Path('/proc/vmstat').read_text(), 'memory_psi': Path('/proc/pressure/memory').read_text()}
path = root / 'bench/results/local-8060s-single-copy' / f'{label}-memory.json'
with path.open('x') as output:
    json.dump(record, output, indent=2)
print(path)
