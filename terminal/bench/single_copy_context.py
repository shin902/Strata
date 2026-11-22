#!/usr/bin/env python3
"""Fresh-process context-boundary inference with paging and memory evidence."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'bench/results/local-8060s-single-copy'
sys.path.insert(0, str(ROOT / 'tools'))
import strata_tokenizer as ST

capacity = int(sys.argv[1])
label = sys.argv[2]
config = json.loads((ROOT / (sys.argv[3] if len(sys.argv) > 3 else 'strata-single-copy.json')).read_text())
folder = Path(config['tokenizer'])
vocab = json.loads((folder / 'vocab.json').read_text())
tokens = [None] * len(vocab)
for token, index in vocab.items():
    tokens[index] = token
tok = ST.Tokenizer(tokens, (folder / 'merges.txt').read_text().split('\n'),
                   json.loads((folder / 'token_type.json').read_text()))
corpus = '\n'.join(f'def transform_{i:06d}(value): return (value * {i % 97 + 1} + {i}) % 100003' for i in range(capacity // 15 + 100))
header = tok.encode('<|im_start|>user\nReview the following code.\n', parse_special=True)
tail = tok.encode('\nWrite a detailed technical review of this code. Explain testing and reliability.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
ids = header + tok.encode(corpus)[:capacity - 128 - len(header) - len(tail)] + tail
assert len(ids) == capacity - 128, (len(ids), capacity)
path = OUT / f'{label}-tokens.txt'
path.write_text(','.join(map(str, ids)))
args = config['args'].copy()
args[args.index('--max-context') + 1] = str(capacity)
args += ['--tokens-file', str(path), '--max-new', '128']
env = os.environ.copy()
env.update(config['env'])
env['LD_LIBRARY_PATH'] = ':'.join(config['lib_dirs']) + ':' + env.get('LD_LIBRARY_PATH', '')


def counters(path):
    return {line.split()[0].rstrip(':'): int(line.split()[1]) for line in Path(path).read_text().splitlines()}


def sample():
    memory = counters('/proc/meminfo')
    vm = counters('/proc/vmstat')
    return {'time': time.time(), 'available_kib': memory['MemAvailable'],
            'swap_used_kib': memory['SwapTotal'] - memory['SwapFree'],
            'memory_kib': memory,
            'vm': {k: v for k, v in vm.items() if k.startswith(('pgscan_', 'compact_', 'allocstall_'))},
            'pswpout': vm['pswpout'], 'pswpin': vm['pswpin'],
            'gtt_bytes': int(Path('/sys/class/drm/card1/device/mem_info_gtt_used').read_text()),
            'psi': Path('/proc/pressure/memory').read_text()}


rows = [sample()]
blocked = None
swap_limit_pages = 32768 if 'reference' in sys.argv[4:] else 8192
with (OUT / f'{label}.out').open('w') as stdout, (OUT / f'{label}.log').open('w') as stderr:
    process = subprocess.Popen([config['exe'], *args], stdout=stdout, stderr=stderr, env=env, cwd=ROOT)
    try:
        while process.poll() is None:
            time.sleep(0.2)
            row = sample()
            status = Path(f'/proc/{process.pid}/status')
            if status.exists():
                row['engine_memory'] = {line.split(':')[0]: int(line.split()[1])
                                        for line in status.read_text().splitlines()
                                        if line.startswith(('VmRSS:', 'RssAnon:', 'RssFile:', 'VmSwap:'))}
            rows.append(row)
            if row['available_kib'] < 2 * 1048576 or row['pswpout'] - rows[0]['pswpout'] > swap_limit_pages:
                blocked = f'RAM safety floor or >{swap_limit_pages * 4 / 1024:.0f} MiB system swap-out'
                process.terminate()
                break
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
rows.append(sample())
text = (OUT / f'{label}.out').read_text()
rates = {kind: {'tokens': int(n), 'ms': float(ms), 'tps': float(tps)}
         for kind, n, ms, tps in re.findall(r'(decode|prefill)\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s', text)}
summary = {'capacity': capacity, 'prompt_tokens': len(ids), 'max_new': 128, 'exit_code': process.returncode,
           'blocked': blocked, 'rates': rates, 'min_available_gib': min(r['available_kib'] for r in rows) / 1048576,
           'swap_out_pages': rows[-1]['pswpout'] - rows[0]['pswpout'],
           'swap_in_pages': rows[-1]['pswpin'] - rows[0]['pswpin'],
           'peak_gtt_gib': max(r['gtt_bytes'] for r in rows) / 1073741824}
(OUT / f'{label}-samples.json').write_text(json.dumps(rows))
(OUT / f'{label}-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
assert process.returncode == 0 and not blocked, summary
assert rates['prefill']['tokens'] == len(ids) - 1, summary
assert rates['decode']['tokens'] == 128, summary
