"""Verify exact server-rendered context boundary and record paging during inference."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'bench/results/local-8060s-single-copy'
sys.path.insert(0, str(ROOT / 'tools'))
import strata_tokenizer as ST

folder = Path('/home/shi/Strata-data/packs/iq3_s/tokenizer')
vocab = json.loads((folder / 'vocab.json').read_text())
tokens = [None] * len(vocab)
for text, index in vocab.items():
    tokens[index] = text
tok = ST.Tokenizer(tokens, (folder / 'merges.txt').read_text().split('\n'), json.loads((folder / 'token_type.json').read_text()))
capacity = int(sys.argv[1])
label = sys.argv[2]
pid = int(subprocess.check_output(['pgrep', '-f', '^/home/shi/Strata/build-halo-714/strata --serve']).split()[0])


def post(endpoint, payload):
    req = urllib.request.Request('http://127.0.0.1:8080' + endpoint, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


corpus = '\n'.join(f'def transform_{i:06d}(value): return (value * {i % 97 + 1} + {i}) % 100003' for i in range(capacity // 15 + 100))
source_ids = tok.encode(corpus)
instruction = '\nWrite a detailed technical essay of at least 2000 words about the code, testing and software reliability. Do not conclude early.'
payload = {'model': 'strata', 'max_tokens': 128, 'temperature': 0, 'thinking': {'type': 'disabled'}}
# The public server reserves eight positions beyond requested output; do not weaken its admission check.
reserved = 8
target = capacity - payload['max_tokens'] - reserved
length = target - 100
prefix = str(uuid.uuid4()) + '\n'
for attempt in range(8):
    payload['messages'] = [{'role': 'user', 'content': prefix + tok.decode(source_ids[:length]) + instruction}]
    count = post('/v1/messages/count_tokens', payload)['input_tokens']
    if count == target:
        break
    length += target - count
else:
    if count == target + 1:
        payload['messages'][0]['content'] = payload['messages'][0]['content'].rstrip('.')
        count = post('/v1/messages/count_tokens', payload)['input_tokens']
    if count != target:
        raise RuntimeError(f'Could not size prompt: {count} != {target}')
(OUT / f'{label}-request.json').write_text(json.dumps(payload))


def sample():
    memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    vm = {line.split()[0]: int(line.split()[1]) for line in Path('/proc/vmstat').read_text().splitlines()}
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'time': time.time(), 'available_kib': memory['MemAvailable'], 'swap_used_kib': memory['SwapTotal'] - memory['SwapFree'],
            'pswpout': vm['pswpout'], 'pswpin': vm['pswpin'], 'major_faults': int(fields[9]),
            'psi': Path('/proc/pressure/memory').read_text(),
            'engine_status': Path(f'/proc/{pid}/status').read_text(),
            'engine_io': Path(f'/proc/{pid}/io').read_text()}


rows = [sample()]
stop = threading.Event()


def monitor():
    while not stop.wait(0.2):
        rows.append(sample())


thread = threading.Thread(target=monitor)
thread.start()
try:
    answer = post('/v1/messages', payload)
finally:
    stop.set()
    thread.join()
    rows.append(sample())
    (OUT / f'{label}-samples.json').write_text(json.dumps(rows))
(OUT / f'{label}-answer.json').write_text(json.dumps(answer, indent=2))
summary = {'capacity': capacity, 'counted_input': count, 'server_reserved_positions': reserved, 'usage': answer['usage'], 'stop_reason': answer['stop_reason'],
           'wall_s': rows[-1]['time'] - rows[0]['time'],
           'min_available_gib': min(r['available_kib'] for r in rows) / 1048576,
           'swap_out_pages': rows[-1]['pswpout'] - rows[0]['pswpout'], 'swap_in_pages': rows[-1]['pswpin'] - rows[0]['pswpin'],
           'engine_major_faults': rows[-1]['major_faults'] - rows[0]['major_faults'],
           'swap_used_before_kib': rows[0]['swap_used_kib'],
           'peak_swap_used_kib': max(r['swap_used_kib'] for r in rows),
           'swap_used_after_kib': rows[-1]['swap_used_kib']}
(OUT / f'{label}-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
assert answer['usage']['input_tokens'] == target, summary
assert answer['usage']['output_tokens'] == 128, summary
assert answer['stop_reason'] == 'max_tokens', summary
assert answer['usage'].get('cache_read_input_tokens', 0) == 0, summary
# Existing swapped pages can be read/written without consuming additional swap slots.
assert summary['peak_swap_used_kib'] <= summary['swap_used_before_kib'], summary
