"""Measure client-observed visible output TPS in the first five seconds; not SSE-event TPS."""
import bisect
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import strata_tokenizer as ST
folder = Path('/home/shi/Strata-data/packs/iq3_s/tokenizer')
vocab = json.loads((folder / 'vocab.json').read_text())
tokens = [None] * len(vocab)
for text, index in vocab.items():
    tokens[index] = text
tok = ST.Tokenizer(tokens, (folder / 'merges.txt').read_text().split('\n'), json.loads((folder / 'token_type.json').read_text()))
rows = []
cases = [('essay', 'Write a detailed technical essay of at least 2000 words about software reliability, testing and concurrency. Use ASCII text only. Do not conclude early.')]
if 'code' in sys.argv:
    cases += [
        ('python', 'Generate a complete Python module with at least 20 functions for parsing, validating, filtering and aggregating JSON records. Include type hints, docstrings and unittest tests. Output only code. Use ASCII only.'),
        ('typescript', 'Generate a complete TypeScript module with at least 20 typed functions for parsing, validating, filtering and aggregating JSON records. Include interfaces, error handling and example calls. Output only code. Use ASCII only.'),
        ('sql', 'Generate SQL migrations for 30 related business tables. For every table include primary keys, foreign keys, timestamps, indexes and insert examples. Output only SQL code. Use ASCII only.'),
    ]
for repeat, (case, prompt) in [(repeat, case) for repeat in range(3) for case in cases]:
    payload = {'model': 'strata', 'messages': [{'role': 'user', 'content': str(uuid.uuid4()) + '\n' + prompt}],
               'max_tokens': 512, 'temperature': 0, 'reasoning_effort': 'none', 'stream': True,
               'stream_options': {'include_usage': True}}
    request = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    arrivals = []
    content = ''
    usage = timings = None
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            received = time.perf_counter()
            if not raw.startswith(b'data: ') or raw[6:].strip() == b'[DONE]':
                continue
            chunk = json.loads(raw[6:])
            usage = chunk.get('usage') or usage
            timings = chunk.get('timings') or timings
            for choice in chunk.get('choices', []):
                text = (choice.get('delta') or {}).get('content', '')
                if text:
                    content += text
                    arrivals.append((len(content.encode()), received))
    encoded = tok.encode(content)
    assert b''.join(tok.token_bytes(i) for i in encoded) == content.encode()
    byte_ends = [b for b, _ in arrivals]
    token_times = []
    offset = 0
    for token in encoded:
        offset += len(tok.token_bytes(token))
        token_times.append(arrivals[bisect.bisect_left(byte_ends, offset)][1])
    first = token_times[0]
    assert token_times[-1] - first >= 5, 'Response too short for a five-second observation'
    windows = []
    for step in range(41):
        start = first + step / 10
        count = bisect.bisect_left(token_times, start + 1) - bisect.bisect_left(token_times, start)
        windows.append({'start_s': step / 10, 'visible_tokens': count, 'tps': count})
    peak = max(windows, key=lambda x: x['tps'])
    first_five = bisect.bisect_left(token_times, first + 5)
    row = {'case': case, 'repeat': repeat, 'ttft_s': first - started, 'peak_1s': peak,
           'first_5s_visible_tokens': first_five, 'first_5s_tps': first_five / 5,
           'visible_output_tokens': len(encoded), 'usage': usage, 'timings': timings,
           'windows': windows, 'token_arrival_s': [t - first for t in token_times]}
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k not in ('windows', 'token_arrival_s')}, ensure_ascii=False), flush=True)
OUT = ROOT / 'bench/results/local-8060s-single-copy' / ('code-generation-tps.json' if 'code' in sys.argv else 'initial-peak-tps.json')
with OUT.open('x') as output:
    json.dump({'definition': 'Visible text retokenized with model tokenizer; arrival times from SSE bytes; 1s windows stepped 0.1s within first 5s after first visible token. Network/MTP batching can affect peaks.', 'rows': rows}, output, indent=2)
print('saved:', OUT)
