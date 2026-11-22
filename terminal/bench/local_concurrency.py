#!/usr/bin/env python3
"""Measure solo PP/TG and two-client PP/PP, TG/TG, PP/TG through the real HTTP API."""
import concurrent.futures
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import strata_tokenizer as ST

BASE = 'http://127.0.0.1:8080'
OUT = ROOT / 'bench/results/local-8060s-concurrency'
OUT.mkdir(parents=True, exist_ok=True)
label = sys.argv[1]
folder = Path('/home/shi/Strata-data/packs/iq3_s/tokenizer')
vocab = json.loads((folder / 'vocab.json').read_text())
tokens = [None] * len(vocab)
for token, index in vocab.items():
    tokens[index] = token
tokenizer = ST.Tokenizer(tokens, (folder / 'merges.txt').read_text().split('\n'),
                         json.loads((folder / 'token_type.json').read_text()))
corpus = '\n'.join(f'def transform_{i:05d}(value): return (value * {i % 97 + 1} + {i}) % 100003' for i in range(1500))
long_text = tokenizer.decode(tokenizer.encode(corpus)[:int(sys.argv[2]) if len(sys.argv) > 2 else 4096])
instruction = '\nWrite a detailed technical essay of at least 2000 words about software reliability, testing, concurrency and memory management. Do not conclude early.'


def fetch_json(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return json.load(response)


def request(kind, first_token=None, stream=True):
    prompt = str(uuid.uuid4()) + '\n' + (long_text if kind == 'pp' else '') + instruction
    payload = {'model': 'strata', 'messages': [{'role': 'user', 'content': prompt}],
               'max_tokens': 16 if kind == 'pp' and 'long-generation' not in sys.argv else 512, 'temperature': 0,
               'reasoning_effort': 'none', 'stream': stream,
               'stream_options': {'include_usage': True}, 'cache_prompt': False}
    started = time.perf_counter()
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    row = {'kind': kind, 'start': started, 'first': None, 'timings': None, 'usage': None, 'finish': None}
    with urllib.request.urlopen(req, timeout=600) as response:
        if not stream:
            answer = json.load(response)
            row.update(timings=answer['timings'], usage=answer['usage'], finish=answer['choices'][0]['finish_reason'])
        else:
            for raw in response:
                if not raw.startswith(b'data: '):
                    continue
                text = raw[6:].strip()
                if text == b'[DONE]':
                    break
                chunk = json.loads(text)
                if chunk.get('usage'):
                    row['usage'] = chunk['usage']
                if chunk.get('timings'):
                    row['timings'] = chunk['timings']
                for choice in chunk.get('choices', []):
                    if choice.get('finish_reason'):
                        row['finish'] = choice['finish_reason']
                    delta = choice.get('delta') or {}
                    if (delta.get('content') or delta.get('reasoning_content')) and row['first'] is None:
                        row['first'] = time.perf_counter()
                        if first_token is not None:
                            first_token.set()
    row['end'] = time.perf_counter()
    assert row['usage'] and row['usage']['completion_tokens'] == payload['max_tokens'], row
    if row['timings']:
        assert row['timings'].get('cache_n', 0) < 32, row
    return row


rows = []
fetch_json('/health')
request('tg', stream=False)
for repeat in range(3):
    for kind in ('pp', 'tg'):
        clients = [request(kind, stream=False)]
        row = {'case': 'solo_' + kind, 'repeat': repeat, 'clients': clients,
               'wall': clients[0]['end'] - clients[0]['start']}
        rows.append(row)
        (OUT / (label + '.json')).write_text(json.dumps({'label': label, 'status': fetch_json('/v1/status'), 'rows': rows}, indent=2))
        print(json.dumps(row), flush=True)
    cases = [] if 'solo-only' in sys.argv else [('pp2', ('pp', 'pp')), ('tg2', ('tg', 'tg')), ('mixed', ('tg', 'pp'))]
    for case, kinds in cases:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            if case == 'mixed':
                first_token = threading.Event()
                first = pool.submit(request, kinds[0], first_token)
                if not first_token.wait(120):
                    raise RuntimeError('decode did not start')
                time.sleep(0.5)
                second = pool.submit(request, kinds[1])
            else:
                barrier = threading.Barrier(2)
                def send(kind):
                    barrier.wait()
                    return request(kind)
                first, second = [pool.submit(send, kind) for kind in kinds]
            clients = [first.result(), second.result()]
        row = {'case': case, 'repeat': repeat, 'clients': clients, 'wall': time.perf_counter() - started}
        rows.append(row)
        (OUT / (label + '.json')).write_text(json.dumps({'label': label, 'status': fetch_json('/v1/status'), 'rows': rows}, indent=2))
        print(json.dumps(row), flush=True)
summary = {}
for case in ('solo_pp', 'solo_tg', 'pp2', 'tg2', 'mixed'):
    selected = [r for r in rows if r['case'] == case]
    if not selected:
        continue
    summary[case] = {'wall_s': statistics.median(r['wall'] for r in selected),
                     'input_tps_wall': statistics.median(sum(c['usage']['prompt_tokens'] for c in r['clients']) / r['wall'] for r in selected),
                     'output_tps_wall': statistics.median(sum(c['usage']['completion_tokens'] for c in r['clients']) / r['wall'] for r in selected)}
    if case.startswith('solo'):
        key = 'prompt_per_second' if case == 'solo_pp' else 'predicted_per_second'
        summary[case]['engine_tps'] = statistics.median(r['clients'][0]['timings'][key] for r in selected)
(OUT / (label + '-summary.json')).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
