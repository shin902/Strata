"""Check measured RAM savings at equal GPU-cache capacity, not source text or allocation promises."""
import json
from pathlib import Path

root = Path(__file__).parent / 'results/local-8060s-single-copy'
before = json.loads((root / 'final-arena-control-memory.json').read_text())
after = json.loads((root / 'final-single-copy-control-memory.json').read_text())
saved_gib = (after['meminfo_kib']['MemAvailable'] - before['meminfo_kib']['MemAvailable']) / 1048576
swap_growth_kib = (after['meminfo_kib']['SwapTotal'] - after['meminfo_kib']['SwapFree']) - (before['meminfo_kib']['SwapTotal'] - before['meminfo_kib']['SwapFree'])
assert saved_gib >= 8, f'Expected >=8 GiB net savings after separate prefill buffers, observed {saved_gib:.2f}'
assert after['engine_kib']['VmSwap'] == 0, 'Engine pages were swapped out'
print(f'Equal-cache net RAM saving: {saved_gib:.2f} GiB; swap growth: {swap_growth_kib / 1024:.2f} MiB')
