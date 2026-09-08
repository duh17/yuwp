import asyncio, hashlib, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path.cwd()))
from benchmarks.lib.english_parity import load_commands, load_dev_cases, ordered_blocks, run_trial, summarize_trials
ROOT = pathlib.Path('/tmp/yuwp-english-parity')
FIXTURES = pathlib.Path('benchmarks/fixtures/english-parity')

async def main():
    assert (ROOT / 'dev-run.exit').exists()
    original = [json.loads(line) for line in (ROOT / 'dev-run.jsonl').read_text().splitlines()]
    assert original[-1]['type'] == 'summary' and original[-1]['elapsed_s'] > 0
    assert json.loads((ROOT / 'parakeet-repaired-readiness.json').read_text())['result'] == 'ready'
    receipt = json.loads((ROOT / 'parakeet-retry-receipt.json').read_text())
    for path, expected in receipt['hashes'].items():
        assert hashlib.file_digest(pathlib.Path(path).open('rb'), 'sha256').hexdigest() == expected, path
    commands = load_commands(ROOT / 'parakeet-retry-commands.json')
    cases = load_dev_cases(FIXTURES / 'dev.jsonl', FIXTURES / 'diagnostics.jsonl')
    assert len(cases) == 27
    warmup = next(pair for pair in cases if not pair[1]['diagnostic'] and pair[1]['kind'] == 'short')
    lookup = {case.id: (case, meta) for case, meta in cases}
    order = ordered_blocks(list(lookup))
    rows = []
    started = time.monotonic()
    deadline = started + 3598
    with (ROOT / 'parakeet-retry.jsonl').open('x') as output:
        def emit(row):
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
        emit(dict(receipt, type='receipt', planned_trials=27, label='NON-ACCEPTANCE', no_load_control=True))
        for index, (cid, _) in enumerate(order):
            case, meta = lookup[cid]
            if time.monotonic() >= deadline:
                row = dict(system='Parakeet', case_id=cid, metadata=meta, status='not-run', error='one-hour budget exhausted', metrics={})
            else:
                row = await run_trial('Parakeet', commands['Parakeet'], case, meta, warmup,
                                      deadline=deadline, stderr_path=ROOT / f'parakeet-retry.{index:03d}.stderr',
                                      load_status='diagnostic-under-load')
            row.update(type='trial', experiment='P1-resolver-repaired', label='NON-ACCEPTANCE', no_load_control=True)
            rows.append(row)
            emit(row)
        summary = summarize_trials(rows)
        summary.update(type='summary', elapsed_s=time.monotonic() - started, planned_trials=27,
                       complete=all(row['status'] == 'ok' for row in rows),
                       comparison='separate noninterleaved uncontrolled dev diagnostic; no latency parity inference')
        emit(summary)
    return int(any(row['status'] != 'ok' for row in rows))

sys.exit(asyncio.run(main()))
