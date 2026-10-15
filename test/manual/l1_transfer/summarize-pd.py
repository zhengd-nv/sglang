"""Validate model token equality and PD data-plane evidence from test-pd.py."""
import argparse
import json
from pathlib import Path
import re


def summarize(root):
    results = json.loads((root / 'results.json').read_text())
    assert len(results) == 6, 'incomplete request matrix'
    prefill = (root / 'prefill.log').read_text()
    decode = (root / 'decode.log').read_text()
    writes = re.findall(r'NVLINK_PD_WRITE room=(\d+) pages=(\d+) ranges=(\d+) bytes=(\d+) last=(True|False)', prefill)
    completed = re.findall(r'NVLINK_PD_COMPLETE room=(\d+) exports=0 imports=0', decode)
    report = []
    for result in results:
        case = result['case']
        room = str(10000 + case)
        source = json.loads((root / f'case-{case}-reference.json').read_text())
        target = json.loads((root / f'case-{case}-decode.json').read_text())
        ids = lambda response: [t[1] for t in response['meta_info']['output_token_logprobs']]
        assert ids(source) == ids(target) and len(ids(source)) == 24
        chunks = [w for w in writes if w[0] == room]
        assert chunks and sum(c[4] == 'True' for c in chunks) == 1
        assert completed.count(room) == 1
        pages = sum(int(c[1]) for c in chunks)
        assert pages == source['meta_info']['prompt_tokens'], (room, pages)
        transferred = sum(int(c[3]) for c in chunks)
        assert transferred > 0
        report.append(dict(case=case, room=room, prompt_tokens=pages, generated_tokens=24,
                           chunks=len(chunks), ranges=sum(int(c[2]) for c in chunks),
                           transferred_bytes=transferred, token_ids_equal=True))
    assert report[2]['chunks'] > 1, 'long prompt did not exercise chunked transfer'
    summary = dict(passed=True, cases=report, total_bytes=sum(r['transferred_bytes'] for r in report),
                   scope='same-node TP1 BF16 NHD page_size=1, serial Qwen2-1.5B requests')
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    summarize(parser.parse_args().output)
