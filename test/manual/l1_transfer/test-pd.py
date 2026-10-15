"""Launch TP1 P/D and a reference server; compare greedy output token IDs."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    processes = []
    common = [sys.executable, '-m', 'sglang.launch_server', '--model-path', args.model,
              '--host', '0.0.0.0', '--tp-size', '1', '--dtype', 'bfloat16',
              '--max-running-requests', '1', '--max-total-tokens', '4096',
              '--context-length', '4096', '--mem-fraction-static', '0.15',
              '--disable-radix-cache', '--page-size', '1', '--attention-backend', 'flashinfer',
              '--chunked-prefill-size', '128', '--cuda-graph-max-bs-decode', '1',
              '--cuda-graph-backend-prefill', 'full', '--cuda-graph-bs-prefill', '128',
              '--skip-server-warmup', '--disable-overlap-schedule']
    try:
        for gpu, role, port in [(0, 'prefill', 31000), (1, 'decode', 31001), (2, 'reference', 31002)]:
            cmd = common + ['--port', str(port)]
            if role != 'reference':
                cmd += ['--disaggregation-mode', role, '--disaggregation-transfer-backend', 'nvlink',
                        '--disaggregation-bootstrap-port', '31003', '--disaggregation-decode-allocation-policy', 'early']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), SGLANG_ENABLE_POST_CAPTURE_KV_SIZING='1')
            (out / f'{role}-command.json').write_text(json.dumps(cmd))
            log = (out / f'{role}.log').open('w')
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((p, log))
        deadline = time.monotonic() + 900
        for port in (31000, 31001, 31002):
            while True:
                if any(p.poll() is not None for p, _ in processes):
                    raise RuntimeError('server exited during startup; inspect role logs')
                try:
                    if requests.get(f'http://127.0.0.1:{port}/health', timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server readiness')
                time.sleep(2)
        print('ALL_SERVERS_READY', flush=True)
        def generate(port, payload):
            response = requests.post(f'http://127.0.0.1:{port}/generate', json=payload, timeout=180)
            response.raise_for_status()
            return response.json()
        prompts = ['The capital of France is', 'Write the first five prime numbers:',
                   ' '.join(['The quick brown fox jumps over the lazy dog.'] * 40) + ' Summarize this in one sentence:',
                   'The capital of France is', 'In Python, a list is', 'The opposite of cold is']
        results = []
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for i, prompt in enumerate(prompts):
                payload = {'text': prompt, 'sampling_params': {'temperature': 0, 'max_new_tokens': 24, 'ignore_eos': True},
                           'return_logprob': True, 'logprob_start_len': -1}
                reference = generate(31002, payload)
                pd = dict(payload, bootstrap_host='127.0.0.1', bootstrap_port=31003, bootstrap_room=10000+i)
                p_payload = dict(pd, sampling_params=dict(payload['sampling_params'], max_new_tokens=1))
                pf = pool.submit(generate, 31000, p_payload)
                decoded = generate(31001, pd)
                prefilled = pf.result()
                for role, value in [('reference',reference),('decode',decoded),('prefill',prefilled)]:
                    (out / f'case-{i}-{role}.json').write_text(json.dumps(value, indent=2))
                ids = lambda value: [item[1] for item in value['meta_info']['output_token_logprobs']]
                result = {'case': i, 'prompt_tokens': reference['meta_info']['prompt_tokens'],
                          'reference_ids': ids(reference), 'decode_ids': ids(decoded),
                          'match': ids(reference) == ids(decoded)}
                results.append(result)
                (out/'results.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
                assert result['match'], f'token mismatch case {i}'
        print('PD_CORRECTNESS_PASS cases=6', flush=True)
    finally:
        for p, log in processes:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p, log in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            log.close()


if __name__ == '__main__':
    main()
