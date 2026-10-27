# SPDX-License-Identifier: Apache-2.0
"""Three HTTP servers started with --l1-config act as sources and targets of each other.

Checks candidate failover, fallback after a source dies, /server_info exposure and,
with --smoke-seconds, a random multi-client run with exact token comparison.
Deterministic FA4 keeps outputs comparable across batch sizes and instances.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time

import requests

SERVER_KWARGS = dict(tp_size=1, dtype='bfloat16', max_running_requests=4, max_total_tokens=4096,
                     context_length=4096, mem_fraction_static=0.15, page_size=128, attention_backend='fa4',
                     enable_deterministic_inference=True, chunked_prefill_size=128, cuda_graph_max_bs_decode=4,
                     cuda_graph_backend_prefill='full', cuda_graph_bs_prefill=[128],
                     disable_overlap_schedule=True, log_level='info')
LAUNCH = ('import json, sys\n'
          'from sglang.launch_server import run_server\n'
          'from sglang.srt.server_args import ServerArgs\n'
          'run_server(ServerArgs(**json.loads(sys.argv[1])))\n')
HTTP_PORT, SERVICE_PORT = 32100, 33100
PREFIX, NEW = 256, 32


def identity_for(model):
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    weights = {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
               for p in Path(model).glob('*.safetensors')}
    return dict(weights_revision=digest(weights),
                model_config_digest=digest(json.loads((Path(model) / 'config.json').read_text())),
                position_config_digest=digest({'position': 'default', 'origin': 0}), adapter_revision='none',
                context_digest=digest({'dtype': 'bfloat16', 'context_length': 4096}))


class Server:
    def __init__(self, index, model, identity, output):
        self.index = index
        self.url = f'http://127.0.0.1:{HTTP_PORT + index}'
        self.service = dict(host='127.0.0.1', port=SERVICE_PORT + index)
        config = dict(identity=identity, concurrent=True, service_host='127.0.0.1',
                      service_port=SERVICE_PORT + index)
        kwargs = dict(SERVER_KWARGS, model_path=model, port=HTTP_PORT + index, l1_config=json.dumps(config))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index))
        self.log = open(Path(output) / f'server-{index}.log', 'w')
        self.process = subprocess.Popen([sys.executable, '-c', LAUNCH, json.dumps(kwargs)], env=env,
                                        stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self, timeout=900):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert self.process.poll() is None, f'server {self.index} exited'
            try:
                if requests.get(self.url + '/health', timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f'server {self.index} not ready')

    def generate(self, tokens, *, sources=None, rid=None, new=NEW):
        body = dict(input_ids=tokens, rid=rid, return_logprob=True, logprob_start_len=-1,
                    sampling_params=dict(temperature=0, max_new_tokens=new, ignore_eos=True))
        if sources is not None:
            body['l1_restore'] = dict(mode='nonblocking', sources=sources, wait_timeout_ms=60000)
        response = requests.post(self.url + '/generate', json=body, timeout=300)
        response.raise_for_status()
        result = response.json()
        ids = [x[1] for x in result['meta_info']['output_token_logprobs']]
        restore = result['meta_info'].get('l1_restore')
        return ids, (restore[0] if restore else None), result['meta_info']

    def flush(self):
        return requests.post(self.url + '/flush_cache', timeout=60).status_code == 200

    def l1_state(self):
        return requests.get(self.url + '/server_info', timeout=60).json()['internal_states'][0]['l1_peer']

    def kill(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()


def source(server, prefix=PREFIX):
    return dict(server.service, prefix_tokens=prefix)


def assert_quiescent(servers):
    for s in servers:
        stats = s.l1_state()['stats']
        busy = {k: stats[k] for k in ('source_leases', 'source_pending', 'exports', 'imports', 'pinned_pages',
                                      'queued_requests', 'unadmitted_requests', 'ready_consumers')}
        assert not any(busy.values()) and not stats['target_active'], (s.index, busy)
        assert s.flush(), f'server {s.index} refused flush'
        assert s.l1_state()['stats']['available_tokens'] == 4096, s.index


def smoke(servers, families, references, seconds, output):
    """Random targets, shuffled source lists and seeding; every output must match its reference."""
    counts, failures, lock = {}, [], threading.Lock()
    stop = time.monotonic() + seconds

    def client(seed):
        rng = random.Random(seed)
        n = 0
        while time.monotonic() < stop and not failures:
            n += 1
            f = rng.randrange(len(families))
            if rng.random() < 0.2:  # seed a family prefix on a random instance
                rng.choice(servers).generate(families[f][:PREFIX + 128], new=1)
                continue
            target = rng.choice(servers)
            others = [s for s in servers if s is not target]
            rng.shuffle(others)
            try:
                ids, restore, _ = target.generate(families[f], sources=[source(s) for s in others],
                                                  rid=f'smoke-{seed}-{n}')
                # Split FINISHED into real cross-GPU copies and target-local hits.
                kind = 'remote' if restore['transferred_bytes'] else 'local'
                key = (restore['state'], restore['reason'] or kind)
                if ids != references[f]:
                    raise AssertionError(f'token mismatch family {f} target {target.index} {restore}')
            except BaseException as exc:
                with lock:
                    failures.append(repr(exc))
                return
            with lock:
                counts[key] = counts.get(key, 0) + 1

    threads = [threading.Thread(target=client, args=(seed,)) for seed in range(6)]
    for t in threads: t.start()
    for t in threads: t.join()
    summary = dict(seconds=seconds, counts={f'{k[0]}:{k[1]}': v for k, v in counts.items()}, failures=failures)
    (Path(output) / 'smoke.json').write_text(json.dumps(summary, indent=2))
    assert not failures, failures
    assert counts.get(('FINISHED', 'remote'), 0) > 0, counts
    print('L1_HTTP_SMOKE_PASS', json.dumps(summary['counts']), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke-seconds', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = tokenizer.encode('The quick brown fox jumps over the lazy dog. ' * 80)
    # Families share nothing; each has a 2-page prefix and a 7-token suffix. Sixteen
    # families overflow the 4096-token pools, so caches churn and remote copies recur.
    families = [[(t + 997 * f) % 30000 for t in base[:PREFIX + 7]] for f in range(16)]
    identity = identity_for(args.model)
    servers = [Server(i, args.model, identity, out) for i in range(3)]
    try:
        for s in servers: s.wait_ready()
        a, b, c = servers
        states = [s.l1_state() for s in servers]
        assert [x['service']['port'] for x in states] == [SERVICE_PORT + i for i in range(3)], states
        assert len({x['contract_digest'] for x in states}) == 1, states
        print('L1_HTTP_SERVER_INFO_PASS', flush=True)

        references = []
        for f in families:
            assert c.flush()
            references.append(c.generate(f)[0])
        for s in servers: assert s.flush()

        # Candidate failover: C misses, A holds the prefix.
        a.generate(families[0][:PREFIX + 128], new=1)
        ids, restore, meta = b.generate(families[0], sources=[source(c), source(a)], rid='failover')
        assert ids == references[0], 'failover output differs'
        assert restore['state'] == 'FINISHED' and restore['transferred_bytes'] > 0, restore
        assert [x['port'] for x in restore['attempts']] == [c.service['port']], restore
        assert meta['cached_tokens'] >= PREFIX, meta
        print('L1_HTTP_FAILOVER_PASS', json.dumps(restore['attempts']), flush=True)
        assert_quiescent(servers)

        if args.smoke_seconds:
            smoke(servers, families, references, args.smoke_seconds, out)
            assert_quiescent(servers)

        # A dead source is unreachable before any request is sent: recompute instead of failing.
        a.generate(families[1][:PREFIX + 128], new=1)
        a.kill()
        ids, restore, _ = b.generate(families[1], sources=[source(a)], rid='dead-source')
        assert ids == references[1], 'fallback output differs'
        assert restore['state'] == 'FALLBACK' and restore['reason'] == 'SOURCE_REJECTED', restore
        assert 'unreachable' in restore['attempts'][0]['error'], restore
        assert_quiescent([b, c])
        print('L1_HTTP_DEAD_SOURCE_PASS', json.dumps(restore['attempts']), flush=True)
        print('L1_HTTP_PASS', flush=True)
    finally:
        for s in servers:
            s.kill()


if __name__ == '__main__':
    main()
