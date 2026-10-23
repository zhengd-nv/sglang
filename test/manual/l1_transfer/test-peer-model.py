"""Real Engine prefix restore, explicit scheduler control, no PD or host KV stage."""
import argparse
import asyncio
import hashlib
import json
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
from pathlib import Path
import traceback


def actor(pipe, gpu, model, output, concurrent=False, deterministic=False, mapping_peer_device=-1):
    os.setsid()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if gpu == 1 and mapping_peer_device >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu},{mapping_peer_device}'
        os.environ['SGLANG_L1_MAPPING_PEER_DEVICE'] = '1'  # ordinal within this process
    else:
        os.environ.pop('SGLANG_L1_MAPPING_PEER_DEVICE', None)
    log = open(Path(output) / f'engine-{gpu}.log', 'w', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    engine = None
    try:
        from sglang import Engine
        engine = Engine(model_path=model, tp_size=1, dtype='bfloat16',
                        max_running_requests=4 if concurrent and gpu == 1 else 1, max_total_tokens=4096, context_length=4096,
                        mem_fraction_static=0.15, page_size=128 if deterministic else 16, attention_backend='fa4' if deterministic else 'flashinfer',
                        enable_deterministic_inference=deterministic,
                        chunked_prefill_size=128, cuda_graph_max_bs_decode=4 if concurrent and gpu == 1 else 1,
                        cuda_graph_backend_prefill='full', cuda_graph_bs_prefill=[128],
                        disable_overlap_schedule=True, port=32000+gpu, log_level='info')
        # Explicit lock permits socket migration to the observation thread;
        # no two threads ever access Engine's control socket concurrently.
        rpc_lock = threading.Lock()
        original_peer = engine.l1_peer
        def locked_peer(*args, **kwargs):
            with rpc_lock:
                return original_peer(*args, **kwargs)
        engine.l1_peer = locked_peer
        pipe.send({'ready': True})
        while True:
            command = pipe.recv()
            op = command.pop('op')
            if op == 'stop':
                break
            try:
                if op == 'peer':
                    answer = engine.l1_peer(command['action'], **command.get('payload', {}))
                elif op == 'generate':
                    observations = []
                    done = threading.Event()
                    rid = command.get('rid')
                    cancelled = threading.Event()
                    def observe():
                        while not done.wait(0.01):
                            before = time.monotonic()
                            try:
                                stats = engine.l1_peer('stats')
                                observations.append(dict(stats, latency=time.monotonic()-before))
                                if (command.get('cancel_state') == stats.get('wait_state')
                                        and not cancelled.is_set()):
                                    cancelled.set()
                                    engine.loop.call_soon_threadsafe(engine.tokenizer_manager.abort_request, rid)
                            except Exception as exc:
                                observations.append({'error': str(exc)})
                                break
                    monitor = None
                    if command.get('l1_restore', {}).get('mode') == 'nonblocking':
                        monitor = threading.Thread(target=observe, daemon=True)
                        monitor.start()
                    try:
                        answer = engine.generate(input_ids=command['tokens'], rid=rid,
                        sampling_params={'temperature':0, 'max_new_tokens':command.get('new',24), 'ignore_eos':True},
                        return_logprob=True, logprob_start_len=-1,
                            l1_restore=command.get('l1_restore'))
                    except Exception:
                        if not command.get('terminal_reason'):
                            raise
                        terminal = engine.l1_peer('request_result', rid=rid)
                        if terminal['reason'] != command['terminal_reason']:
                            raise
                        answer = {'meta_info': {'l1_restore': terminal}}
                    finally:
                        done.set()
                        if monitor is not None:
                            monitor.join(5)
                            assert not monitor.is_alive(), 'control query blocked scheduler'
                    if monitor is not None:
                        answer['control_observations'] = observations
                    if command.get('terminal_reason'):
                        assert answer['meta_info']['l1_restore']['reason'] == command['terminal_reason'], answer
                        if command.get('cancel_state'):
                            assert cancelled.is_set(), 'requested cancellation phase was not observed'
                elif op == 'concurrent_group':
                    async def group():
                        observations, answers = [], []
                        started = time.monotonic()
                        async def generate_one(i, item):
                            if item.get('delay'):
                                await asyncio.sleep(item['delay'])
                            try:
                                result = await engine.async_generate(
                                    input_ids=item['tokens'], rid=item['rid'],
                                    sampling_params={'temperature': 0, 'max_new_tokens': item['new'], 'ignore_eos': True},
                                    return_logprob=True, logprob_start_len=-1,
                                    l1_restore=item.get('l1_restore'))
                            except Exception:
                                if not item.get('terminal_reason'):
                                    raise
                                result = {'meta_info': {'l1_restore': engine.l1_peer('request_result', rid=item['rid'])}}
                            if item.get('terminal_reason'):
                                assert result['meta_info']['l1_restore']['reason'] == item['terminal_reason'], result
                            answers.append(dict(index=i, elapsed=time.monotonic()-started, result=result))
                        tasks = [asyncio.create_task(generate_one(i, item)) for i, item in enumerate(command['items'])]
                        while not all(t.done() for t in tasks):
                            await asyncio.sleep(0.01)
                            stats = engine.l1_peer('stats')
                            observations.append(dict(stats, elapsed=time.monotonic()-started))
                            for item in command['items']:
                                if item.get('cancel_state') == stats['wait_state'] and not item.get('cancelled') and item.get('cancel_state'):
                                    item['cancelled'] = True
                                    engine.tokenizer_manager.abort_request(item['rid'])
                        await asyncio.gather(*tasks)
                        return dict(answers=sorted(answers, key=lambda x: x['index']), observations=observations,
                                    elapsed=time.monotonic()-started,
                                    cancelled=[item["rid"] for item in command["items"] if item.get("cancelled")])
                    answer = engine.loop.run_until_complete(group())
                elif op == 'flush':
                    answer = engine.flush_cache().success
                else:
                    raise ValueError(op)
                pipe.send({'answer':answer})
            except Exception:
                pipe.send({'error':traceback.format_exc()})
    except BaseException:
        pipe.send({'error':traceback.format_exc()})
    finally:
        if engine is not None:
            engine.shutdown()


def delayed_peer(address):
    """Test-only proxy: delay each ACK so both wait phases are observable."""
    from sglang.srt.mem_cache.l1_transfer.peer.control import receive, send
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0)); listener.listen(1)
    port = listener.getsockname()[1]
    errors = []
    def run():
        try:
            with listener, socket.create_connection((address['host'], address['port']), timeout=30) as source:
                target, _ = listener.accept()
                with target:
                    target.settimeout(30)
                    for _ in range(2):
                        send(source, receive(target))
                        response = receive(source)
                        time.sleep(0.25)
                        send(target, response)
        except BaseException:
            errors.append(traceback.format_exc())
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return dict(host='127.0.0.1', port=port), worker, errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--request-restore', action='store_true',
                        help='Use generate(l1_restore=...) with automatic consumer release')
    parser.add_argument('--nonblocking', action='store_true')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic FA4 for exact concurrent token comparisons')
    parser.add_argument('--concurrent', action='store_true', help='Enable target private handoff and concurrent ordinary requests')
    parser.add_argument('--lifecycle', action='store_true', help='Also validate safe failures and cancellation')
    parser.add_argument('--mapping-peer-device', type=int, default=-1,
                        help='Physical local peer exposed to the target process; -1 disables guard')
    args=parser.parse_args()
    if args.mapping_peer_device < -1 or args.mapping_peer_device == 1:
        parser.error('mapping peer must be -1 or a distinct nonnegative local device')
    if args.deterministic and not args.concurrent:
        parser.error("--deterministic requires --concurrent")
    if args.concurrent: args.nonblocking = True
    if args.nonblocking: args.request_restore = True
    if args.lifecycle and not args.nonblocking:
        parser.error('--lifecycle requires --nonblocking')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    token_count = 320 if args.deterministic else 160
    prefix_tokens = 256 if args.deterministic else 112
    request_tokens = prefix_tokens + 7
    partial_tokens = 128 if args.deterministic else 32
    tokens=tokenizer.encode('The quick brown fox jumps over the lazy dog. '*40)[:token_count]
    assert len(tokens)==token_count
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    weights={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
             for p in Path(args.model).glob('*.safetensors')}
    assert weights
    identity=dict(weights_revision=digest(weights),model_config_digest=digest(json.loads((Path(args.model)/'config.json').read_text())),
                  position_config_digest=digest({'position':'default','origin':0}),adapter_revision='none',
                  context_digest=digest({'dtype':'bfloat16','context_length':4096}))
    (out/'identity.json').write_text(json.dumps(identity,indent=2))
    ctx=mp.get_context('spawn');workers=[];pipes=[]
    def recv(i):
        assert pipes[i].poll(900), f'actor {i} timed out'
        answer=pipes[i].recv()
        if 'error' in answer:raise RuntimeError(answer['error'])
        return answer.get('answer',answer)
    def call(i,op,**kw):
        pipes[i].send(dict(op=op,**kw));return recv(i)
    def peer(i,action,**payload):return call(i,'peer',action=action,payload=payload)
    try:
        for gpu in range(4 if args.deterministic else 3):
            a,b=ctx.Pipe();p=ctx.Process(target=actor,args=(b,gpu,args.model,str(out),args.concurrent,args.deterministic and gpu<3,args.mapping_peer_device))
            p.start();b.close();pipes.append(a);workers.append(p)
        for i in range(len(workers)):assert recv(i)['ready']
        if args.deterministic:
            # GPU3 has no L1 initialization: isolate ordinary FlashInfer cache
            # numerics without any transfer, target reservation or shared pages.
            ordinary = list(tokens[:119]); ordinary[0] = 314
            assert call(3, 'flush')
            cold = call(3, 'generate', tokens=ordinary+[401], new=256)
            assert call(3, 'flush')
            call(3, 'generate', tokens=ordinary+[400], new=1)
            warm = call(3, 'generate', tokens=ordinary+[401], new=256)
            cold_ids = [x[1] for x in cold['meta_info']['output_token_logprobs']]
            warm_ids = [x[1] for x in warm['meta_info']['output_token_logprobs']]
            mismatch = next((i for i, pair in enumerate(zip(cold_ids, warm_ids)) if pair[0] != pair[1]), None)
            (out/'flashinfer-no-l1-control.json').write_text(json.dumps(dict(cold=cold, warm=warm, first_mismatch=mismatch), indent=2))
            print('FLASHINFER_NO_L1_CONTROL first_mismatch=', mismatch, flush=True)
        for i in (0,1):peer(i,'init' ,identity=identity,concurrent=args.concurrent and i==1)
        mapping_config = [peer(i, 'stats').get('mapping_peer_device') for i in (0, 1)]
        assert mapping_config == [None, 1 if args.mapping_peer_device >= 0 else None]
        (out/'mapping-config.json').write_text(json.dumps(mapping_config))
        call(0,'generate',tokens=tokens,new=1)
        service_address = peer(0,'start_service') if args.nonblocking else None
        rows=[]
        for case,local in [('cold',0),('partial',partial_tokens),('all_local',prefix_tokens),('repeat_cold',0)]:
            assert call(1,'flush')
            assert call(2,'flush')
            if local:
                call(1,'generate',tokens=tokens[:local]+[777],new=1)
            request=tokens[:request_tokens]
            reference=call(2,'generate',tokens=request)
            address = {'host': '127.0.0.1', 'port': 1}  # unused for all-local
            proxy = None
            if local<prefix_tokens:
                if args.nonblocking:
                    address, proxy, errors = delayed_peer(service_address)
                else:
                    address=peer(0,'listen')
                    pipes[0].send(dict(op='peer',action='serve'))
            if args.request_restore:
                context = dict(source=address, prefix_tokens=prefix_tokens,
                               **({'mode':'nonblocking'} if args.nonblocking else {}))
                source_observations = []
                if args.nonblocking:
                    pipes[1].send(dict(op='generate', tokens=request, l1_restore=context))
                    deadline = time.monotonic() + 120
                    while not pipes[1].poll(0.01):
                        assert time.monotonic() < deadline, 'request restore timed out'
                        before = time.monotonic()
                        stats = peer(0, 'stats')
                        source_observations.append(dict(stats, latency=time.monotonic()-before))
                    actual = recv(1)
                    actual['source_control_observations'] = source_observations
                else:
                    actual=call(1,'generate',tokens=request,l1_restore=context)
                restored=actual['meta_info']['l1_restore']
                assert restored['consumer_released'], restored
            else:
                restored=peer(1,'restore',tokens=tokens[:prefix_tokens],**address)
                # Target consumer pin must survive until ordinary generate completes.
                assert not call(1,'flush'), 'flush accepted a live target reservation'
                actual=call(1,'generate',tokens=request)
            if local<prefix_tokens:
                source=peer(0,'stats') if args.nonblocking else recv(0)
                if proxy is not None:
                    proxy.join(5)
                    assert not proxy.is_alive() and not errors, errors
                    observations = actual['control_observations']
                    states = {x.get('wait_state') for x in observations}
                    assert {'WAIT_ACQUIRE','WAIT_RELEASE'} <= states, observations
                    assert not any('error' in x for x in observations), observations
                    assert any(x['source_leases'] == 1 for x in source_observations), source_observations
                assert source['source_leases']==source['exports']==0
            assert restored['local_tokens']==local,restored
            assert restored['restored_tokens']==prefix_tokens
            assert restored['pinned_pages']==prefix_tokens // (128 if args.deterministic else 16)
            assert restored['imports']==restored['batches']==0
            ids=lambda x:[t[1] for t in x['meta_info']['output_token_logprobs']]
            assert ids(reference)==ids(actual),(case,ids(reference),ids(actual))
            assert actual['meta_info']['cached_tokens']==prefix_tokens,actual['meta_info']
            released=peer(1,'stats') if args.request_restore else peer(1,'finish')
            assert released['pinned_pages']==0 and not released['target_active']
            assert call(1,'flush')
            assert peer(1,'stats')['available_tokens']==4096
            row=dict(restored,case=case,mode='nonblocking' if args.nonblocking else ('request' if args.request_restore else 'explicit'),
                     cached_tokens=actual['meta_info']['cached_tokens'],
                     generated_tokens=len(ids(actual)),tokens_equal=True,consumer_released=True,
                     max_output_logprob_delta=max(abs(a[0]-b[0]) for a,b in zip(
                         reference['meta_info']['output_token_logprobs'],
                         actual['meta_info']['output_token_logprobs'])))
            rows.append(row)
            (out/f'{case}-reference.json').write_text(json.dumps(reference,indent=2))
            (out/f'{case}-restored.json').write_text(json.dumps(actual,indent=2))
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
        if args.concurrent:
            concurrency = []
            for scenario in ('distinct', 'competing_prefix', 'capacity_wait', 'chunked', 'cancel_acquire', 'cancel_running'):
                assert call(1, 'flush')
                address, proxy, errors = delayed_peer(service_address)
                ordinary = list(tokens[:request_tokens])
                ordinary[0] = 314
                shared = tokens[:request_tokens] if scenario != 'chunked' else tokens[:prefix_tokens] + tokens * 4
                items = [dict(rid=f'{scenario}-ordinary-{i}', tokens=ordinary+[400+i],
                              new=(1400 if args.deterministic else 1800) if scenario == 'capacity_wait' else 256) for i in range(2 if scenario == 'capacity_wait' else 3)]
                if scenario == 'capacity_wait':
                    for i, item in enumerate(items): item['tokens'][0] = 4000+i
                # A competing local request publishes the remote prefix while
                # acquire is pending. Private ownership must survive dedup.
                if scenario == 'competing_prefix':
                    items[0]['tokens'] = tokens[:request_tokens]
                items.append(dict(rid=f'{scenario}-shared', tokens=shared, new=512 if scenario == 'capacity_wait' else 24,
                                  delay=0.03 if scenario == 'capacity_wait' else 0,
                                  l1_restore=dict(source=address, prefix_tokens=prefix_tokens, mode='nonblocking')))
                if scenario.startswith('cancel_'):
                    items[-1].update(terminal_reason='CANCELLED',
                                     cancel_state='WAIT_ACQUIRE' if scenario == 'cancel_acquire' else 'READY', new=512)
                # Submit restore first so competing_prefix still uses remote READ.
                if scenario != 'capacity_wait':
                    items = [items[-1]] + items[:-1]
                    for item in items[1:]: item['delay'] = 0.03
                actual = call(1, 'concurrent_group', items=items)
                proxy.join(5)
                assert not proxy.is_alive() and not errors, errors
                (out/f'concurrent-{scenario}-actual.json').write_text(json.dumps(actual, indent=2))
                for item, answer in zip(items, actual['answers']):
                    if item.get('terminal_reason'):
                        assert item['rid'] in actual['cancelled']
                        continue
                    assert call(2, 'flush')
                    reference = call(2, 'generate', tokens=item['tokens'], new=item['new'])
                    (out/f"{item['rid']}-reference.json").write_text(json.dumps(reference, indent=2))
                    get_ids = lambda result: [x[1] for x in result['meta_info']['output_token_logprobs']]
                    assert get_ids(answer['result']) == get_ids(reference), (scenario, item['rid'])
                shared_result = next(x['result'] for x in actual['answers'] if 'l1_restore' in x['result']['meta_info'])
                restored = shared_result['meta_info']['l1_restore']
                assert restored['consumer_released']
                if scenario.startswith('cancel_'):
                    assert restored['reason'] == 'CANCELLED'
                elif scenario == 'capacity_wait':
                    assert restored['wait_polls']['WAIT_CAPACITY'] > 0
                else:
                    assert restored['overlap_forwards']['WAIT_ACQUIRE'] > 0, restored
                    assert restored['overlap_forwards']['WAIT_RELEASE'] > 0, restored
                    assert restored['transferred_bytes'] > 0, restored
                target, source = peer(1, 'stats'), peer(0, 'stats')
                assert target['pinned_pages'] == target['imports'] == target['batches'] == 0, target
                assert not target['target_active']
                assert source['source_leases'] == source['exports'] == 0, source
                assert call(1, 'flush')
                assert peer(1, 'stats')['available_tokens'] == 4096
                concurrency.append(dict(scenario=scenario, items=items, actual=actual, target=target, source=source))
                (out/'concurrency.json').write_text(json.dumps(concurrency, indent=2))
                print('PEER_CONCURRENT_CASE_PASS', scenario, restored.get('overlap_forwards'), flush=True)
            print('PEER_CONCURRENT_PASS cases=6', flush=True)
        if args.lifecycle:
            lifecycle = []
            for case, cancel_state, timeout, reason in [
                ('cancel_acquire', 'WAIT_ACQUIRE', 30000, 'CANCELLED'),
                ('cancel_release', 'WAIT_RELEASE', 30000, 'CANCELLED'),
                ('deadline_acquire', None, 50, 'WAIT_TIMEOUT'),
                ('source_miss', None, 30000, 'SOURCE_REJECTED'),
                ('cancel_running', 'READY', 30000, 'CANCELLED'),
            ]:
                assert call(1, 'flush')
                address, proxy, errors = (service_address, None, [])
                if case != 'source_miss':
                    address, proxy, errors = delayed_peer(service_address)
                request = list(tokens[:request_tokens])
                if case == 'source_miss':
                    request[0] = (request[0] + 1) % 10000
                answer = call(1, 'generate', tokens=request, rid=case,
                              new=512 if case == 'cancel_running' else 24,
                              cancel_state=cancel_state, terminal_reason=reason,
                              l1_restore=dict(source=address, prefix_tokens=prefix_tokens, mode='nonblocking',
                                              wait_timeout_ms=timeout))
                if proxy is not None:
                    proxy.join(5)
                    assert not proxy.is_alive() and not errors, errors
                target, source = peer(1, 'stats'), peer(0, 'stats')
                assert target['pinned_pages'] == target['imports'] == target['batches'] == 0, target
                assert not target['target_active'] and target['wait_state'] is None, target
                assert source['source_leases'] == source['exports'] == 0, source
                assert call(1, 'flush')
                assert peer(1, 'stats')['available_tokens'] == 4096
                # A safe failure must leave both Engines usable.
                call(1, 'generate', tokens=tokens[:23], new=1)
                lifecycle.append(dict(case=case, result=answer, target=target, source=source))
                (out/'lifecycle.json').write_text(json.dumps(lifecycle, indent=2))
            print('PEER_LIFECYCLE_PASS cases=5', flush=True)
        assert call(0,'flush')
        assert peer(0,'stats')['available_tokens']==4096
        print('PEER_MODEL_PASS cases=4',flush=True)
    finally:
        for i,p in enumerate(workers):
            if p.is_alive():
                try:pipes[i].send({'op':'stop'})
                except (BrokenPipeError,EOFError):pass
        for p in workers:
            p.join(30)
            if p.is_alive():os.killpg(p.pid, signal.SIGTERM);p.join(10)
            if p.is_alive():os.killpg(p.pid, signal.SIGKILL);p.join()


if __name__=='__main__':main()
