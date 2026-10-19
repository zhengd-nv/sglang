"""Real Engine prefix restore, explicit scheduler control, no PD or host KV stage."""
import argparse
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


def actor(pipe, gpu, model, output):
    os.setsid()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    log = open(Path(output) / f'engine-{gpu}.log', 'w', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    engine = None
    try:
        from sglang import Engine
        engine = Engine(model_path=model, tp_size=1, dtype='bfloat16',
                        max_running_requests=1, max_total_tokens=4096, context_length=4096,
                        mem_fraction_static=0.15, page_size=16, attention_backend='flashinfer',
                        chunked_prefill_size=128, cuda_graph_max_bs_decode=1,
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
                    def observe():
                        while not done.wait(0.01):
                            before = time.monotonic()
                            try:
                                stats = engine.l1_peer('stats')
                                observations.append(dict(stats, latency=time.monotonic()-before))
                            except Exception as exc:
                                observations.append({'error': str(exc)})
                                break
                    monitor = None
                    if command.get('l1_restore', {}).get('mode') == 'nonblocking':
                        monitor = threading.Thread(target=observe, daemon=True)
                        monitor.start()
                    try:
                        answer = engine.generate(input_ids=command['tokens'],
                        sampling_params={'temperature':0, 'max_new_tokens':command.get('new',24), 'ignore_eos':True},
                        return_logprob=True, logprob_start_len=-1,
                            l1_restore=command.get('l1_restore'))
                    finally:
                        done.set()
                        if monitor is not None:
                            monitor.join(5)
                            assert not monitor.is_alive(), 'control query blocked scheduler'
                    if monitor is not None:
                        answer['control_observations'] = observations
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
    args=parser.parse_args()
    if args.nonblocking: args.request_restore = True
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    tokens=tokenizer.encode('The quick brown fox jumps over the lazy dog. '*30)[:160]
    assert len(tokens)==160
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
        for gpu in range(3):
            a,b=ctx.Pipe();p=ctx.Process(target=actor,args=(b,gpu,args.model,str(out)))
            p.start();b.close();pipes.append(a);workers.append(p)
        for i in range(3):assert recv(i)['ready']
        for i in (0,1):peer(i,'init',identity=identity)
        call(0,'generate',tokens=tokens,new=1)
        service_address = peer(0,'start_service') if args.nonblocking else None
        rows=[]
        for case,local in [('cold',0),('partial',32),('all_local',112),('repeat_cold',0)]:
            assert call(1,'flush')
            assert call(2,'flush')
            if local:
                call(1,'generate',tokens=tokens[:local]+[777],new=1)
            request=tokens[:119]
            reference=call(2,'generate',tokens=request)
            address = {'host': '127.0.0.1', 'port': 1}  # unused for all-local
            proxy = None
            if local<112:
                if args.nonblocking:
                    address, proxy, errors = delayed_peer(service_address)
                else:
                    address=peer(0,'listen')
                    pipes[0].send(dict(op='peer',action='serve'))
            if args.request_restore:
                context = dict(source=address, prefix_tokens=112,
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
                restored=peer(1,'restore',tokens=tokens[:112],**address)
                # Target consumer pin must survive until ordinary generate completes.
                assert not call(1,'flush'), 'flush accepted a live target reservation'
                actual=call(1,'generate',tokens=request)
            if local<112:
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
            assert restored['restored_tokens']==112
            assert restored['pinned_pages']==7
            assert restored['imports']==restored['batches']==0
            ids=lambda x:[t[1] for t in x['meta_info']['output_token_logprobs']]
            assert ids(reference)==ids(actual),(case,ids(reference),ids(actual))
            assert actual['meta_info']['cached_tokens']==112,actual['meta_info']
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
