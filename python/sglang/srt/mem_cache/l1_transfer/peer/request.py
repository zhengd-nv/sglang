# SPDX-License-Identifier: Apache-2.0
"""Synchronous Python Engine convenience path, not scheduler admission control."""


MAX_SOURCES = 8  # Arbitrary bound on one request's failover chain.


def _parse_sources(restore):
    """Return [(host, port, prefix_tokens)] in caller preference order."""
    if 'sources' in restore:
        if 'source' in restore or 'prefix_tokens' in restore:
            raise ValueError('l1_restore takes either sources or source with prefix_tokens')
        if restore.get('mode') != 'nonblocking':
            raise ValueError('l1_restore sources require nonblocking mode')
        sources = restore['sources']
        if not isinstance(sources, list) or not 0 < len(sources) <= MAX_SOURCES:
            raise ValueError(f'sources must be a list of 1..{MAX_SOURCES} candidates')
        if any(not isinstance(s, dict) or set(s) != {'host', 'port', 'prefix_tokens'} for s in sources):
            raise ValueError('each source requires exactly host, port and prefix_tokens')
        parsed = [(s['host'], s['port'], s['prefix_tokens']) for s in sources]
    else:
        if not {'source', 'prefix_tokens'} <= set(restore):
            raise ValueError('l1_restore requires exactly source and prefix_tokens')
        source = restore['source']
        if not isinstance(source, dict) or set(source) != {'host', 'port'}:
            raise ValueError('source requires a nonempty host and integer port in 1..65535')
        parsed = [(source['host'], source['port'], restore['prefix_tokens'])]
    for host, port, _ in parsed:
        if (not isinstance(host, str) or not host
                or type(port) is not int or not 1 <= port <= 65535):
            raise ValueError('source requires a nonempty host and integer port in 1..65535')
    if len({(host, port) for host, port, _ in parsed}) != len(parsed):
        raise ValueError('sources must not repeat an endpoint')
    return parsed


def restore_payload(arguments, restore, page_size):
    """Validate the narrow demo API before acquiring any remote ownership."""
    if (not isinstance(restore, dict)
            or set(restore) - {'source', 'sources', 'prefix_tokens', 'mode', 'wait_timeout_ms'}
            or restore.get('mode', 'synchronous') not in ('synchronous', 'nonblocking')):
        raise ValueError('l1_restore requires exactly source and prefix_tokens')
    if 'wait_timeout_ms' in restore:
        timeout = restore['wait_timeout_ms']
        if (restore.get('mode') != 'nonblocking' or type(timeout) is not int
                or not 0 < timeout <= 3600000):
            raise ValueError('wait_timeout_ms requires nonblocking mode and integer 1..3600000')
    sources = _parse_sources(restore)
    tokens = arguments.get('input_ids')
    if (not isinstance(tokens, list) or not tokens
            or any(type(t) is not int or not 0 <= t < 2**31 for t in tokens)):
        raise ValueError('l1_restore requires one nonempty list of token IDs')
    for _, _, prefix in sources:
        if (type(page_size) is not int or page_size <= 0 or type(prefix) is not int
                or not 0 < prefix < len(tokens) or prefix % page_size):
            raise ValueError('prefix_tokens must contain complete pages and be shorter than input_ids')
    if arguments.get('stream'):
        raise ValueError('l1_restore does not support streaming')
    for field in ('prompt', 'image_data', 'audio_data', 'video_data', 'mm_hashes',
                  'mm_content_hashes', 'lora_path', 'session_id', 'session_params',
                  'cache_salt', 'bootstrap_host', 'bootstrap_port', 'bootstrap_room',
                  'kv_hints'):
        if arguments.get(field) is not None:
            raise ValueError(f'l1_restore does not support {field}')
    if isinstance(arguments.get('rid'), list):
        raise ValueError('l1_restore requires a single request ID')
    sampling = arguments.get('sampling_params')
    if sampling is not None and (not isinstance(sampling, dict)
                                or type(sampling.get('n', 1)) is not int
                                or sampling.get('n', 1) != 1):
        raise ValueError('l1_restore requires one sampling configuration with n=1')
    if restore.get('mode') == 'nonblocking' and sampling is not None:
        if any(sampling.get(k) is not None for k in ('json_schema', 'regex', 'ebnf', 'structural_tag')):
            raise ValueError('nonblocking L1 restore does not support grammar constraints')
    return dict(candidates=[dict(host=host, port=port, tokens=list(tokens[:prefix]))
                            for host, port, prefix in sources])


def generate_with_restore(engine, arguments, restore):
    """Both peers must be initialized; caller exclusively owns this Engine.

    Only a successful, terminal non-streaming generate may trigger finish.
    An exception after restore stops this Engine instead of guessing whether
    request/GPU consumption ended. There is no automatic retry or fallback.
    """
    candidate, = restore_payload(arguments, restore, engine.server_args.page_size)['candidates']
    restored = engine.l1_peer('restore', **candidate)
    try:
        result = engine.generate(**arguments)
        released = engine.l1_peer('finish')
    except BaseException:
        engine.shutdown()
        raise
    result['meta_info']['l1_restore'] = dict(
        restored, consumer_released=(not released['target_active']
                                     and released['pinned_pages'] == 0))
    return result
