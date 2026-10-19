# SPDX-License-Identifier: Apache-2.0
"""Synchronous Python Engine convenience path, not scheduler admission control."""


def restore_payload(arguments, restore, page_size):
    """Validate the narrow demo API before acquiring any remote ownership."""
    if not isinstance(restore, dict) or set(restore) != {'source', 'prefix_tokens'}:
        raise ValueError('l1_restore requires exactly source and prefix_tokens')
    source = restore['source']
    if (not isinstance(source, dict) or set(source) != {'host', 'port'}
            or not isinstance(source['host'], str) or not source['host']
            or type(source['port']) is not int or not 1 <= source['port'] <= 65535):
        raise ValueError('source requires a nonempty host and integer port in 1..65535')
    tokens = arguments.get('input_ids')
    if (not isinstance(tokens, list) or not tokens
            or any(type(t) is not int or not 0 <= t < 2**31 for t in tokens)):
        raise ValueError('l1_restore requires one nonempty list of token IDs')
    prefix = restore['prefix_tokens']
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
    return dict(tokens=list(tokens[:prefix]), **source)


def generate_with_restore(engine, arguments, restore):
    """Both peers must be initialized; caller exclusively owns this Engine.

    Only a successful, terminal non-streaming generate may trigger finish.
    An exception after restore stops this Engine instead of guessing whether
    request/GPU consumption ended. There is no automatic retry or fallback.
    """
    payload = restore_payload(arguments, restore, engine.server_args.page_size)
    restored = engine.l1_peer('restore', **payload)
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
