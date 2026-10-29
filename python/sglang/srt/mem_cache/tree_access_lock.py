# SPDX-License-Identifier: Apache-2.0
"""One RLock around every public cache and tree-core method (L1 off-thread sources).

The scheduler keeps calling the cache as before; each call holds the lock only for
its own duration, so a source worker can find, lock and pin a prefix between two
scheduler cache calls instead of waiting for a scheduler turn. Same approach as
the M10 tree-lock Pin of the sglang2 ablation (l1-axes ``tree_pins.py``).
"""
import functools
import threading

# Internal entry points other components call directly on the tree core.
_CORE_INTERNALS = ('_add_new_node', '_split_node', '_remove_leaf_from_parent')


def _locked(bound, lock):
    @functools.wraps(bound)
    def call(*args, **kwargs):
        with lock:
            return bound(*args, **kwargs)
    return call


def _wrap_public_methods(obj, lock):
    for name in dir(type(obj)):
        if name.startswith('_'):
            continue
        attr = getattr(type(obj), name, None)
        if not callable(attr) or isinstance(attr, (staticmethod, classmethod, type)):
            continue
        setattr(obj, name, _locked(getattr(obj, name), lock))


def install_tree_access_lock(cache):
    """Wrap ``cache`` and its tree core; install after the source key index."""
    lock = threading.RLock()
    _wrap_public_methods(cache, lock)
    _wrap_public_methods(cache.tree_core, lock)
    for name in _CORE_INTERNALS:
        setattr(cache.tree_core, name, _locked(getattr(cache.tree_core, name), lock))
    cache.source_leases.tree_lock = lock
    return lock
