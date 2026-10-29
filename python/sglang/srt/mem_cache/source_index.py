# SPDX-License-Identifier: Apache-2.0
"""Chained page key -> radix node index for L1 source leases.

A target names a prefix by its last page key alone: the key chains over every
earlier page, so finding it at the right depth identifies the whole prefix.
Installed only on L1 sources; it hashes every new device node at insert, the
same keys HiCache storage computes (``compute_node_hash_values``). With a
``ready_stream`` each new node also records an event there, so a reader off the
scheduler thread waits for exactly the forwards that wrote the path.
"""
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.utils import compute_node_hash_values, get_hash_str


def prefix_page_keys(tokens, page_size):
    """Page keys of complete pages, matching an un-namespaced source tree."""
    from array import array
    from sglang.srt.mem_cache.radix_cache import RadixKey
    # Same element width as tree keys; the native hash reads the raw buffer.
    return get_hash_str(RadixKey(array('q', tokens)), None, page_size=page_size)


class SourceKeyIndex:
    def __init__(self, core, page_size):
        if not isinstance(core, UnifiedTreeCore):
            raise ValueError('L1 source key index requires the Python UnifiedTreeCore')
        self.core, self.page_size = core, page_size
        self.ready_stream = None
        self._nodes = {}
        core.backfill_missing_hash_values()
        stack = [core.root_node]
        while stack:
            node = stack.pop()
            if node is not core.root_node:
                self._register(node)
            stack.extend(node.children.values())
        add, split, remove = core._add_new_node, core._split_node, core._remove_leaf_from_parent
        reset = core.reset

        def reset_tree():
            reset()
            self._nodes.clear()

        core.reset = reset_tree

        def add_new_node(*args, **kwargs):
            node = add(*args, **kwargs)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, page_size)
            if self.ready_stream is not None:
                import torch
                # Insert follows the launch of the forward that wrote this node's KV.
                node._l1_ready = torch.cuda.Event()
                node._l1_ready.record(self.ready_stream)
            self._register(node)
            return node

        def split_node(*args, **kwargs):
            new_node, action = split(*args, **kwargs)
            new_node._l1_ready = getattr(args[1] if len(args) > 1 else kwargs['child'], '_l1_ready', None)
            self._register(new_node)  # its pages moved out of the child
            return new_node, action

        def remove_leaf_from_parent(node):
            self._unregister(node)
            remove(node)

        core._add_new_node = add_new_node
        core._split_node = split_node
        core._remove_leaf_from_parent = remove_leaf_from_parent

    def _register(self, node):
        for key in node.hash_value or ():
            self._nodes[key] = node

    def _unregister(self, node):
        for key in node.hash_value or ():
            if self._nodes.get(key) is node:
                del self._nodes[key]

    def __len__(self):
        return len(self._nodes)

    def ready_events(self, node):
        """Insert-time events of ``node`` and its ancestors; nodes without one predate it."""
        events = []
        while node is not None and node is not self.core.root_node:
            event = getattr(node, '_l1_ready', None)
            if event is not None:
                events.append(event)
            node = node.parent
        return tuple(events)

    def find(self, key, end, tier='hbm'):
        """Node holding the page that ends at token ``end``, resident in ``tier``
        (``hbm`` device value, ``egm`` host value) along the whole path; ValueError on any miss."""
        node = self._nodes.get(key)
        if node is None:
            raise ValueError('prefix key not in tree')
        hashes = node.hash_value or []
        try:
            offset = hashes.index(key)
        except ValueError:
            raise ValueError('stale prefix key') from None
        depth, child = 0, node
        while child is not self.core.root_node:
            parent = child.parent
            if parent is None or parent.children.get(child.key.child_key(self.page_size)) is not child:
                raise ValueError('prefix key node is detached')
            data = child.component_data[ComponentType.FULL]
            if (data.value if tier == 'hbm' else data.host_value) is None:
                raise ValueError(f'prefix not resident in {tier}')
            if child is not node:
                depth += len(child.key)
            child = parent
        if depth + (offset + 1) * self.page_size != end:
            raise ValueError('prefix key at a different depth')
        return node

    def touch(self, node):
        """Refresh the path's recency as a token match would; a leased prefix is hot."""
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import get_and_increase_time_counter
        now = get_and_increase_time_counter()
        while node is not None:
            node.last_access_time = now
            now -= 0.00001
            node = node.parent
