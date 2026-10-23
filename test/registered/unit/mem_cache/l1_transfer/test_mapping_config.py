import os
import unittest
from unittest.mock import patch

from sglang.srt.mem_cache.l1_transfer.peer.scheduler import _mapping_peer_device


class MappingConfigTests(unittest.TestCase):
    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_mapping_peer_device(1))

    def test_explicit_local_ordinal(self):
        with patch.dict(os.environ, {'SGLANG_L1_MAPPING_PEER_DEVICE': '2'}):
            self.assertEqual(_mapping_peer_device(1), 2)

    def test_invalid_or_same_device(self):
        for value in ('', '-1', '1', '1.5', 'auto'):
            with self.subTest(value=value), patch.dict(os.environ, {'SGLANG_L1_MAPPING_PEER_DEVICE': value}):
                with self.assertRaises(ValueError):
                    _mapping_peer_device(1)
