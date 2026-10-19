# L1 transfer and Engine sharing tests

Start with the in-repository documentation:

- [Implementation and ownership](../../../python/sglang/srt/mem_cache/l1_transfer/IMPLEMENTATION.md)
- [Test commands, prerequisites and validated results](../../../python/sglang/srt/mem_cache/l1_transfer/TESTING.md)
- [Transport build, execution and provenance](../../../python/sglang/srt/mem_cache/l1_transfer/README.md)

`run-tests.sh` exercises components on two allocated NVLink-connected nodes.
`run-peer-model.sh` compares generation with three ordinary Engines on one node.
Both require a configured framework runtime and fresh absolute output paths;
see the testing guide before launching them.
