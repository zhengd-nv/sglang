# SPDX-License-Identifier: Apache-2.0
"""Conservative TP1 admission budget, independent of cache hit estimates.

Charge maximum prompt+output, page rounded. Allocated rows and locked cache
prefixes are already absent from available+evictable space; charge only future
growth for admitted rows. New requests pay their entire bound (even cache hits).
All updates happen on the scheduler owner thread. No speculative overcommit.
"""


class RequestCapacity:
    def __init__(self, cache):
        self.cache = cache
        self.allocator = cache.token_to_kv_pool_allocator
        self.page = self.allocator.page_size
        self.active = {}

    def rounded(self, tokens):
        return (tokens + self.page - 1) // self.page * self.page

    def bound(self, req):
        if getattr(req.sampling_params, 'beam_width', None) not in (None, 1):
            raise ValueError('concurrent L1 does not support beam search')
        output = req.sampling_params.max_new_tokens
        if type(output) is not int or output < 0:
            raise ValueError('concurrent L1 requires bounded max_new_tokens')
        return self.rounded(len(req.origin_input_ids) + output)

    def prune(self):
        # Called between scheduler turns, after normal output processing freed
        # terminal rows. Retractions keep their full admission entitlement.
        self.active = {k: v for k, v in self.active.items() if not v[0].finished()}

    def remaining(self):
        return sum(max(0, bound - self.rounded(req.kv.kv_allocated_len))
                   for req, bound in self.active.values())

    def available(self):
        return self.allocator.available_size() + self.cache.evictable_size()

    def can_admit(self, req, reserved=0):
        if req.rid in self.active:
            return True
        return self.bound(req) + self.remaining() + reserved <= self.available()

    def admit(self, req):
        self.active.setdefault(req.rid, (req, self.bound(req)))
