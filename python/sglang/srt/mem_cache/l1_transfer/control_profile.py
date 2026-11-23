"""Explicit SlotGuard hash-Pin control profile over unchanged data ABI v1.

Selecting a profile never probes peers or downgrades. SlotGuard PREPARED is
strictly local: the target's existing group rendezvous still authorizes COPY.
"""

import uuid
from enum import Enum

from . import slotguard_control_wire, slotguard_protocol
from .slot_guard import LeaseIdentity

PinRequest = slotguard_protocol.PinReq
PinResponse = slotguard_protocol.PinPreparedResp
UnpinRequest = slotguard_protocol.UnpinReq
UnpinResponse = slotguard_protocol.UnpinResp


class ControlProfile(Enum):
    SLOTGUARD_HASH_V3 = "slotguard-hash-v3"

    @property
    def types(self):
        return (PinRequest, PinResponse, UnpinRequest, UnpinResponse)

    @property
    def version(self):
        return slotguard_protocol.CONTROL_VERSION

    @property
    def paths(self):
        return ("/v3/slotguard-hash/pin", "/v3/slotguard-hash/unpin")

    @property
    def codec(self):
        return slotguard_control_wire

    def decode(self, body, expected, *, max_body_bytes, request_context=None):
        result = slotguard_control_wire.decode(
            body,
            expected,
            max_body_bytes=max_body_bytes,
            request_context=request_context,
        )
        if result.protocol_version != self.version:
            raise ValueError("message does not belong to selected control profile")
        return result

    def endpoints(self, request):
        if (
            type(request) not in (PinRequest, UnpinRequest)
            or request.protocol_version != self.version
        ):
            raise ValueError("request does not belong to selected control profile")
        if type(request) is PinRequest:
            return (
                request.result.source,
                request.result.query.target,
                request.result.transfer_id,
            )
        ref = request.reference
        return ref.source, ref.target, ref.identity.transfer_id

    def make_pin(self, result, *, identity=None):
        """Worker-only full-content construction. A future admission owner may
        pre-reserve the cheap lease identity before allocating private slots.
        PinReq validates that supplied identity against the complete result.
        """
        if identity is None:
            identity = LeaseIdentity(
                result.source.incarnation,
                result.query.target.incarnation,
                result.transfer_id,
                uuid.uuid4().hex,
            )
        from .hash_pin import HashLookupResult

        return slotguard_protocol.PinReq(HashLookupResult.from_lookup(result), identity)

    def pin_matches(self, response, request):
        return (
            type(response) is PinResponse
            and type(request) is PinRequest
            and response.request.reference == request.reference
        )

    def prepared(self, response):
        return response.status is slotguard_protocol.PinStatus.PREPARED

    def make_unpin(self, response):
        return slotguard_protocol.UnpinReq(
            response.request.reference,
            response.lease_id,
            True,
            True,
            response.protocol_version,
            response.capability,
        )

    def released(self, response, request):
        statuses = slotguard_protocol.UnpinStatus
        return (
            type(response) is UnpinResponse
            and response.request == request
            and response.status in (statuses.RELEASED, statuses.ALREADY_RELEASED)
        )


def require_profile(profile):
    if type(profile) is not ControlProfile:
        raise ValueError("explicit supported control profile required")
    return profile
