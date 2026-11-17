"""Reviewed, exact runtime identities; artifact identities are never Git SHAs.

The 512K archive contains frozen c1 sources plus only six reviewed files. Its
observer capacity is unchanged; protocol/page bounds are an independent proof.
These static entries do not replace fresh eight-rank source/process receipts.
"""


def require(condition, message):
    if not condition:
        raise AssertionError(message)


PROTOCOL_RUNTIME = "c1a96f6bb6c8fb534d1fb04656a8d6e30d377d20"
PROTOCOL_SOURCES = {
    "slotguard_protocol.py": "4a8de236d98747947ad98a1121367487cdaff29627a0c4d55b5e633f25d91519",
    "slotguard_control_wire.py": "3a53fbfefc08fe1c0fbad8515d4f6134f5da5451260aaccf217a8963ac15da8f",
    "control_wire.py": "42a8c7073cee1029949e07003738f3d9d5a920736af298e8bfcd846181a3f09a",
    "wire_codec.py": "9109f8d4c807e2236ce509a251838ec359679ece703c3dcc65e9a5582aa39e8c",
    "slotguard_service.py": "418370dddb4dceea1f7b3ed235d0a7ba31f770fd8969ff3655b5eb131e8875a7",
    "tp_admission.py": "fa8369852e6df4d850d3f3bb8e9ac095764e72750baf84989e42f4641f1405b8",
    "nvlink_backend.py": "4dacda5967e5d0df982632913b8393f944a2d852c5c66a1effa1fea631d309e8",
}
PROTOCOL_CAPABILITIES = dict(
    max_prefix_tokens=300000,
    max_restore_pages=4096,
    metadata_payload_bytes=4 * 1024 * 1024,
    control_payload_bytes=16 * 1024 * 1024,
    lease_duration_ms=30000,
)
RUNTIME_512 = (
    "artifact-sha256:ffa48714aa7f9f3c5334493a4ad766a49211d716cf732acda76d4f4bf956c238"
)
SOURCES_512 = dict(
    PROTOCOL_SOURCES,
    **{
        "slotguard_protocol.py": "cb7b819fe2e7bc5eb02baf47d657ecd0d9981113450dbf828b11fedefe2a090c",
        "slotguard_control_wire.py": "7c659e2ced485b5814c720090a8080c935132060895f7638b1a476985b461301",
        "slotguard_limits.py": "5ffa7aaf66287f2f5417d6a9aeef999c76d32b1e36bdcadfc788c2a40d2810c2",
        "config.py": "4032bed1e774b225eee6a31f030220be06aa52f4a1ff64c2fa6c2d9069ed2312",
        "serving.py": "0e7a463b1fb472fc32ebb179bacc475ff6e07633497f2c6eef5449be4a7a3b0f",
        "slotguard_receipts.py": "290cb6f4ffa0b07548351f35e3baebdc8e4c04aa1d29eab348b3598d640357f6",
    },
)
CAPABILITIES_512 = dict(
    PROTOCOL_CAPABILITIES, max_prefix_tokens=524288, max_restore_pages=8192
)
RUNTIME_PROOFS = {
    PROTOCOL_RUNTIME: dict(
        source_hashes=PROTOCOL_SOURCES, capabilities=PROTOCOL_CAPABILITIES
    ),
    RUNTIME_512: dict(source_hashes=SOURCES_512, capabilities=CAPABILITIES_512),
}


def runtime_proof(identity):
    require(identity in RUNTIME_PROOFS, "unreviewed protocol runtime identity")
    return RUNTIME_PROOFS[identity]


def manifest_runtime(manifest):
    """Require both peers to use one exact proof, including configured limits."""
    identity = manifest["evidence"]["revisions"]["serving"]
    proof = runtime_proof(identity)
    engines, configs = manifest["evidence"]["engines"], manifest["configs"]
    require(
        len(engines) == len(configs) == 2
        and all(engine["serving_sha"] == identity for engine in engines),
        "mixed or missing peer protocol runtime proof",
    )
    if identity == RUNTIME_512:
        require(
            all(
                type(config["slotguard_limits"][name]) is int
                and config["slotguard_limits"][name] == 8192
                for config in configs
                for name in ("max_pin_pages", "max_remote_pinned_pages")
            ),
            "512K runtime requires exact configured 8192 page limits on both peers",
        )
    return identity, proof


def require_shape(identity, length, hit, page_size=64):
    capabilities = runtime_proof(identity)["capabilities"]
    require(
        length <= capabilities["max_prefix_tokens"]
        and hit // page_size <= capabilities["max_restore_pages"],
        "UNSUPPORTED_PROTOCOL: long-context workload exceeds reviewed PinReq protocol limits",
    )


def receipt_runtime(manifest):
    """Only the exact dual-peer 512K proof extends historical receipt bounds."""
    engines = manifest.get("evidence", {}).get("engines", [])
    identity = manifest.get("evidence", {}).get("revisions", {}).get("serving")
    if identity == RUNTIME_512 or any(
        engine.get("serving_sha") == RUNTIME_512 for engine in engines
    ):
        return manifest_runtime(manifest)[0]
    return None


def receipt_page_limit(identity=None):
    if identity is None:
        return 4096
    return runtime_proof(identity)["capabilities"]["max_restore_pages"]
