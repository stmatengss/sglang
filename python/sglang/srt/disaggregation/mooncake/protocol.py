# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mooncake Transfer Engine protocol catalog for PD disaggregation.

Mooncake supports multiple transports (RDMA, TCP, EFA, NVLink, ...). SGLang
exposes them as:

* ``--disaggregation-transfer-backend mooncake_<protocol>`` aliases
  (e.g. ``mooncake_tcp``, ``mooncake_efa``), rewritten to ``mooncake``.
* ``--disaggregation-mooncake-protocol`` for the engine-wide default.
* Per-path flags (``--disaggregation-mooncake-kv-protocol``, ``-aux-``,
  ``-state-``, ``-staging-``) forwarded as TENT ``transport_hint`` values.

See https://kvcache-ai.github.io/Mooncake/getting_started/supported-protocols.html
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Dict, Optional, Tuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# PD transfer paths that can pin a Mooncake transport independently.
MOONCAKE_TRANSFER_PATHS: Tuple[str, ...] = ("kv", "aux", "state", "staging")

PATH_PROTOCOL_ATTR = {
    "kv": "disaggregation_mooncake_kv_protocol",
    "aux": "disaggregation_mooncake_aux_protocol",
    "state": "disaggregation_mooncake_state_protocol",
    "staging": "disaggregation_mooncake_staging_protocol",
}


@dataclasses.dataclass(frozen=True)
class MooncakeProtocolSpec:
    """How to select one Mooncake transport from SGLang."""

    name: str
    # Value passed to TransferEngine.initialize() / MOONCAKE_PROTOCOL.
    mooncake_protocol: str
    # Extra Mooncake/SGLang env vars required to actually install this transport.
    extra_env: Dict[str, str] = dataclasses.field(default_factory=dict)
    # Transports that do not use an IB/RDMA HCA.
    clear_ib_device: bool = False
    # Optional SGLANG_MOONCAKE_CUSTOM_MEM_POOL value (set only if unset).
    custom_mem_pool: Optional[str] = None
    # Enable the Ascend+Mooncake integration path.
    ascend: bool = False


# Order is argparse-facing: common production protocols first.
_PROTOCOL_SPECS: Tuple[MooncakeProtocolSpec, ...] = (
    MooncakeProtocolSpec(name="rdma", mooncake_protocol="rdma"),
    MooncakeProtocolSpec(
        name="tcp",
        mooncake_protocol="tcp",
        extra_env={"MC_FORCE_TCP": "1"},
        clear_ib_device=True,
    ),
    MooncakeProtocolSpec(name="efa", mooncake_protocol="efa"),
    MooncakeProtocolSpec(
        name="nvlink",
        mooncake_protocol="nvlink",
        extra_env={"MC_FORCE_MNNVL": "1"},
        clear_ib_device=True,
        custom_mem_pool="NVLINK",
    ),
    MooncakeProtocolSpec(
        name="nvlink_intra",
        mooncake_protocol="nvlink_intra",
        extra_env={"MC_INTRANODE_NVLINK": "1"},
        clear_ib_device=True,
        custom_mem_pool="INTRA_NODE_NVLINK",
    ),
    MooncakeProtocolSpec(
        name="hip",
        mooncake_protocol="hip",
        clear_ib_device=True,
    ),
    MooncakeProtocolSpec(
        name="barex",
        mooncake_protocol="barex",
        custom_mem_pool="BAREX",
    ),
    MooncakeProtocolSpec(
        name="ascend",
        mooncake_protocol="ascend",
        clear_ib_device=True,
        ascend=True,
    ),
    MooncakeProtocolSpec(
        name="musa",
        mooncake_protocol="musa",
        extra_env={"MC_FORCE_MUSA": "1"},
        clear_ib_device=True,
    ),
    MooncakeProtocolSpec(
        name="cxl",
        mooncake_protocol="cxl",
        clear_ib_device=True,
    ),
    MooncakeProtocolSpec(name="nvmeof", mooncake_protocol="nvmeof"),
    MooncakeProtocolSpec(name="tpu", mooncake_protocol="tpu"),
    MooncakeProtocolSpec(name="mpcomm", mooncake_protocol="mpcomm"),
)

MOONCAKE_PROTOCOL_SPECS: Dict[str, MooncakeProtocolSpec] = {
    spec.name: spec for spec in _PROTOCOL_SPECS
}

MOONCAKE_TRANSFER_PROTOCOL_CHOICES: Tuple[str, ...] = tuple(
    spec.name for spec in _PROTOCOL_SPECS
)

MOONCAKE_BACKEND_ALIAS_PREFIX = "mooncake_"


def is_mooncake_backend(backend: Optional[str]) -> bool:
    """True for ``mooncake`` and ``mooncake_<protocol>`` aliases."""
    if not backend:
        return False
    if backend == "mooncake":
        return True
    return parse_mooncake_backend_alias(backend) is not None


def parse_mooncake_backend_alias(backend: Optional[str]) -> Optional[str]:
    """Return the protocol name if ``backend`` is ``mooncake_<protocol>``."""
    if not backend or not backend.startswith(MOONCAKE_BACKEND_ALIAS_PREFIX):
        return None
    protocol = backend[len(MOONCAKE_BACKEND_ALIAS_PREFIX) :]
    if protocol in MOONCAKE_PROTOCOL_SPECS:
        return protocol
    return None


def get_mooncake_protocol_spec(protocol: str) -> MooncakeProtocolSpec:
    spec = MOONCAKE_PROTOCOL_SPECS.get(protocol)
    if spec is None:
        raise ValueError(
            f"Unsupported Mooncake transfer protocol {protocol!r}. "
            f"Supported: {list(MOONCAKE_TRANSFER_PROTOCOL_CHOICES)}"
        )
    return spec


def validate_mooncake_protocol(protocol: Optional[str], flag: str) -> None:
    if protocol is None or protocol == "":
        return
    if protocol not in MOONCAKE_PROTOCOL_SPECS:
        raise ValueError(
            f"{flag} must be one of {list(MOONCAKE_TRANSFER_PROTOCOL_CHOICES)}, "
            f"got {protocol!r}"
        )


# Force-flags owned by protocol specs. Cleared when applying a protocol that
# does not set them, so a previous alias (e.g. mooncake_tcp) cannot stick.
_OWNED_FORCE_ENV: Tuple[str, ...] = (
    "MC_FORCE_TCP",
    "MC_FORCE_MNNVL",
    "MC_INTRANODE_NVLINK",
    "MC_FORCE_MUSA",
)


def apply_mooncake_protocol(protocol: str) -> MooncakeProtocolSpec:
    """Install env vars so TransferEngine.initialize() selects ``protocol``.

    CLI/alias selection wins over previously inherited values for the keys this
    spec owns. ``SGLANG_MOONCAKE_CUSTOM_MEM_POOL`` is set only when unset, so an
    explicit user override is preserved.
    """
    spec = get_mooncake_protocol_spec(protocol)
    envs.MOONCAKE_PROTOCOL.set(spec.mooncake_protocol)
    for key in _OWNED_FORCE_ENV:
        if key not in spec.extra_env:
            os.environ.pop(key, None)
    for key, value in spec.extra_env.items():
        os.environ[key] = value
    if spec.custom_mem_pool and not envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.is_set():
        envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.set(spec.custom_mem_pool)
    if spec.ascend:
        envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE.set(True)
    logger.info(
        "Applied Mooncake PD protocol %s (MOONCAKE_PROTOCOL=%s, extra_env=%s)",
        spec.name,
        spec.mooncake_protocol,
        spec.extra_env,
    )
    return spec


def protocol_clears_ib_device(protocol: Optional[str]) -> bool:
    if not protocol:
        return False
    spec = MOONCAKE_PROTOCOL_SPECS.get(protocol)
    return bool(spec and spec.clear_ib_device)


def resolve_path_transport_hint(server_args, path: str) -> str:
    """Return the TENT ``transport_hint`` for a PD transfer path.

    Per-path flag wins; otherwise the engine-wide
    ``--disaggregation-mooncake-protocol`` is used. An empty string leaves
    selection to Mooncake policy / the initialized protocol.
    """
    if path not in PATH_PROTOCOL_ATTR:
        raise ValueError(
            f"Unknown Mooncake PD transfer path {path!r}. "
            f"Supported: {list(MOONCAKE_TRANSFER_PATHS)}"
        )
    per_path = getattr(server_args, PATH_PROTOCOL_ATTR[path], None)
    if per_path:
        return per_path
    global_protocol = getattr(server_args, "disaggregation_mooncake_protocol", None)
    return global_protocol or ""
