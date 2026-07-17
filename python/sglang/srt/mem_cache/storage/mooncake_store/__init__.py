from __future__ import annotations

import logging

from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


def _mooncake_factory(ctx):
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_radix_cache import (
        MooncakeRadixCache,
    )

    return MooncakeRadixCache(
        params=ctx.params,
        model_config=ctx.model_config,
        server_args=ctx.server_args,
        tp_rank=ctx.tp_rank,
        tp_size=ctx.tp_size,
        tp_group=ctx.tp_group,
    )


try:
    register_radix_cache_backend("mooncake", _mooncake_factory)
except ValueError as exc:
    logger.debug("mooncake backend already registered: %s", exc)
