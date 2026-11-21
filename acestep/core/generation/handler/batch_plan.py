"""Batch-size planning from VRAM / host-memory budgets.

The DiT phase is where the batch dimension costs VRAM: VAE decoding is
batch-sequential with an adaptive window, the LM's KV cache is preallocated and
its scheduler queues sequences that do not fit.  Measured on an RTX 4060 8 GB
(turbo DiT, bf16, flash attention; ``qnn/results/opt/LOG.md`` round 3), the
peak is reached in ``prepare_condition`` while the whole DiT is on the GPU::

    peak = allocated before the DiT phase + DiT weights + B * (0.041 + 8e-6 * frames) GB

(frames = 25 per second of audio; residual of the fit <= 0.01 GB over
10-240 s and B = 1-6).  Out of memory was hit with 0.12 GB of predicted
headroom left (10 s x 8) and avoided with 0.20 GB, hence a 0.3 GB margin.

The plan is computed from the *current* state so that it gives the same answer
before the LM runs (nothing of the DiT loaded yet) and after it (DiT partly
prefetched): weights already on the GPU are inside the allocated memory,
weights still on the CPU are added explicitly.

    budget = effective free VRAM - DiT weights still on CPU - margin
    batch  = floor(budget / per_item(frames))

When the duration is not known yet (the LM will choose it), the tier's maximum
duration is assumed so that the batch never has to shrink after the LM has
produced per-item codes.  Host RAM is checked as well (decoded waveforms and
their copies, ~1.5 MB per second of audio per item, measured as VmHWM growth).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

LATENT_FPS = 25
DIT_ACT_ITEM_GB = 0.041
DIT_ACT_ITEM_GB_PER_FRAME = 8e-6
VRAM_MARGIN_GB = 0.3
HOST_GB_PER_ITEM_SECOND = 0.0015
HOST_RESERVE_GB = 1.0
FALLBACK_MAX_DURATION_S = 600.0


def per_item_vram_gb(duration_s: float, dit_type: str = "turbo") -> float:
    """DiT-phase VRAM of one extra batch item for *duration_s* seconds of audio."""
    gb = DIT_ACT_ITEM_GB + DIT_ACT_ITEM_GB_PER_FRAME * duration_s * LATENT_FPS
    if dit_type.startswith("xl_"):
        gb *= 1.7  # 4B DiT: wider hidden states
    if dit_type.endswith("_base") or dit_type == "base":
        gb *= 2.0  # CFG: conditional + unconditional forward
    return gb


def _module_gb_off_device(module) -> float:
    if module is None:
        return 0.0
    try:
        from acestep.core.generation.handler import pinned_offload

        return pinned_offload.module_bytes(module, only_on_cpu=True) / 2**30
    except Exception:
        return 0.0


def pending_weights_gb(host: Any) -> float:
    """Weights the DiT phase will still upload to the GPU (0 when nothing is offloaded)."""
    if not getattr(host, "offload_to_cpu", False):
        return 0.0
    dit = _module_gb_off_device(getattr(host, "model", None)) if getattr(host, "offload_dit_to_cpu", False) else 0.0
    # With a resident DiT the text encoder is the largest module loaded during generation.
    text = _module_gb_off_device(getattr(host, "text_encoder", None))
    return max(dit, text)


def host_available_gb() -> Optional[float]:
    """MemAvailable from /proc/meminfo (Linux), else psutil, else None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return None


def _max_duration_s() -> float:
    try:
        from acestep.gpu_config import get_global_gpu_config

        cfg = get_global_gpu_config()
        if cfg is not None and getattr(cfg, "max_duration_with_lm", None):
            return float(max(cfg.max_duration_with_lm, getattr(cfg, "max_duration_without_lm", 0) or 0))
    except Exception:
        pass
    return FALLBACK_MAX_DURATION_S


def plan_batch_size(host: Any, batch_size: int, audio_duration: Optional[float] = None,
                    free_vram_gb: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
    """Largest batch <= *batch_size* that fits VRAM and host RAM; returns (batch, details).

    *free_vram_gb* is the effective free VRAM (device free + allocator cache);
    the caller supplies it so tests and the CUDA device index stay explicit.
    """
    info: Dict[str, Any] = {"requested": batch_size}
    if batch_size <= 1 or free_vram_gb is None:
        return max(1, batch_size), info
    try:
        duration = float(audio_duration) if audio_duration is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        duration = _max_duration_s()
        info["duration_assumed"] = True
    dit_type = ""
    try:
        from acestep.gpu_config import get_dit_type_from_path

        dit_type = get_dit_type_from_path((getattr(host, "last_init_params", None) or {}).get("config_path", ""))
    except Exception:
        pass
    pending = pending_weights_gb(host)
    item = per_item_vram_gb(duration, dit_type or "turbo")
    budget = free_vram_gb - pending - VRAM_MARGIN_GB
    vram_max = max(1, math.floor(budget / item)) if budget > 0 else 1
    info.update(duration_s=round(duration, 1), free_gb=round(free_vram_gb, 2), pending_weights_gb=round(pending, 2),
                per_item_gb=round(item, 3), vram_max=vram_max)
    planned = min(batch_size, vram_max)
    host_avail = host_available_gb()
    if host_avail is not None:
        host_max = max(1, math.floor((host_avail - HOST_RESERVE_GB) / (HOST_GB_PER_ITEM_SECOND * duration)))
        info.update(host_available_gb=round(host_avail, 2), host_max=host_max)
        planned = min(planned, host_max)
    info["planned"] = planned
    return planned, info
