"""Memory and VRAM helper methods for handler decomposition."""

import os
from typing import Optional

import torch
from loguru import logger

from acestep.core.generation.handler import batch_plan

from acestep.gpu_config import (
    cuda_supports_bfloat16,
    get_effective_free_vram_gb,
    is_rocm_available,
)


def _is_cuda_device(device: object) -> bool:
    """Return whether a device identifier refers to any CUDA device."""
    if device is None:
        return False
    try:
        return torch.device(str(device)).type == "cuda"
    except (TypeError, RuntimeError, ValueError):
        return isinstance(device, str) and device.split(":", 1)[0] == "cuda"

def _cuda_device_index(device: object) -> int:
    """Return the CUDA device index implied by a device identifier."""
    if isinstance(device, torch.device):
        return 0 if device.index is None else device.index
    try:
        parsed = torch.device(str(device))
        if parsed.type == "cuda":
            return 0 if parsed.index is None else parsed.index
    except (TypeError, RuntimeError, ValueError):
        pass
    if isinstance(device, str) and device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


class MemoryUtilsMixin:
    """Mixin containing memory sizing and VRAM guard helpers.

    Depends on host members:
    - Attribute: ``device``.
    """

    def is_silence(self, audio: torch.Tensor) -> bool:
        """Return True when audio is effectively silent."""
        return bool(torch.all(audio.abs() < 1e-6))

    def _get_system_memory_gb(self) -> Optional[float]:
        """Return total system RAM in GB when available."""
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            if page_size and page_count:
                return (page_size * page_count) / (1024**3)
        except (ValueError, OSError, AttributeError):
            return None
        return None

    def _get_effective_mps_memory_gb(self) -> Optional[float]:
        """Best-effort MPS memory estimate (recommended max or system RAM)."""
        if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
            try:
                return torch.mps.recommended_max_memory() / (1024**3)
            except Exception:
                pass
        system_gb = self._get_system_memory_gb()
        if system_gb is None:
            return None
        return system_gb * 0.75

    VAE_DECODE_MAX_CHUNK_SIZE = 512
    # CUDA: chunk size is derived from free VRAM (see _get_auto_decode_chunk_size).
    # Capped at 1024 frames: per-frame decode cost is flat from ~256 frames on, and
    # beyond 1024 the only gain is less overlap (87.5% -> 90% useful frames), while
    # very large windows OOM on fragmented allocator pools (seen at 1352 frames on
    # an 8 GB card for a 120 s song).
    VAE_DECODE_MAX_CHUNK_SIZE_CUDA = 1024
    # Measured Oobleck decode peak (allocated): 2.58 MiB per latent frame in bf16,
    # linear from 256 to 1280 frames (qnn/results/opt/vae_mem_curve).
    VAE_DECODE_GB_PER_FRAME_BF16 = 2.58 / 1024

    def _get_auto_decode_chunk_size(self) -> int:
        """Choose a conservative VAE decode chunk size based on available memory."""
        override = os.environ.get("ACESTEP_VAE_DECODE_CHUNK_SIZE")
        if override:
            try:
                value = int(override)
                if value > 0:
                    return value
            except ValueError:
                pass

        max_chunk = self.VAE_DECODE_MAX_CHUNK_SIZE

        if self.device == "mps":
            mem_gb = self._get_effective_mps_memory_gb()
            if mem_gb is not None:
                if mem_gb >= 48:
                    return min(1536, max_chunk)
                if mem_gb >= 24:
                    return min(1024, max_chunk)
            return min(512, max_chunk)

        if _is_cuda_device(self.device):
            try:
                free_gb = get_effective_free_vram_gb(_cuda_device_index(self.device))
            except Exception:
                free_gb = 0
            logger.debug(f"[_get_auto_decode_chunk_size] Effective free VRAM: {free_gb:.2f} GB")
            # Size the window from the memory actually available instead of fixed
            # tiers: decoding cost is proportional to decoded frames, and small
            # windows waste up to 50% on the overlap (128-frame windows keep only
            # 64).  When DiT is offloaded an 8 GB card has ~5 GB free, enough to
            # decode a 30 s song (750 frames) in one pass (0.88 s -> 0.47 s).
            try:
                elem = torch.empty(0, dtype=self._get_vae_dtype()).element_size()
            except Exception:
                elem = 2
            per_frame_gb = getattr(self, "VAE_DECODE_GB_PER_FRAME_BF16", 2.58 / 1024) * elem / 2
            # "effective free" counts the allocator's cached blocks, which may be
            # fragmented; keep a 40% margin on top of the 0.5 GB reserve.
            budget_gb = max(0.0, free_gb - 0.5) * 0.6
            frames = int(budget_gb / per_frame_gb)
            cap = getattr(self, "VAE_DECODE_MAX_CHUNK_SIZE_CUDA", 1024)
            return max(128, min(cap, frames))
        return min(256, max_chunk)

    def _should_offload_wav_to_cpu(self) -> bool:
        """Decide whether to offload decoded wavs to CPU for memory safety."""
        override = os.environ.get("ACESTEP_MPS_DECODE_OFFLOAD")
        if override:
            return override.lower() in ("1", "true", "yes")
        if self.device == "mps":
            mem_gb = self._get_effective_mps_memory_gb()
            if mem_gb is not None and mem_gb >= 32:
                return False
            return True
        if _is_cuda_device(self.device):
            try:
                free_gb = get_effective_free_vram_gb(_cuda_device_index(self.device))
                logger.debug(f"[_should_offload_wav_to_cpu] Effective free VRAM: {free_gb:.2f} GB")
                if free_gb >= 24.0:
                    return False
            except Exception:
                pass
        return True

    def plan_batch_size(self, batch_size: int, audio_duration: Optional[float] = None) -> int:
        """Largest batch <= ``batch_size`` that fits the DiT phase in VRAM and host RAM.

        Gives the same answer before and after the LM runs (weights already
        prefetched count as allocated, weights still on CPU are added), so
        ``generate_music`` can size the batch before the LM writes per-item
        codes.  ``audio_duration`` <= 0 / None assumes the tier's maximum.
        See :mod:`batch_plan` for the measured memory model.
        """
        if batch_size <= 1:
            return max(1, batch_size)
        device = self.device
        if not _is_cuda_device(device):
            return batch_size
        try:
            free_gb = get_effective_free_vram_gb(_cuda_device_index(device))
        except Exception:
            return batch_size
        planned, info = batch_plan.plan_batch_size(self, batch_size, audio_duration, free_vram_gb=free_gb)
        try:
            idx = _cuda_device_index(device)
            dev_free, dev_total = torch.cuda.mem_get_info(idx)
            info.update(alloc_gb=round(torch.cuda.memory_allocated(idx) / 2**30, 2),
                        reserved_gb=round(torch.cuda.memory_reserved(idx) / 2**30, 2),
                        device_free_gb=round(dev_free / 2**30, 2), device_total_gb=round(dev_total / 2**30, 2))
        except Exception:
            pass
        if planned < batch_size:
            logger.warning(
                f"[VRAM guard] batch_size {batch_size} -> {planned}: {info}"
            )
        else:
            logger.debug(f"[VRAM guard] batch_size {batch_size} fits: {info}")
        return planned

    def _vram_guard_reduce_batch(
        self,
        batch_size: int,
        audio_duration: Optional[float] = None,
        use_lm: bool = False,
    ) -> int:
        """Auto-reduce batch_size when VRAM / host RAM is too tight (see ``plan_batch_size``)."""
        return MemoryUtilsMixin.plan_batch_size(self, batch_size, audio_duration)

    def _get_vae_dtype(self, device: Optional[str] = None) -> torch.dtype:
        """Get VAE dtype based on target device and GPU tier."""
        target_device = device or self.device
        if _is_cuda_device(target_device):
            if is_rocm_available():
                # On ROCm, defer to self.dtype which is already set to a safe
                # value (float32 by default, or ACESTEP_ROCM_DTYPE override).
                return getattr(self, "dtype", torch.float32)
            if cuda_supports_bfloat16(_cuda_device_index(target_device)):
                return torch.bfloat16
            return torch.float16
        if target_device == "xpu":
            return torch.bfloat16
        if target_device == "mps":
            return torch.float16
        if target_device == "cpu":
            return torch.float32
        return self.dtype
