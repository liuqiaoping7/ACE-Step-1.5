"""Context manager for temporary model loading/offloading."""

import os
import time
from contextlib import contextmanager

from loguru import logger

from acestep.core.generation.handler import pinned_offload


class InitServiceOffloadContextMixin:
    """Context-managed model load/offload behavior for CPU offload mode."""

    @contextmanager
    def _load_model_context(self, model_name: str):
        """Load a model to device for the context and offload back to CPU on exit."""
        if not self.offload_to_cpu:
            yield
            return

        if model_name == "model" and not self.offload_dit_to_cpu:
            model = getattr(self, model_name, None)
            if model is not None:
                try:
                    param = next(model.parameters())
                    if param.device.type == "cpu":
                        logger.info(f"[_load_model_context] Moving {model_name} to {self.device} (persistent)")
                        self._recursive_to_device(model, self.device, self.dtype)
                        self._release_system_memory()
                        if hasattr(self, "silence_latent"):
                            self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)
                except StopIteration:
                    pass
            yield
            return

        model = getattr(self, model_name, None)
        if model is None:
            yield
            return

        rss_before = self._get_rss_mb()
        logger.info(f"[_load_model_context] Loading {model_name} to {self.device} (RSS: {rss_before:.0f} MB)")
        start_time = time.time()
        load_dtype = self._get_vae_dtype() if model_name == "vae" else self.dtype
        # Fast path: pinned host masters, async H2D, no D2H on offload.
        use_pinned = pinned_offload.supports(model, self.device)
        if use_pinned:
            pinned_offload.load_to_device(model, self.device, load_dtype)
        else:
            self._recursive_to_device(model, self.device, load_dtype)

        if model_name == "model" and hasattr(self, "silence_latent"):
            self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

        load_time = time.time() - start_time
        self.current_offload_cost += load_time

        # Free old CPU tensor storage after moving to GPU (not counted in load_time).
        # The pinned path reuses persistent host masters, so there is nothing to reclaim.
        if not use_pinned:
            self._release_system_memory()
        rss_after = self._get_rss_mb()
        logger.info(
            f"[_load_model_context] Loaded {model_name} to {self.device} in {load_time:.4f}s "
            f"(RSS: {rss_before:.0f} -> {rss_after:.0f} MB, delta: {rss_after - rss_before:+.0f} MB)"
        )

        try:
            yield
        finally:
            rss_before = self._get_rss_mb()
            logger.info(
                f"[_load_model_context] Offloading {model_name} to CPU (RSS: {rss_before:.0f} MB)"
            )
            start_time = time.time()
            if use_pinned:
                pinned_offload.offload_to_cpu(model)
            elif model_name == "vae":
                self._recursive_to_device(model, "cpu", self._get_vae_dtype("cpu"))
            else:
                self._recursive_to_device(model, "cpu")

            offload_time = time.time() - start_time
            self.current_offload_cost += offload_time

            # Aggressively reclaim memory: GPU cache + Python GC + OS heap trim
            # (not counted in offload_time to keep timing accurate).  The pinned
            # path allocates no new host storage, so only the GPU cache is flushed.
            if use_pinned:
                self._empty_cache()
            else:
                self._release_system_memory()
            rss_after = self._get_rss_mb()
            logger.info(
                f"[_load_model_context] Offloaded {model_name} to CPU in {offload_time:.4f}s "
                f"(RSS: {rss_before:.0f} -> {rss_after:.0f} MB, delta: {rss_after - rss_before:+.0f} MB)"
            )

    @contextmanager
    def _load_model_parts_context(self, parts):
        """Temporarily load only the given submodules of ``self.model`` to the device.

        Used when a small part of the DiT checkpoint is needed (e.g. the FSQ
        quantizer + detokenizer, ~0.2 GB) so the full 4.5 GB model is not
        uploaded.  A no-op unless the DiT itself is offloaded to CPU.
        """
        if not (self.offload_to_cpu and self.offload_dit_to_cpu):
            yield
            return
        # Parts may already be resident because of an async DiT prefetch.
        pinned_offload.wait_prefetch(self.model)
        to_load = []
        for part in parts:
            try:
                if next(part.parameters()).device.type == "cpu":
                    to_load.append(part)
            except StopIteration:
                continue
        for part in to_load:
            if pinned_offload.supports(part, self.device):
                pinned_offload.load_to_device(part, self.device, self.dtype)
            else:
                self._recursive_to_device(part, self.device, self.dtype)
        try:
            yield
        finally:
            for part in to_load:
                if pinned_offload.supports(part, self.device):
                    pinned_offload.offload_to_cpu(part)
                else:
                    self._recursive_to_device(part, "cpu")

    def prefetch_dit_async(self) -> bool:
        """Start uploading most of the DiT checkpoint on a side stream (DiT-offload mode).

        Called before the LM runs so the ~3.3 GB H2D copy of the decoder,
        tokenizer and detokenizer overlaps LM decoding instead of stalling the
        GPU later.  The condition encoder (~1.1 GB) is left to the normal load
        so the text encoder still fits next to the LM.  The text encoder and the
        VAE are loaded one at a time, and prefetched + text encoder (~4.4 GB)
        stays below the full DiT that diffusion needs anyway, so the prefetch
        does not raise the peak.  Skipped when free VRAM would not leave room
        for the larger of the two plus a 0.25 GB margin.
        Disable with ``ACESTEP_DIT_PREFETCH=0``.
        """
        if not (getattr(self, "offload_to_cpu", False) and getattr(self, "offload_dit_to_cpu", False)):
            return False
        if os.environ.get("ACESTEP_DIT_PREFETCH", "1") == "0":
            return False
        model = getattr(self, "model", None)
        if model is None or not pinned_offload.supports(model, self.device):
            return False
        parts = [getattr(model, n, None) for n in ("decoder", "tokenizer", "detokenizer")]
        parts = [p for p in parts if p is not None]
        need = sum(pinned_offload.module_bytes(p, only_on_cpu=True) for p in parts)
        if need == 0:
            return False
        try:
            from acestep.gpu_config import get_effective_free_vram_gb

            free = get_effective_free_vram_gb() * (1024 ** 3)
        except Exception:
            return False
        largest = 0
        for name in ("text_encoder", "vae"):
            m = getattr(self, name, None)
            if m is not None:
                largest = max(largest, pinned_offload.module_bytes(m))
        reserve = largest + int(0.25 * 1024 ** 3)
        if free - need < reserve:
            logger.info(
                f"[prefetch_dit_async] Skipped: need {need / 1024**3:.2f} GB + reserve "
                f"{reserve / 1024**3:.2f} GB, free {free / 1024**3:.2f} GB"
            )
            return False
        for part in parts:
            pinned_offload.prefetch_to_device(part, self.device, self.dtype)
        logger.info(f"[prefetch_dit_async] Prefetching {need / 1024**3:.2f} GB of DiT weights")
        return True

    def release_dit_prefetch(self) -> None:
        """Offload DiT weights left on the GPU by a prefetch that was never consumed."""
        model = getattr(self, "model", None)
        if model is None or not (getattr(self, "offload_to_cpu", False) and getattr(self, "offload_dit_to_cpu", False)):
            return
        if not pinned_offload.supports(model, self.device):
            return
        if pinned_offload.module_bytes(model, only_on_cpu=True) == pinned_offload.module_bytes(model):
            return  # nothing resident
        pinned_offload.offload_to_cpu(model)
        self._empty_cache()

    def _offload_dit_condition_parts(self) -> None:
        """Free the DiT parts not needed by the sampling loop (DiT-offload mode only).

        The condition encoder (lyric/timbre, ~1.13 GB) and the audio tokenizer /
        detokenizer (~0.4 GB) run once in ``prepare_condition``; only ``decoder``
        is used by the diffusion steps.  With pinned masters the offload is a
        pointer swap.  This is what lets a 240 s song fit on an 8 GB card next to
        the resident LM (it OOMed by ~0.1 GB before).
        """
        if not (getattr(self, "offload_to_cpu", False) and getattr(self, "offload_dit_to_cpu", False)):
            return
        if os.environ.get("ACESTEP_OFFLOAD_DIT_COND_PARTS", "1") == "0":
            return
        model = getattr(self, "model", None)
        if model is None or not pinned_offload.supports(model, self.device):
            return
        for name in ("encoder", "tokenizer", "detokenizer"):
            part = getattr(model, name, None)
            if part is not None:
                pinned_offload.offload_to_cpu(part)
        self._empty_cache()
