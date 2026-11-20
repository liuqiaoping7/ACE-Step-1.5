"""Context manager for temporary model loading/offloading."""

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
        use_pinned = pinned_offload.supports(model)
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
        to_load = []
        for part in parts:
            try:
                if next(part.parameters()).device.type == "cpu":
                    to_load.append(part)
            except StopIteration:
                continue
        for part in to_load:
            if pinned_offload.supports(part):
                pinned_offload.load_to_device(part, self.device, self.dtype)
            else:
                self._recursive_to_device(part, self.device, self.dtype)
        try:
            yield
        finally:
            for part in to_load:
                if pinned_offload.supports(part):
                    pinned_offload.offload_to_cpu(part)
                else:
                    self._recursive_to_device(part, "cpu")
