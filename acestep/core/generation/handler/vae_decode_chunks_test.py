"""Unit tests for ``VaeDecodeChunksMixin`` chunk/fallback behavior."""

import unittest

import torch

from acestep.core.generation.handler.vae_decode_test_helpers import _ChunksHost


class VaeDecodeChunksMixinTests(unittest.TestCase):
    """Verify critical chunk decode paths and OOM fallback chain semantics."""

    def test_batch_sequential_decode_for_multi_sample_input(self):
        """Batch size > 1 should decode per sample then concatenate."""
        host = _ChunksHost()
        latents = torch.zeros(2, 4, 6)
        out = host._tiled_decode_inner(latents, chunk_size=10, overlap=2, offload_wav_to_cpu=False)
        self.assertEqual(tuple(out.shape), (2, 2, 12))

    def test_direct_decode_for_short_latents(self):
        """Short latents should take direct decode path without tiling loop."""
        host = _ChunksHost()
        latents = torch.zeros(1, 4, 6)
        out = host._tiled_decode_inner(latents, chunk_size=10, overlap=2, offload_wav_to_cpu=False)
        self.assertEqual(tuple(out.shape), (1, 2, 12))

    def test_offload_oom_retries_smaller_window_on_gpu_before_cpu(self):
        """An OOM at a large window halves the window on GPU instead of decoding on CPU."""
        host = _ChunksHost()
        calls = []

        def _offload(latents, bsz, frames, stride, overlap, num_steps):
            calls.append(stride + 2 * overlap)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("fragmented")
            return torch.ones(1, 2, 8)

        host._tiled_decode_offload_cpu = _offload
        host._decode_on_cpu = lambda latents: self.fail("must not fall back to CPU")
        out = host._tiled_decode_inner(torch.zeros(1, 4, 3000), chunk_size=1024, overlap=64, offload_wav_to_cpu=True)
        self.assertEqual(calls, [1024, 512])
        self.assertEqual(tuple(out.shape), (1, 2, 8))

    def test_oom_at_minimum_window_falls_back_to_cpu(self):
        host = _ChunksHost()

        def _offload(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError("oom")

        host._tiled_decode_offload_cpu = _offload
        host._decode_on_cpu = lambda latents: torch.full((1, 2, 3), 7.0)
        out = host._tiled_decode_inner(torch.zeros(1, 4, 3000), chunk_size=128, overlap=64, offload_wav_to_cpu=True)
        self.assertTrue(torch.all(out == 7.0))

    def test_overlap_adjustment_reduces_invalid_overlap(self):
        """Invalid overlap should be reduced until stride becomes positive."""
        host = _ChunksHost()

        def _capture_gpu(latents, stride, overlap, num_steps):
            """Capture overlap argument passed to GPU decode path."""
            _ = latents, stride, num_steps
            host.recorded["overlap"] = overlap
            return torch.ones(1, 2, 4)

        host._tiled_decode_gpu = _capture_gpu
        out = host._tiled_decode_inner(torch.zeros(1, 4, 10), chunk_size=4, overlap=3, offload_wav_to_cpu=False)
        self.assertEqual(host.recorded["overlap"], 1)
        self.assertEqual(tuple(out.shape), (1, 2, 4))

    def test_oom_fallback_gpu_to_offload_path(self):
        """GPU OOM should fallback to offload path before full CPU fallback."""
        host = _ChunksHost()

        def _gpu_oom(*args, **kwargs):
            """Raise OOM to force GPU fallback chain."""
            _ = args, kwargs
            raise torch.cuda.OutOfMemoryError("gpu oom")

        def _offload_ok(*args, **kwargs):
            """Return sentinel tensor from offload path."""
            _ = args, kwargs
            return torch.ones(1, 2, 5)

        host._tiled_decode_gpu = _gpu_oom
        host._tiled_decode_offload_cpu = _offload_ok
        out = host._tiled_decode_inner(torch.zeros(1, 4, 20), chunk_size=8, overlap=2, offload_wav_to_cpu=False)
        self.assertEqual(tuple(out.shape), (1, 2, 5))
        self.assertEqual(host.decode_on_cpu_calls, 0)

    def test_oom_fallback_chain_reaches_decode_on_cpu(self):
        """Repeated OOMs should end at full CPU decode fallback."""
        host = _ChunksHost()

        def _oom(*args, **kwargs):
            """Raise OOM for all tiled decode branches."""
            _ = args, kwargs
            raise torch.cuda.OutOfMemoryError("oom")

        host._tiled_decode_gpu = _oom
        host._tiled_decode_offload_cpu = _oom
        out = host._tiled_decode_inner(torch.zeros(1, 4, 20), chunk_size=8, overlap=2, offload_wav_to_cpu=False)
        self.assertTrue(torch.equal(out, torch.full((1, 2, 7), 9.0)))
        self.assertEqual(host.decode_on_cpu_calls, 1)


if __name__ == "__main__":
    unittest.main()
