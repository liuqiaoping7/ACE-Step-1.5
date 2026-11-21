"""Tests for VRAM / host-memory batch planning (batch_plan.py, MemoryUtilsMixin.plan_batch_size)."""

import types
import unittest
from unittest.mock import patch

import torch

from acestep.core.generation.handler import batch_plan
from acestep.core.generation.handler import memory_utils
from acestep.core.generation.handler.memory_utils import MemoryUtilsMixin

# RTX 4060 8 GB, LM 0.6B (vllm) resident, offload + DiT offload: effective free VRAM
# before the DiT phase and the turbo DiT weights still on the CPU.
FREE_8GB = 5.12  # measured at planning time (LM 0.6B resident: 2.09 GB allocated)
DIT_GB = 4.46
PREFETCHED_GB = 3.33  # decoder + tokenizer + detokenizer prefetched during the LM


def _host(**kw):
    base = dict(device="cuda", offload_to_cpu=True, offload_dit_to_cpu=True, model=None, text_encoder=None,
                last_init_params={"config_path": "acestep-v15-turbo"})
    base.update(kw)
    return types.SimpleNamespace(**base)


class PlanBatchSizeTests(unittest.TestCase):
    def _plan(self, batch, duration, free=FREE_8GB, pending=DIT_GB, host_avail=7.0, host=None):
        with patch.object(batch_plan, "pending_weights_gb", return_value=pending), \
                patch.object(batch_plan, "host_available_gb", return_value=host_avail):
            return batch_plan.plan_batch_size(host or _host(), batch, duration, free_vram_gb=free)

    def test_limits_on_8gb(self):
        """Free VRAM before the first generation; each batch ran without OOM (qnn/results/opt/PLAN2_*)."""
        self.assertEqual(self._plan(8, 10)[0], 8)
        self.assertEqual(self._plan(8, 30)[0], 7)
        self.assertEqual(self._plan(8, 120)[0], 5)
        self.assertEqual(self._plan(8, 240)[0], 4)

    def test_derived_weight_leak_would_have_cost_batch(self):
        """0.17 GB of VAE weight-norm weights left on the GPU (fixed) shrink every budget."""
        self.assertLess(self._plan(8, 240, free=FREE_8GB - 0.17)[0], self._plan(8, 240)[0])

    def test_same_answer_before_and_after_dit_prefetch(self):
        """Prefetched weights move from 'pending' into 'allocated': the plan must not change."""
        for d in (10, 30, 120, 240, 480):
            before = self._plan(8, d)[0]
            after = self._plan(8, d, free=FREE_8GB - PREFETCHED_GB, pending=DIT_GB - PREFETCHED_GB)[0]
            self.assertEqual(before, after, d)

    def test_never_exceeds_request_and_never_below_one(self):
        self.assertEqual(self._plan(2, 30)[0], 2)
        self.assertEqual(self._plan(4, 30, free=DIT_GB + 0.1)[0], 1)

    def test_unknown_duration_assumes_tier_maximum(self):
        with patch.object(batch_plan, "_max_duration_s", return_value=480.0):
            planned, info = self._plan(8, None)
        self.assertTrue(info["duration_assumed"])
        self.assertEqual(planned, self._plan(8, 480)[0])
        self.assertLessEqual(planned, self._plan(8, 240)[0])

    def test_host_memory_caps_batch(self):
        planned, info = self._plan(8, 240, host_avail=1.8)  # (1.8 - 1.0) / (0.0015 * 240) = 2.2
        self.assertEqual(planned, 2)
        self.assertEqual(info["host_max"], 2)

    def test_resident_dit_needs_no_upload(self):
        """Without DiT offload the whole effective free VRAM goes to activations."""
        planned, info = self._plan(16, 30, free=1.2, pending=0.0)
        self.assertEqual(info["pending_weights_gb"], 0.0)
        self.assertEqual(planned, 16)

    def test_cfg_and_xl_models_cost_more_per_item(self):
        base = batch_plan.per_item_vram_gb(60, "turbo")
        self.assertAlmostEqual(batch_plan.per_item_vram_gb(60, "base"), 2 * base)
        self.assertAlmostEqual(batch_plan.per_item_vram_gb(60, "xl_turbo"), 1.7 * base)

    def test_batch_one_and_missing_free_vram_pass_through(self):
        self.assertEqual(batch_plan.plan_batch_size(_host(), 1, 240, free_vram_gb=0.1)[0], 1)
        self.assertEqual(batch_plan.plan_batch_size(_host(), 3, 240, free_vram_gb=None)[0], 3)


class PendingWeightsTests(unittest.TestCase):
    def test_counts_only_cpu_resident_weights_when_offloading(self):
        model = torch.nn.Linear(256, 256, bias=False)  # 256 KiB fp32 on CPU
        host = _host(model=model)
        self.assertAlmostEqual(batch_plan.pending_weights_gb(host), 256 * 256 * 4 / 2**30)
        self.assertEqual(batch_plan.pending_weights_gb(_host(model=model, offload_to_cpu=False)), 0.0)

    def test_text_encoder_counts_when_dit_is_resident(self):
        enc = torch.nn.Linear(128, 128, bias=False)
        host = _host(offload_dit_to_cpu=False, model=torch.nn.Linear(512, 512), text_encoder=enc)
        self.assertAlmostEqual(batch_plan.pending_weights_gb(host), 128 * 128 * 4 / 2**30)


class MixinTests(unittest.TestCase):
    def test_guard_uses_planner_and_device_index(self):
        host = _host(device="cuda:1")
        with patch.object(memory_utils, "get_effective_free_vram_gb", return_value=FREE_8GB) as free_mock, \
                patch.object(batch_plan, "pending_weights_gb", return_value=DIT_GB), \
                patch.object(batch_plan, "host_available_gb", return_value=7.0):
            self.assertEqual(MemoryUtilsMixin._vram_guard_reduce_batch(host, 8, audio_duration=240), 4)
        free_mock.assert_called_once_with(1)

    def test_non_cuda_device_is_not_limited(self):
        self.assertEqual(MemoryUtilsMixin.plan_batch_size(_host(device="cpu"), 8, 240), 8)


if __name__ == "__main__":
    unittest.main()
