"""Tests for pinned-master CPU<->GPU offload."""

import os
import unittest

import torch
from torch import nn

from acestep.core.generation.handler import pinned_offload as po


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.emb.weight  # tied
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.arange(4, dtype=torch.float32))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PinnedOffloadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = _Net()
        self.ref = {k: v.clone() for k, v in self.net.state_dict().items()}

    def _assert_on(self, device_type):
        for t in list(self.net.parameters()) + list(self.net.buffers()):
            self.assertEqual(t.device.type, device_type)

    def test_round_trip_preserves_values_and_ties(self):
        self.assertTrue(po.supports(self.net))
        for _ in range(3):
            po.load_to_device(self.net, "cuda")
            self._assert_on("cuda")
            self.assertIs(self.net.head.weight, self.net.emb.weight)
            po.offload_to_cpu(self.net)
            self._assert_on("cpu")
            for k, v in self.net.state_dict().items():
                self.assertTrue(torch.equal(v, self.ref[k]), k)
        self.assertTrue(self.net.lin.weight.is_pinned())

    def test_offload_does_not_reallocate_host_memory(self):
        po.load_to_device(self.net, "cuda")
        po.offload_to_cpu(self.net)
        ptr = self.net.lin.weight.data_ptr()
        po.load_to_device(self.net, "cuda")
        po.offload_to_cpu(self.net)
        self.assertEqual(ptr, self.net.lin.weight.data_ptr())

    def test_in_place_update_on_gpu_is_written_back(self):
        po.load_to_device(self.net, "cuda")
        with torch.no_grad():
            self.net.lin.weight.add_(1.0)
        po.offload_to_cpu(self.net)
        self.assertTrue(torch.equal(self.net.lin.weight, self.ref["lin.weight"] + 1.0))

    def test_works_inside_inference_mode(self):
        with torch.inference_mode():
            po.load_to_device(self.net, "cuda")
            y = self.net.lin(torch.ones(1, 4, device="cuda"))
            po.offload_to_cpu(self.net)
        self.assertEqual(y.shape, (1, 4))
        self.assertTrue(torch.equal(self.net.lin.weight, self.ref["lin.weight"]))

    def test_dtype_cast_on_load(self):
        po.load_to_device(self.net, "cuda", torch.bfloat16)
        self.assertEqual(self.net.lin.weight.dtype, torch.bfloat16)
        po.offload_to_cpu(self.net)
        self.assertEqual(self.net.lin.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(self.net.lin.weight, self.ref["lin.weight"].to(torch.bfloat16)))

    def test_disabled_by_env(self):
        os.environ["ACESTEP_PINNED_OFFLOAD"] = "0"
        try:
            self.assertFalse(po.supports(self.net))
        finally:
            del os.environ["ACESTEP_PINNED_OFFLOAD"]


if __name__ == "__main__":
    unittest.main()
