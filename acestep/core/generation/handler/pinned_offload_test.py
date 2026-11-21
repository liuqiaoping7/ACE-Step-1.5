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

    def test_prefetch_then_full_load_adopts_resident_tensors(self):
        po.load_to_device(self.net, "cuda"); po.offload_to_cpu(self.net)  # create masters
        master_ptr = self.net.lin.weight.data_ptr()
        po.prefetch_to_device(self.net.lin, "cuda")
        self.assertTrue(po.is_prefetched(self.net))
        po.load_to_device(self.net, "cuda")  # waits for the prefetch, loads the rest
        self.assertFalse(po.is_prefetched(self.net))
        self._assert_on("cuda")
        y = self.net.lin(torch.ones(1, 4, device="cuda"))
        ref = torch.ones(1, 4) @ self.ref["lin.weight"].T + self.ref["lin.bias"]
        torch.testing.assert_close(y.cpu(), ref)
        po.offload_to_cpu(self.net)
        self.assertEqual(self.net.lin.weight.data_ptr(), master_ptr)  # no new host copy

    def test_submodule_load_then_full_offload_shares_masters(self):
        po.load_to_device(self.net, "cuda"); po.offload_to_cpu(self.net)
        ptr = self.net.lin.weight.data_ptr()
        po.load_to_device(self.net.lin, "cuda")
        po.offload_to_cpu(self.net)  # whole-model offload of a submodule-loaded tensor
        self.assertEqual(self.net.lin.weight.data_ptr(), ptr)
        for k, v in self.net.state_dict().items():
            self.assertTrue(torch.equal(v, self.ref[k]), k)

    def test_module_bytes(self):
        n = sum(t.numel() * t.element_size() for t in list(self.net.parameters()) + list(self.net.buffers()))
        self.assertEqual(po.module_bytes(self.net), n)
        po.load_to_device(self.net.lin, "cuda")
        self.assertLess(po.module_bytes(self.net, only_on_cpu=True), n)
        po.offload_to_cpu(self.net)

    def test_cpu_target_is_not_supported(self):
        self.assertFalse(po.supports(self.net, "cpu"))
        self.assertTrue(po.supports(self.net, "cuda"))

    def test_disabled_by_env(self):
        os.environ["ACESTEP_PINNED_OFFLOAD"] = "0"
        try:
            self.assertFalse(po.supports(self.net))
        finally:
            del os.environ["ACESTEP_PINNED_OFFLOAD"]



@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class ReleaseDerivedWeightsTests(unittest.TestCase):
    """Legacy weight_norm caches ``weight`` as a plain attribute that offloading used to leave on the GPU."""

    def test_offload_releases_weight_norm_weight_and_forward_recomputes_it(self):
        import warnings
        from torch.nn.utils import weight_norm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            conv = weight_norm(torch.nn.Conv1d(8, 8, 3)).cuda()
        x = torch.randn(1, 8, 16, device="cuda")
        ref = conv(x)
        self.assertTrue(conv.weight.is_cuda)
        po.offload_to_cpu(conv)
        released = po.release_derived_weights(conv)
        self.assertEqual(released, 8 * 8 * 3 * 4)
        self.assertIsNone(conv.weight)
        self.assertEqual(po.release_derived_weights(conv), 0)
        po.load_to_device(conv, "cuda")
        torch.testing.assert_close(conv(x), ref, rtol=0, atol=0)

if __name__ == "__main__":
    unittest.main()
