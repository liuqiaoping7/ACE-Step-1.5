"""Tests for the fused Triton Snake activation."""

import unittest

import torch

from acestep.models.common import fused_snake as fs


@unittest.skipUnless(fs.fused_snake_enabled(), "CUDA + Triton required")
class FusedSnakeTests(unittest.TestCase):
    def _ref(self, x, alpha, beta, logscale):
        # fp32 math on the given (possibly bf16) inputs, as the kernel does
        a = torch.exp(alpha.float()) if logscale else alpha.float()
        b = torch.exp(beta.float()) if logscale else beta.float()
        xs = x.reshape(x.shape[0], x.shape[1], -1).float()
        y = xs + (b.float() + 1e-9).reciprocal() * torch.sin(a.float() * xs).pow(2)
        return y.reshape(x.shape)

    def test_matches_fp32_reference(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for shape in [(1, 4, 1000), (2, 8, 4097), (1, 3, 5, 7)]:
                for logscale in (True, False):
                    x = torch.randn(*shape, device="cuda", dtype=dtype) * 3
                    alpha = torch.randn(1, shape[1], 1, device="cuda", dtype=dtype) * 0.5
                    beta = torch.randn(1, shape[1], 1, device="cuda", dtype=dtype) * 0.5
                    if not logscale:
                        alpha, beta = alpha.abs() + 0.5, beta.abs() + 0.5
                    y = fs.snake_fused(x, alpha, beta, logscale)
                    ref = self._ref(x, alpha, beta, logscale)
                    self.assertEqual(y.shape, x.shape)
                    self.assertEqual(y.dtype, dtype)
                    if dtype == torch.float32:
                        torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
                    else:
                        # math is fp32 with one final rounding: within 1 bf16 ulp
                        err = (y.float() - ref).abs()
                        self.assertTrue(bool((err <= ref.abs() * 2.0 ** -7 + 1e-6).all()),
                                        f"max err {err.max().item()}")

    def test_patch_vae_uses_fused_forward_and_falls_back_on_cpu(self):
        from diffusers.models.autoencoders.autoencoder_oobleck import Snake1d

        m = Snake1d(6).cuda()
        with torch.no_grad():
            m.alpha.normal_(0, 0.3); m.beta.normal_(0, 0.3)
        holder = torch.nn.Sequential(m)
        x = torch.randn(2, 6, 333, device="cuda")
        with torch.no_grad():
            ref = m(x)
            self.assertEqual(fs.patch_vae_snake(holder), 1)
            self.assertEqual(fs.patch_vae_snake(holder), 0)  # idempotent
            torch.testing.assert_close(m(x), ref, rtol=1e-5, atol=1e-5)
            m.cpu()
            torch.testing.assert_close(m(x.cpu()), ref.cpu(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
