"""Fused Snake activation for the Oobleck VAE (Triton).

diffusers' ``Snake1d`` computes ``x + 1/(beta+1e-9) * sin(alpha*x)**2`` as five
separate element-wise kernels (mul, sin, pow, mul, add), each reading and writing
the full activation in bf16.  In the VAE decoder these activations reach
128 ch x 1.44 M samples (368 MB) and the element-wise kernels take ~60% of the
decode time on an RTX 4060.  This module replaces the forward of every Snake1d
instance with a single Triton kernel (one read, one write; math in fp32).

``patch_vae_snake(vae)`` is a no-op when Triton/CUDA are unavailable or when
``ACESTEP_FUSED_SNAKE=0``.
"""

from __future__ import annotations

import os
import types

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _snake_kernel(x_ptr, y_ptr, alpha_ptr, inv_beta_ptr, C, T, BLOCK: tl.constexpr):
        row = tl.program_id(0)  # over B*C rows
        blk = tl.program_id(1)  # over T blocks
        c = row % C
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T
        base = row.to(tl.int64) * T
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(alpha_ptr + c)
        ib = tl.load(inv_beta_ptr + c)
        s = tl.sin(a * x)
        y = x + ib * s * s
        tl.store(y_ptr + base + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_snake_enabled() -> bool:
    return _HAS_TRITON and torch.cuda.is_available() and os.environ.get("ACESTEP_FUSED_SNAKE", "1") != "0"


def snake_fused(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, logscale: bool) -> torch.Tensor:
    """y = x + 1/(beta+1e-9) * sin(alpha*x)^2 with per-channel alpha/beta ([1, C, 1])."""
    shape = x.shape
    B, C = shape[0], shape[1]
    xs = x.reshape(B, C, -1)
    if not xs.is_contiguous():
        xs = xs.contiguous()
    T = xs.shape[2]
    a = alpha.reshape(C).float()
    b = beta.reshape(C).float()
    if logscale:
        a = torch.exp(a)
        b = torch.exp(b)
    inv_b = (b + 1e-9).reciprocal()
    y = torch.empty_like(xs)
    BLOCK = 1024
    grid = (B * C, triton.cdiv(T, BLOCK))
    _snake_kernel[grid](xs, y, a.contiguous(), inv_b.contiguous(), C, T, BLOCK=BLOCK)
    return y.reshape(shape)


def _fused_forward(self, hidden_states):
    if hidden_states.is_cuda and not torch.is_grad_enabled():
        return snake_fused(hidden_states, self.alpha, self.beta, self.logscale)
    return self._acestep_orig_forward(hidden_states)


def patch_vae_snake(vae) -> int:
    """Patch every diffusers Snake1d in *vae* to use the fused kernel. Returns #patched."""
    if vae is None or not fused_snake_enabled():
        return 0
    n = 0
    for m in vae.modules():
        if type(m).__name__ == "Snake1d" and hasattr(m, "alpha") and hasattr(m, "beta"):
            if getattr(m, "_acestep_fused_snake", False):
                continue
            m._acestep_orig_forward = m.forward
            m.forward = types.MethodType(_fused_forward, m)
            m._acestep_fused_snake = True
            n += 1
    return n
