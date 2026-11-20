"""Kernel-level profile of one DiT step and one full VAE decode, plus measured bf16 GEMM peak.

Usage: python qnn/perf/kernel_profile.py
Output: qnn/results/opt/kernel_profile/{result.json,dit_kernels.txt,vae_kernels.txt}
"""
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402

OUT = os.path.join(ROOT, "qnn", "results", "opt", "kernel_profile")


def gemm_peak():
    res = {}
    for (m, k, n) in [(4096, 4096, 4096), (375, 2048, 6144), (375, 6144, 2048), (750, 2048, 2048)]:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        it = 50
        t = time.perf_counter()
        for _ in range(it):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / it
        res[f"{m}x{k}x{n}"] = round(2 * m * k * n / dt / 1e12, 1)
    # bandwidth
    x = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        y.copy_(x)
    torch.cuda.synchronize()
    res["copy_GBps"] = round(2 * x.numel() * 20 / (time.perf_counter() - t) / 1e9, 1)
    return res


def categorize(name):
    n = name.lower()
    if "gemm" in n or "sm80_xmma" in n or "cutlass" in n or "ampere" in n or "sm90" in n or "matmul" in n or "gemv" in n:
        return "gemm"
    if "flash" in n or "fmha" in n or "attention" in n or "sdpa" in n or "efficient" in n:
        return "attention"
    if "conv" in n or "implicit" in n or "winograd" in n or "fft" in n or "cudnn" in n or "xmma_fprop" in n or "dgrad" in n:
        return "conv"
    if "copy" in n or "memcpy" in n or "cat" in n:
        return "copy/cat"
    if "reduce" in n or "norm" in n or "mean" in n:
        return "reduce/norm"
    return "elementwise/other"


def summarize(prof, path):
    rows, cats = [], {}
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        if not t or e.key.startswith("aten::") or e.key.startswith("cuda") or e.device_type.name != "CUDA":
            continue
        rows.append((t / 1000, e.count, e.key))
        c = categorize(e.key)
        cats[c] = cats.get(c, 0) + t / 1000
    rows.sort(reverse=True)
    with open(path, "w") as f:
        for t, c, k in rows[:40]:
            f.write(f"{t:9.2f} ms {c:5d}  {k[:150]}\n")
    total = sum(cats.values())
    return {"total_ms": round(total, 2), "by_category_ms": {k: round(v, 2) for k, v in sorted(cats.items(), key=lambda x: -x[1])}}


def main():
    os.makedirs(OUT, exist_ok=True)
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=False, offload_dit_to_cpu=False)
    assert ok, st
    h.text_encoder.to("cpu"); torch.cuda.empty_cache()
    res = {"gemm_TFLOPS_and_bw": gemm_peak()}
    print(res, flush=True)
    m = h.model
    dev, dt = h.device, h.dtype
    T = 750
    xt = torch.randn(1, T, 64, device=dev, dtype=dt)
    ctx = torch.randn(1, T, 128, device=dev, dtype=dt)
    enc = torch.randn(1, 164, 2048, device=dev, dtype=dt)
    enc_mask = torch.ones(1, 164, device=dev, dtype=torch.bool)
    att = torch.ones(1, T, device=dev, dtype=dt)
    t1 = torch.ones(1, device=dev, dtype=dt)

    def dit_step():
        return m.decoder(hidden_states=xt, timestep=t1, timestep_r=t1, attention_mask=att,
                         encoder_hidden_states=enc, encoder_attention_mask=enc_mask,
                         context_latents=ctx, use_cache=False)[0]

    with torch.inference_mode():
        for _ in range(3):
            dit_step()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            dit_step()
        torch.cuda.synchronize()
        res["dit_step_ms"] = round((time.perf_counter() - t) / 10 * 1000, 2)
        flops = 2 * 1.5104e9 * (T // 2)  # 24 layers' params x tokens (ignores attention score FLOPs)
        res["dit_step_achieved_TFLOPS_linear_only"] = round(flops / (res["dit_step_ms"] / 1000) / 1e12, 1)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            dit_step(); torch.cuda.synchronize()
        res["dit_step_kernels"] = summarize(prof, os.path.join(OUT, "dit_kernels.txt"))
        print("DiT", res["dit_step_ms"], res["dit_step_kernels"], flush=True)

        h.model.to("cpu"); torch.cuda.empty_cache()
        lat = torch.randn(1, 64, T, device=dev, dtype=h.vae.dtype)
        torch.backends.cudnn.deterministic = True
        for _ in range(2):
            h.vae.decode(lat)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(3):
            h.vae.decode(lat)
        torch.cuda.synchronize()
        res["vae_full_decode_ms"] = round((time.perf_counter() - t) / 3 * 1000, 1)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            h.vae.decode(lat); torch.cuda.synchronize()
        res["vae_kernels"] = summarize(prof, os.path.join(OUT, "vae_kernels.txt"))
        print("VAE", res["vae_full_decode_ms"], res["vae_kernels"], flush=True)
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
