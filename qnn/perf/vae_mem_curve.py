"""Peak VRAM of a single Oobleck VAE decode window vs window length (bf16, fused Snake, cudnn.deterministic).

Measures what _tiled_decode_inner actually holds: one window decode plus the CPU-offloaded output.
Usage: python qnn/perf/vae_mem_curve.py
"""
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402


def main():
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=False, offload_dit_to_cpu=False)
    assert ok, st
    h.model.to("cpu"); h.text_encoder.to("cpu"); torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    res = {}
    with torch.inference_mode():
        for n in (128, 256, 384, 512, 768, 1024, 1280):
            lat = torch.randn(1, 64, n, device="cuda", dtype=h.vae.dtype)
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t = time.perf_counter()
            try:
                out = h.vae.decode(lat).sample
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                peak = (torch.cuda.max_memory_allocated() - base) / 2**30
                res[n] = {"peak_gb": round(peak, 3), "gb_per_frame_MiB": round(peak * 1024 / n, 3),
                          "s": round(dt, 3), "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
                del out
            except torch.cuda.OutOfMemoryError as e:
                res[n] = {"oom": True}
                torch.cuda.empty_cache()
            print(n, res[n], flush=True)
    os.makedirs(os.path.join(ROOT, "qnn", "results", "opt", "vae_mem_curve"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "qnn", "results", "opt", "vae_mem_curve", "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
