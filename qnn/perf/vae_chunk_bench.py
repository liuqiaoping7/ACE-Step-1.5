"""VAE decode microbenchmark: chunk size / overlap / CPU offload vs time, peak VRAM, error.

Decodes a real 30 s latent (from qnn/results/opt/*/run0_wav.pt re-encoded) with the
handler's own tiled decode at several chunk sizes, plus a full (untiled) decode.

Usage: python qnn/perf/vae_chunk_bench.py [--frames 750]
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402


def timed(fn, reps=5):
    ts = []
    out = None
    for _ in range(reps):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    return out, statistics.median(ts), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=750)
    ap.add_argument("--wav", default=os.path.join(ROOT, "qnn", "results", "opt", "O5_no_gc_dit", "run0_wav.pt"))
    args = ap.parse_args()
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=False, offload_dit_to_cpu=False)
    assert ok, st
    # free the DiT / text encoder so the VAE has the GPU to itself (as after DiT offload)
    h.model.to("cpu"); h.text_encoder.to("cpu"); torch.cuda.empty_cache()
    wav = torch.load(args.wav).float()
    torch.manual_seed(0)
    with torch.inference_mode():
        lat = h._encode_audio_to_latents(wav.to(h.device))  # [T, 64]
    lat = lat[: args.frames].T.unsqueeze(0).contiguous().to(h.vae.dtype)  # [1, 64, T]
    print("latent", tuple(lat.shape), lat.dtype, "free GB", torch.cuda.mem_get_info()[0] / 2**30)

    res = {}
    with torch.inference_mode():
        ref, t, pk = timed(lambda: h.tiled_decode(lat, chunk_size=128, overlap=64, offload_wav_to_cpu=True))
        ref = ref.float().cpu()
        res["chunk128_ov64->32_cpu(current)"] = {"s": round(t, 4), "peak_gb": round(pk, 3), "maxabs_vs_ref": 0.0}
        for cs, ov, off in [(128, 64, False), (256, 64, True), (256, 64, False), (384, 64, False),
                            (512, 64, False), (768, 64, False)]:
            out, t, pk = timed(lambda: h.tiled_decode(lat, chunk_size=cs, overlap=ov, offload_wav_to_cpu=off))
            d = (out.float().cpu() - ref).abs()
            res[f"chunk{cs}_ov{ov}_{'cpu' if off else 'gpu'}"] = {
                "s": round(t, 4), "peak_gb": round(pk, 3), "maxabs_vs_ref": round(d.max().item(), 5),
                "meanabs_vs_ref": round(d.mean().item(), 6)}
            print(cs, ov, off, res[f"chunk{cs}_ov{ov}_{'cpu' if off else 'gpu'}"], flush=True)
        out, t, pk = timed(lambda: h.vae.decode(lat).sample)
        d = (out.float().cpu() - ref).abs()
        res["full_untiled"] = {"s": round(t, 4), "peak_gb": round(pk, 3), "maxabs_vs_ref": round(d.max().item(), 5),
                               "meanabs_vs_ref": round(d.mean().item(), 6)}
    print(json.dumps(res, indent=1))
    out_dir = os.path.join(ROOT, "qnn", "results", "opt", "vae_chunk_bench")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
