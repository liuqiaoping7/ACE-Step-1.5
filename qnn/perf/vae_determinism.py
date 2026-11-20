"""Is the VAE decode deterministic?  Compare repeated decodes, with/without cudnn.deterministic.

Usage: python qnn/perf/vae_determinism.py
"""
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402


def run(h, lat, chunk, reps=4):
    outs, ts = [], []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        o = h.tiled_decode(lat, chunk_size=chunk, overlap=64, offload_wav_to_cpu=True)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        outs.append(o.float().cpu())
    diffs = [(outs[0] - o).abs().max().item() for o in outs[1:]]
    return {"s": round(statistics.median(ts), 4), "max_run_to_run_diff": max(diffs),
            "bit_identical": all(torch.equal(outs[0], o) for o in outs[1:])}


def main():
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=False, offload_dit_to_cpu=False)
    assert ok, st
    h.model.to("cpu"); h.text_encoder.to("cpu"); torch.cuda.empty_cache()
    wav = torch.load(os.path.join(ROOT, "qnn", "results", "opt", "O5_no_gc_dit", "run0_wav.pt")).float()
    torch.manual_seed(0)
    with torch.inference_mode():
        lat = h._encode_audio_to_latents(wav.to(h.device))
    lat = lat.T.unsqueeze(0).contiguous().to(h.vae.dtype)
    res = {"cudnn.benchmark": torch.backends.cudnn.benchmark, "cudnn.deterministic": torch.backends.cudnn.deterministic}
    with torch.inference_mode():
        for det in (False, True):
            torch.backends.cudnn.deterministic = det
            for chunk in (128, 384, 1300):
                res[f"det={det}_chunk{chunk}"] = run(h, lat, chunk)
                print(f"det={det} chunk={chunk}", res[f"det={det}_chunk{chunk}"], flush=True)
    os.makedirs(os.path.join(ROOT, "qnn", "results", "opt", "vae_determinism"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "qnn", "results", "opt", "vae_determinism", "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
