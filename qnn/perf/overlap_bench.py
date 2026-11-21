"""Upper bound for cross-request pipelining: LM (request k+1) concurrent with DiT diffusion (request k).

The LM runs on the default stream in the main thread (nano-vllm CUDA graphs); the 8 DiT decoder
steps run in a second thread on a non-blocking side stream.  Both models are resident, exactly as in
the offload pipeline while the DiT is prefetched (LM weights + KV stay on the GPU throughout).
Measures solo LM, solo DiT and concurrent wall time.  VAE is not included (its window would not fit
next to the resident DiT + LM on 8 GB).

Usage: python qnn/perf/overlap_bench.py --durations 30 120 240
Output: qnn/results/opt/overlap_bench/result.json
"""
import argparse
import json
import os
import sys
import threading
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.core.generation.handler import pinned_offload as po  # noqa: E402

CAPTION = "An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal"
LYRICS = "[Verse]\nCity lights are calling me\n[Chorus]\nWe are dancing in the night"


def run_lm(llm, dur):
    r = llm.generate_with_stop_condition(caption=CAPTION, lyrics=LYRICS, infer_type="llm_dit", temperature=0.85,
                                         cfg_scale=2.0, target_duration=dur, use_constrained_decoding=True,
                                         batch_size=1, seeds=[42])
    assert r.get("success", True), r.get("error")


def make_dit(h, frames):
    dec, dev, dt = h.model.decoder, h.device, h.dtype
    g = torch.Generator(device="cuda").manual_seed(frames)
    xt = torch.randn(1, frames, 64, device=dev, dtype=dt, generator=g)
    ctx = torch.randn(1, frames, 128, device=dev, dtype=dt, generator=g)
    enc = torch.randn(1, 164, 2048, device=dev, dtype=dt, generator=g)
    enc_mask = torch.ones(1, 164, device=dev, dtype=torch.bool)
    att = torch.ones(1, frames, device=dev, dtype=dt)

    def run(stream=None):
        with torch.inference_mode(), torch.cuda.stream(stream or torch.cuda.current_stream()):
            for i in range(8):
                t = torch.full((1,), 1.0 - i / 8, device=dev, dtype=dt)
                dec(hidden_states=xt, timestep=t, timestep_r=t, attention_mask=att, encoder_hidden_states=enc,
                    encoder_attention_mask=enc_mask, context_latents=ctx, use_cache=False)
            (stream or torch.cuda.current_stream()).synchronize()
    return run


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", type=int, nargs="+", default=[30, 120, 240])
    args = ap.parse_args()
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=True, compile_model=False,
                                  offload_to_cpu=True, offload_dit_to_cpu=True)
    assert ok, st
    llm = LLMHandler()
    st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                            backend="vllm", device="cuda", offload_to_cpu=True, dtype=None)
    assert ok, st
    po.load_to_device(h.model.decoder, "cuda", h.dtype)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    side = torch.cuda.Stream()
    res = {}
    for dur in args.durations:
        dit = make_dit(h, dur * 25)
        run_lm(llm, dur); dit()  # warmup
        lm_s = min(timed(lambda: run_lm(llm, dur)) for _ in range(2))
        dit_s = min(timed(dit) for _ in range(2))
        conc = []
        for _ in range(2):
            out = {}

            def worker():
                t = time.perf_counter(); dit(side); out["dit"] = time.perf_counter() - t
            th = threading.Thread(target=worker)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            th.start(); t1 = time.perf_counter(); run_lm(llm, dur); out["lm"] = time.perf_counter() - t1
            th.join(); torch.cuda.synchronize(); out["wall"] = time.perf_counter() - t0
            conc.append(out)
        best = min(conc, key=lambda o: o["wall"])
        res[dur] = {"lm_solo_s": round(lm_s, 3), "dit8_solo_s": round(dit_s, 3), "serial_sum_s": round(lm_s + dit_s, 3),
                    "concurrent_wall_s": round(best["wall"], 3), "lm_in_conc_s": round(best["lm"], 3),
                    "dit_in_conc_s": round(best["dit"], 3),
                    "saved_s": round(lm_s + dit_s - best["wall"], 3),
                    "ideal_saved_s": round(min(lm_s, dit_s), 3),
                    "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        print("OVERLAP", dur, json.dumps(res[dur]), flush=True)
    os.makedirs(os.path.join(ROOT, "qnn", "results", "opt", "overlap_bench"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "qnn", "results", "opt", "overlap_bench", "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
