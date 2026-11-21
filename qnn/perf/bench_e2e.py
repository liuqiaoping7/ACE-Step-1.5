"""End-to-end benchmark for ACE-Step 1.5 (text2music, --batch songs per call, default 1).

Cold start (model init / compile) is timed separately; then 1 warmup + N measured runs
in the same process.  Saves per-run timings, peak VRAM, audio and LM codes for correctness.

Usage:
  python qnn/perf/bench_e2e.py --tag B0 --lm acestep-5Hz-lm-0.6B --backend pt --offload
Output: qnn/results/opt/<tag>/result.json (+ run0.wav, run0_codes.txt)
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import wave

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

CAPTION = "An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal"
LYRICS = ("[Verse]\nCity lights are calling me\nNeon rivers in the street\n"
          "[Chorus]\nWe are dancing in the night\nHold me close and hold me tight")


def git_rev():
    try:
        rev = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def host_mem_gb():
    out = {}
    for line in open("/proc/self/status"):
        k = line.split(":")[0]
        if k in ("VmRSS", "VmHWM", "VmSwap"):
            out[k] = round(int(line.split()[1]) / 2**20, 2)
    return out


def save_wav(path, t):
    x = t.float().clamp(-1, 1).cpu().numpy().T
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lm", default="acestep-5Hz-lm-0.6B", help="'none' disables the LM")
    ap.add_argument("--backend", default="pt", choices=["pt", "vllm"])
    ap.add_argument("--offload", action="store_true", help="offload_to_cpu (DiT handler + LM)")
    ap.add_argument("--offload-dit", action="store_true")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--flash", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1, help="GenerationConfig.batch_size (songs per call)")
    ap.add_argument("--no-batch-guard", action="store_true",
                    help="bypass handler._vram_guard_reduce_batch (it measures free VRAM after the DiT prefetch and caps batch>tier limit to 1)")
    ap.add_argument("--lm-chunk", type=int, default=8, help="GenerationConfig.lm_batch_chunk_size")
    ap.add_argument("--cprofile", action="store_true", help="cProfile the last measured run")
    ap.add_argument("--phases", action="store_true", help="synchronised wall time per pipeline function (last run)")
    ap.add_argument("--nsys-range", action="store_true", help="cudaProfilerStart/Stop around the last run (nsys --capture-range=cudaProfilerApi)")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "qnn", "results", "opt", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=args.flash, compile_model=args.compile,
                                  offload_to_cpu=args.offload, offload_dit_to_cpu=args.offload_dit,
                                  quantization=args.quant)
    assert ok, st
    t_dit_init = time.perf_counter() - t0
    llm = None
    t_lm_init = 0.0
    if args.lm != "none":
        t1 = time.perf_counter()
        llm = LLMHandler()
        st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path=args.lm,
                                backend=args.backend, device="cuda", offload_to_cpu=args.offload, dtype=None)
        assert ok, st
        t_lm_init = time.perf_counter() - t1

    use_lm = llm is not None
    PH = {}
    PK, PK_STACK = {}, []  # per-phase peak allocated bytes (absolute), see wrap()

    def wrap(obj, attr, name):
        if obj is None or not hasattr(obj, attr):
            return
        fn = getattr(obj, attr)
        import functools

        @functools.wraps(fn)  # keep the signature: inference.py filters kwargs via inspect.signature
        def w(*a, **k):
            # Per-phase peak VRAM, nesting-safe: each frame keeps the max seen so far; a child resets
            # the device counter on entry and hands its peak up to the parent on exit.
            torch.cuda.synchronize(); t_ = time.perf_counter()
            if PK_STACK:
                PK_STACK[-1] = max(PK_STACK[-1], torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            PK_STACK.append(torch.cuda.memory_allocated())
            try:
                return fn(*a, **k)
            finally:
                torch.cuda.synchronize()
                d = PH.setdefault(name, [0, 0.0]); d[0] += 1; d[1] += time.perf_counter() - t_
                pk = max(PK_STACK.pop(), torch.cuda.max_memory_allocated())
                PK[name] = max(PK.get(name, 0), pk)
                if PK_STACK:
                    PK_STACK[-1] = max(PK_STACK[-1], pk)
        setattr(obj, attr, w)

    if args.phases:
        import acestep.inference as inf_mod
        for obj, attr, name in [
            (llm, "generate_with_stop_condition", "LM.generate_with_stop_condition"),
            (h, "generate_music", "H.generate_music(DiT total)"),
            (h, "_prepare_generate_music_service_inputs", "H.prepare_service_inputs"),
            (h, "service_generate", "H.service_generate"),
            (h, "_prepare_batch", "H._prepare_batch"),
            (h, "_prepare_target_latents_and_wavs", "H.prepare_target_latents"),
            (h, "_decode_audio_codes_to_latents", "H.decode_codes"),
            (h, "preprocess_batch", "H.preprocess_batch"),
            (h, "infer_text_embeddings", "H.text_encoder"),
            (h, "infer_refer_latent", "H.refer_latent"),
            (h.model, "prepare_condition", "M.prepare_condition"),
            (h.model, "generate_audio", "M.generate_audio"),
            (h, "_decode_generate_music_pred_latents", "H.vae_decode_total"),
            (h, "tiled_decode", "H.tiled_decode"),
            (h, "_build_generate_music_success_payload", "H.build_payload"),
        ]:
            wrap(obj, attr, name)
    if args.no_batch_guard:
        h._vram_guard_reduce_batch = lambda bs, *a, **k: bs
    params = GenerationParams(task_type="text2music", caption=CAPTION, lyrics=LYRICS, duration=args.duration,
                              inference_steps=8, shift=3.0, seed=args.seed, thinking=use_lm,
                              use_cot_metas=use_lm, use_cot_caption=use_lm, use_cot_language=use_lm,
                              **({} if use_lm else dict(bpm=100, keyscale="E♭ minor", timesignature="4",
                                                        vocal_language="en")))
    cfg = GenerationConfig(batch_size=args.batch, use_random_seed=False, audio_format="wav",
                           seeds=[args.seed + k for k in range(args.batch)] if args.batch > 1 else None,
                           lm_batch_chunk_size=args.lm_chunk)
    runs = []
    for i in range(args.runs + 1):  # first = warmup
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)  # LM sampling uses global RNG
        np.random.seed(args.seed)
        import random as _r; _r.seed(args.seed)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        prof = None
        if args.cprofile and i == args.runs:
            import cProfile; prof = cProfile.Profile(); prof.enable()
        nsys_on = args.nsys_range and i == args.runs
        if nsys_on:
            torch.cuda.cudart().cudaProfilerStart()
        t = time.perf_counter()
        with torch.cuda.nvtx.range(f"generate_music_run{i}"):
            res = generate_music(h, llm, params, cfg, save_dir=None)
        torch.cuda.synchronize()
        if nsys_on:
            torch.cuda.cudart().cudaProfilerStop()
        wall = time.perf_counter() - t
        if prof is not None:
            prof.disable()
            import pstats, io
            sio = io.StringIO(); pstats.Stats(prof, stream=sio).sort_stats("cumulative").print_stats(60)
            open(os.path.join(out_dir, "cprofile_cum.txt"), "w").write(sio.getvalue())
            sio = io.StringIO(); pstats.Stats(prof, stream=sio).sort_stats("tottime").print_stats(40)
            open(os.path.join(out_dir, "cprofile_tot.txt"), "w").write(sio.getvalue())
            prof.dump_stats(os.path.join(out_dir, "cprofile.prof"))
        assert res.success, res.error
        a = res.audios[0]
        codes = a["params"].get("audio_codes") or ""
        tc = {k: round(float(v), 4) for k, v in (res.extra_outputs.get("time_costs") or {}).items()
              if isinstance(v, (int, float))}
        run_peak = max([torch.cuda.max_memory_allocated()] + list(PK.values()))  # phases reset the counter
        r = {"run": i - 1, "warmup": i == 0, "wall_s": round(wall, 4), "time_costs": tc,
             "peak_alloc_gb": round(run_peak / 2**30, 3),
             "phase_peak_gb": {k: round(v / 2**30, 3) for k, v in PK.items()},
             "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
             "host_gb": host_mem_gb(),
             "codes_md5": hashlib.md5(codes.encode()).hexdigest()[:10], "n_codes": codes.count("<|audio_code_"),
             "wav_md5": hashlib.md5(a["tensor"].float().cpu().numpy().tobytes()).hexdigest()[:10],
             "n_audios": len(res.audios),
             "items": [{"n_codes": (x["params"].get("audio_codes") or "").count("<|audio_code_"),
                        "codes_md5": hashlib.md5((x["params"].get("audio_codes") or "").encode()).hexdigest()[:10],
                        "wav_md5": hashlib.md5(x["tensor"].float().cpu().numpy().tobytes()).hexdigest()[:10]}
                       for x in res.audios]}
        runs.append(r)
        print(json.dumps(r), flush=True)
        if args.phases and i == args.runs:
            with open(os.path.join(out_dir, "phases.json"), "w") as f:
                json.dump({k: {"calls": v[0], "s": round(v[1], 4)} for k, v in PH.items()}, f, indent=1)
            print("PHASES", json.dumps({k: round(v[1], 3) for k, v in PH.items()}), flush=True)
        PH.clear(); PK.clear()
        if i == 1:
            save_wav(os.path.join(out_dir, "run0.wav"), a["tensor"])
            torch.save(a["tensor"].float().cpu(), os.path.join(out_dir, "run0_wav.pt"))
            with open(os.path.join(out_dir, "run0_codes.txt"), "w") as f:
                f.write(codes)
    meas = [r for r in runs if not r["warmup"]]
    walls = [r["wall_s"] for r in meas]
    keys = sorted({k for r in meas for k in r["time_costs"]})
    summary = {
        "wall_median_s": round(statistics.median(walls), 3),
        "wall_min_s": round(min(walls), 3), "wall_max_s": round(max(walls), 3),
        "wall_cv": round(statistics.pstdev(walls) / statistics.mean(walls), 4) if len(walls) > 1 else 0,
        "time_costs_median": {k: round(statistics.median(r["time_costs"].get(k, 0) for r in meas), 3) for k in keys},
        "peak_alloc_gb_max": max(r["peak_alloc_gb"] for r in meas),
        "peak_reserved_gb_max": max(r["peak_reserved_gb"] for r in meas),
        "host_rss_peak_gb": max(r["host_gb"].get("VmHWM", 0) for r in runs),
        "host_swap_gb_max": max(r["host_gb"].get("VmSwap", 0) for r in runs),
        "cold_start_s": {"dit_init": round(t_dit_init, 2), "lm_init": round(t_lm_init, 2),
                         "warmup_run": runs[0]["wall_s"]},
    }
    meta = {"commit": git_rev(), "cmd": " ".join(sys.argv), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "python": platform.python_version(),
            "args": vars(args)}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"meta": meta, "summary": summary, "runs": runs}, f, indent=1)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
