"""Compare LM output quality between the pt and nano-vllm (vllm) backends.

gen:   python qnn/perf/lm_backend_quality.py gen --backend pt|vllm
       -> qnn/results/opt/lm_backend_quality/<backend>.json (codes + metadata per prompt/seed)
score: python qnn/perf/lm_backend_quality.py score
       -> scores both sets with the same pt HF scorer (repo PMI reward, lm_score.py)
          plus code statistics and metadata validity; writes summary.json
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from acestep.llm_inference import LLMHandler  # noqa: E402

OUT = os.path.join(ROOT, "qnn", "results", "opt", "lm_backend_quality")
PROMPTS = [
    ("An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal",
     "[Verse]\nCity lights are calling me\nNeon rivers in the street\n[Chorus]\nWe are dancing in the night\nHold me close and hold me tight"),
    ("A calm solo piano ballad, slow tempo, soft and emotional, with gentle reverb", "[Instrumental]"),
    ("Aggressive heavy metal with distorted guitars, double-kick drums and a screaming male vocal",
     "[Verse]\nFire in the sky tonight\nBurning all we knew\n[Chorus]\nRise up, rise up\nBreak the chains"),
    ("Laid-back lo-fi hip hop beat with vinyl crackle, jazzy piano chords and a relaxed male rap",
     "[Verse]\nSunday morning coffee slow\nWatching how the rivers flow\n[Chorus]\nTake it easy, let it go"),
]
SEEDS = [int(x) for x in os.environ.get("LMQ_SEEDS", "11,22,33").split(",")]
LEVELS = np.array([8, 8, 8, 5, 5, 5])
BASIS = np.cumprod([1] + list(LEVELS[:-1]))


def parse_codes(s):
    return [int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", s or "")]


def gen(backend):
    llm = LLMHandler()
    st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                            backend=backend, device="cuda", offload_to_cpu=False, dtype=None)
    assert ok, st
    rows = []
    for pi, (cap, lyr) in enumerate(PROMPTS):
        for seed in SEEDS:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            t = time.perf_counter()
            r = llm.generate_with_stop_condition(caption=cap, lyrics=lyr, infer_type="llm_dit", temperature=0.85,
                                                 cfg_scale=2.0, negative_prompt="NO USER INPUT", top_p=0.9,
                                                 target_duration=30, use_cot_metas=True, use_cot_caption=True,
                                                 use_cot_language=True)
            dt = time.perf_counter() - t
            assert r["success"], r.get("error")
            rows.append({"prompt": pi, "seed": seed, "seconds": round(dt, 3), "metadata": r["metadata"],
                         "audio_codes": r["audio_codes"]})
            print(backend, pi, seed, round(dt, 2), len(parse_codes(r["audio_codes"])), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, f"{backend}.json"), "w"), indent=1, ensure_ascii=False)


def greedy(backend):
    """Near-greedy decode (temperature 1e-3, no top-p) to compare backends without sampling noise."""
    llm = LLMHandler()
    st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                            backend=backend, device="cuda", offload_to_cpu=False, dtype=None)
    assert ok, st
    rows = []
    for pi, (cap, lyr) in enumerate(PROMPTS):
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        r = llm.generate_with_stop_condition(caption=cap, lyrics=lyr, infer_type="llm_dit", temperature=1e-3,
                                             cfg_scale=2.0, negative_prompt="NO USER INPUT", top_p=None,
                                             target_duration=30, use_cot_metas=True, use_cot_caption=True,
                                             use_cot_language=True)
        # controlled: all metas fixed -> Phase 1 skipped, identical Phase 2 prompt on both backends
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        um = {"bpm": "100", "duration": "30", "keyscale": "C major", "timesignature": "4"}
        r2 = llm.generate_with_stop_condition(caption=cap, lyrics=lyr, infer_type="llm_dit", temperature=1e-3,
                                              cfg_scale=2.0, negative_prompt="NO USER INPUT", top_p=None,
                                              target_duration=30, user_metadata=um, use_cot_metas=True,
                                              use_cot_caption=False, use_cot_language=False)
        rows.append({"prompt": pi, "metadata": r["metadata"], "audio_codes": r["audio_codes"],
                     "fixed_meta_codes": r2["audio_codes"]})
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, f"greedy_{backend}.json"), "w"), indent=1, ensure_ascii=False)


def greedy_compare():
    a = json.load(open(os.path.join(OUT, "greedy_pt.json")))
    b = json.load(open(os.path.join(OUT, "greedy_vllm.json")))
    res = []
    for x, y in zip(a, b):
        ca, cb = parse_codes(x["audio_codes"]), parse_codes(y["audio_codes"])
        n = min(len(ca), len(cb))
        first_div = next((i for i in range(n) if ca[i] != cb[i]), n)
        lva, lvb = (np.asarray(ca[:n])[:, None] // BASIS) % LEVELS, (np.asarray(cb[:n])[:, None] // BASIS) % LEVELS
        fa, fb = parse_codes(x["fixed_meta_codes"]), parse_codes(y["fixed_meta_codes"])
        m2 = min(len(fa), len(fb))
        fdiv = next((i for i in range(m2) if fa[i] != fb[i]), m2)
        lfa, lfb = (np.asarray(fa[:m2])[:, None] // BASIS) % LEVELS, (np.asarray(fb[:m2])[:, None] // BASIS) % LEVELS
        res.append({"prompt": x["prompt"],
                    "fixed_meta_identical_prefix": fdiv, "fixed_meta_n": m2,
                    "fixed_meta_exact_match": round(float(np.mean(np.asarray(fa[:m2]) == np.asarray(fb[:m2]))), 3),
                    "fixed_meta_per_dim_match": round(float((lfa == lfb).mean()), 3),
                    "duration_pt_vs_vllm": (x["metadata"].get("duration"), y["metadata"].get("duration")),
                    "same_metadata": {k: x["metadata"].get(k) == y["metadata"].get(k)
                                      for k in ("bpm", "keyscale", "timesignature", "duration", "language", "caption")},
                    "codes_identical_prefix": first_div, "codes_exact_match": round(float(np.mean(np.asarray(ca[:n]) == np.asarray(cb[:n]))), 3),
                    "per_dim_match": round(float((lva == lvb).mean()), 3)})
        print(res[-1], flush=True)
    json.dump(res, open(os.path.join(OUT, "greedy_compare.json"), "w"), indent=1)


def code_stats(codes):
    c = np.asarray(codes)
    lv = (c[:, None] // BASIS) % LEVELS
    ent = []
    for d in range(6):
        p = np.bincount(lv[:, d], minlength=LEVELS[d]) / len(c)
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum() / math.log2(LEVELS[d])))
    return {"unique_frac": len(set(codes)) / len(codes), "repeat_rate": float((c[1:] == c[:-1]).mean()),
            "norm_entropy": float(np.mean(ent))}


def score():
    from acestep.core.scoring.lm_score import calculate_pmi_score_per_condition
    llm = LLMHandler()
    st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                            backend="pt", device="cuda", offload_to_cpu=False, dtype=None)
    assert ok, st
    summary = {}
    for backend in ("pt", "vllm"):
        rows = json.load(open(os.path.join(OUT, f"{backend}.json")))
        per = []
        for r in rows:
            cap, lyr = PROMPTS[r["prompt"]]
            md = r["metadata"] or {}
            meta = {k: md.get(k) for k in ("bpm", "keyscale", "timesignature", "duration", "language") if md.get(k)}
            scores, reward, _ = calculate_pmi_score_per_condition(
                llm, r["audio_codes"], caption=md.get("caption") or cap,
                lyrics=lyr if lyr != "[Instrumental]" else "", metadata=meta)
            codes = parse_codes(r["audio_codes"])
            s = code_stats(codes)
            per.append({"prompt": r["prompt"], "seed": r["seed"], "reward": reward,
                        "caption_pmi": scores.get("caption"), "lyrics_pmi": scores.get("lyrics"),
                        "n_codes": len(codes), "bpm": md.get("bpm"), "keyscale": md.get("keyscale"),
                        "metas_complete": all(md.get(k) for k in ("bpm", "keyscale", "timesignature", "duration")),
                        "seconds": r["seconds"], **s})
        def m(k):
            v = [x[k] for x in per if x[k] is not None]
            return round(statistics.mean(v), 4), round(statistics.pstdev(v), 4)
        summary[backend] = {"n": len(per), "reward_mean_std": m("reward"), "caption_pmi": m("caption_pmi"),
                            "lyrics_pmi": m("lyrics_pmi"), "unique_frac": m("unique_frac"),
                            "repeat_rate": m("repeat_rate"), "norm_entropy": m("norm_entropy"),
                            "all_150_codes": all(x["n_codes"] == 150 for x in per),
                            "metas_complete_frac": sum(x["metas_complete"] for x in per) / len(per),
                            "lm_seconds_mean": m("seconds"), "per_sample": per}
        print(backend, {k: v for k, v in summary[backend].items() if k != "per_sample"}, flush=True)
    # paired difference by (prompt, seed) is not meaningful (different RNG paths); compare distributions
    a = [x["reward"] for x in summary["pt"]["per_sample"]]
    b = [x["reward"] for x in summary["vllm"]["per_sample"]]
    diff = statistics.mean(b) - statistics.mean(a)
    se = math.sqrt(statistics.pvariance(a) / len(a) + statistics.pvariance(b) / len(b))
    summary["reward_diff_vllm_minus_pt"] = {"mean": round(diff, 4), "se": round(se, 4)}
    print("reward diff vllm-pt", summary["reward_diff_vllm_minus_pt"])
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "score", "greedy", "greedy_compare"])
    ap.add_argument("--backend", default="pt")
    a = ap.parse_args()
    {"gen": lambda: gen(a.backend), "score": score, "greedy": lambda: greedy(a.backend),
     "greedy_compare": greedy_compare}[a.cmd]()
