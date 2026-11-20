"""Trace module I/O shapes and GPU time for one ACE-Step text2music run (DiT path).

Usage: python qnn/trace_shapes.py [--duration 30] [--codes-from-lm 0]
Hooks: text encoder, condition encoder (lyric / timbre), audio tokenizer / detokenizer,
DiT decoder (per step), VAE encode / decode.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402


def shape_of(x):
    if isinstance(x, torch.Tensor):
        return f"{tuple(x.shape)}:{str(x.dtype).replace('torch.', '')}"
    if isinstance(x, (list, tuple)):
        return [shape_of(v) for v in x if v is not None][:4]
    if hasattr(x, "last_hidden_state"):
        return shape_of(x.last_hidden_state)
    if hasattr(x, "sample"):
        return shape_of(x.sample)
    return type(x).__name__


records = defaultdict(list)


def hook(name, module):
    def pre(mod, args, kwargs):
        torch.cuda.synchronize()
        mod._t0 = time.perf_counter()
        ins = {k: shape_of(v) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        if args:
            ins["args"] = [shape_of(a) for a in args if isinstance(a, torch.Tensor)]
        mod._ins = ins

    def post(mod, args, kwargs, out):
        torch.cuda.synchronize()
        records[name].append({
            "ms": round((time.perf_counter() - mod._t0) * 1000, 2),
            "in": mod._ins,
            "out": shape_of(out),
        })

    module.register_forward_pre_hook(pre, with_kwargs=True)
    module.register_forward_hook(post, with_kwargs=True)


def wrap_method(obj, attr, name):
    fn = getattr(obj, attr)

    def wrapped(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        records[name].append({
            "ms": round((time.perf_counter() - t0) * 1000, 2),
            "in": [shape_of(x) for x in a if isinstance(x, torch.Tensor)],
            "out": shape_of(out),
        })
        return out

    setattr(obj, attr, wrapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--lm", default="", help="e.g. acestep-5Hz-lm-0.6B; empty = DiT only")
    ap.add_argument("--out", default=os.path.join(ROOT, "qnn", "logs", "trace_shapes.json"))
    args = ap.parse_args()

    h = AceStepHandler()
    status, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                      use_flash_attention=False, compile_model=False,
                                      offload_to_cpu=bool(args.lm), offload_dit_to_cpu=False)
    assert ok, status
    llm = None
    if args.lm:
        llm = LLMHandler()
        st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path=args.lm,
                                backend="pt", device="cuda", offload_to_cpu=True, dtype=None)
        assert ok, st
        lm_calls = defaultdict(int)

        def lm_pre(mod, a, k):
            ids = k.get("input_ids", a[0] if a else None)
            if ids is not None:
                key = "prefill" if ids.shape[-1] > 1 else "decode"
                lm_calls[key] += 1
                if key == "prefill":
                    records["LM prefill input_ids"].append({"ms": 0, "in": shape_of(ids), "out": ""})
        llm.llm.register_forward_pre_hook(lm_pre, with_kwargs=True)
    m = h.model
    hook("text_encoder(Qwen3-Embedding-0.6B)", h.text_encoder)
    hook("condition_encoder", m.encoder)
    hook("  lyric_encoder", m.encoder.lyric_encoder)
    hook("  timbre_encoder", m.encoder.timbre_encoder)
    hook("audio_tokenizer(pool+FSQ)", m.tokenizer)
    hook("audio_detokenizer", m.detokenizer)
    hook("DiT decoder (per step)", m.decoder)
    wrap_method(h.vae, "decode", "VAE decode (per tile)")
    wrap_method(h.vae, "encode", "VAE encode")

    params = GenerationParams(
        task_type="text2music",
        caption="An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal",
        lyrics="[Verse]\nCity lights are calling me\n[Chorus]\nWe are dancing in the night",
        vocal_language="unknown" if args.lm else "en",
        bpm=None if args.lm else 120, keyscale="" if args.lm else "C Major",
        timesignature="" if args.lm else "4",
        duration=args.duration, inference_steps=8, seed=42, thinking=bool(args.lm),
        use_cot_metas=bool(args.lm), use_cot_caption=bool(args.lm), use_cot_language=bool(args.lm), shift=3.0,
    )
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    out_dir = os.path.join(ROOT, "qnn", "outputs")
    for run in ("warmup", "measure"):
        records.clear()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if llm is not None:
            lm_calls.clear()
        res = generate_music(h, llm, params, cfg, save_dir=out_dir)
        wall = time.perf_counter() - t0
        print(f"[{run}] success={res.success} wall={wall:.3f}s peak_alloc={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
    summary = {name: {"calls": len(v), "total_ms": round(sum(r["ms"] for r in v), 1), "first": v[0]}
               for name, v in records.items()}
    summary["_wall_s"] = round(wall, 3)
    if llm is not None:
        summary["_lm_forward_calls"] = dict(lm_calls)
        summary["_lm_metadata"] = res.extra_outputs.get("lm_metadata") if hasattr(res, "extra_outputs") else None
    summary["_time_costs"] = res.extra_outputs.get("time_costs", {}) if hasattr(res, "extra_outputs") else {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
