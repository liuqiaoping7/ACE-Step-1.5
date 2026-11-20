"""Full data-flow trace for one ACE-Step text2music run (0.6B LM thinking + turbo DiT + VAE).

Records for every hooked module / function: call count, input/output shapes, dtypes and value stats,
plus the actual prompt texts, LM output text, token counts and model structure / parameter counts.

Usage: python qnn/dataflow_trace.py [--duration 30]
Output: qnn/logs/dataflow_trace.json, qnn/logs/model_structure.txt
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

REC = OrderedDict()
TEXTS = OrderedDict()


def desc(x):
    if isinstance(x, torch.Tensor):
        d = {"shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", "")}
        if x.numel() and (x.is_floating_point()):
            xf = x.detach().float()
            d.update(mean=round(xf.mean().item(), 4), std=round(xf.std().item(), 4) if x.numel() > 1 else 0,
                     min=round(xf.min().item(), 4), max=round(xf.max().item(), 4))
        elif x.numel() and x.dtype in (torch.bool,):
            d.update(true_frac=round(x.float().mean().item(), 4))
        elif x.numel():
            d.update(min=int(x.min().item()), max=int(x.max().item()))
        return d
    if isinstance(x, (list, tuple)):
        return [desc(v) for v in x[:6] if v is not None]
    if isinstance(x, dict):
        return {k: desc(v) for k, v in list(x.items())[:30] if isinstance(v, (torch.Tensor, list, tuple))}
    for attr in ("last_hidden_state", "logits", "sample", "latent_dist"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if attr == "latent_dist":
                return {"latent_dist.mean": desc(v.mean), "latent_dist.std": desc(v.std)}
            return {attr: desc(v)}
    return type(x).__name__


def rec(name, entry, keep=2):
    r = REC.setdefault(name, {"calls": 0, "ms_total": 0.0, "samples": []})
    r["calls"] += 1
    r["ms_total"] = round(r["ms_total"] + entry.pop("ms", 0), 2)
    if len(r["samples"]) < keep:
        r["samples"].append(entry)


def hook(name, mod, keep=1):
    def pre(m, args, kwargs):
        torch.cuda.synchronize()
        m._t0 = time.perf_counter()
        m._in = {"args": desc([a for a in args if isinstance(a, torch.Tensor)]),
                 **{k: desc(v) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}}

    def post(m, args, kwargs, out):
        torch.cuda.synchronize()
        rec(name, {"ms": (time.perf_counter() - m._t0) * 1000, "in": m._in, "out": desc(out)}, keep)

    mod.register_forward_pre_hook(pre, with_kwargs=True)
    mod.register_forward_hook(post, with_kwargs=True)


def wrap(obj, attr, name, keep=2, text=False):
    fn = getattr(obj, attr)

    def w(*a, **k):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        e = {"ms": (time.perf_counter() - t0) * 1000,
             "in": desc([x for x in a if isinstance(x, torch.Tensor)]) or None,
             "kwargs": {kk: desc(vv) for kk, vv in k.items() if isinstance(vv, torch.Tensor)} or None,
             "out": desc(out) if not isinstance(out, str) else f"<str len={len(out)}>"}
        if text:
            TEXTS.setdefault(name, []).append(out if isinstance(out, str) else str(out)[:4000])
        rec(name, e, keep)
        return out

    setattr(obj, attr, w)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def structure(h, llm):
    lines = []
    m = h.model
    lines.append(f"== AceStepConditionGenerationModel  total {nparams(m)/1e6:.1f}M")
    for name, child in m.named_children():
        lines.append(f"  {name:<12} {type(child).__name__:<32} {nparams(child)/1e6:8.1f}M")
        for n2, c2 in child.named_children():
            if isinstance(c2, torch.nn.ModuleList):
                lines.append(f"    {n2:<22} ModuleList x{len(c2)}  {nparams(c2)/1e6:8.1f}M  (each {nparams(c2[0])/1e6:.2f}M)")
            else:
                lines.append(f"    {n2:<22} {type(c2).__name__:<28} {nparams(c2)/1e6:8.2f}M")
    lines.append("\n== DiT layer[0] / layer[1]")
    lines.append(repr(m.decoder.layers[0]))
    lines.append(f"attention_type: {[l.attention_type for l in m.decoder.layers[:4]]} ...")
    lines.append("\n== decoder.proj_in / time_embed / proj_out")
    lines.append(repr(m.decoder.proj_in)); lines.append(repr(m.decoder.time_embed)); lines.append(repr(m.decoder.proj_out))
    lines.append("\n== encoder (condition) lyric_encoder.layers[0]")
    lines.append(repr(m.encoder.lyric_encoder.layers[0]))
    lines.append("\n== tokenizer / detokenizer")
    lines.append(repr(m.tokenizer.quantizer))
    lines.append(f"tokenizer.attention_pooler layers: {len(m.tokenizer.attention_pooler.layers)}; detokenizer layers: {len(m.detokenizer.layers)}")
    lines.append(f"\n== VAE {type(h.vae).__name__} total {nparams(h.vae)/1e6:.1f}M  encoder {nparams(h.vae.encoder)/1e6:.1f}M decoder {nparams(h.vae.decoder)/1e6:.1f}M")
    lines.append(repr(h.vae.decoder)[:6000])
    lines.append(f"\n== Text encoder {type(h.text_encoder).__name__} {nparams(h.text_encoder)/1e6:.1f}M")
    lines.append(repr(h.text_encoder)[:2500])
    if llm is not None and llm.llm is not None:
        lines.append(f"\n== 5Hz LM {type(llm.llm).__name__} {nparams(llm.llm)/1e6:.1f}M")
        lines.append(repr(llm.llm)[:2500])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30)
    args = ap.parse_args()

    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=True, offload_dit_to_cpu=False)
    assert ok, st
    llm = LLMHandler()
    st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                            backend="pt", device="cuda", offload_to_cpu=True, dtype=None)
    assert ok, st

    with open(os.path.join(ROOT, "qnn", "logs", "model_structure.txt"), "w") as f:
        f.write(structure(h, llm))

    m = h.model
    # ---- LM side
    wrap(llm, "build_formatted_prompt", "LM.prompt_phase1(cot)", text=True)
    wrap(llm, "build_formatted_prompt_with_cot", "LM.prompt_phase2(codes)", text=True, keep=4)
    wrap(llm, "parse_lm_output", "LM.parse_lm_output", text=True)
    lm_calls = {"prefill": [], "decode": 0}

    def lm_pre(mod, a, k):
        ids = k.get("input_ids", a[0] if a else None)
        if ids is not None:
            if ids.shape[-1] > 1:
                lm_calls["prefill"].append(list(ids.shape))
            else:
                lm_calls["decode"] += 1
    llm.llm.register_forward_pre_hook(lm_pre, with_kwargs=True)
    # ---- DiT handler side
    wrap(h, "preprocess_batch", "H.preprocess_batch(batch->tensors)")
    wrap(h, "infer_text_embeddings", "H.infer_text_embeddings")
    wrap(h, "infer_lyric_embeddings", "H.infer_lyric_embeddings")
    wrap(h, "_decode_audio_codes_to_latents", "H.decode_codes_to_25Hz_hints")
    wrap(h, "infer_refer_latent", "H.infer_refer_latent")
    tok = h.text_tokenizer
    orig_tok_call = tok.__call__

    # capture texts fed into the DiT text tokenizer
    def tok_call(self, text, *a, **k):
        out = orig_tok_call(text, *a, **k)
        TEXTS.setdefault("DiT.text_tokenizer_inputs", []).append(
            {"text": text if isinstance(text, str) else str(text)[:3000],
             "n_tokens": int(out["input_ids"].shape[-1]) if hasattr(out["input_ids"], "shape") else None})
        return out
    h.text_tokenizer = _TokProxy(tok, tok_call)

    hook("TextEncoder(Qwen3-Embedding)", h.text_encoder)
    hook("Cond.text_projector", m.encoder.text_projector)
    hook("Cond.lyric_encoder", m.encoder.lyric_encoder)
    hook("Cond.lyric_encoder.embed_tokens(Linear)", m.encoder.lyric_encoder.embed_tokens)
    hook("Cond.timbre_encoder", m.encoder.timbre_encoder)
    hook("Cond.encoder(total)", m.encoder)
    hook("Tok.audio_tokenizer", m.tokenizer)
    hook("Tok.attention_pooler", m.tokenizer.attention_pooler)
    hook("Detok.detokenizer", m.detokenizer)
    wrap(m.tokenizer.quantizer, "get_output_from_indices", "FSQ.get_output_from_indices")
    wrap(m, "prepare_condition", "M.prepare_condition")
    wrap(m, "generate_audio", "M.generate_audio", keep=1)
    d = m.decoder
    hook("DiT(per step)", d, keep=2)
    hook("DiT.proj_in(Conv1d patch2)", d.proj_in)
    hook("DiT.time_embed(t)", d.time_embed)
    hook("DiT.time_embed_r(t-r)", d.time_embed_r)
    hook("DiT.condition_embedder", d.condition_embedder)
    hook("DiT.layers[0](sliding)", d.layers[0])
    hook("DiT.layers[1](full)", d.layers[1])
    hook("DiT.layers[0].self_attn", d.layers[0].self_attn)
    hook("DiT.layers[0].cross_attn", d.layers[0].cross_attn)
    hook("DiT.layers[0].mlp", d.layers[0].mlp)
    hook("DiT.proj_out(ConvT1d)", d.proj_out)
    wrap(h.vae, "decode", "VAE.decode(per tile)")
    wrap(h, "tiled_decode", "H.tiled_decode")

    params = GenerationParams(
        task_type="text2music",
        caption="An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal",
        lyrics="[Verse]\nCity lights are calling me\nNeon rivers in the street\n[Chorus]\nWe are dancing in the night\nHold me close and hold me tight",
        duration=args.duration, inference_steps=8, seed=42, thinking=True, shift=3.0,
    )
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    t0 = time.perf_counter()
    res = generate_music(h, llm, params, cfg, save_dir=os.path.join(ROOT, "qnn", "outputs", "dataflow"))
    wall = time.perf_counter() - t0
    assert res.success, res.error
    a = res.audios[0]
    out = {
        "wall_s": round(wall, 2),
        "time_costs": res.extra_outputs.get("time_costs"),
        "lm_calls": {"prefill_shapes": lm_calls["prefill"], "decode_steps": lm_calls["decode"]},
        "lm_metadata": res.extra_outputs.get("lm_metadata"),
        "final_audio": {"tensor": desc(a["tensor"]), "sample_rate": a.get("sample_rate"), "path": a.get("path")},
        "audio_codes_head": (a["params"].get("audio_codes") or "")[:400],
        "n_audio_codes": (a["params"].get("audio_codes") or "").count("<|audio_code_"),
        "records": REC,
        "texts": TEXTS,
    }
    with open(os.path.join(ROOT, "qnn", "logs", "dataflow_trace.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print("done", wall)


class _TokProxy:
    """Forward attribute access to a HF tokenizer while spying on __call__."""

    def __init__(self, tok, call):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "_call", call)

    def __call__(self, *a, **k):
        return self._call(self._tok, *a, **k)

    def __getattr__(self, k):
        return getattr(self._tok, k)


if __name__ == "__main__":
    main()
