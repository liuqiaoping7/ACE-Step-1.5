"""Empirical analysis of ACE-Step 5Hz LM audio codes.

A. LM embedding geometry (CPU): do the LM's <|audio_code_i|> embeddings reflect the FSQ grid?
B. Round trip (GPU): LM codes -> DiT -> wav -> VAE encode -> FSQ tokenize -> codes'.
   Measures code statistics, DiT adherence to LM codes, and detokenizer hint vs real latent.

Usage: python qnn/analyze_codes.py [--lm acestep-5Hz-lm-0.6B] [--skip-gen]
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

LEVELS = np.array([8, 8, 8, 5, 5, 5])
BASIS = np.cumprod([1] + list(LEVELS[:-1]))  # [1, 8, 64, 512, 2560, 12800]
N_CODES = int(np.prod(LEVELS))  # 64000
CODE_BASE = 151669  # token id of <|audio_code_0|>


def idx_to_levels(idx):
    idx = np.asarray(idx)[..., None]
    return (idx // BASIS) % LEVELS


def part_a(lm):
    path = os.path.join(ROOT, "checkpoints", lm, "model.safetensors")
    with safe_open(path, "pt") as f:
        key = [k for k in f.keys() if k.endswith("embed_tokens.weight")][0]
        emb = f.get_tensor(key).float()
    text = emb[:CODE_BASE]
    codes = emb[CODE_BASE:CODE_BASE + N_CODES]
    unused = emb[CODE_BASE + N_CODES:]
    out = {
        "embed_shape": list(emb.shape),
        "norm_text_mean": text.norm(dim=1).mean().item(),
        "norm_codes_mean": codes.norm(dim=1).mean().item(),
        "norm_codes_std": codes.norm(dim=1).std().item(),
        "norm_unused_mean": unused.norm(dim=1).mean().item() if len(unused) else None,
        "n_unused": len(unused),
    }
    cn = torch.nn.functional.normalize(codes, dim=1)
    un = torch.nn.functional.normalize(unused, dim=1)
    # mean pairwise cosine of unused tokens (≈1 if they share an init / never trained)
    if len(unused) > 1:
        s = un[:512] @ un[:512].T
        out["unused_mean_pairwise_cos"] = ((s.sum() - s.diagonal().sum()) / (s.numel() - len(s))).item()
    # cosine vs FSQ grid L1 distance
    rng = np.random.default_rng(0)
    a = rng.integers(0, N_CODES, 200000)
    b = rng.integers(0, N_CODES, 200000)
    la, lb = idx_to_levels(a), idx_to_levels(b)
    l1 = np.abs(la - lb).sum(-1)
    cos = (cn[a] * cn[b]).sum(1).numpy()
    out["cos_vs_grid_l1"] = {int(d): round(float(cos[l1 == d].mean()), 4) for d in range(0, 16) if (l1 == d).sum() > 50}
    out["spearman_cos_vs_l1"] = float(np.corrcoef(np.argsort(np.argsort(cos)), np.argsort(np.argsort(l1)))[0, 1])
    # grid neighbours (differ by 1 level in one dim) vs random
    base = rng.integers(0, N_CODES, 50000)
    lv = idx_to_levels(base)
    dim = rng.integers(0, 6, 50000)
    step = np.where(lv[np.arange(50000), dim] < LEVELS[dim] - 1, 1, -1)
    nb = base + step * BASIS[dim]
    out["cos_grid_neighbour_by_dim"] = {
        int(d): round(float((cn[base[dim == d]] * cn[nb[dim == d]]).sum(1).mean()), 4) for d in range(6)}
    out["cos_random_pair"] = round(float(cos.mean()), 4)
    # nearest neighbour (in embedding space) of 2000 codes: how far on the grid?
    q = rng.integers(0, N_CODES, 2000)
    sims = cn[q] @ cn.T
    sims[torch.arange(2000), torch.as_tensor(q)] = -2
    nn_idx = sims.argmax(1).numpy()
    nn_l1 = np.abs(idx_to_levels(q) - idx_to_levels(nn_idx)).sum(-1)
    out["nn_grid_l1_hist"] = {int(k): int(v) for k, v in zip(*np.unique(nn_l1, return_counts=True))}
    out["nn_grid_l1_random_expect"] = round(float(np.abs(la - lb).sum(-1).mean()), 2)
    return out


def code_stats(codes):
    codes = np.asarray(codes)
    lv = idx_to_levels(codes)
    ent = []
    for d in range(6):
        p = np.bincount(lv[:, d], minlength=LEVELS[d]) / len(codes)
        p = p[p > 0]
        ent.append(round(float(-(p * np.log2(p)).sum()), 2))
    return {
        "n": int(len(codes)),
        "unique": int(len(np.unique(codes))),
        "consecutive_repeat_rate": round(float((codes[1:] == codes[:-1]).mean()), 3),
        "per_dim_entropy_bits": ent,
        "per_dim_max_bits": [round(math.log2(l), 2) for l in LEVELS],
        "per_dim_level_hist": [np.bincount(lv[:, d], minlength=LEVELS[d]).tolist() for d in range(6)],
    }


def part_b(lm, prompts, out_dir):
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=True, offload_dit_to_cpu=False)
    assert ok, st
    llm = LLMHandler()
    st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path=lm,
                            backend="pt", device="cuda", offload_to_cpu=True, dtype=None)
    assert ok, st

    results = []
    all_lm_codes = []
    for i, (cap, lyr) in enumerate(prompts):
        params = GenerationParams(task_type="text2music", caption=cap, lyrics=lyr, duration=30,
                                  inference_steps=8, seed=1000 + i, thinking=True, shift=3.0)
        cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
        res = generate_music(h, llm, params, cfg, save_dir=out_dir)
        assert res.success, res.error
        audio = res.audios[0]
        code_str = audio["params"].get("audio_codes") or ""
        lm_codes = [int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str)]
        all_lm_codes += lm_codes
        wav = audio["tensor"] if "tensor" in audio else None
        if wav is None:
            import torchaudio
            wav, sr = torchaudio.load(audio["path"])
            assert sr == 48000
        wav = wav.float()
        if wav.dim() == 3:
            wav = wav[0]
        with torch.inference_mode():
            with h._load_model_context("vae"):
                lat = h._encode_audio_to_latents(wav.to(h.device))  # [T, 64]
            with h._load_model_context("model"):
                lat_b = lat.unsqueeze(0).to(h.dtype)
                mask = torch.ones(1, lat_b.shape[1], device=h.device, dtype=h.dtype)
                _, idx, _ = h.model.tokenize(lat_b, h.silence_latent, mask)
                re_codes = idx.flatten().cpu().tolist()
                # hint from LM codes vs actual latent of the generated audio
                ids = torch.tensor(lm_codes, device=h.device).view(1, -1, 1)
                q = h.model.tokenizer.quantizer.get_output_from_indices(ids).to(h.dtype)
                hint = h.model.detokenizer(q)[0]
                T = min(hint.shape[0], lat.shape[0])
                hint_f, lat_f = hint[:T].float(), lat[:T].float().to(h.device)
                cos_t = torch.nn.functional.cosine_similarity(hint_f, lat_f, dim=1)
                # hint from re-tokenised codes (tokenizer round trip on real latent)
                q2 = h.model.tokenizer.quantizer.get_output_from_indices(idx.to(h.device)).to(h.dtype)
                hint2 = h.model.detokenizer(q2)[0][:T].float()
                cos_rt = torch.nn.functional.cosine_similarity(hint2, lat_f, dim=1)
        n = min(len(lm_codes), len(re_codes))
        a, b = np.array(lm_codes[:n]), np.array(re_codes[:n])
        la, lb = idx_to_levels(a), idx_to_levels(b)
        # shift tolerance: best match within ±1 code frame (0.2 s)
        exact_pm1 = np.mean([(a[k] == b[max(0, k - 1):k + 2]).any() for k in range(n)])
        results.append({
            "caption": cap[:60],
            "lm_metas": res.extra_outputs.get("lm_metadata"),
            "n_lm_codes": len(lm_codes), "n_re_codes": len(re_codes),
            "exact_match": round(float((a == b).mean()), 3),
            "exact_match_within_1frame": round(float(exact_pm1), 3),
            "per_dim_match": [round(float((la[:, d] == lb[:, d]).mean()), 3) for d in range(6)],
            "per_dim_abs_err": [round(float(np.abs(la[:, d] - lb[:, d]).mean()), 3) for d in range(6)],
            "chance_per_dim_match": [round(1 / l, 3) for l in LEVELS],
            "cos(hint_from_lm_codes, generated_latent)": round(cos_t.mean().item(), 3),
            "cos(hint_from_retokenised, generated_latent)": round(cos_rt.mean().item(), 3),
            "lm_code_stats": code_stats(lm_codes),
            "retokenised_code_stats": code_stats(re_codes),
        })
        print(json.dumps(results[-1], ensure_ascii=False, default=str)[:600])
    return {"songs": results, "pooled_lm_code_stats": code_stats(all_lm_codes)}


PROMPTS = [
    ("An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal",
     "[Verse]\nCity lights are calling me\nNeon rivers in the street\n[Chorus]\nWe are dancing in the night\nHold me close and hold me tight"),
    ("A calm solo piano ballad, slow tempo, soft and emotional, with gentle reverb", "[Instrumental]"),
    ("Aggressive heavy metal with distorted guitars, double-kick drums and a screaming male vocal",
     "[Verse]\nFire in the sky tonight\nBurning all we knew\n[Chorus]\nRise up, rise up\nBreak the chains"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="acestep-5Hz-lm-0.6B")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "qnn", "logs", "analyze_codes.json"))
    args = ap.parse_args()
    report = {"A_embedding": {}}
    for lm in ["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B"]:
        report["A_embedding"][lm] = part_a(lm)
        print(lm, json.dumps(report["A_embedding"][lm], default=str)[:800])
    if not args.skip_gen:
        report["B_roundtrip"] = part_b(args.lm, PROMPTS, os.path.join(ROOT, "qnn", "outputs", "analyze"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
