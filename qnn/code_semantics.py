"""What do 5Hz codes carry, and how predictable are they for the LM?

1. Generate S1 (LM thinking) -> codes C1, metas M1.  S2: same prompt, other seed (different codes).
2. Cover: feed C1 with new captions (piano / metal) and no LM -> S3, S4.
   Compare chroma (harmony/melody), onset envelope (rhythm) and per-dim code match vs S1, against S2.
3. LM teacher-forced NLL of C1 under: full prompt (+CoT), prompt without CoT, unconditional prompt.
   bits/code, top-1 / top-10 accuracy, how often the argmax is a grid neighbour of the target.

Usage: python qnn/code_semantics.py
"""
import json
import math
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

LEVELS = np.array([8, 8, 8, 5, 5, 5])
BASIS = np.cumprod([1] + list(LEVELS[:-1]))
CODE_BASE = 151669
SR = 48000


def lv(idx):
    return (np.asarray(idx)[..., None] // BASIS) % LEVELS


def parse_codes(s):
    return [int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", s or "")]


def chroma(wav, hop=4800, n_fft=16384):
    """12-bin chroma at 10 Hz from a mono 48 kHz waveform (pure torch)."""
    x = wav.mean(0).float()
    spec = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs() ** 2
    freqs = torch.fft.rfftfreq(n_fft, 1 / SR)
    sel = (freqs > 60) & (freqs < 2500)
    pc = (torch.round(12 * torch.log2(freqs[sel] / 440.0)) + 9) % 12
    C = torch.zeros(12, spec.shape[1])
    C.index_add_(0, pc.long(), spec[sel])
    C = C / (C.norm(dim=0, keepdim=True) + 1e-9)
    return C  # [12, frames]


def onset_env(wav, hop=480, n_fft=2048):
    x = wav.mean(0).float()
    spec = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs()
    flux = torch.relu(torch.log1p(spec[:, 1:]) - torch.log1p(spec[:, :-1])).sum(0)
    return (flux - flux.mean()) / (flux.std() + 1e-9)  # 100 Hz


def sim(w_a, w_b):
    ca, cb = chroma(w_a), chroma(w_b)
    n = min(ca.shape[1], cb.shape[1])
    frame_cos = (ca[:, :n] * cb[:, :n]).sum(0).mean().item()
    # global key profile similarity (time-averaged chroma)
    ga, gb = ca.mean(1), cb.mean(1)
    key_cos = torch.nn.functional.cosine_similarity(ga, gb, dim=0).item()
    oa, ob = onset_env(w_a), onset_env(w_b)
    m = min(len(oa), len(ob))
    # rhythm: best correlation within ±100 ms lag
    best = max(float((oa[max(0, l):m + min(0, l)] * ob[max(0, -l):m - max(0, l)]).mean()) for l in range(-10, 11))
    return {"chroma_frame_cos": round(frame_cos, 3), "key_profile_cos": round(key_cos, 3), "onset_corr_max": round(best, 3)}


def retokenize(h, wav):
    with torch.inference_mode():
        with h._load_model_context("vae"):
            lat = h._encode_audio_to_latents(wav.to(h.device))
        with h._load_model_context("model"):
            x = lat.unsqueeze(0).to(h.dtype)
            m = torch.ones(1, x.shape[1], device=h.device, dtype=h.dtype)
            _, idx, _ = h.model.tokenize(x, h.silence_latent, m)
    return idx.flatten().cpu().numpy()


def code_match(a, b):
    n = min(len(a), len(b))
    la, lb = lv(np.asarray(a[:n])), lv(np.asarray(b[:n]))
    return {"exact": round(float((np.asarray(a[:n]) == np.asarray(b[:n])).mean()), 3),
            "per_dim_mean": round(float((la == lb).mean()), 3)}


def gen(h, llm, **kw):
    p = GenerationParams(task_type="text2music", duration=30, inference_steps=8, shift=3.0, **kw)
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    r = generate_music(h, llm, p, cfg, save_dir=os.path.join(ROOT, "qnn", "outputs", "semantics"))
    assert r.success, r.error
    a = r.audios[0]
    return a["tensor"].float(), parse_codes(a["params"].get("audio_codes", "")), r.extra_outputs.get("lm_metadata") or {}


def lm_nll(llm, prompt, codes):
    tok = llm.llm_tokenizer
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    c_ids = torch.tensor([CODE_BASE + c for c in codes])
    ids = torch.cat([p_ids, c_ids]).unsqueeze(0)
    with llm._load_model_context():
        model = llm.llm
        dev = next(model.parameters()).device
        with torch.inference_mode():
            logits = model(input_ids=ids.to(dev)).logits[0, len(p_ids) - 1:-1].float()
    code_logits = logits[:, CODE_BASE:CODE_BASE + 64000]  # restrict to valid codes (as at inference)
    logp = torch.log_softmax(code_logits, -1)
    tgt = torch.tensor(codes, device=logp.device)
    nll = -logp[torch.arange(len(codes)), tgt]
    top10 = code_logits.topk(10, dim=-1).indices
    am = top10[:, 0].cpu().numpy()
    tgt_np = np.asarray(codes)
    l1 = np.abs(lv(am) - lv(tgt_np)).sum(-1)
    bits = (nll / math.log(2)).cpu().numpy()
    return {
        "bits_per_code": round(float(bits.mean()), 2),
        "bits_first_25": round(float(bits[:25].mean()), 2),
        "bits_rest": round(float(bits[25:].mean()), 2),
        "perplexity": round(float(np.exp(nll.mean().item())), 1),
        "top1_acc": round(float((am == tgt_np).mean()), 3),
        "top10_acc": round(float((top10.cpu().numpy() == tgt_np[:, None]).any(1).mean()), 3),
        "argmax_grid_l1_le1": round(float((l1 <= 1).mean()), 3),
        "argmax_grid_l1_mean": round(float(l1.mean()), 2),
    }


def main():
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=True, offload_dit_to_cpu=False)
    assert ok, st
    llm = LLMHandler()
    st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                            backend="pt", device="cuda", offload_to_cpu=True, dtype=None)
    assert ok, st

    cap = "An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal"
    lyr = ("[Verse]\nCity lights are calling me\nNeon rivers in the street\n"
           "[Chorus]\nWe are dancing in the night\nHold me close and hold me tight")
    w1, c1, m1 = gen(h, llm, caption=cap, lyrics=lyr, seed=7, thinking=True)
    w2, c2, m2 = gen(h, llm, caption=cap, lyrics=lyr, seed=8, thinking=True)
    metas = dict(bpm=m1.get("bpm"), keyscale=m1.get("keyscale", ""), timesignature=str(m1.get("timesignature", "")),
                 vocal_language=m1.get("language", "en") or "en")
    code_str = "".join(f"<|audio_code_{c}|>" for c in c1)
    covers = {
        "piano": "A soft solo acoustic piano arrangement, gentle and intimate, no drums",
        "metal": "Heavy metal with distorted electric guitars, double-kick drums and aggressive male vocal",
    }
    wc = {}
    for k, c in covers.items():
        wc[k], _, _ = gen(h, None, caption=c, lyrics=lyr, seed=7, thinking=False, audio_codes=code_str,
                          use_cot_metas=False, use_cot_caption=False, use_cot_language=False, **metas)
    # no-codes baseline with the piano caption (same metas), to separate "caption effect" from "codes effect"
    wb, _, _ = gen(h, None, caption=covers["piano"], lyrics=lyr, seed=7, thinking=False,
                   use_cot_metas=False, use_cot_caption=False, use_cot_language=False, **metas)

    r1 = retokenize(h, w1)
    report = {
        "S1_metas": {k: m1.get(k) for k in ["bpm", "keyscale", "timesignature", "language", "duration"]},
        "S2_metas": {k: m2.get(k) for k in ["bpm", "keyscale", "timesignature", "language", "duration"]},
        "similarity_to_S1": {
            "S2_same_prompt_other_codes": {**sim(w1, w2), **code_match(r1, retokenize(h, w2))},
            "cover_piano_same_codes": {**sim(w1, wc["piano"]), **code_match(r1, retokenize(h, wc["piano"]))},
            "cover_metal_same_codes": {**sim(w1, wc["metal"]), **code_match(r1, retokenize(h, wc["metal"]))},
            "piano_no_codes_same_metas": {**sim(w1, wb), **code_match(r1, retokenize(h, wb))},
        },
        "cover_piano_vs_cover_metal": sim(wc["piano"], wc["metal"]),
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))

    cot = llm._format_metadata_as_cot(m1)
    cap1 = m1.get("caption", cap)
    prompts = {
        "full(caption+lyrics+CoT)": llm.build_formatted_prompt_with_cot(cap1, lyr, cot),
        "no_CoT(caption+lyrics)": llm.build_formatted_prompt_with_cot(cap1, lyr, "<think>\n\n</think>"),
        "uncond(NO USER INPUT)": llm.build_formatted_prompt_with_cot(cap1, lyr, cot, is_negative_prompt=True),
        "mismatched(metal caption+CoT of S1)": llm.build_formatted_prompt_with_cot(covers["metal"], lyr, cot),
    }
    report["lm_nll_on_S1_codes"] = {k: lm_nll(llm, p, c1) for k, p in prompts.items()}
    report["lm_nll_on_S1_codes"]["_uniform_bits"] = round(math.log2(64000), 2)
    print(json.dumps(report["lm_nll_on_S1_codes"], indent=1))
    out = os.path.join(ROOT, "qnn", "logs", "code_semantics.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
