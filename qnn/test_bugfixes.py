"""End-to-end checks for the two bug fixes.

1. Vocal language: the LM-detected language must reach the DiT lyric prompt
   ("# Languages\\n<lang>"); a user-specified language must win.
2. chunk_mask_mode="auto" now feeds 2.0 to the DiT (was silently 1.0).
   A/B against "explicit" (== old behaviour for full-song generation) with identical
   codes / seeds; metrics: context mask value, code adherence, repo PMI reward, latent stats.

Usage: python qnn/test_bugfixes.py
"""
import json
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
from acestep.core.scoring.lm_score import calculate_pmi_score_per_condition  # noqa: E402

LEVELS = np.array([8, 8, 8, 5, 5, 5])
BASIS = np.cumprod([1] + list(LEVELS[:-1]))
OUT = os.path.join(ROOT, "qnn", "outputs", "bugfix")

SPY = {"lyric_prompts": [], "ctx_mask_values": []}


class _TokProxy:
    def __init__(self, tok):
        object.__setattr__(self, "_tok", tok)

    def __call__(self, text, *a, **k):
        if isinstance(text, str) and text.startswith("# Languages"):
            SPY["lyric_prompts"].append(text.split("\n")[1])
        return self._tok(text, *a, **k)

    def __getattr__(self, k):
        return getattr(self._tok, k)


def parse_codes(s):
    return [int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", s or "")]


def lv(idx):
    return (np.asarray(idx)[..., None] // BASIS) % LEVELS


def retokenize(h, wav):
    with torch.inference_mode():
        with h._load_model_context("vae"):
            lat = h._encode_audio_to_latents(wav.float().to(h.device))
        with h._load_model_context("model"):
            x = lat.unsqueeze(0).to(h.dtype)
            m = torch.ones(1, x.shape[1], device=h.device, dtype=h.dtype)
            _, idx, _ = h.model.tokenize(x, h.silence_latent, m)
    return idx.flatten().cpu().numpy()


def gen(h, llm, **kw):
    p = GenerationParams(task_type="text2music", duration=30, inference_steps=8, shift=3.0, **kw)
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    r = generate_music(h, llm, p, cfg, save_dir=OUT)
    assert r.success, r.error
    a = r.audios[0]
    return a, r.extra_outputs


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
    h.text_tokenizer = _TokProxy(h.text_tokenizer)

    orig_prep = h.model.prepare_condition

    def prep_spy(*a, **k):
        cm = k.get("chunk_masks")
        if cm is not None:
            SPY["ctx_mask_values"].append(sorted(set(cm.float().unique().tolist())))
        return orig_prep(*a, **k)
    h.model.prepare_condition = prep_spy

    report = {"language": [], "mask_ab": []}

    # ---------------- 1. vocal language propagation ----------------
    lang_cases = [
        ("en_lyrics_unknown", dict(caption="Bright synth-pop with a female vocal",
                                   lyrics="[Verse]\nCity lights are calling me\n[Chorus]\nWe are dancing in the night",
                                   vocal_language="unknown")),
        ("zh_lyrics_unknown", dict(caption="温柔的中文流行抒情歌，钢琴与弦乐，女声",
                                   lyrics="[Verse]\n夜色温柔落在窗前\n思念随风飘向远方\n[Chorus]\n我在等你回来",
                                   vocal_language="unknown")),
        ("en_lyrics_user_ja", dict(caption="Bright synth-pop with a female vocal",
                                   lyrics="[Verse]\nCity lights are calling me\n[Chorus]\nWe are dancing in the night",
                                   vocal_language="ja")),
    ]
    for name, kw in lang_cases:
        SPY["lyric_prompts"].clear()
        a, extra = gen(h, llm, seed=11, thinking=True, **kw)
        report["language"].append({
            "case": name, "user_vocal_language": kw["vocal_language"],
            "lm_language": (extra.get("lm_metadata") or {}).get("language"),
            "dit_lyric_prompt_language": SPY["lyric_prompts"][-1] if SPY["lyric_prompts"] else None,
        })
        print(report["language"][-1], flush=True)

    # ---------------- 2. chunk mask A/B ----------------
    prompts = [
        ("An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal",
         "[Verse]\nCity lights are calling me\nNeon rivers in the street\n[Chorus]\nWe are dancing in the night\nHold me close and hold me tight"),
        ("A calm solo piano ballad, slow tempo, soft and emotional, with gentle reverb", "[Instrumental]"),
        ("Aggressive heavy metal with distorted guitars, double-kick drums and a screaming male vocal",
         "[Verse]\nFire in the sky tonight\nBurning all we knew\n[Chorus]\nRise up, rise up\nBreak the chains"),
    ]
    for pi, (cap, lyr) in enumerate(prompts):
        for seed in (101, 202):
            # LM codes once (thinking), then DiT twice with identical codes / seed
            a0, extra0 = gen(h, llm, caption=cap, lyrics=lyr, seed=seed, thinking=True, chunk_mask_mode="explicit")
            codes = parse_codes(a0["params"].get("audio_codes"))
            meta = extra0.get("lm_metadata") or {}
            code_str = "".join(f"<|audio_code_{c}|>" for c in codes)
            common = dict(caption=meta.get("caption", cap), lyrics=lyr, seed=seed, thinking=False,
                          use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
                          bpm=meta.get("bpm"), keyscale=meta.get("keyscale", ""),
                          timesignature=str(meta.get("timesignature", "")),
                          vocal_language=meta.get("language", "unknown") or "unknown", audio_codes=code_str)
            row = {"prompt": pi, "seed": seed}
            lat = {}
            for path, extra_kw in (("lm_codes", common),
                                   ("no_lm", {k: v for k, v in common.items() if k != "audio_codes"})):
                for mode in ("explicit", "auto"):
                    SPY["ctx_mask_values"].clear()
                    a, _ = gen(h, None, chunk_mask_mode=mode, **extra_kw)
                    wav = a["tensor"]
                    re_codes = retokenize(h, wav)
                    lat[(path, mode)] = re_codes
                    r = {"mask_values_seen": SPY["ctx_mask_values"][-1] if SPY["ctx_mask_values"] else None,
                         "wav_rms": round(float(wav.float().pow(2).mean().sqrt()), 4),
                         "wav_peak": round(float(wav.float().abs().max()), 4)}
                    if path == "lm_codes":
                        n = min(len(codes), len(re_codes))
                        r["code_per_dim_match"] = round(float((lv(codes[:n]) == lv(re_codes[:n])).mean()), 3)
                        r["code_exact"] = round(float((np.asarray(codes[:n]) == re_codes[:n]).mean()), 3)
                    code_s = "".join(f"<|audio_code_{int(c)}|>" for c in re_codes)
                    scores, reward, _ = calculate_pmi_score_per_condition(
                        llm, code_s, caption=common["caption"], lyrics=lyr if lyr != "[Instrumental]" else "",
                        metadata={k: meta.get(k) for k in ("bpm", "keyscale", "timesignature", "duration") if meta.get(k)})
                    r["pmi_reward"] = round(float(reward), 4)
                    r["pmi_scores"] = {k: round(float(v), 3) for k, v in scores.items()}
                    row[f"{path}/{mode}"] = r
            for path in ("lm_codes", "no_lm"):
                a_, b_ = lat[(path, "explicit")], lat[(path, "auto")]
                n = min(len(a_), len(b_))
                row[f"{path}/explicit_vs_auto_code_per_dim_match"] = round(float((lv(a_[:n]) == lv(b_[:n])).mean()), 3)
            report["mask_ab"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    # summary
    summ = {}
    for path in ("lm_codes", "no_lm"):
        for mode in ("explicit", "auto"):
            rows = [r[f"{path}/{mode}"] for r in report["mask_ab"]]
            s = {"pmi_reward_mean": round(float(np.mean([x["pmi_reward"] for x in rows])), 4),
                 "wav_rms_mean": round(float(np.mean([x["wav_rms"] for x in rows])), 4)}
            if path == "lm_codes":
                s["code_per_dim_match_mean"] = round(float(np.mean([x["code_per_dim_match"] for x in rows])), 3)
            summ[f"{path}/{mode}"] = s
        summ[f"{path}/explicit_vs_auto_similarity"] = round(float(np.mean(
            [r[f"{path}/explicit_vs_auto_code_per_dim_match"] for r in report["mask_ab"]])), 3)
    report["summary"] = summ
    print(json.dumps(summ, indent=1))
    with open(os.path.join(ROOT, "qnn", "logs", "test_bugfixes.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
