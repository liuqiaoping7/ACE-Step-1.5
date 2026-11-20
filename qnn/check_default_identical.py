"""Generate with *default* chunk_mask_mode and dump waveforms, to compare code versions.

Case A: text2music without LM.  Case B: text2music with fixed audio codes (auto-switches to cover).
Codes are obtained deterministically by tokenizing a reference wav.

Usage: python qnn/check_default_identical.py <out.pt> <reference.wav>
"""
import os
import sys

import wave

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402


def main(out_path, ref_wav):
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=True, offload_dit_to_cpu=False)
    assert ok, st
    with wave.open(ref_wav, "rb") as w:
        assert w.getframerate() == 48000 and w.getsampwidth() == 2
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    wav = torch.from_numpy(data.T.astype(np.float32) / 32768.0)
    torch.manual_seed(0)  # VAE encode uses latent_dist.sample(); fix it so the codes are reproducible
    torch.cuda.manual_seed_all(0)
    with torch.inference_mode():
        with h._load_model_context("vae"):
            lat = h._encode_audio_to_latents(wav.to(h.device))
        with h._load_model_context("model"):
            x = lat.unsqueeze(0).to(h.dtype)
            m = torch.ones(1, x.shape[1], device=h.device, dtype=h.dtype)
            _, idx, _ = h.model.tokenize(x, h.silence_latent, m)
    print("codes_head", idx.flatten()[:8].tolist())
    codes = "".join(f"<|audio_code_{int(c)}|>" for c in idx.flatten().tolist())
    common = dict(caption="An upbeat synth-pop song with punchy drums, bright synth leads and a female vocal",
                  lyrics="[Verse]\nCity lights are calling me\n[Chorus]\nWe are dancing in the night",
                  bpm=100, keyscale="E♭ minor", timesignature="4", vocal_language="en",
                  duration=30, inference_steps=8, shift=3.0, seed=4242, thinking=False,
                  use_cot_metas=False, use_cot_caption=False, use_cot_language=False)
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    res = {}
    for name, extra in (("A_text2music_noLM", {}), ("B_with_codes", {"audio_codes": codes})):
        r = generate_music(h, None, GenerationParams(task_type="text2music", **common, **extra), cfg, save_dir=None)
        assert r.success, r.error
        res[name] = r.audios[0]["tensor"].float().cpu()
    torch.save(res, out_path)
    print("saved", out_path, {k: tuple(v.shape) for k, v in res.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
