"""DiT single-step time / peak VRAM / output difference: SDPA (4D masks) vs flash_attention_2.

Same weights, same inputs; switches config._attn_implementation at runtime.
Usage: python qnn/perf/dit_attn_bench.py
"""
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler  # noqa: E402


def set_impl(model, impl):
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = impl


def main():
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=False, offload_dit_to_cpu=False)
    assert ok, st
    h.text_encoder.to("cpu"); h.vae.to("cpu")
    m = h.model
    m.encoder.to("cpu"); m.tokenizer.to("cpu"); m.detokenizer.to("cpu")
    torch.cuda.empty_cache()
    dec, dev, dt = m.decoder, h.device, h.dtype
    res = {}
    with torch.inference_mode():
        for T in (750, 1500, 3000, 6000):
            g = torch.Generator(device="cuda").manual_seed(T)
            xt = torch.randn(1, T, 64, device=dev, dtype=dt, generator=g)
            ctx = torch.randn(1, T, 128, device=dev, dtype=dt, generator=g)
            enc = torch.randn(1, 164, 2048, device=dev, dtype=dt, generator=g)
            enc_mask = torch.ones(1, 164, device=dev, dtype=torch.bool)
            att = torch.ones(1, T, device=dev, dtype=dt)
            t1 = torch.full((1,), 0.75, device=dev, dtype=dt)
            outs = {}
            for impl in ("sdpa", "flash_attention_2"):
                set_impl(m, impl)

                def step():
                    return dec(hidden_states=xt, timestep=t1, timestep_r=t1, attention_mask=att,
                               encoder_hidden_states=enc, encoder_attention_mask=enc_mask,
                               context_latents=ctx, use_cache=False)[0]
                try:
                    for _ in range(2):
                        step()
                    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    t = time.perf_counter()
                    for _ in range(3):
                        out = step()
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t) / 3 * 1000
                    outs[impl] = out.float()
                    res[f"{T}_{impl}"] = {"step_ms": round(ms, 1),
                                          "act_peak_gb": round((torch.cuda.max_memory_allocated() - base) / 2**30, 3)}
                except Exception as e:
                    res[f"{T}_{impl}"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(T, impl, res[f"{T}_{impl}"], flush=True)
            if len(outs) == 2:
                d = (outs["sdpa"] - outs["flash_attention_2"]).abs()
                rel = d.norm() / outs["sdpa"].norm()
                res[f"{T}_diff"] = {"maxabs": round(d.max().item(), 4), "rel_l2": round(rel.item(), 5)}
                print(T, "diff", res[f"{T}_diff"], flush=True)
    os.makedirs(os.path.join(ROOT, "qnn", "results", "opt", "dit_attn_bench"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "qnn", "results", "opt", "dit_attn_bench", "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
