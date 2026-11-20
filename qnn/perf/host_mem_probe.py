"""Host-memory probe: RSS breakdown and pinned-allocator usage during generation.

Usage: python qnn/perf/host_mem_probe.py [--duration 30] [--runs 2]
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
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.core.generation.handler import pinned_offload as po  # noqa: E402


def proc_mem():
    out = {}
    for line in open("/proc/self/status"):
        k = line.split(":")[0]
        if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"):
            out[k] = round(int(line.split()[1]) / 1024 / 1024, 2)  # GiB
    return out


def master_bytes(module):
    tot, seen = 0, set()
    for owner, kind, name, t in po._iter_slots(module):
        m = po._get_master(owner, kind, name, t)
        if m is not None and m.data_ptr() not in seen:
            seen.add(m.data_ptr()); tot += m.numel() * m.element_size()
    return round(tot / 2**30, 2)


def host_stats():
    try:
        s = torch.cuda.host_memory_stats()
        return {k: round(v / 2**30, 2) for k, v in s.items()
                if k in ("allocated_bytes.current", "reserved_bytes.current", "allocated_bytes.peak", "reserved_bytes.peak")}
    except Exception as e:
        return {"err": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    peak = {"VmRSS": 0}
    stop = False

    def sampler():
        while not stop:
            m = proc_mem()
            if m.get("VmRSS", 0) > peak["VmRSS"]:
                peak.update(m)
            time.sleep(0.2)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    rep = {"start": proc_mem()}
    h = AceStepHandler()
    st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda",
                                  use_flash_attention=False, compile_model=False,
                                  offload_to_cpu=True, offload_dit_to_cpu=True)
    assert ok, st
    rep["after_dit_init"] = proc_mem()
    llm = LLMHandler()
    st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                            backend="vllm", device="cuda", offload_to_cpu=True, dtype=None)
    assert ok, st
    rep["after_lm_init"] = proc_mem()
    p = GenerationParams(task_type="text2music", caption="An upbeat synth-pop song with punchy drums",
                         lyrics="[Verse]\nCity lights\n[Chorus]\nWe are dancing", duration=args.duration,
                         inference_steps=8, shift=3.0, seed=42, thinking=True)
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    for i in range(args.runs):
        torch.manual_seed(42)
        generate_music(h, llm, p, cfg, save_dir=None)
        rep[f"after_run{i}"] = proc_mem()
        rep[f"host_alloc_run{i}"] = host_stats()
    stop = True
    rep["peak"] = dict(peak)
    rep["exact_master_GiB"] = {n: master_bytes(getattr(h, n)) for n in ("model", "text_encoder", "vae")}
    rep["module_GiB"] = {n: round(po.module_bytes(getattr(h, n)) / 2**30, 2) for n in ("model", "text_encoder", "vae")}
    print("PROBE", json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
