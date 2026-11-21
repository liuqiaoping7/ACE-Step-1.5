"""VRAM retained across generations: allocated-memory delta after one generate_music call, with the
Python allocation stacks of the blocks still alive (torch.cuda.memory._record_memory_history).

Found the legacy weight_norm `weight` attributes of the Oobleck VAE left on the GPU after offload (0.17 GiB).
Usage: python qnn/perf/vram_leak_probe.py
"""
import gc, os, sys, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.insert(0, ROOT)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
h = AceStepHandler()
st, ok = h.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda", use_flash_attention=True,
                              compile_model=False, offload_to_cpu=True, offload_dit_to_cpu=True); assert ok, st
llm = LLMHandler()
st, ok = llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B", backend="vllm",
                        device="cuda", offload_to_cpu=True, dtype=None); assert ok, st
def snap():
    gc.collect(); out = {}
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                out[id(o)] = o
        except Exception: pass
    return out
before = snap(); a0 = torch.cuda.memory_allocated()
snap0 = torch.cuda.memory._snapshot()
live0 = {(b["address"]) for seg in snap0["segments"] for b in seg["blocks"] if b["state"]=="active_allocated"}
torch.cuda.memory._record_memory_history(max_entries=200000)
p = GenerationParams(task_type="text2music", caption="synth pop", lyrics="[Verse]\nla la", duration=30, thinking=True)
res = generate_music(h, llm, p, GenerationConfig(batch_size=4, use_random_seed=False, seeds=[1,2,3,4], audio_format="wav"), save_dir=None)
print("success", res.success, "n", len(res.audios))
del res; gc.collect()
after = snap(); a1 = torch.cuda.memory_allocated()
print("alloc delta GB", round((a1 - a0) / 2**30, 3))
snap1 = torch.cuda.memory._snapshot()
import collections
agg = collections.Counter()
for seg in snap1["segments"]:
    for b in seg["blocks"]:
        if b["state"] == "active_allocated" and b["address"] not in live0:
            fr = [f for f in b.get("frames", []) if f["filename"].endswith(".py") and "/torch/" not in f["filename"]]
            key = " <- ".join(f"{os.path.basename(f['filename'])}:{f['line']}:{f['name']}" for f in fr[:6]) or "(no frames)"
            agg[key] += b["size"]
for k, v in agg.most_common(10):
    print(round(v / 2**20, 1), "MiB", k)
new = [t for k, t in after.items() if k not in before]
new.sort(key=lambda t: -t.numel() * t.element_size())
for t in new[:12]:
    refs = [type(r).__name__ + (":" + ",".join(list(r.keys())[:6]) if isinstance(r, dict) else "") for r in gc.get_referrers(t) if r is not after and r is not new][:3]
    print(round(t.numel() * t.element_size() / 2**20, 1), "MiB", tuple(t.shape), t.dtype, refs)
