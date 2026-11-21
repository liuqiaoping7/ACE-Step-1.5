"""Tabulate batch sweep results: qnn/results/opt/{F_*,BAT_*,BATNG_*}/result.json.

Usage: python qnn/perf/batch_table.py  (prints a markdown table)
"""
import glob
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPT = os.path.join(ROOT, "qnn", "results", "opt")


def main():
    rows = []
    for p in glob.glob(os.path.join(OPT, "BAT*_*s_b*", "result.json")):
        tag = os.path.basename(os.path.dirname(p))
        m = re.match(r"(BATNG|BAT)_(\d+)s_b(\d+)", tag)
        d = json.load(open(p))
        s, runs = d["summary"], [r for r in d["runs"] if not r["warmup"]]
        n = min(r.get("n_audios", 1) for r in runs)
        tc = s["time_costs_median"]
        rows.append((int(m.group(2)), int(m.group(3)), m.group(1), n, s["wall_median_s"], tc.get("lm_phase1_time", 0),
                     tc.get("lm_phase2_time", 0), tc.get("dit_diffusion_time_cost", 0), tc.get("dit_vae_decode_time_cost", 0),
                     s["peak_alloc_gb_max"], s["peak_reserved_gb_max"], s["host_rss_peak_gb"], s["host_swap_gb_max"],
                     sorted({i["n_codes"] for r in runs for i in r.get("items", [])})))
    rows.sort()
    base = {r[0]: r[4] for r in rows if r[1] == 1}
    print("| 长度 | batch | guard | 实得 | 墙钟 s | 每首 s | 吞吐× | LM P1 | LM P2 | DiT | VAE | 显存 alloc/resv GB | RSS GB | swap | 码数 |")
    print("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
    for (dur, b, g, n, w, p1, p2, dit, vae, pa, pr, rss, sw, nc) in rows:
        per = w / n
        x = f"{base[dur] / per:.2f}" if dur in base else "-"
        print(f"| {dur} | {b} | {'off' if g == 'BATNG' else 'on'} | {n} | {w:.2f} | {per:.2f} | {x} | {p1:.2f} | {p2:.2f} | "
              f"{dit:.2f} | {vae:.2f} | {pa:.2f}/{pr:.2f} | {rss:.2f} | {sw} | {nc} |")


if __name__ == "__main__":
    main()
