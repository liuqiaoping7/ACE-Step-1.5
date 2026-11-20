"""Summarise an nsys sqlite export into a coarse GPU timeline.

Prints, per time bin: kernel-busy ms, H2D ms / GB, D2H ms, and fully idle ms,
plus totals and the largest idle gaps.  Bins are relative to the first event.

Usage: python qnn/perf/nsys_timeline.py <file.sqlite> [bin_ms]
"""
import sqlite3
import sys


def intervals(cur, sql):
    return [(s, e, *rest) for s, e, *rest in cur.execute(sql)]


def union_len(iv, lo, hi):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted((max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def main(path, bin_ms=250):
    con = sqlite3.connect(path)
    cur = con.cursor()
    k = intervals(cur, "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL")
    m = intervals(cur, "SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    h2d = [(s, e, b) for s, e, kind, b in m if kind == 1]
    d2h = [(s, e, b) for s, e, kind, b in m if kind == 2]
    all_iv = [(s, e) for s, e in k] + [(s, e) for s, e, *_ in m]
    t0 = min(s for s, _ in all_iv)
    t1 = max(e for _, e in all_iv)
    span = (t1 - t0) / 1e6
    busy_k = union_len(k, t0, t1) / 1e6
    busy_h = union_len([(s, e) for s, e, _ in h2d], t0, t1) / 1e6
    busy_any = union_len(all_iv, t0, t1) / 1e6
    print(f"span {span:.1f} ms | kernel busy {busy_k:.1f} | H2D busy {busy_h:.1f} ({sum(b for *_, b in h2d)/1e9:.2f} GB)"
          f" | D2H {union_len([(s, e) for s, e, _ in d2h], t0, t1)/1e6:.1f} ms ({sum(b for *_, b in d2h)/1e9:.3f} GB)"
          f" | any busy {busy_any:.1f} | idle {span - busy_any:.1f}")
    print(f"{'bin(ms)':>10} {'kern':>7} {'h2d':>7} {'h2dGB':>6} {'idle':>7}")
    step = bin_ms * 1e6
    b = t0
    while b < t1:
        e = b + step
        kk = union_len(k, b, e) / 1e6
        hh = union_len([(s, x) for s, x, _ in h2d], b, e) / 1e6
        gb = sum(bt * (min(x, e) - max(s, b)) / max(x - s, 1) for s, x, bt in h2d if x > b and s < e) / 1e9
        idle = (min(e, t1) - b) / 1e6 - union_len(all_iv, b, e) / 1e6
        print(f"{(b - t0)/1e6:10.0f} {kk:7.1f} {hh:7.1f} {gb:6.2f} {idle:7.1f}")
        b = e
    # largest idle gaps
    ev = sorted(all_iv)
    gaps, cur_e = [], ev[0][1]
    for s, e in ev[1:]:
        if s > cur_e:
            gaps.append(((s - cur_e) / 1e6, (cur_e - t0) / 1e6))
        cur_e = max(cur_e, e)
    gaps.sort(reverse=True)
    print("largest idle gaps (ms @ offset ms):", [(round(g, 1), round(o)) for g, o in gaps[:10]])
    nv = list(cur.execute("SELECT text, start, end FROM NVTX_EVENTS WHERE text IS NOT NULL")) if \
        cur.execute("SELECT name FROM sqlite_master WHERE name='NVTX_EVENTS'").fetchone() else []
    for text, s, e in nv:
        if e:
            print(f"NVTX {text}: {(s - t0)/1e6:.0f} -> {(e - t0)/1e6:.0f} ms ({(e - s)/1e6:.0f} ms)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 250)
