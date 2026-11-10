"""Figure 1 for the paper: (a) model-free retrieval by distance, (b) pooled Gemma 4 agents by distance.

  python -m bugdial.figures --locate out/merged --agents out/agents_all/episodes.jsonl --out paper/fig/figure1.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

XS = ["0", "1", "2", "3+"]


def bucket(d):
    return "none" if d is None else ("3+" if d >= 3 else str(d))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def series(hits):
    """hits: {bucket: [bool,...]} -> xs, ys, lo, hi, ns"""
    xs, ys, lo, hi, ns = [], [], [], [], []
    for i, b in enumerate(XS):
        h = hits.get(b, [])
        if not h:
            continue
        k, n = sum(h), len(h)
        l, u = wilson(k, n)
        xs.append(i); ys.append(k / n); lo.append(max(0, k / n - l)); hi.append(max(0, u - k / n)); ns.append(n)
    return xs, ys, lo, hi, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locate", required=True)
    ap.add_argument("--agents", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loc = [json.loads(l) for p in Path(a.locate).glob("*_locate.jsonl") for l in open(p)]
    loc = [r for r in loc if r["variant"] == "issue"]
    A = [json.loads(l) for l in open(a.agents)]
    A = [e for e in A if e["kind"] == "synth" and e["variant"] == "issue"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), dpi=170)
    left = [("bm25", "BM25", "#555555", "-"), ("emb", "released embeddings", "#9a9a9a", ":"),
            ("undir_bm25", "BM25 within 3 hops, any direction", "#1f77b4", "--"),
            ("dir_bm25", "BM25 within 3 hops, downstream", "#1f77b4", "-"),
            ("ppr_undir", "PPR+BM25, released graph", "#d62728", "--"),
            ("ppr_dir", "PPR+BM25, direction recovered", "#d62728", "-")]
    ax = axes[0]
    ns_left = {}
    for j, (m, lab, col, ls) in enumerate(left):
        hits = {}
        for r in loc:
            if r["method"] == m:
                hits.setdefault(bucket(r["d"]), []).append(r["rank"] <= 10)
        xs, ys, lo, hi, ns = series(hits)
        off = (j - 2.5) * 0.035
        ax.errorbar([x + off for x in xs], ys, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=0.8, capsize=2, alpha=0.8)
        ax.plot([x + off for x in xs], ys, color=col, ls=ls, marker="o", ms=3.5, lw=1.6, label=lab)
        ns_left = dict(zip([XS[x] for x in xs], ns))
    ax.set_xticks(range(4), [f"{x}\n(n={ns_left.get(x, 0)})" for x in XS])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("symptom-to-cause distance d (call hops)")
    ax.set_ylabel("faulty function in top 10")
    ax.set_title("(a) retrieval, 454 tasks, report without the name", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)

    right = [("text", "grep + read", "#555555", "-"), ("graph", "+ official graph tools offered", "#9467bd", ":"),
             ("ctx_short_bm25", "+ BM25 shortlist in prompt", "#ff7f0e", "--"),
             ("ctx_short_dir", "+ downstream shortlist in prompt", "#2ca02c", "-")]
    ax = axes[1]
    ns_right = {}
    for j, (arm, lab, col, ls) in enumerate(right):
        hits = {}
        for e in A:
            if e["arm"] == arm:
                hits.setdefault(bucket(e["d"]), []).append(e["rank"] == 1)
        xs, ys, lo, hi, ns = series(hits)
        off = (j - 1.5) * 0.04
        ax.errorbar([x + off for x in xs], ys, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=0.8, capsize=2, alpha=0.8)
        ax.plot([x + off for x in xs], ys, color=col, ls=ls, marker="o", ms=3.5, lw=1.6, label=lab)
        ns_right = dict(zip([XS[x] for x in xs], ns))
    ax.set_xticks(range(4), [f"{x}\n(n={ns_right.get(x, 0)})" for x in XS])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("symptom-to-cause distance d (call hops)")
    ax.set_ylabel("agent's first answer is the faulty function")
    ax.set_title("(b) Gemma 4 E4B + E2B agents, 8 tool calls, report without the name", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
