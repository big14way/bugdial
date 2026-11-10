"""Aggregate run outputs into the paper's tables and figures.

  python -m bugdial.report --in out/kaggle --fig paper/fig
Reads whatever exists of: *_mutants.jsonl, *_tasks.jsonl, *_locate.jsonl, real_tasks.jsonl,
graph_stats.jsonl, episodes_*.jsonl. Prints markdown tables; writes PNG figures if matplotlib exists.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def jl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def bucket(d):
    return "none" if d is None else ("3+" if d >= 3 else str(d))


ORDER = ["0", "1", "2", "3+", "none"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--fig", default="paper/fig")
    a = ap.parse_args()
    D = Path(a.inp)
    out = []

    # ---- graph stats
    gs = D / "graph_stats.jsonl"
    if gs.exists():
        rows = [r for r in jl(gs) if "released_edges" in r]
        by = defaultdict(list)
        for r in rows:
            by[r["graph"].split("_")[0]].append(r)
        out.append("### Released graphs (all commits)\n\n| repo | graphs | released edges | reciprocal | pairs | recovered one-way | both | none |\n|---|---|---|---|---|---|---|---|")
        for repo, rr in sorted(by.items()):
            E = sum(r["released_edges"] for r in rr); R = sum(r["reciprocal_edges"] for r in rr)
            P = sum(r["pairs"] for r in rr); O = sum(r["recovered_one_way"] for r in rr)
            B = sum(r["recovered_both"] for r in rr); N = sum(r["recovered_none"] for r in rr)
            out.append(f"| {repo} | {len(rr)} | {E} | {R/E:.1%} | {P} | {O/P:.1%} | {B/P:.1%} | {N/P:.1%} |")

    # ---- planting yield
    for mp in sorted(D.glob("*_mutants.jsonl")):
        rows = jl(mp)
        st = Counter(r["status"] for r in rows)
        ops = Counter(r["mutation"]["op"] for r in rows if r["status"] == "killed")
        att = sum(len(r.get("attempts", [])) for r in rows)
        out.append(f"\n**{mp.stem}**: functions tried {len(rows)}, statuses {dict(st)}, mutants run {att}, kept by operator {dict(ops)}")

    # ---- tasks
    tasks = []
    for tp in sorted(p for p in D.glob("*_tasks.jsonl") if not p.name.startswith("real")):
        tasks += jl(tp)
    if tasks:
        c = Counter((t["repo"], bucket(t["d"])) for t in tasks)
        out.append("\n### Synthetic tasks by distance\n\n| repo | " + " | ".join(ORDER) + " | total | report names v |\n|---|" + "---|" * (len(ORDER) + 2))
        for repo in sorted({t["repo"] for t in tasks}):
            tt = [t for t in tasks if t["repo"] == repo]
            out.append(f"| {repo} | " + " | ".join(str(c[(repo, b)]) for b in ORDER) + f" | {len(tt)} | {sum(t['name_in_issue'] for t in tt)} |")

    # ---- model-free localization
    loc = []
    for lp in sorted(D.glob("*_locate.jsonl")):
        loc += jl(lp)
    curves = {}
    if loc:
        agg = defaultdict(list)
        for r in loc:
            agg[(r["variant"], bucket(r["d"]), r["method"])].append(r["rank"])
        _MO = ["bm25", "emb", "graph_undir", "graph_dir", "undir_bm25", "dir_bm25", "ppr_undir", "ppr_dir"]
        methods = sorted({r["method"] for r in loc}, key=lambda m: _MO.index(m) if m in _MO else 9)
        for var in ("issue", "issue_clue"):
            out.append(f"\n### Model-free acc@10 by distance ({'no clue' if var == 'issue' else 'clue: report names v'})\n\n| d | n | " + " | ".join(methods) + " |\n|---|---|" + "---|" * len(methods))
            for b in ORDER:
                rs = agg.get((var, b, methods[0]))
                if not rs:
                    continue
                cells = []
                for m in methods:
                    x = agg[(var, b, m)]
                    k = sum(v <= 10 for v in x)
                    cells.append(f"{k/len(x):.2f}")
                    curves.setdefault((var, m), []).append((b, k / len(x), wilson(k, len(x)), len(x)))
                out.append(f"| {b} | {len(rs)} | " + " | ".join(cells) + " |")
        # paired sign tests on the same tasks (hit@10), no-clue variant
        hit = {(r["task_id"], r["method"]): r["rank"] <= 10 for r in loc if r["variant"] == "issue"}
        dmap = {r["task_id"]: bucket(r["d"]) for r in loc}
        pairs = [("undir_bm25", "dir_bm25"), ("ppr_undir", "ppr_dir"), ("graph_undir", "graph_dir"),
                 ("bm25", "dir_bm25"), ("bm25", "ppr_dir"), ("emb", "bm25")]
        out.append("\n### Paired comparisons, no clue (B wins / A wins on tasks where they disagree; exact sign test)\n\n| A | B | d | B wins | A wins | p |\n|---|---|---|---|---|---|")
        for A, B in pairs:
            if not any(k[1] == A for k in hit) or not any(k[1] == B for k in hit):
                continue
            for b in ["1", "2", "3+", "1-3+"]:
                ids = [t for t, bb in dmap.items() if (bb == b or (b == "1-3+" and bb in ("1", "2", "3+")))]
                bw = sum(1 for t in ids if hit.get((t, B)) and not hit.get((t, A)))
                aw = sum(1 for t in ids if hit.get((t, A)) and not hit.get((t, B)))
                n = aw + bw
                p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(aw, bw) + 1)) / 2 ** n) if n else 1.0
                out.append(f"| {A} | {B} | {b} | {bw} | {aw} | {p:.3g} |")

    # ---- real tasks
    rt = D / "real_tasks.jsonl"
    if rt.exists():
        R = [r for r in jl(rt) if r["n_gold"]]
        out.append(f"\n### Real public tasks (gold mapped: {len(R)})\n")
        out.append(f"- report names a gold function: {sum(bool(r['named']) for r in R)}/{len(R)}")
        cb = Counter(bucket(r["d"]) for r in R)
        out.append("- distance: " + ", ".join(f"d={b}: {cb[b]}" for b in ORDER))
        meth = [k[5:] for k in R[0] if k.startswith("rank_")] if R else []
        if meth:
            out.append("\n| d | n | " + " | ".join(meth) + " |\n|---|---|" + "---|" * len(meth))
            for b in ORDER + ["ALL"]:
                rr = [r for r in R if b == "ALL" or bucket(r["d"]) == b]
                if rr:
                    out.append(f"| {b} | {len(rr)} | " + " | ".join(f"{sum(r['rank_'+m] <= 10 for r in rr)/len(rr):.2f}" for m in meth) + " |")

    # ---- agents
    for ep in sorted(D.glob("episodes_*.jsonl")):
        E = [e for e in jl(ep) if "arm" in e]
        model = ep.stem.removeprefix("episodes_")
        err = sum(bool(e.get("error")) for e in E)
        out.append(f"\n### Agent {model}: {len(E)} episodes, errors {err}\n")
        for kind in ("synth", "real"):
            EE = [e for e in E if e["kind"] == kind]
            if not EE:
                continue
            _AO = ["text", "graph", "callee", "ctx_callee", "ctx_short_undir", "ctx_short_dir"]
            arms = sorted({e["arm"] for e in EE}, key=lambda x: _AO.index(x) if x in _AO else 9)
            for var in sorted({e["variant"] for e in EE}):
                out.append(f"\n{kind} / {var}: acc@1 (acc@3) by d\n\n| d | n | " + " | ".join(arms) + " | mean calls |\n|---|---|" + "---|" * (len(arms) + 1))
                for b in ORDER + ["ALL"]:
                    cells, n = [], 0
                    for arm in arms:
                        x = [e for e in EE if e["variant"] == var and e["arm"] == arm and (b == "ALL" or bucket(e["d"]) == b)]
                        if not x:
                            cells.append("-")
                            continue
                        n = len(x)
                        a1 = sum(e["rank"] == 1 for e in x) / n
                        a3 = sum(e["rank"] is not None for e in x) / n
                        cells.append(f"{a1:.2f} ({a3:.2f})")
                        if kind == "synth":
                            curves.setdefault((f"agent:{model}:{var}", arm), []).append((b, a3, wilson(round(a3 * n), n), n))
                    calls = [e["steps"] for e in EE if e["variant"] == var and (b == "ALL" or bucket(e["d"]) == b)]
                    if n:
                        out.append(f"| {b} | {n} | " + " | ".join(cells) + f" | {sum(calls)/max(1,len(calls)):.1f} |")
                # tool usage
                use = Counter(c[0] for e in EE if e["variant"] == var for c in e["calls"])
                out.append(f"\ntool calls: {dict(use)}")

    print("\n".join(out))
    Path(a.fig).mkdir(parents=True, exist_ok=True)
    (Path(a.fig) / "tables.md").write_text("\n".join(out))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("(no matplotlib: figures skipped)")
        return
    groups = sorted({k[0] for k in curves})
    for g in groups:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=160)
        for (gg, m), pts in sorted(curves.items()):
            if gg != g:
                continue
            pts = [p for p in pts if p[0] in ("0", "1", "2", "3+")]
            if not pts:
                continue
            xs = [ORDER.index(p[0]) for p in pts]
            ys = [p[1] for p in pts]
            lo = [max(0.0, p[1] - p[2][0]) for p in pts]
            hi = [max(0.0, p[2][1] - p[1]) for p in pts]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=m)
        ax.set_xticks(range(4), ["0", "1", "2", "3+"])
        ax.set_xlabel("symptom-to-cause distance d (call hops)")
        ax.set_ylabel("acc@10" if not g.startswith("agent") else "acc@3")
        ax.set_ylim(0, 1)
        ax.set_title(g.replace("issue_clue", "clue on").replace("issue", "clue off"), fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(a.fig) / f"curve_{g.replace(':', '_')}.png")
        plt.close(fig)
    paper_figure(curves, Path(a.fig), plt)
    print("figures in", a.fig)


def paper_figure(curves, outdir: Path, plt):
    """Figure 1: (a) model-free, direction pairs; (b) agents by tool arm, if present."""
    style = {  # name: (label, colour, linestyle)
        "bm25": ("BM25", "#555555", "-"),
        "emb": ("released embeddings", "#9a9a9a", ":"),
        "undir_bm25": ("BM25 within 3 hops, undirected", "#1f77b4", "--"),
        "dir_bm25": ("BM25 within 3 hops, downstream", "#1f77b4", "-"),
        "ppr_undir": ("PPR + BM25, undirected", "#d62728", "--"),
        "ppr_dir": ("PPR + BM25, directed", "#d62728", "-"),
    }
    agent_groups = sorted(g for g in {k[0] for k in curves} if g.startswith("agent:") and g.endswith(":issue"))
    ncols = 1 + (1 if agent_groups else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 3.6), dpi=170, squeeze=False)
    xs_lab = ["0", "1", "2", "3+"]

    def draw(ax, pts, label, color, ls, off=0.0):
        pts = [p for p in pts if p[0] in xs_lab]
        if not pts:
            return
        xs = [xs_lab.index(p[0]) + off for p in pts]
        ys = [p[1] for p in pts]
        lo = [max(0.0, p[1] - p[2][0]) for p in pts]
        hi = [max(0.0, p[2][1] - p[1]) for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=color, capsize=2, elinewidth=0.8, alpha=0.8)
        ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=3.5, lw=1.6, label=label)

    ax = axes[0][0]
    for i, (m, (lab, col, ls)) in enumerate(style.items()):
        if ("issue", m) in curves:
            draw(ax, curves[("issue", m)], lab, col, ls, off=(i - 2.5) * 0.03)
    n_by = {p[0]: p[3] for p in curves.get(("issue", "bm25"), [])}
    ax.set_xticks(range(4), [f"{x}\n(n={n_by.get(x, 0)})" for x in xs_lab])
    ax.set_xlabel("symptom-to-cause distance d (call hops)")
    ax.set_ylabel("bug in top 10 (acc@10)")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) model-free localization, report without the name", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    if agent_groups:
        ax = axes[0][1]
        acol = {"text": ("grep + read", "#555555"), "graph": ("+ official graph tools", "#9467bd"),
                "callee": ("+ downstream callees", "#2ca02c")}
        for g in agent_groups:
            model = g.split(":")[1].replace("gemma-4-", "").replace("-it-qat-w4a16-ct", "").upper()
            ls = "-" if "E4B" in model else "--"
            for j, (arm, (lab, col)) in enumerate(acol.items()):
                if (g, arm) in curves:
                    draw(ax, curves[(g, arm)], f"{model} {lab}", col, ls, off=(j - 1) * 0.04)
        ax.set_xticks(range(4), xs_lab)
        ax.set_xlabel("symptom-to-cause distance d (call hops)")
        ax.set_ylabel("bug among the agent's 3 answers (acc@3)")
        ax.set_ylim(0, 1.02)
        ax.set_title("(b) Gemma 4 agents, 8 tool calls, report without the name", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "figure1.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
