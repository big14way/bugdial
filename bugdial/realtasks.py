"""How far is the bug in the REAL public tasks?

For each public task: map the gold patch's hunks to graph function nodes, find the library
symbols the issue text mentions, and measure the directed call distance from those mentions
to the gold function(s), plus whether the issue names the gold function outright.

This uses one graph per repo (the latest task's commit) as an approximation of each task's own
graph; node ids are stable across commits for most functions, and unmapped tasks are reported.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from .graph import CodeGraph, mentioned_nodes
from .tasks import word_in

HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@ ?(.*)$")
FILE = re.compile(r"^\+\+\+ b/(\S+)")
DEF = re.compile(r"\b(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)|\bclass\s+([A-Za-z_][A-Za-z0-9_]*)")


def gold_functions(cg: CodeGraph, patch: str) -> set[str]:
    """Gold edit sites: for each hunk, the function whose text contains the hunk's removed/context lines."""
    gold = set()
    cur_file = None
    hunks: list[tuple[str, list[str], str]] = []
    block: list[str] = []
    header = ""
    for ln in patch.splitlines():
        m = FILE.match(ln)
        if m:
            if block and cur_file:
                hunks.append((cur_file, block, header))
            cur_file, block = m.group(1), []
            continue
        h = HUNK.match(ln)
        if h:
            if block and cur_file:
                hunks.append((cur_file, block, header))
            block, header = [], h.group(2)
            continue
        if ln.startswith(("-", " ")) and not ln.startswith("---"):
            block.append(ln[1:])
        elif ln.startswith("+") and not ln.startswith("+++"):
            block.append("\x00" + ln[1:])  # added line marker
    if block and cur_file:
        hunks.append((cur_file, block, header))
    for f, blk, header in hunks:
        if not f.endswith(".py") or "/tests/" in "/" + f or f.startswith(("tests/", "docs", "docs_src/")):
            continue
        mod = f[:-3].removeprefix("src/").replace("/", ".").removesuffix(".__init__")
        cands = [n for n in cg.library if n.startswith(mod + ".")
                 and cg.text[n].lstrip().startswith(("def ", "async def "))]
        removed = [b.strip() for b in blk if not b.startswith("\x00") and b.strip()]
        # removed lines are the most specific evidence; fall back to context lines
        changed_ctx = [b.strip() for b in blk if b.strip() and not b.startswith("\x00")]
        best = None
        for key in (removed if removed else changed_ctx,):
            if not key:
                continue
            for n in cands:
                body = {l.strip() for l in cg.text[n].splitlines()}
                hit = sum(k in body for k in key)
                if hit and (best is None or (hit, -len(cg.text[n])) > (best[0], -len(cg.text[best[1]]))):
                    best = (hit, n)
        if best is None:
            m = DEF.search(header or "")
            name = (m.group(1) or m.group(2)) if m else None
            if name:
                hits = [n for n in cands if n.split(".")[-1] == name]
                if len(hits) == 1:
                    best = (0, hits[0])
        if best:
            gold.add(best[1])
    return gold


PKG = {"Textualize/rich": "rich", "psf/requests": "requests", "fastapi/fastapi": "fastapi", "encode/httpx": "httpx"}


def analyse_per_task(tasks_path: str, comp: Path, with_methods: bool = True):
    """Exact version for Kaggle: each task's own graph and embeddings (graphs/<instance_id>.json)."""
    from .locate import Locator, METHODS
    T = [json.loads(l) for l in open(tasks_path)]
    out = []
    for t in T:
        pkg = PKG.get(t["repo"])
        g = comp / "graphs" / f"{pkg}_{t['base_commit']}.json"
        e = comp / "embeddings" / f"{pkg}_{t['base_commit']}.npz"
        if not g.exists():
            g = comp / "graphs" / f"{t['instance_id']}.json"
            e = comp / "embeddings" / f"{t['instance_id']}.npz"
        if not g.exists():
            continue
        cg = CodeGraph(g, pkg)
        row = _one(cg, t)
        if with_methods and row["gold"]:
            loc = Locator(cg, str(e) if e.exists() else None)
            text = t["problem_statement"] + "\n" + (t.get("hints_text") or "")
            seeds = mentioned_nodes(cg, text)
            for m in METHODS:
                order = loc.rank(m, text, seeds)
                pos = {n: i + 1 for i, n in enumerate(order)}
                row[f"rank_{m}"] = min(pos.get(gf, len(order) + 1) for gf in row["gold"])
        out.append(row)
    return out


def _one(cg: CodeGraph, t: dict) -> dict:
    gold = gold_functions(cg, t["patch"])
    text = t["problem_statement"] + "\n" + (t.get("hints_text") or "")
    anchors = mentioned_nodes(cg, text)
    named = any(word_in(g.split(".")[-1], text) for g in gold) if gold else None
    d = None
    for a in anchors:
        dd = cg.callee_dist(a, max_d=10)
        for g in gold:
            if g in dd and (d is None or dd[g] < d):
                d = dd[g]
    return {"instance_id": t["instance_id"], "repo": t["repo"], "gold": sorted(gold),
            "n_gold": len(gold), "anchors": len(anchors), "named": named, "d": d}


def analyse(tasks_path: str, graphs: dict[str, tuple[str, str]]):
    T = [json.loads(l) for l in open(tasks_path)]
    out = []
    cgs = {repo: CodeGraph(path, pkg) for repo, (path, pkg) in graphs.items()}
    for t in T:
        cg = cgs.get(t["repo"])
        if cg is None:
            continue
        out.append(_one(cg, t))
        continue
        gold = gold_functions(cg, t["patch"])
        text = t["problem_statement"] + "\n" + (t.get("hints_text") or "")
        anchors = mentioned_nodes(cg, text)
        named = any(word_in(g.split(".")[-1], text) for g in gold) if gold else None
        d = None
        for a in anchors:
            dd = cg.callee_dist(a, max_d=10)
            for g in gold:
                if g in dd and (d is None or dd[g] < d):
                    d = dd[g]
        out.append({"instance_id": t["instance_id"], "repo": t["repo"], "gold": sorted(gold),
                    "n_gold": len(gold), "anchors": len(anchors), "named": named, "d": d})
    return out


def report(rows: list[dict]) -> str:
    mapped = [r for r in rows if r["n_gold"]]
    lines = [f"tasks analysed {len(rows)}, gold mapped {len(mapped)}",
             f"issue names a gold function: {Counter(r['named'] for r in mapped)}"]

    def b(d):
        return "none" if d is None else ("3+" if d >= 3 else str(d))
    lines.append(f"directed distance mention->gold: {Counter(b(r['d']) for r in mapped)}")
    meth = [k[5:] for k in (mapped[0] if mapped else {}) if k.startswith("rank_")]
    for bucket in ["0", "1", "2", "3+", "none", "ALL"]:
        rr = [r for r in mapped if bucket == "ALL" or b(r["d"]) == bucket]
        if not rr or not meth:
            continue
        parts = [f"{m}={sum(r['rank_'+m] <= 10 for r in rr)/len(rr):.2f}" for m in meth]
        lines.append(f"  acc@10 d={bucket:<4} n={len(rr):<3} " + " ".join(parts))
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp")
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--out", default="out/real_tasks_distance.jsonl")
    a = ap.parse_args()
    if a.comp:
        rows = analyse_per_task(a.tasks, Path(a.comp))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(report(rows))
        return 0
    graphs = {
        "Textualize/rich": ("data/graphs/graphs/rich_19c67b9a3479841e9133bea94607c89ee931d3fc.json", "rich"),
        "psf/requests": ("data/graphs/graphs/requests_6f205ff422bccd5e4c4fc0b64c5f3e7df5181db6.json", "requests"),
        "fastapi/fastapi": ("data/graphs/graphs/fastapi_f1d750fddad90e2faad6c3cc5c85c7687b787e89.json", "fastapi"),
    }
    rows = analyse("data/tasks.jsonl", graphs)
    Path("out").mkdir(exist_ok=True)
    Path("out/real_tasks_distance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    mapped = [r for r in rows if r["n_gold"]]
    print(f"tasks analysed {len(rows)}, gold mapped {len(mapped)}")
    print("issue names a gold function:", Counter(r["named"] for r in mapped))
    print("issue mentions >=1 library symbol:", sum(r["anchors"] > 0 for r in mapped))

    def b(d):
        return "unreachable/none" if d is None else ("3+" if d >= 3 else str(d))
    print("directed distance from mentioned symbols to gold:", Counter(b(r["d"]) for r in mapped))
    print("…among issues that do NOT name the gold function:",
          Counter(b(r["d"]) for r in mapped if r["named"] is False))
    for repo in sorted({r["repo"] for r in mapped}):
        rr = [r for r in mapped if r["repo"] == repo]
        print(f"  {repo}: n={len(rr)} named={sum(bool(r['named']) for r in rr)} "
              f"d0={sum(r['d']==0 for r in rr)} d1={sum(r['d']==1 for r in rr)} d2+={sum((r['d'] or 0)>=2 for r in rr)} none={sum(r['d'] is None for r in rr)}")


if __name__ == "__main__":
    sys.exit(main())
