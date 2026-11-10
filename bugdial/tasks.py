"""Turn killed mutants into localization tasks with a measured symptom-to-cause distance.

The issue is written from one failing test: the reproduction is the test body, the error is
the assertion/exception message. Distance d is a property of (issue text, graph): the shortest
directed call-path from any library symbol the issue mentions to the buggy function v.
A clue variant appends one sentence naming v (lexical cue on).
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

from .graph import CodeGraph, find_span, mentioned_nodes, module_file

NOISE = re.compile(r"^(?:[\w./\\-]+\.py:\d+(?::|\s|$)|_+ .* _+$|=+.*=+$)")


def clean_message(msg: str, max_lines: int = 18) -> str:
    """The error a user would paste: pytest's `E ` lines (assertion/exception), no code echo, no frames."""
    head, _, body = msg.partition("\n@@BODY@@\n")
    raw = [ln[1:].rstrip() for ln in body.splitlines() if ln.startswith("E ")]
    ind = min((len(s) - len(s.lstrip()) for s in raw if s.strip()), default=0)
    elines = [s[ind:] for s in raw]
    if not elines:
        elines = [l.rstrip() for l in head.splitlines() if l.strip() and not NOISE.match(l.strip())]
    dedup = []
    for s in elines:
        if s.strip() and (not dedup or dedup[-1] != s):
            dedup.append(s)
    if len(dedup) > max_lines:
        dedup = dedup[:max_lines] + ["..."]
    return "\n".join(dedup)[:1500]


def test_source(repo: Path, test_id: str, max_lines: int = 40) -> str | None:
    hit = module_file(repo, test_id)
    if not hit:
        return None
    path, qual = hit
    tree = ast.parse(path.read_text())
    body, node = tree.body, None
    for q in qual:
        node = next((b for b in body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and b.name == q), None)
        if node is None:
            return None
        body = node.body
    lines = path.read_text().splitlines()[node.lineno - 1: node.end_lineno]
    ind = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    lines = [l[ind:] for l in lines]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["    ..."]
    return "\n".join(lines)


def make_issue(test_src: str, err: str, entry_hint: str | None) -> str:
    who = f"`{entry_hint}`" if entry_hint else "the library"
    return (f"Unexpected behaviour from {who}\n\n"
            "I'm getting a wrong result. Here is a minimal reproduction taken from my code:\n\n"
            f"```python\n{test_src}\n```\n\n"
            "It fails with:\n\n"
            f"```\n{err}\n```\n")


def clue_sentence(v: str) -> str:
    return f"\nI suspect the problem is in `{'.'.join(v.split('.')[-2:])}`.\n"


def entry_points(cg: CodeGraph, t: str) -> set[str]:
    """First library nodes reached from test t, walking through non-library helpers."""
    seen, frontier, hits = {t}, [t], set()
    while frontier:
        nxt = []
        for u in frontier:
            for x in cg.Gd.successors(u):
                if x in seen:
                    continue
                seen.add(x)
                if x in cg.library:
                    hits.add(x)
                elif not x.startswith(cg.package + "."):
                    nxt.append(x)
        frontier = nxt
    return hits


def word_in(name: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text) is not None


def dbucket(d):
    return "none" if d is None else ("3+" if d >= 3 else str(d))


def build(repo: Path, cg: CodeGraph, rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    tasks = []
    counts: dict[str, int] = {}
    rows = list(rows)
    rng.shuffle(rows)
    for r in rows:
        if r.get("status") != "killed":
            continue
        v = r["v"]
        short = v.split(".")[-1]
        options = []
        for f in r["fails"]:
            t = f["test"]
            src = test_source(repo, t)
            if not src:
                continue
            err = clean_message(f["msg"])
            if not err.strip():
                continue
            entries = entry_points(cg, t)
            body = make_issue(src, err, None)
            anchors = mentioned_nodes(cg, body)
            # anchors the reader can see = symbols named in the text; graph entry points of the test that are
            # mentioned in the text are the natural "symptom" functions
            dists = {}
            for a in anchors:
                dd = cg.callee_dist(a, max_d=10)
                if v in dd:
                    dists[a] = dd[v]
            d = min(dists.values()) if dists else None
            und = None
            for a in anchors:
                du = cg.undirected_dist(a, max_d=10)
                if v in du:
                    und = du[v] if und is None else min(und, du[v])
            options.append({"test": t, "issue": body, "anchors": anchors, "d": d, "d_undirected": und,
                            "entries": sorted(entries), "name_in_issue": word_in(short, body)})
        if not options:
            continue
        # Prefer reproductions whose text does not already name v (unless d == 0, where it must), then
        # balance the distance buckets greedily: pick the reachable bucket with the fewest tasks so far.
        rng.shuffle(options)
        ok = [o for o in options if o["d"] is not None and (o["d"] == 0 or not o["name_in_issue"])] or \
             [o for o in options if o["d"] is not None] or options
        o = min(ok, key=lambda o: (counts.get(dbucket(o["d"]), 0), o["name_in_issue"]))
        counts[dbucket(o["d"])] = counts.get(dbucket(o["d"]), 0) + 1
        tasks.append({
            "task_id": f"{cg.package}__{v}__{r['mutation']['op']}_{r['mutation']['line']}",
            "repo": cg.package, "v": v, "span": r["span"], "mutation": r["mutation"],
            "bug_patch": r["bug_patch"], "fix_patch": r["fix_patch"],
            "test": o["test"], "n_failing_tests": len(r["fails"]),
            "issue": o["issue"], "issue_clue": o["issue"] + clue_sentence(v),
            "anchors": o["anchors"], "d": o["d"], "d_undirected": o["d_undirected"],
            "name_in_issue": o["name_in_issue"], "entries": o["entries"],
        })
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cg = CodeGraph(a.graph, a.package)
    rows = [json.loads(l) for l in open(a.mutants)]
    tasks = build(Path(a.repo).resolve(), cg, rows, a.seed)
    Path(a.out).write_text("".join(json.dumps(t) + "\n" for t in tasks))
    from collections import Counter
    print(len(tasks), "tasks; d:", sorted(Counter(t["d"] for t in tasks).items(), key=lambda x: (x[0] is None, x[0] or 0)),
          "; name_in_issue:", sum(t["name_in_issue"] for t in tasks))


if __name__ == "__main__":
    main()
