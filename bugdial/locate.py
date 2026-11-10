"""Model-free localization baselines over the released graph and embeddings.

Every method ranks the library's functions for one issue; we score the rank of the buggy function v.
  bm25        lexical BM25 over function source (what grep/keyword search approximates)
  emb         released node embeddings, seeded from symbols the issue mentions (search_similar_code)
  graph_undir hop distance from mentioned symbols, both directions (get_code_neighbors style)
  graph_dir   hop distance following calls downstream from mentioned symbols (callees only)
  ppr_dir     personalized PageRank on the directed call graph, fused with BM25 (RRF)
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

from .graph import CodeGraph, mentioned_nodes

SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def toks(s: str) -> list[str]:
    out = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s):
        lw = w.lower()
        out.append(lw)
        parts = [p.lower() for p in SPLIT.findall(w)]
        if len(parts) > 1:
            out.extend(parts)
    return [t for t in out if len(t) > 1]


class Locator:
    def __init__(self, cg: CodeGraph, emb_npz: str | None):
        self.cg = cg
        self.fns = sorted(n for n in cg.library if cg.text[n].lstrip().startswith(("def ", "async def ")))
        docs = [toks(n.replace(".", " ") + " " + cg.text[n]) for n in self.fns]
        self.bm = BM25Okapi(docs)
        self.E = {}
        if emb_npz:
            z = np.load(emb_npz)
            for k in z.files:
                key = k[:-4] if k.endswith(".npy") else k
                vec = z[k].astype(np.float32)
                nrm = np.linalg.norm(vec)
                if nrm > 0:
                    self.E[key] = vec / nrm
        self.Ud = cg.L.to_undirected(as_view=True)

    def _bm25(self, text: str) -> dict[str, float]:
        sc = self.bm.get_scores(toks(text))
        return dict(zip(self.fns, sc))

    @staticmethod
    def _order(score: dict[str, float]) -> list[str]:
        return [k for k, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]

    def rank(self, method: str, issue: str, seeds: list[str] | None = None) -> list[str]:
        bm = self._bm25(issue)
        bm_order = self._order(bm)
        if method == "bm25":
            return bm_order
        if seeds is None:
            seeds = mentioned_nodes(self.cg, issue)
        bm_rank = {n: i for i, n in enumerate(bm_order)}
        if method == "emb":
            S = [self.E[s] for s in seeds if s in self.E]
            if not S:
                return bm_order
            M = np.stack(S)
            sc = {}
            for n in self.fns:
                e = self.E.get(n)
                sc[n] = float((M @ e).max()) if e is not None else -1.0
            # seeds themselves first is what the tool would show; keep them at their similarity (1.0)
            return [k for k, _ in sorted(sc.items(), key=lambda kv: (-kv[1], bm_rank[kv[0]]))]
        if method in ("graph_undir", "graph_dir"):
            dist: dict[str, int] = {}
            for s in seeds:
                dd = (nx.single_source_shortest_path_length(self.cg.Ld, s, cutoff=10) if method == "graph_dir"
                      else nx.single_source_shortest_path_length(self.Ud, s, cutoff=10)) if s in self.cg.L else {}
                for n, d in dd.items():
                    if n not in dist or d < dist[n]:
                        dist[n] = d
            return sorted(self.fns, key=lambda n: (dist.get(n, 99), bm_rank[n]))
        if method in ("ppr_dir", "ppr_undir"):
            G = self.cg.Ld if method == "ppr_dir" else self.cg.L
            pers = {s: 1.0 for s in seeds if s in G}
            if not pers:
                return bm_order
            pr = nx.pagerank(G, alpha=0.85, personalization=pers, max_iter=200)
            pr_order = sorted(self.fns, key=lambda n: (-pr.get(n, 0.0), bm_rank[n]))
            pr_rank = {n: i for i, n in enumerate(pr_order)}
            k = 10
            fused = {n: 1 / (k + pr_rank[n]) + 1 / (k + bm_rank[n]) for n in self.fns}
            return self._order(fused)
        if method in ("dir_bm25", "undir_bm25"):
            # search space = functions within K hops of what the report mentions (downstream only for dir),
            # ranked by text relevance; everything else after, also by text relevance
            K = 3
            reach: set[str] = set()
            for s in seeds:
                if s in self.cg.L:
                    G = self.cg.Ld if method == "dir_bm25" else self.Ud
                    reach |= set(nx.single_source_shortest_path_length(G, s, cutoff=K))
            return sorted(self.fns, key=lambda n: (n not in reach, bm_rank[n]))
        raise ValueError(method)


METHODS = ["bm25", "emb", "graph_undir", "graph_dir", "undir_bm25", "dir_bm25", "ppr_undir", "ppr_dir"]


def evaluate(loc: Locator, tasks: list[dict], methods=METHODS) -> list[dict]:
    rows = []
    for t in tasks:
        for variant in ("issue", "issue_clue"):
            text = t[variant]
            seeds = mentioned_nodes(loc.cg, text)
            for m in methods:
                order = loc.rank(m, text, seeds)
                r = order.index(t["v"]) + 1 if t["v"] in order else len(order) + 1
                rows.append({"task_id": t["task_id"], "repo": t["repo"], "d": t["d"], "variant": variant,
                             "name_in_issue": t["name_in_issue"], "method": m, "rank": r})
    return rows


def summarize(rows: list[dict]) -> str:
    def bucket(d):
        return "none" if d is None else ("3+" if d >= 3 else str(d))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], bucket(r["d"]), r["method"])].append(r["rank"])
    lines = ["variant      d     n   method       acc@1  acc@5  acc@10  MRR"]
    for (var, b, m), rs in sorted(agg.items()):
        n = len(rs)
        a1 = sum(x <= 1 for x in rs) / n
        a5 = sum(x <= 5 for x in rs) / n
        a10 = sum(x <= 10 for x in rs) / n
        mrr = sum(1 / x for x in rs) / n
        lines.append(f"{var:<12} {b:<5} {n:<4} {m:<12} {a1:.2f}   {a5:.2f}   {a10:.2f}   {mrr:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--emb")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cg = CodeGraph(a.graph, a.package)
    loc = Locator(cg, a.emb)
    tasks = [json.loads(l) for l in open(a.tasks)]
    rows = evaluate(loc, tasks)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(summarize(rows))


if __name__ == "__main__":
    main()
