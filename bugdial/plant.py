"""Plant one-line bugs in library functions and keep the ones that break existing tests.

Usage:
  python -m bugdial.plant --repo repos/rich --graph data/graphs/.../rich_<sha>.json --package rich \
      --out out/rich_mutants.jsonl --workers 8 --limit 600 --seed 0
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool, current_process
from pathlib import Path

from .graph import CodeGraph, find_span, import_root, pytest_id
from . import mutate as M

PY = sys.executable


def run_pytest(repo: Path, ids: list[str], timeout: int) -> tuple[dict[str, dict], str]:
    """Run pytest on the given node ids. Returns {graph_style_test_id: {"ok": bool, "msg": str}}."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as fh:
        xml = fh.name
    env = dict(os.environ, PYTHONPATH=f"{import_root(repo)}{os.pathsep}{repo}", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               COLUMNS="80", TERM="dumb", NO_COLOR="1")
    env.pop("FORCE_COLOR", None)
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "--tb=short",
           "-W", "ignore", f"--junitxml={xml}", *ids]
    try:
        subprocess.run(cmd, cwd=repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, "timeout"
    res: dict[str, dict] = {}
    try:
        root = ET.parse(xml).getroot()
    except Exception:
        return {}, "noxml"
    finally:
        Path(xml).unlink(missing_ok=True)
    for tc in root.iter("testcase"):
        tid = (tc.get("classname", "") + "." + tc.get("name", "").split("[")[0]).strip(".")
        bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        skipped = tc.find("skipped") is not None
        r = res.setdefault(tid, {"ok": True, "msg": "", "skipped": True})
        r["skipped"] = r["skipped"] and skipped
        if bad is not None:
            r["ok"] = False
            if not r["msg"]:
                r["msg"] = ((bad.get("message") or "") + "\n@@BODY@@\n" + (bad.text or ""))[:6000]
    return res, "ok"


# ---- worker state -------------------------------------------------------------
_W: dict = {}


def _init(repo: str, work_root: str, graph: str, package: str, baseline: dict, timeout: int, max_tests: int):
    idx = current_process()._identity[0] if current_process()._identity else 0
    dst = Path(work_root) / f"w{idx}"
    if not dst.exists():
        shutil.copytree(repo, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    _W.update(repo=dst, cg=CodeGraph(graph, package), baseline=baseline, timeout=timeout, max_tests=max_tests)


def _try_function(args):
    v, seed, tries = args
    repo: Path = _W["repo"]
    cg: CodeGraph = _W["cg"]
    span = find_span(repo, v)
    if span is None:
        return {"v": v, "status": "nospan"}
    path = repo / span.file
    src = path.read_text()
    import ast
    fn = M.find_function(ast.parse(src), span.start, span.end)
    if fn is None:
        return {"v": v, "status": "nofn"}
    cands = M.candidates(src, fn)
    if not cands:
        return {"v": v, "status": "nocands"}
    # tests that reach v in the graph and pass at baseline, nearest first
    reach = [t for t in cg.tests_reaching(v) if _W["baseline"].get(t) is True]
    if not reach:
        return {"v": v, "status": "notests"}
    rev = cg.Gd.reverse(copy=False)
    import networkx as nx
    dist = nx.single_source_shortest_path_length(rev, v)
    reach.sort(key=lambda t: (dist.get(t, 99), t))
    reach = reach[: _W["max_tests"]]
    ids = [p for p in (pytest_id(repo, t) for t in reach) if p]
    rng = random.Random(f"{seed}:{v}")
    attempts = []
    for m in M.pick(cands, rng, tries):
        try:
            mutated = M.apply(src, m)
        except Exception as e:  # noqa: BLE001
            attempts.append({"mutation": m.to_dict(), "status": f"apply_error:{type(e).__name__}"})
            continue
        path.write_text(mutated)
        t0 = time.time()
        try:
            res, st = run_pytest(repo, ids, _W["timeout"])
        finally:
            path.write_text(src)
        dt = round(time.time() - t0, 2)
        fails = [{"test": t, "msg": r["msg"]} for t, r in res.items()
                 if not r["ok"] and _W["baseline"].get(t) is True]
        att = {"mutation": m.to_dict(), "status": st, "secs": dt, "n_fail": len(fails), "n_run": len(res)}
        attempts.append(att)
        if st == "ok" and fails:
            bug_patch = "".join(difflib.unified_diff(src.splitlines(True), mutated.splitlines(True),
                                                     f"a/{span.file}", f"b/{span.file}"))
            fix_patch = "".join(difflib.unified_diff(mutated.splitlines(True), src.splitlines(True),
                                                     f"a/{span.file}", f"b/{span.file}"))
            return {"v": v, "status": "killed", "span": span.__dict__, "mutation": m.to_dict(),
                    "bug_patch": bug_patch, "fix_patch": fix_patch, "fails": fails,
                    "selected": len(ids), "attempts": attempts}
    return {"v": v, "status": "survived", "attempts": attempts, "selected": len(ids)}


def baseline_run(repo: Path, cg: CodeGraph, timeout: int) -> dict[str, bool]:
    ids = sorted({p for p in (pytest_id(repo, t) for t in cg.tests) if p})
    files = sorted({i.split("::")[0] for i in ids})
    res, st = run_pytest(repo, files, timeout)
    print(f"baseline: {st}, {len(res)} test functions, {sum(r['ok'] for r in res.values())} passing", flush=True)
    bad = [(t, r["msg"]) for t, r in res.items() if not r["ok"]][:3]
    for t, m in bad:
        print(f"  baseline failure {t}:\n    " + m[:700].replace("\n", "\n    "), flush=True)
    return {t: (r["ok"] and not r["skipped"]) for t, r in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-tests", type=int, default=120)
    ap.add_argument("--prev", help="earlier mutants.jsonl: only retry functions that survived there")
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    cg = CodeGraph(a.graph, a.package)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bl_path = out.with_suffix(".baseline.json")
    if bl_path.exists():
        baseline = json.loads(bl_path.read_text())
    else:
        baseline = baseline_run(repo, cg, timeout=1800)
        bl_path.write_text(json.dumps(baseline))

    done = set()
    if out.exists():
        done = {json.loads(l)["v"] for l in out.open()}
    if a.prev and Path(a.prev).exists():
        prev = [json.loads(l) for l in open(a.prev)]
        done |= {r["v"] for r in prev if r["status"] != "survived"}
        print(f"prev: skipping {len(done)} functions already decided", flush=True)
    fns = sorted(n for n in cg.library if cg.text[n].lstrip().startswith(("def ", "async def ")) and n.split(".")[-1] not in ("__repr__", "__rich_repr__"))
    rng = random.Random(a.seed)
    rng.shuffle(fns)
    todo = [f for f in fns if f not in done][: a.limit]
    print(f"library functions: {len(cg.library)}, todo: {len(todo)}", flush=True)
    work_root = Path(tempfile.gettempdir()) / f"bugdial_{a.package}"
    work_root.mkdir(exist_ok=True)
    stats = {}
    t0 = time.time()
    with Pool(a.workers, initializer=_init,
              initargs=(str(repo), str(work_root), a.graph, a.package, baseline, a.timeout, a.max_tests)) as pool, \
            out.open("a") as fh:
        for i, r in enumerate(pool.imap_unordered(_try_function, [(v, a.seed, a.tries) for v in todo]), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            if i % 20 == 0:
                print(f"[{i}/{len(todo)}] {stats} {time.time()-t0:.0f}s", flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
