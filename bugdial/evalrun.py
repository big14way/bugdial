"""Run the localization agent over synthetic and real tasks against a local vLLM server.

  python -m bugdial.evalrun --comp <comp_dir> --tasks a.jsonl b.jsonl --real <real_tasks.jsonl> \
     --model gemma-4-e4b-it-qat-w4a16-ct --base-url http://127.0.0.1:8000/v1 --out episodes.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .agent import ARMS, RepoView, run_episode
from .graph import CodeGraph

SHAS = {"rich": "19c67b9a3479841e9133bea94607c89ee931d3fc",
        "requests": "6f205ff422bccd5e4c4fc0b64c5f3e7df5181db6",
        "fastapi": "f1d750fddad90e2faad6c3cc5c85c7687b787e89"}
INSTANCE = {"rich": "rich_4079", "requests": "requests_7505", "fastapi": "fastapi_15800"}
PKG = {"Textualize/rich": "rich", "psf/requests": "requests", "fastapi/fastapi": "fastapi"}


class _NS(dict):
    """dict with attribute access, recursively (mimics the openai response objects we read)."""
    def __getattr__(self, k):
        v = self.get(k)
        if isinstance(v, dict):
            return _NS(v)
        if isinstance(v, list):
            return [_NS(x) if isinstance(x, dict) else x for x in v]
        return v


class MiniClient:
    """Minimal OpenAI-compatible chat client over urllib (no extra packages on offline GPU sessions)."""
    def __init__(self, base_url: str, timeout: int = 600):
        from types import SimpleNamespace
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **body):
        import urllib.request
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return _NS(data)


def load_files(tgz: Path) -> dict[str, str]:
    """All .py files of a snapshot, keyed by path relative to the repo root (the dir holding pyproject/setup)."""
    raw = {}
    roots = []
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name.removeprefix("./")
            if name.rsplit("/", 1)[-1] in ("pyproject.toml", "setup.py", "setup.cfg"):
                roots.append(name.rsplit("/", 1)[0] if "/" in name else "")
            if name.endswith(".py"):
                try:
                    raw[name] = tf.extractfile(m).read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    pass
    root = min(roots, key=lambda r: r.count("/")) if roots else ""
    pre = root + "/" if root else ""
    return {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}


def load_emb(npz: Path) -> dict[str, np.ndarray]:
    if not npz.exists():
        return {}
    z = np.load(npz)
    out = {}
    for k in z.files:
        v = z[k].astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            out[k[:-4] if k.endswith(".npy") else k] = v / n
    return out


def mutated_view(base: dict[str, str], task: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Files with the planted bug applied, and the buggy function's mutated text for read_function."""
    m, sp = task["mutation"], task["span"]
    path = sp["file"]
    files = dict(base)
    if path not in files:
        return files, {}
    lines = files[path].splitlines(keepends=True)
    ln = lines[m["line"] - 1]
    if ln[m["col"]:m["end_col"]] == m["old"]:
        lines[m["line"] - 1] = ln[:m["col"]] + m["new"] + ln[m["end_col"]:]
    files[path] = "".join(lines)
    fn_lines = files[path].splitlines()[sp["start"] - 1: sp["end"]]
    ind = len(fn_lines[0]) - len(fn_lines[0].lstrip()) if fn_lines else 0
    txt = "\n".join(l[ind:] if l[:ind].strip() == "" else l for l in fn_lines)
    return files, {task["v"]: txt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--real")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="text,graph,callee")
    ap.add_argument("--variants", default="issue,issue_clue")
    ap.add_argument("--max-per-bucket", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    client = MiniClient(a.base_url)
    comp = Path(a.comp)

    # ---- synthetic tasks, balanced by distance bucket
    rng = random.Random(a.seed)
    synth = [json.loads(l) for p in a.tasks for l in open(p)]
    buckets: dict[str, list] = {}
    for t in synth:
        b = "none" if t["d"] is None else ("3+" if t["d"] >= 3 else str(t["d"]))
        buckets.setdefault(f"{t['repo']}:{b}", []).append(t)
    chosen = []
    for k, ts in sorted(buckets.items()):
        rng.shuffle(ts)
        chosen += ts[: a.max_per_bucket]
    print("synthetic chosen:", {k: min(len(v), a.max_per_bucket) for k, v in sorted(buckets.items())}, flush=True)

    views, cache = {}, {}
    for pkg in sorted({t["repo"] for t in chosen} | ({"rich", "requests", "fastapi"} if a.real else set())):
        cg = CodeGraph(comp / "graphs" / f"{pkg}_{SHAS[pkg]}.json", pkg)
        files = load_files(comp / "snapshots" / f"{INSTANCE[pkg]}.tgz")
        print(pkg, "files", len(files), "e.g.", sorted(files)[:3], flush=True)
        views[pkg] = (cg, files, load_emb(comp / "embeddings" / f"{pkg}_{SHAS[pkg]}.npz"))

    jobs = []
    for t in chosen:
        for arm in a.arms.split(","):
            for var in a.variants.split(","):
                jobs.append(("synth", t, arm, var))

    # ---- real tasks: each task's own graph/snapshot (gold from realtasks output)
    real = []
    if a.real:
        R = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.real)}
        T = [json.loads(l) for l in open(comp / "tasks.jsonl")]
        for t in T:
            r = R.get(t["instance_id"])
            if not r or not r["gold"] or t["repo"] not in PKG:
                continue
            real.append({"task_id": t["instance_id"], "repo": PKG[t["repo"]], "v": None, "gold": r["gold"],
                         "d": r["d"], "named": r["named"], "base_commit": t["base_commit"],
                         "issue": t["problem_statement"] + ("\n\nHints:\n" + t["hints_text"] if t.get("hints_text") else "")})
        for t in real:
            for arm in a.arms.split(","):
                jobs.append(("real", t, arm, "issue"))
    print("jobs:", len(jobs), flush=True)

    lock = threading.Lock()
    real_views: dict[str, RepoView] = {}

    def view_for(kind, t):
        pkg = t["repo"]
        if kind == "synth":
            cg, files, emb = views[pkg]
            mfiles, override = mutated_view(files, t)
            return RepoView(mfiles, cg, emb), override
        key = t["task_id"]
        with lock:
            if key not in real_views:
                g = comp / "graphs" / f"{pkg}_{t['base_commit']}.json"
                cg = CodeGraph(g, pkg)
                files = load_files(comp / "snapshots" / f"{t['task_id']}.tgz")
                real_views[key] = RepoView(files, cg, load_emb(comp / "embeddings" / f"{pkg}_{t['base_commit']}.npz"))
            return real_views[key], None

    def work(job):
        kind, t, arm, var = job
        view, override = view_for(kind, t)
        ep = run_episode(client, a.model, view, t, arm, var, a.max_steps, override)
        gold = [t["v"]] if kind == "synth" else t["gold"]
        rank = next((i + 1 for i, r in enumerate(ep.resolved) if r in gold), None)
        return {"kind": kind, "task_id": t["task_id"], "repo": t["repo"], "d": t["d"], "arm": arm, "variant": var,
                "model": a.model, "rank": rank, "submitted": ep.submitted, "resolved": ep.resolved,
                "calls": ep.calls, "steps": ep.steps, "error": ep.error, "secs": round(ep.secs, 1),
                "empty_retry": getattr(ep, "empty_retry", False),
                "named": t.get("named", t.get("name_in_issue"))}

    done = 0
    with open(a.out, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as e:  # noqa: BLE001
                row = {"error": f"driver: {type(e).__name__}: {e}"[:300]}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    print("eval done", done, flush=True)


if __name__ == "__main__":
    main()
