"""Package bugdial/ into a self-contained Kaggle script kernel.

  python kaggle_jobs/build.py plant   -> kaggle_jobs/plant/{kernel.py,kernel-metadata.json}
Then:  kaggle kernels push -p kaggle_jobs/plant
"""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "godswillidolor"


def pack_src() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted((ROOT / "bugdial").glob("*.py")):
            tf.add(p, arcname=f"bugdial/{p.name}")
    return base64.b64encode(buf.getvalue()).decode()


PLANT = r'''
import base64, glob, io, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
SRC_B64 = "__SRC__"
CONFIG = json.loads(__CONFIG__)
W = Path("/kaggle/working"); SRC = W / "src"; OUT = W / "out"
SRC.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(SRC_B64))).extractall(SRC)
sys.path.insert(0, str(SRC))
T0 = time.time()

def sh(cmd, **kw):
    print(f"[{time.time()-T0:7.0f}s] $", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(list(map(str, cmd)), **kw)
    print(f"[{time.time()-T0:7.0f}s] exit {r.returncode}", flush=True)
    return r

hits = glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True)
COMP = Path(hits[0]).parent
print("COMP", COMP, sorted(os.listdir(COMP))[:20], "cpus", os.cpu_count(), sys.version, flush=True)
sh([sys.executable, "-m", "pip", "install", "-q", "rank-bm25", "pytest", "networkx", *CONFIG.get("pip", [])])
ENV = dict(os.environ, PYTHONPATH=str(SRC))

REPOS = {
    "rich": dict(instance="rich_4079", sha="19c67b9a3479841e9133bea94607c89ee931d3fc"),
    "requests": dict(instance="requests_7505", sha="6f205ff422bccd5e4c4fc0b64c5f3e7df5181db6"),
    "fastapi": dict(instance="fastapi_15800", sha="f1d750fddad90e2faad6c3cc5c85c7687b787e89"),
}

if CONFIG.get("graphstats", True):
    from bugdial.graph import CodeGraph
    PK = {"rich": "rich", "requests": "requests", "fastapi": "fastapi", "httpx": "httpx"}
    with open(OUT / "graph_stats.jsonl", "w") as fh:
        for g in sorted((COMP / "graphs").glob("*_*.json")):
            stem = g.stem
            pkg = stem.split("_")[0]
            if pkg not in PK or len(stem.split("_")[-1]) != 40:
                continue
            try:
                st = CodeGraph(g, PK[pkg]).edge_stats()
            except Exception as e:  # noqa: BLE001
                st = {"error": str(e)[:200]}
            fh.write(json.dumps({"graph": stem, **st}) + "\n")
    print("graph stats written", flush=True)

if CONFIG.get("realtasks", True):
    sh([sys.executable, "-m", "bugdial.realtasks", "--comp", COMP, "--tasks", COMP / "tasks.jsonl",
        "--out", OUT / "real_tasks.jsonl"], cwd=SRC, env=ENV)

for pkg in CONFIG["repos"]:
    r = REPOS[pkg]
    dst = Path("/tmp/repos") / pkg
    if not dst.exists():
        dst.mkdir(parents=True)
        sh(["tar", "-xzf", COMP / "snapshots" / f"{r['instance']}.tgz", "-C", dst])
        kids = [p for p in dst.iterdir()]
        if len(kids) == 1 and kids[0].is_dir():
            inner = kids[0]
            for p in inner.iterdir():
                shutil.move(str(p), dst / p.name)
            inner.rmdir()
    print(pkg, "repo:", sorted(os.listdir(dst))[:25], flush=True)
    if (dst / "requirements-dev.txt").exists():
        print((dst / "requirements-dev.txt").read_text()[:800], flush=True)
    for extra in CONFIG.get("pip_" + pkg, []):
        sh([sys.executable, "-m", "pip", "install", "-q", extra])
    if CONFIG.get("wheels_" + pkg):
        # the harness's own recipe: dependencies from the competition's offline wheelhouse
        sh([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", COMP / "wheels", "-e", dst])
        sh([sys.executable, "-m", "pip", "uninstall", "-y", pkg])  # keep the snapshot on PYTHONPATH authoritative
        for extra in CONFIG.get("wheels_extra_" + pkg, []):
            sh([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", COMP / "wheels", extra])
    graph = COMP / "graphs" / f"{pkg}_{r['sha']}.json"
    emb = COMP / "embeddings" / f"{pkg}_{r['sha']}.npz"
    mut = OUT / f"{pkg}_mutants.jsonl"
    sh([sys.executable, "-m", "bugdial.plant", "--repo", dst, "--graph", graph, "--package", pkg, "--out", mut,
        "--workers", str(os.cpu_count() or 4), "--limit", str(CONFIG["limit"].get(pkg, 300)), "--seed", "0",
        "--timeout", "150", "--max-tests", str(CONFIG.get("max_tests", 120))], cwd=SRC, env=ENV)
    tasks = OUT / f"{pkg}_tasks.jsonl"
    sh([sys.executable, "-m", "bugdial.tasks", "--repo", dst, "--graph", graph, "--package", pkg,
        "--mutants", mut, "--out", tasks], cwd=SRC, env=ENV)
    sh([sys.executable, "-m", "bugdial.locate", "--graph", graph, "--package", pkg, "--emb", emb,
        "--tasks", tasks, "--out", OUT / f"{pkg}_locate.jsonl"], cwd=SRC, env=ENV)
print("ALL DONE", time.time() - T0, flush=True)
'''


EVAL = r'''
import base64, glob, io, json, os, shutil, subprocess, sys, tarfile, time, importlib
from pathlib import Path
SRC_B64 = "__SRC__"
CONFIG = json.loads(__CONFIG__)
W = Path("/kaggle/working"); SRC = W / "src"; OUT = W / "out"
SRC.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(SRC_B64))).extractall(SRC)
sys.path.insert(0, str(SRC))
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

for k, v in {"TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
             "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OPENAI_API_KEY": "EMPTY"}.items():
    os.environ[k] = v

# wheelhouse install, as in the official starter notebook
whl_dirs = sorted({str(Path(p).parent) for p in glob.glob("/kaggle/input/**/vllm-*.whl", recursive=True)})
log("wheelhouse", whl_dirs)
WH = Path(whl_dirs[0])
for pth in glob.glob("/usr/local/lib/python*/dist-packages/*cutlass*.pth") + glob.glob("/usr/local/lib/python*/site-packages/*cutlass*.pth"):
    try: os.unlink(pth)
    except OSError: pass
tmp = Path("/tmp/wheelhouse"); tmp.mkdir(exist_ok=True)
for w in WH.glob("*.whl"):
    if "cutlass" in w.name.lower():
        continue
    name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
    t = tmp / name
    if not t.exists():
        try: os.symlink(w, t)
        except OSError: shutil.copyfile(w, t)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *sorted(map(str, tmp.glob("*.whl")))], check=True)
importlib.invalidate_caches()
log("wheels installed")

import torch
from adk_submission import VllmConfig, VllmServer
COMP = Path(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True)[0]).parent
prev = {Path(p).name: Path(p) for p in glob.glob("/kaggle/input/**/*_tasks.jsonl", recursive=True)}
real = glob.glob("/kaggle/input/**/real_tasks.jsonl", recursive=True)
log("COMP", COMP, "inputs", list(prev), real)
gpus = torch.cuda.device_count()
tp = 4 if gpus >= 4 else (2 if gpus >= 2 else 1)
log("gpus", gpus, [torch.cuda.get_device_name(i) for i in range(gpus)])
ENV = dict(os.environ, PYTHONPATH=str(SRC))
for model in CONFIG["models"]:
    std = Path(f"/kaggle/input/models/google/gemma-4/other/{model}/2")
    if (std / "config.json").exists():
        mp = std
    else:
        cfgs = glob.glob(f"/kaggle/input/**/{model}/**/config.json", recursive=True)
        mp = Path(sorted(cfgs, key=len)[0]).parent
    log("model", model, mp)
    cfg = VllmConfig(model=str(mp), port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
                     max_model_len=32768, dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto",
                     gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=False, max_loras=1,
                     max_lora_rank=128, scheduler_reserve_full_isl=False, tensor_parallel_size=tp,
                     startup_timeout=60 * 20,
                     extra_args=(("--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')
                                 if model.startswith("gemma-4-12b") else ()))
    server = VllmServer(cfg)
    server.start()
    import urllib.request
    served = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/v1/models").read())["data"][0]["id"]
    log("served", served)
    args = [sys.executable, "-m", "bugdial.evalrun", "--comp", str(COMP), "--model", served,
            "--out", str(OUT / f"episodes_{model}.jsonl"), "--arms", CONFIG.get("arms", "text,graph,callee"),
            "--variants", CONFIG.get("variants", "issue,issue_clue"),
            "--max-per-bucket", str(CONFIG.get("max_per_bucket", 60)), "--workers", str(CONFIG.get("workers", 24)),
            "--max-steps", str(CONFIG.get("max_steps", 8))]
    tasks = [str(p) for n, p in sorted(prev.items()) if n.split("_")[0] in CONFIG.get("repos", ["rich", "requests"])]
    if tasks:
        args += ["--tasks", *tasks]
    if real and CONFIG.get("real", True):
        args += ["--real", real[0]]
    log("$", " ".join(args))
    subprocess.run(args, cwd=SRC, env=ENV)
    server.stop()
    log("done", model)
log("ALL DONE")
'''


def build(job: str, config: dict, kind: str = "plant", sources: list[str] | None = None):
    d = ROOT / "kaggle_jobs" / job
    d.mkdir(parents=True, exist_ok=True)
    tmpl = PLANT if kind == "plant" else EVAL
    code = tmpl.replace("__SRC__", pack_src()).replace("__CONFIG__", repr(json.dumps(config)))
    (d / "kernel.py").write_text(code)
    meta = {
        "id": f"{USER}/bugdial-{job}",
        "title": f"bugdial-{job}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": kind == "eval",
        "enable_internet": kind != "eval",
        "competition_sources": ["gemma-4-developer-agent"],
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"] if kind == "eval" else [],
        "kernel_sources": sources or [],
        "model_sources": [f"google/gemma-4/Other/{m}/2" for m in config.get("models", [])],
    }
    if kind == "eval":
        meta["machine_shape"] = config.get("machine_shape", "NvidiaL4")
        meta["docker_image"] = ("gcr.io/kaggle-private-byod/python@sha256:"
                                "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("built", d, (d / "kernel.py").stat().st_size, "bytes")


if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else "plant"
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    if job.startswith("eval"):
        cfg = {"models": ["gemma-4-e4b-it-qat-w4a16-ct"], "repos": ["rich", "requests"],
               "arms": "text,graph,callee", "variants": "issue,issue_clue", "max_per_bucket": 60,
               "workers": 24, "max_steps": 8, "real": True}
        cfg.update(extra)
        build(job, cfg, kind="eval", sources=extra.get("sources", [f"{USER}/bugdial-plant"]))
    else:
        cfg = {
            "repos": ["rich", "requests"],
            "limit": {"rich": 520, "requests": 250, "fastapi": 300},
            "pip_requests": ["pytest-httpbin", "pytest-mock", "trustme"],
            "max_tests": 120,
            "realtasks": True,
            "graphstats": True,
        }
        cfg.update(extra)
        build(job, cfg)
