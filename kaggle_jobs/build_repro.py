"""Build the reproduction notebook (kaggle_jobs/repro/) from bugdial/ and push with `kaggle kernels push`."""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pack() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted((ROOT / "bugdial").glob("*.py")):
            tf.add(p, arcname=f"bugdial/{p.name}")
    return base64.b64encode(buf.getvalue()).decode()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# BugDial: how far is the bug? Reproduction notebook\n\n"
       "Reproduces every number and the figure in the Paper Track writeup *How Far Is the Bug?* from the "
       "[BugDial dataset](https://www.kaggle.com/datasets/godswillidolor/bugdial) and the competition data. "
       "Code: https://github.com/big14way/bugdial (packed into the next cell so the notebook runs offline).\n\n"
       "Steps: (1) audit the released code graphs; (2) recompute the model-free ranks for all 454 tasks; "
       "(3) print the tables; (4) draw Figure 1. Planting the bugs (running the repositories' test suites) and "
       "the Gemma 4 agent runs are in the repository (`bugdial.plant`, `bugdial.evalrun`); their outputs are in the dataset."),
    code("import base64, io, tarfile, sys, glob, json, os\nfrom pathlib import Path\n"
         f"SRC = '{pack()}'\n"
         "tarfile.open(fileobj=io.BytesIO(base64.b64decode(SRC))).extractall('/kaggle/working/src')\n"
         "sys.path.insert(0, '/kaggle/working/src')\n"
         "DATA = Path(glob.glob('/kaggle/input/**/rich_tasks.jsonl', recursive=True)[0]).parent\n"
         "COMP = Path(glob.glob('/kaggle/input/**/tasks.jsonl', recursive=True)[0]).parent\n"
         "print('dataset:', DATA); print('competition data:', COMP)"),
    md("## 1. The released graphs carry no direction\n\nFor every released graph, count edges whose reverse is also present, "
       "and recover direction from node source text."),
    code("from bugdial.graph import CodeGraph\n"
         "tot = {'graphs': 0, 'edges': 0, 'reciprocal': 0, 'pairs': 0, 'oriented': 0}\n"
         "for g in sorted((COMP / 'graphs').glob('*_*.json')):\n"
         "    pkg, sha = g.stem.split('_', 1)\n"
         "    if len(sha) != 40: continue\n"
         "    st = CodeGraph(g, pkg).edge_stats()\n"
         "    tot['graphs'] += 1; tot['edges'] += st['released_edges']; tot['reciprocal'] += st['reciprocal_edges']\n"
         "    tot['pairs'] += st['pairs']; tot['oriented'] += st['recovered_one_way'] + st['recovered_both']\n"
         "print(tot)\nprint(f\"reciprocal: {tot['reciprocal']/tot['edges']:.1%}   direction recovered: {tot['oriented']/tot['pairs']:.1%}\")"),
    md("## 2. Recompute model-free localization for all 454 tasks"),
    code("from bugdial.graph import CodeGraph\nfrom bugdial.locate import Locator, evaluate\n"
         "SHA = {'rich': '19c67b9a3479841e9133bea94607c89ee931d3fc', 'requests': '6f205ff422bccd5e4c4fc0b64c5f3e7df5181db6',\n"
         "       'fastapi': 'f1d750fddad90e2faad6c3cc5c85c7687b787e89'}\n"
         "OUT = Path('/kaggle/working/recomputed'); OUT.mkdir(exist_ok=True)\n"
         "for pkg, sha in SHA.items():\n"
         "    cg = CodeGraph(COMP / 'graphs' / f'{pkg}_{sha}.json', pkg)\n"
         "    loc = Locator(cg, str(COMP / 'embeddings' / f'{pkg}_{sha}.npz'))\n"
         "    tasks = [json.loads(l) for l in open(DATA / f'{pkg}_tasks.jsonl')]\n"
         "    rows = evaluate(loc, tasks)\n"
         "    (OUT / f'{pkg}_locate.jsonl').write_text(''.join(json.dumps(r) + '\\n' for r in rows))\n"
         "    same = sum(1 for r, s in zip(rows, (json.loads(l) for l in open(DATA / f'{pkg}_locate.jsonl'))) if r['rank'] == s['rank'])\n"
         "    print(pkg, len(tasks), 'tasks;', same, 'of', len(rows), 'ranks identical to the dataset')\n"
         "import shutil\nfor f in DATA.glob('*.jsonl'):\n"
         "    if not f.name.endswith('_locate.jsonl'): shutil.copy(f, OUT / f.name)"),
    md("## 3. Tables (retrieval, paired sign tests, public-task map, agents)"),
    code("import subprocess\nprint(subprocess.run([sys.executable, '-m', 'bugdial.report', '--in', str(OUT), '--fig', '/kaggle/working/fig'],\n"
         "                     cwd='/kaggle/working/src', capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '/kaggle/working/src'}).stdout)"),
    md("## 4. Figure 1"),
    code("subprocess.run([sys.executable, '-m', 'bugdial.figures', '--locate', str(OUT), '--agents', str(OUT), '--out', '/kaggle/working/figure1.png'],\n"
         "               cwd='/kaggle/working/src', env={**os.environ, 'PYTHONPATH': '/kaggle/working/src'})\n"
         "from IPython.display import Image\nImage('/kaggle/working/figure1.png')"),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
d = ROOT / "kaggle_jobs" / "repro"
d.mkdir(parents=True, exist_ok=True)
(d / "bugdial-repro.ipynb").write_text(json.dumps(nb, indent=1))
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": "godswillidolor/bugdial-how-far-is-the-bug",
    "title": "BugDial: how far is the bug (reproduction)",
    "code_file": "bugdial-repro.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "competition_sources": ["gemma-4-developer-agent"],
    "dataset_sources": ["godswillidolor/bugdial"],
    "kernel_sources": [],
    "model_sources": [],
}, indent=2))
print("built", d)
