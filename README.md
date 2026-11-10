# BugDial: how far is the bug?

A controlled, executable benchmark for **repository-level bug localization** in which the
**symptom-to-cause distance** is an experimental variable instead of an accident of the data.
Built for the Kaggle *Gemma 4 Developer Agent* Paper Track, on the competition's own repositories
and commits.

## Idea

Observational studies on natural issues cannot tell when a code graph should help an agent:
how far the bug sits from what the reporter mentions, whether the reporter names the faulty
function, and how hard the code is are all entangled. BugDial controls them:

1. **Plant** a one-token bug in a function `v` (8 AST operators: comparison flip, membership flip,
   `and`/`or`, `+`/`-`, off-by-one, boolean flip, negated condition, dropped `not`).
2. **Verify** it: run the repository's own tests that reach `v` in the call graph and pass on the
   clean code; keep the bug only if one of them fails.
3. **Write the report** from one failing test (the test body as the reproduction, the assertion or
   exception message as the observed error), so the report talks about what the user called,
   not about `v`. A *clue* variant adds one sentence naming `v`.
4. **Measure distance** `d`: the shortest directed call path from anything the report mentions to `v`.

Then compare localization methods (lexical search, the released embeddings, undirected graph
expansion, direction-aware traversal) and Gemma 4 agents with different tool sets, as a function
of `d` and of the clue.

## A finding about the released code graph

In the competition's released graphs, **every `calls` edge also appears reversed**: at the latest
public commits, 5,786 of 5,786 edges for rich, 1,884 of 1,884 for requests and 2,752 of 2,752 for
fastapi are reciprocal. The graph therefore carries no direction, and a neighbour query cannot
tell callers from callees. `bugdial.graph.CodeGraph` recovers direction from each node's source
text (`u -> v` when `u`'s code calls `v` by name, contains member `v`, or references class `v`);
this assigns a direction to about 99% of connected pairs.

## Layout

| path | what it does |
|---|---|
| `bugdial/graph.py` | load the released graph, recover edge direction, map nodes to source spans and pytest ids |
| `bugdial/mutate.py` | single-token bug operators, located with the AST and applied as exact text edits |
| `bugdial/plant.py` | plant bugs, run graph-selected tests (JUnit XML), keep bugs that break tests |
| `bugdial/tasks.py` | turn a failing test into an issue, measure `d`, add the clue variant, balance buckets |
| `bugdial/locate.py` | model-free localization methods and metrics |
| `bugdial/realtasks.py` | the same distance measured on the 129 public tasks (gold patch mapped to functions) |
| `bugdial/agent.py`, `bugdial/evalrun.py` | Gemma 4 localization agent over vLLM (tool calling), three tool arms |
| `bugdial/report.py` | tables and figures |
| `kaggle_jobs/build.py` | packs `bugdial/` into Kaggle script kernels (CPU planting, GPU evaluation) |

## Running

Test suites and models run on Kaggle, attached to the competition data (snapshots, graphs,
embeddings, offline wheels) and the official Gemma 4 QAT models:

```bash
python kaggle_jobs/build.py plant      # CPU: plant, verify, build tasks, model-free baselines
kaggle kernels push -p kaggle_jobs/plant
python kaggle_jobs/build.py eval       # GPU (4x L4): Gemma 4 agents on the tasks
kaggle kernels push -p kaggle_jobs/eval
python -m bugdial.report --in <downloaded outputs> --fig fig/
```

The model-free parts (`locate`, `report`) also run locally on downloaded task files.

## Status

Work in progress for the Paper Track (deadline 12 Nov 2026). Results, the task dataset and the
paper will be linked here.

## License

Apache-2.0
