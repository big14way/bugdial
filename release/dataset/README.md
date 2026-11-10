# BugDial: how far is the bug?

Executable, test-verified bug-localization tasks with a controlled **symptom-to-cause distance**,
built on the Gemma 4 Developer Agent competition's repositories and commits, plus every result in
the accompanying Paper Track writeup. Code: https://github.com/big14way/bugdial

## Files

| file | rows | what it is |
|---|---|---|
| `{rich,requests,fastapi}_tasks.jsonl` | 282 / 82 / 90 | synthetic tasks (one per buggy function) |
| `{rich,requests,fastapi}_locate.jsonl` | 16 per task | model-free method ranks (8 methods × clue off/on) |
| `episodes_gemma-4-{e4b,e2b}-it-qat-w4a16-ct.jsonl` | 5,110 each | Gemma 4 agent episodes (7 arms; synthetic tasks × clue off/on, public tasks) |
| `real_tasks.jsonl` | 129 | derived labels for the competition's public tasks |
| `graph_stats.jsonl` | 127 | edge audit of every released code graph |

### Task fields
`task_id`, `repo`, `v` (buggy function, graph id), `span` (file and lines), `mutation` (operator, line,
columns, old/new token), `bug_patch` / `fix_patch` (one-line unified diffs), `test` (the failing test
the report was written from), `n_failing_tests`, `issue` (report without the name), `issue_clue`
(same plus one sentence naming `v`), `anchors` (library symbols the report mentions), `d` (directed
call distance from the anchors to `v` in the direction-recovered graph; null if unreachable),
`d_undirected`, `name_in_issue` (the report text already contains `v`'s short name), `entries`.

Repositories and commits: rich `19c67b9a3479841e9133bea94607c89ee931d3fc`, requests
`6f205ff422bccd5e4c4fc0b64c5f3e7df5181db6`, fastapi `f1d750fddad90e2faad6c3cc5c85c7687b787e89`.
Apply `bug_patch` to the repository at that commit to reproduce a task; the listed test fails, and
`fix_patch` restores it.

### Episode fields
`kind` (`synth` or `real`), `task_id`, `repo`, `d`, `arm` (`text`, `graph`, `callee`, `ctx_callee`,
`ctx_short_bm25`, `ctx_short_undir`, `ctx_short_dir`), `variant` (`issue` / `issue_clue`), `rank`
(1-based position of the faulty function in the agent's up-to-3 answers; null if absent),
`submitted`, `resolved`, `calls` (tool name and arguments; arguments removed for public tasks),
`steps`, `secs`, `empty_retry`, `ctx_chars`.

### Public-task labels
For the competition's public tasks we publish only derived labels (instance id, gold function ids
mapped from the reference patch, whether the issue names one, the distance `d`, and method ranks),
as the hosts permit. No problem statements, patches or tests are included; get those from the
competition data.

## Provenance and licences
Code excerpts in tasks (test bodies, one-line patches) come from rich (MIT), requests (Apache-2.0)
and fastapi (MIT) at the commits above and remain under those licences. Annotations and results:
CC BY 4.0.
