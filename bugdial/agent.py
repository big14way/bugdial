"""A small localization agent for Gemma 4 served by vLLM (OpenAI-compatible tool calling).

The agent reads an issue and may call tools for at most `max_steps` turns, then must call
`submit` with up to 3 fully qualified function names, best first.

Tool sets (the experimental arms):
  text   : search_code (grep over the repo), read_function
  graph  : text + get_neighbors (callers AND callees, as the competition tool) + search_similar_code
  callee : text + get_callees (follow calls downstream only) + get_callers
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .graph import CodeGraph

SYSTEM = (
    "You are a software engineer localizing a bug in the Python repository `{repo}`. "
    "Read the issue, use the tools to inspect the code, and find the function whose code must be changed to fix it. "
    "Function names are fully qualified dotted paths such as `{example}`. "
    "You have at most {steps} tool calls. When you are confident (or out of calls), call `submit` with up to 3 "
    "candidate functions, most likely first."
)

TOOL_SPECS = {
    "search_code": {"description": "Search the repository source for a regex or plain string. Returns matching lines as path:line: text (max 40).",
                    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    "read_function": {"description": "Return the source code of a function or method by (partial) dotted name, e.g. Text.wrap or rich.text.Text.wrap.",
                      "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    "get_neighbors": {"description": "List callers and callees of a function in the repository call graph.",
                      "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    "search_similar_code": {"description": "Find functions whose precomputed embeddings are most similar to the given function (pass a function name).",
                            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    "get_callees": {"description": "List the functions that the given function calls, directly (depth 1) and indirectly (depth 2), with their signatures. Bugs that show up in a function usually live in it or downstream in its callees.",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    "get_callers": {"description": "List the functions that call the given function.",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    "submit": {"description": "Submit up to 3 fully qualified function names that most likely contain the bug, best first.",
               "parameters": {"type": "object", "properties": {"functions": {"type": "array", "items": {"type": "string"}}}, "required": ["functions"]}},
}
ARMS = {
    "text": ["search_code", "read_function", "submit"],
    "graph": ["search_code", "read_function", "get_neighbors", "search_similar_code", "submit"],
    "callee": ["search_code", "read_function", "get_callees", "get_callers", "submit"],
}


def sig(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:120]


class RepoView:
    def __init__(self, files: dict[str, str], cg: CodeGraph, emb: dict[str, np.ndarray] | None, max_out: int = 4000):
        self.files, self.cg, self.emb, self.max_out = files, cg, emb or {}, max_out
        self.fns = [n for n in cg.library if cg.text[n].lstrip().startswith(("def ", "async def "))]

    def _resolve(self, name: str) -> str | None:
        """Resolve a model-written name to a graph id. Models often use public import paths
        (requests.Session.request, requests.get), so we try progressively shorter dotted suffixes;
        among several matches we prefer ids that contain the dropped prefix parts, then the shortest."""
        name = name.strip().strip("`").split("(")[0].strip().rstrip(".")
        if not name:
            return None
        if name in self.cg.text:
            return name
        parts = name.split(".")
        for k in range(len(parts)):
            suf = ".".join(parts[k:])
            hits = [n for n in self.cg.library if n.endswith("." + suf) or n == suf] or \
                   [n for n in self.cg.library if n.lower().endswith("." + suf.lower())]
            if hits:
                dropped = [p for p in parts[:k] if p]
                hits.sort(key=lambda n: (-sum(p in n.split(".") for p in dropped), len(n), n))
                return hits[0]
        return None

    def search_code(self, pattern: str) -> str:
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        out = []
        for path in sorted(self.files):
            if not path.endswith(".py"):
                continue
            for i, ln in enumerate(self.files[path].splitlines(), 1):
                if rx.search(ln):
                    out.append(f"{path}:{i}: {ln.strip()[:160]}")
                    if len(out) >= 40:
                        return "\n".join(out) + "\n(truncated)"
        return "\n".join(out) or "no matches"

    def read_function(self, name: str, override: dict[str, str] | None = None) -> str:
        n = self._resolve(name)
        if not n:
            return f"not found: {name}"
        txt = (override or {}).get(n, self.cg.text[n])
        return f"# {n}\n{txt}"[: self.max_out]

    def get_neighbors(self, name: str) -> str:
        n = self._resolve(name)
        if not n:
            return f"not found: {name}"
        nb = sorted(set(self.cg.G.predecessors(n)) | set(self.cg.G.successors(n)))
        nb = [x for x in nb if x in self.cg.library]
        return f"neighbors of {n} ({len(nb)}):\n" + "\n".join(nb[:50])

    def search_similar_code(self, name: str, k: int = 10) -> str:
        n = self._resolve(name)
        if not n or n not in self.emb:
            return f"no embedding for: {name}"
        q = self.emb[n]
        sc = sorted(((float(q @ v), m) for m, v in self.emb.items() if m != n and m in self.cg.library), reverse=True)[:k]
        return "\n".join(f"{m} (sim={s:.3f})" for s, m in sc)

    def get_callees(self, name: str) -> str:
        n = self._resolve(name)
        if not n:
            return f"not found: {name}"
        d = self.cg.callee_dist(n, max_d=2)
        rows = [(dd, m) for m, dd in d.items() if dd > 0 and m in self.cg.library]
        rows.sort()
        lines = [f"[{dd}] {m}: {sig(self.cg.text[m])}" for dd, m in rows[:60]]
        return f"callees of {n} (depth<=2, {len(rows)}):\n" + "\n".join(lines)

    def get_callers(self, name: str) -> str:
        n = self._resolve(name)
        if not n:
            return f"not found: {name}"
        c = sorted(x for x in self.cg.Gd.predecessors(n) if x in self.cg.library)
        return f"callers of {n} ({len(c)}):\n" + "\n".join(c[:50])


@dataclass
class Episode:
    task_id: str
    arm: str
    variant: str
    submitted: list[str] = field(default_factory=list)
    resolved: list[str | None] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    steps: int = 0
    error: str | None = None
    secs: float = 0.0


def run_episode(client, model: str, view: RepoView, task: dict, arm: str, variant: str,
                max_steps: int = 8, override: dict[str, str] | None = None, temperature: float = 0.0) -> Episode:
    ep = Episode(task["task_id"], arm, variant)
    ep.empty_retry = False
    t0 = time.time()
    names = ARMS[arm]
    tools = [{"type": "function", "function": {"name": n, **TOOL_SPECS[n]}} for n in names]
    example = next(iter(sorted(view.fns)), "pkg.module.func")
    msgs = [{"role": "system", "content": SYSTEM.format(repo=task["repo"], example=example, steps=max_steps)},
            {"role": "user", "content": task[variant]}]
    try:
        for step in range(max_steps + 1):
            force = step == max_steps
            kw = {"tool_choice": {"type": "function", "function": {"name": "submit"}}} if force else {"tool_choice": "auto"}
            r = client.chat.completions.create(model=model, messages=msgs, tools=tools, temperature=temperature,
                                               max_tokens=1024, **kw)
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            if not calls:
                # nudge once per turn, as the competition harness does
                msgs.append({"role": "user", "content": "Use a tool, or call submit with your best guesses."})
                continue
            for c in calls:
                ep.steps += 1
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = c.function.name
                ep.calls.append((name, json.dumps(args)[:200]))
                if name == "submit":
                    fs = args.get("functions") or []
                    if isinstance(fs, str):
                        fs = [fs]
                    if not fs and not ep.empty_retry:
                        # one explicit retry: forced tool choice sometimes yields submit({})
                        ep.empty_retry = True
                        msgs.append({"role": "tool", "tool_call_id": c.id,
                                     "content": "error: `functions` must be a non-empty list of fully qualified function names"})
                        r2 = client.chat.completions.create(
                            model=model, messages=msgs, tools=tools, temperature=temperature, max_tokens=1024,
                            tool_choice={"type": "function", "function": {"name": "submit"}})
                        for c2 in (r2.choices[0].message.tool_calls or []):
                            try:
                                fs = json.loads(c2.function.arguments or "{}").get("functions") or []
                            except json.JSONDecodeError:
                                fs = []
                            if isinstance(fs, str):
                                fs = [fs]
                            if fs:
                                break
                    ep.submitted = [str(f) for f in fs][:3]
                    ep.resolved = [view._resolve(f) for f in ep.submitted]
                    ep.secs = time.time() - t0
                    return ep
                if name not in names:
                    out = f"unknown tool {name}"
                elif name == "read_function":
                    out = view.read_function(args.get("name", ""), override)
                else:
                    fn = getattr(view, name)
                    arg = args.get("pattern") if name == "search_code" else args.get("name", "")
                    out = fn(arg or "")
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out[: view.max_out]})
    except Exception as e:  # noqa: BLE001
        ep.error = f"{type(e).__name__}: {e}"[:300]
    ep.secs = time.time() - t0
    return ep
