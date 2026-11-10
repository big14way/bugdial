"""Load the competition's released code graph and map its nodes to source spans."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass
class Span:
    file: str          # path relative to repo root
    start: int         # 1-based first line of the def (decorators excluded)
    end: int           # 1-based last line
    qual: str          # qualified name inside the module, e.g. "Console.print"


def is_test_id(nid: str) -> bool:
    parts = nid.split(".")
    return parts[0] in ("tests", "test", "testing") or any(p.startswith("test_") for p in parts[-2:])


class CodeGraph:
    """Directed call graph from the released NetworkX JSON (edge u -> v means u calls v)."""

    def __init__(self, graph_json: str | Path, package: str):
        g = json.loads(Path(graph_json).read_text())
        self.package = package
        self.text: dict[str, str] = {n["id"]: n.get("text") or "" for n in g["nodes"]}
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.text)
        for e in g["edges"]:
            if e.get("type", "calls") == "calls" and e["source"] in self.text and e["target"] in self.text:
                if e["source"] != e["target"]:
                    self.G.add_edge(e["source"], e["target"])
        self.library = {n for n in self.text if n.startswith(package + ".")}
        self.tests = {n for n in self.text if is_test_id(n) and n.split(".")[-1].startswith("test")}
        self.L = self.G.subgraph(self.library).copy()   # library-only graph, as released (symmetric)
        self.classes = {n for n, t in self.text.items() if t.lstrip().startswith("class ")}
        # Direction recovery. Every released "calls" edge also appears reversed, so the JSON carries no
        # direction. For each connected pair we keep u -> v when u's source calls v by name (`name(`),
        # when u is a class containing member v, or when u references class v by name.
        self.Gd = nx.DiGraph()
        self.Gd.add_nodes_from(self.text)
        for a, b in {tuple(sorted(e)) for e in self.G.edges()}:
            for u, v in ((a, b), (b, a)):
                if self._calls(u, v):
                    self.Gd.add_edge(u, v)
        self.Ld = self.Gd.subgraph(self.library).copy()
        self._short: dict[str, list[str]] = {}
        for n in self.library:
            self._short.setdefault(n.split(".")[-1], []).append(n)

    def _calls(self, u: str, v: str) -> bool:
        if u in self.classes and v.startswith(u + "."):
            return True                      # class contains member
        name = re.escape(v.split(".")[-1])
        txt = self.text.get(u) or ""
        if v in self.classes:
            return re.search(r"(?<![A-Za-z0-9_])" + name + r"(?![A-Za-z0-9_])", txt) is not None
        return re.search(r"(?<![A-Za-z0-9_])" + name + r"\s*\(", txt) is not None

    # ---- distances -------------------------------------------------------
    def callee_dist(self, src: str, max_d: int = 8) -> dict[str, int]:
        """Directed BFS distances following calls (src -> callees) inside the library (repaired graph)."""
        if src not in self.Ld:
            return {}
        return nx.single_source_shortest_path_length(self.Ld, src, cutoff=max_d)

    def edge_stats(self) -> dict:
        E = list(self.G.edges())
        recip = sum(1 for u, v in E if self.G.has_edge(v, u))
        pairs = {tuple(sorted(e)) for e in E}
        one = sum(1 for a, b in pairs if self.Gd.has_edge(a, b) != self.Gd.has_edge(b, a))
        both = sum(1 for a, b in pairs if self.Gd.has_edge(a, b) and self.Gd.has_edge(b, a))
        return {"nodes": len(self.text), "library_nodes": len(self.library), "classes": len(self.classes),
                "released_edges": len(E), "reciprocal_edges": recip, "pairs": len(pairs),
                "recovered_one_way": one, "recovered_both": both, "recovered_none": len(pairs) - one - both,
                "tests": len(self.tests)}

    def undirected_dist(self, src: str, max_d: int = 8) -> dict[str, int]:
        if src not in self.L:
            return {}
        return nx.single_source_shortest_path_length(self.L.to_undirected(as_view=True), src, cutoff=max_d)

    def tests_reaching(self, v: str) -> set[str]:
        """Test functions that can reach v through (recovered, directed) call edges."""
        return {a for a in nx.ancestors(self.Gd, v) if a in self.tests}

    def resolve(self, name: str) -> list[str]:
        """Resolve a symbol mention to library node ids (exact, then dotted suffix)."""
        if name in self.library:
            return [name]
        hits = [n for n in self.library if n.endswith("." + name)]
        return hits


# ---- node id -> file span ----------------------------------------------------

def module_file(repo: Path, nid: str) -> tuple[Path, list[str]] | None:
    """Longest module prefix of nid that exists as a .py file; returns (file, remaining qualname parts)."""
    parts = nid.split(".")
    for root in (repo, repo / "src"):
        for i in range(len(parts), 0, -1):
            base = root.joinpath(*parts[:i])
            for cand in (base.with_suffix(".py"), base / "__init__.py"):
                if cand.is_file():
                    return cand, parts[i:]
    return None


def import_root(repo: Path) -> Path:
    """Directory to put on PYTHONPATH (handles src/ layouts such as requests)."""
    return repo / "src" if (repo / "src").is_dir() else repo


def find_span(repo: Path, nid: str) -> Span | None:
    hit = module_file(repo, nid)
    if not hit:
        return None
    path, qual = hit
    if not qual:
        return None
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return None
    body = tree.body
    node = None
    for q in qual:
        node = next((b for b in body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and b.name == q), None)
        if node is None:
            return None
        body = node.body
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return Span(str(path.relative_to(repo)), node.lineno, node.end_lineno, ".".join(qual))


def pytest_id(repo: Path, test_nid: str) -> str | None:
    hit = module_file(repo, test_nid)
    if not hit:
        return None
    path, qual = hit
    if not qual:
        return None
    return str(path.relative_to(repo)) + "::" + "::".join(qual)


IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def mentioned_nodes(cg: CodeGraph, text: str) -> list[str]:
    """Library nodes a reader could anchor on from the issue text.

    A dotted mention (Text.from_markup, rich.cells.cell_len) resolves by suffix.
    A bare name resolves only if it is unique in the library (avoids 'print' -> 40 nodes).
    """
    seen: dict[str, None] = {}
    for m in IDENT.findall(text):
        if "." in m:
            parts = m.split(".")
            for k in range(len(parts) - 1):
                hits = cg.resolve(".".join(parts[k:]))
                if 0 < len(hits) <= 3:
                    for h in hits:
                        seen[h] = None
                    break
        else:
            hits = cg._short.get(m, [])
            if len(hits) == 1 and len(m) > 3:
                seen[hits[0]] = None
    return list(seen)
