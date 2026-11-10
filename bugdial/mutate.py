"""Single-line procedural bug operators, located with the AST and applied as exact text edits.

Each mutation changes one token range on one line of one function, so the gold fix is a
one-line patch. Operators follow classic mutation testing (and SWE-smith's procedural set).
"""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass, asdict


@dataclass
class Mutation:
    op: str
    line: int        # 1-based line in file
    col: int         # 0-based start col (character offset)
    end_col: int
    old: str
    new: str

    def to_dict(self):
        return asdict(self)


CMP_FLIP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
            ast.Eq: ("==", "!="), ast.NotEq: ("!=", "==")}


def _seg(lines: list[str], lineno: int, a: int, b: int) -> str:
    return lines[lineno - 1][a:b]


def _in_docstring(fn: ast.AST, node: ast.AST) -> bool:
    body = getattr(fn, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        d = body[0]
        return d.lineno <= getattr(node, "lineno", 0) <= d.end_lineno
    return False


def candidates(source: str, fn: ast.FunctionDef) -> list[Mutation]:
    lines = source.splitlines()
    out: list[Mutation] = []

    def single_line(n):
        return getattr(n, "lineno", None) is not None and n.lineno == n.end_lineno

    for node in ast.walk(fn):
        if node is fn or _in_docstring(fn, node):
            continue
        # don't mutate nested defs' signatures/decorators; their bodies are fine
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and single_line(node):
            op = type(node.ops[0])
            left, right = node.left, node.comparators[0]
            if left.end_lineno != right.lineno:
                continue
            gap = _seg(lines, node.lineno, left.end_col_offset, right.col_offset)
            if op in CMP_FLIP:
                old, new = CMP_FLIP[op]
                i = gap.find(old)
                # make sure "<" does not match the start of "<=" and vice versa
                if i >= 0 and gap.strip() == old:
                    a = left.end_col_offset + i
                    out.append(Mutation("compare_flip", node.lineno, a, a + len(old), old, new))
            elif op in (ast.Is, ast.IsNot, ast.In, ast.NotIn):
                old = {ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in"}[op]
                new = {ast.Is: "is not", ast.IsNot: "is", ast.In: "not in", ast.NotIn: "in"}[op]
                if " ".join(gap.split()) == old:
                    a = left.end_col_offset
                    b = right.col_offset
                    out.append(Mutation("membership_flip", node.lineno, a, b, gap, f" {new} "))
        elif isinstance(node, ast.BoolOp) and len(node.values) == 2 and single_line(node):
            l, r = node.values
            gap = _seg(lines, node.lineno, l.end_col_offset, r.col_offset)
            old = "and" if isinstance(node.op, ast.And) else "or"
            if gap.strip() == old:
                a = l.end_col_offset + gap.find(old)
                out.append(Mutation("boolop_swap", node.lineno, a, a + len(old), old, "or" if old == "and" else "and"))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)) and single_line(node):
            l, r = node.left, node.right
            gap = _seg(lines, node.lineno, l.end_col_offset, r.col_offset)
            old = "+" if isinstance(node.op, ast.Add) else "-"
            if gap.strip() == old and not (isinstance(l, ast.Constant) and isinstance(l.value, str)):
                a = l.end_col_offset + gap.find(old)
                out.append(Mutation("arith_swap", node.lineno, a, a + 1, old, "-" if old == "+" else "+"))
        elif isinstance(node, ast.Constant) and type(node.value) is int and single_line(node) and 0 <= node.value <= 64:
            txt = _seg(lines, node.lineno, node.col_offset, node.end_col_offset)
            if txt == str(node.value):
                out.append(Mutation("off_by_one", node.lineno, node.col_offset, node.end_col_offset, txt, str(node.value + 1)))
        elif isinstance(node, ast.Constant) and type(node.value) is bool and single_line(node):
            txt = _seg(lines, node.lineno, node.col_offset, node.end_col_offset)
            if txt in ("True", "False"):
                out.append(Mutation("bool_flip", node.lineno, node.col_offset, node.end_col_offset, txt,
                                    "False" if txt == "True" else "True"))
        elif isinstance(node, (ast.If, ast.While)) and single_line(node.test):
            t = node.test
            txt = _seg(lines, t.lineno, t.col_offset, t.end_col_offset)
            if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
                inner = _seg(lines, t.operand.lineno, t.operand.col_offset, t.operand.end_col_offset)
                if t.operand.lineno == t.lineno:
                    out.append(Mutation("drop_not", t.lineno, t.col_offset, t.end_col_offset, txt, inner))
            else:
                out.append(Mutation("negate_cond", t.lineno, t.col_offset, t.end_col_offset, txt, f"not ({txt})"))
    # dedupe by position
    uniq = {(m.line, m.col, m.end_col, m.new): m for m in out}
    return list(uniq.values())


def apply(source: str, m: Mutation) -> str:
    lines = source.splitlines(keepends=True)
    ln = lines[m.line - 1]
    assert ln[m.col:m.end_col] == m.old, (ln, m)
    lines[m.line - 1] = ln[:m.col] + m.new + ln[m.end_col:]
    out = "".join(lines)
    ast.parse(out)  # must stay valid Python
    return out


def find_function(tree: ast.Module, start: int, end: int):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno == start and n.end_lineno == end:
            return n
    return None


def pick(cands: list[Mutation], rng: random.Random, k: int) -> list[Mutation]:
    c = list(cands)
    rng.shuffle(c)
    return c[:k]
