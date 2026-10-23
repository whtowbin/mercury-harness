```python
import math
import re
from dataclasses import dataclass


class CalcError(ValueError):
    pass


# 1. Tokens
@dataclass
class Token:
    kind: str
    text: str
    pos: int


_TOK = re.compile(
    r"\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|([-+*/%^(),]))"
)


def tokenize(src):
    """Split src into tokens."""
    out, i = [], 0
    while True:
        while i < len(src) and src[i].isspace():
            i += 1
        if i >= len(src):
            out.append(Token("END", "", i))
            return out
        m = _TOK.match(src, i)
        if not m:
            raise CalcError(f"bad character at {i}")
        kind = "NUM" if m[1] else "NAME" if m[2] else "OP"
        text = m[1] or m[2] or m[3]
        out.append(Token(kind, text, m.start(m.lastindex)))
        i = m.end()


# 2. Syntax tree
@dataclass
class Num:
    value: float


@dataclass
class Var:
    name: str


@dataclass
class Unary:
    op: str
    operand: object


@dataclass
class Binary:
    op: str
    left: object
    right: object


@dataclass
class Call:
    name: str
    args: list


# 3. Parser
def parse(src):
    """Parse src to a tree."""
    toks, i = tokenize(src), 0

    def peek():
        return toks[i]

    def take(text=None):
        nonlocal i
        t = toks[i]
        if text is not None and t.text != text:
            raise CalcError(f"expected {text} at {t.pos}")
        i += 1
        return t

    def expr():
        node = term()
        while peek().text in ("+", "-") and peek().kind == "OP":
            node = Binary(take().text, node, term())
        return node

    def term():
        node = unary()
        while peek().text in ("*", "/", "%"):
            node = Binary(take().text, node, unary())
        return node

    def unary():
        if peek().text in ("+", "-"):
            return Unary(take().text, unary())
        return power()

    def power():
        base = primary()
        if peek().text == "^":
            take()
            return Binary("^", base, unary())
        return base

    def primary():
        t = take()
        if t.kind == "NUM":
            return Num(float(t.text))
        if t.kind == "NAME":
            if peek().text == "(":
                take()
                args = []
                if peek().text != ")":
                    args.append(expr())
                    while peek().text == ",":
                        take()
                        args.append(expr())
                take(")")
                return Call(t.text, args)
            return Var(t.text)
        if t.text == "(":
            node = expr()
            take(")")
            return node
        raise CalcError(f"unexpected {t.text!r} at {t.pos}")

    node = expr()
    if peek().kind != "END":
        raise CalcError("trailing input")
    return node


# 4. Evaluator
_F = {
    "sin": (math.sin, 1, 1),
    "cos": (math.cos, 1, 1),
    "sqrt": (None, 1, 1),
    "abs": (abs, 1, 1),
    "min": (lambda *a: min(a), 1, 99),
    "max": (lambda *a: max(a), 1, 99),
    "round": (round, 1, 2),
}


def evaluate(src, env=None):
    """Evaluate src."""
    names = {"pi": math.pi, "e": math.e, **(env or {})}

    def ev(n):
        if isinstance(n, Num):
            return n.value
        if isinstance(n, Var):
            if n.name not in names:
                raise CalcError(n.name)
            return float(names[n.name])
        if isinstance(n, Unary):
            v = ev(n.operand)
            return -v if n.op == "-" else v
        if isinstance(n, Binary):
            a, b = ev(n.left), ev(n.right)
            if n.op in "/%" and b == 0:
                raise CalcError("division by zero")
            return (
                {"+": a + b, "-": a - b, "*": a * b}.get(n.op)
                if n.op in "+-*"
                else (a / b if n.op == "/" else a % b if n.op == "%" else a**b)
            )
        f = _F.get(n.name)
        if f is None or not f[1] <= len(n.args) <= f[2]:
            raise CalcError(n.name)
        args = [ev(a) for a in n.args]
        if n.name == "sqrt":
            if args[0] < 0:
                raise CalcError("sqrt")
            return math.sqrt(args[0])
        if n.name == "round":
            return float(round(args[0], int(args[1])) if len(args) == 2 else round(args[0]))
        return float(f[0](*args))

    return ev(parse(src))


# 5. Printer
_P = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2}


def to_source(n):
    """Print with minimal parentheses."""

    def prec(x):
        if isinstance(x, Binary):
            return 4 if x.op == "^" else _P[x.op]
        if isinstance(x, Unary):
            return 3
        return 5

    def wrap(x, need):
        s = to_source(x)
        return f"({s})" if prec(x) < need else s

    if isinstance(n, Num):
        return repr(n.value)
    if isinstance(n, Var):
        return n.name
    if isinstance(n, Call):
        return f"{n.name}({', '.join(to_source(a) for a in n.args)})"
    if isinstance(n, Unary):
        return n.op + wrap(n.operand, 3)
    if n.op == "^":
        return f"{wrap(n.left, 5)}^{wrap(n.right, 3)}"
    p = _P[n.op]
    return f"{wrap(n.left, p)} {n.op} {wrap(n.right, p + 1)}"


# 6. Self-test
if __name__ == "__main__":
    assert evaluate("1+2") == 3
    print("ALL TESTS PASSED")
```
