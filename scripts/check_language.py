"""Check the repository against the two writing standards it uses.

Standard A is ASD-STE100 Simplified Technical English, and it applies to the text
a person reads in the application: UI strings, error and refusal messages, field
notes and tooltips. The checks here are the mechanical part of that standard.
A procedural sentence has at most 20 words and a descriptive one at most 25, and
a semicolon or a colon must not join two sentences, because one sentence carries
one instruction.

Standard B is the plain-technical-prose rule set, and it applies to the Markdown
documents and to comments and docstrings. No em dash, no figurative verb, and no
first or second person.

Neither standard can be checked completely by a program. Word order, one meaning
per word, and whether a refusal names the action all need a reader. What this
finds is the part that recurs, so a sweep does not have to happen twice.

Run: python scripts/check_language.py [--fix-nothing]
Exit code 1 when anything is reported.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Standard A: the application's own text.
STE_PY = ("core", "routes", "python_scripts")
STE_JS = (
    "static/js/app.js",
    "static/js/core.js",
    "static/js/upload.js",
    "static/js/preprocess.js",
    "static/js/modeling.js",
    "static/js/narrative.js",
    "static/js/inference.js",
    "static/js/init.js",
)

# Standard B: the prose.
PROSE = ("README.md", "CHANGELOG.md", "CONTRIBUTING.md", "docs/reviews/AUDIT_RESUME.md", "deploy/README.md", "examples/README.md")
PROSE_DIRS = ("docs",)

STE_PROCEDURAL_WORDS = 20
STE_DESCRIPTIVE_WORDS = 25

# Verbs that name no mechanism. The rule is to say what the code does instead.
# Phrasings where a listed verb is the exact term rather than a figure of speech.
LITERAL_USES = (
    "holds for",
    "does not hold",
    "the ceiling holds",
    "where the memory bound holds",
    "still holds",
    "drives the page",
    "drives the browser",
)

FIGURATIVE = (
    "land",
    "lands",
    "landed",
    "ship",
    "ships",
    "shipped",
    "drive",
    "drives",
    "driven",
    "leverage",
    "leverages",
    "harness",
    "harnesses",
    "navigate",
    "navigates",
    "unpack",
    "unpacks",
    "flag",
    "flags",
    "flagged",
    "carry",
    "carries",
    "carried",
    "hold",
    "holds",
    "held",
    "sit",
    "sits",
    "ride",
    "rides",
)
FIRST_SECOND = ("we", "our", "us", "ours", "my", "mine")
# "I" is checked against the original case, and only where it is used as a pronoun:
# followed by a space and a lowercase word. Lowercasing the line first made every
# author initial read as first person, and a bare capital test made the ruff rule
# list "E, F, I, B" read as one too.
FIRST_PERSON_I = re.compile(r"(?<![.\-])\bI\s+[a-z]")

# An en dash between digits is a numeric range and is allowed.
EM_DASH = re.compile(r"—")
EN_DASH_NON_NUMERIC = re.compile(r"(?<!\d)–|–(?!\d)")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A semicolon with a clause on each side joins two sentences. A colon that
# introduces a list is correct and is not reported.
JOINED = re.compile(r"[a-z0-9)\"']\s*;\s+[a-z]")
WORD = re.compile(r"[A-Za-z0-9_.%'/-]+")
IMPERATIVE = re.compile(
    r"^(Set|Run|Use|Add|Remove|Open|Close|Select|Enter|Upload|Install|Train|Check|Rename|Ask|Configure|Click|Read|Write|Give|Make|Do)\b"
)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, int, str, str]] = []

    def add(self, path: Path, line: int, rule: str, detail: str) -> None:
        self.rows.append((str(path.relative_to(ROOT)), line, rule, detail))

    def show(self) -> int:
        if not self.rows:
            print("check_language: nothing to report")
            return 0
        for path, line, rule, detail in sorted(self.rows):
            print(f"{path}:{line}: {rule}: {detail}")
        print(f"\ncheck_language: {len(self.rows)} to fix")
        return 1


def _words(text: str) -> int:
    return len(WORD.findall(text))


def check_ste_sentence(report: Report, path: Path, line: int, text: str, where: str) -> None:
    """One message, against the mechanical part of STE."""
    stripped = text.strip()
    if not stripped or len(stripped) < 12:
        return
    if EM_DASH.search(stripped):
        report.add(path, line, "STE em dash", where)
    for sentence in SENTENCE_SPLIT.split(stripped):
        sentence = sentence.strip()
        if not sentence:
            continue
        n = _words(sentence)
        limit = STE_PROCEDURAL_WORDS if IMPERATIVE.match(sentence) else STE_DESCRIPTIVE_WORDS
        if n > limit:
            report.add(path, line, f"STE sentence {n} words over {limit}", sentence[:70])
    if JOINED.search(stripped):
        report.add(path, line, "STE semicolon joins two sentences", stripped[:70])


def _text_of(node: ast.AST) -> str | None:
    """The literal text of a string or an f-string, with the holes marked."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "X" for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _text_of(node.left), _text_of(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _message_nodes(tree: ast.AST):
    """The strings a person actually reads: what is raised, and what is returned as an error.

    A docstring is prose for a reader of the source and belongs to the other
    standard, so scanning every string would report the wrong text.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            call = node.exc
            if isinstance(call, ast.Call):
                for arg in call.args:
                    text = _text_of(arg)
                    if text:
                        yield node.lineno, text
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if isinstance(key, ast.Constant) and key.value in ("error", "detail", "message", "reason"):
                    text = _text_of(value)
                    if text:
                        yield getattr(value, "lineno", node.lineno), text
        elif isinstance(node, ast.Return) and node.value is not None:
            text = _text_of(node.value)
            if text and " " in text and len(text) > 24:
                yield node.lineno, text
        elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
            text = _text_of(node.elts[1])
            if text and " " in text and len(text) > 24:
                yield getattr(node, "lineno", 0), text


def scan_python_messages(report: Report) -> None:
    """Every string a Python error path hands to a person."""
    for d in STE_PY:
        for path in sorted((ROOT / d).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            seen = set()
            for lineno, text in _message_nodes(tree):
                if (lineno, text) in seen or len(text) < 24 or " " not in text:
                    continue
                if "<!DOCTYPE" in text or "<html" in text or "<div" in text:
                    continue
                seen.add((lineno, text))
                check_ste_sentence(report, path, lineno, text, "message")


JS_STRING = re.compile(r"'((?:[^'\\]|\\.){20,}?)'|\"((?:[^\"\\]|\\.){20,}?)\"")
# A run of CSS declarations, an HTML fragment or a selector is not prose.
CSS_LIKE = re.compile(r"^[a-z-]+\s*:\s*[^;]+;|(?:\d+(?:px|rem|em|%)|#[0-9a-fA-F]{3,6}|\bflex\b|\bmargin\b|\bpadding\b)")


def scan_js_messages(report: Report) -> None:
    for rel in STE_JS:
        path = ROOT / rel
        if not path.exists():
            continue
        for i, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if "console." in raw or raw.strip().startswith(("//", "*", "/*")):
                continue
            for m in JS_STRING.finditer(raw):
                text = m.group(1) or m.group(2) or ""
                if (
                    " " in text
                    and not text.startswith(("http", "/", "."))
                    and "<" not in text
                    and not CSS_LIKE.search(text)
                    and "${" not in text
                    and " ? " not in text
                ):
                    check_ste_sentence(report, path, i, text, "message")


def _prose_files() -> list[Path]:
    files = [ROOT / p for p in PROSE if (ROOT / p).exists()]
    for d in PROSE_DIRS:
        files += sorted((ROOT / d).glob("*.md"))
    return files


def scan_prose(report: Report) -> None:
    for path in _prose_files():
        in_code = False
        for i, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if raw.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            line = re.sub(r"`[^`]*`", "", raw)
            if EM_DASH.search(line):
                report.add(path, i, "em dash", line.strip()[:70])
            if EN_DASH_NON_NUMERIC.search(line):
                report.add(path, i, "en dash outside a numeric range", line.strip()[:70])
            low = line.lower()
            if any(phrase in low for phrase in LITERAL_USES):
                continue
            for verb in FIGURATIVE:
                if re.search(rf"\b{verb}\b", low):
                    report.add(path, i, f"figurative verb {verb!r}", line.strip()[:70])
                    break
            if FIRST_PERSON_I.search(line):
                report.add(path, i, "first person 'I'", line.strip()[:70])
            for pron in FIRST_SECOND:
                if re.search(rf"\b{pron}\b", low):
                    report.add(path, i, f"first person {pron!r}", line.strip()[:70])
                    break
            if re.search(r"\bR and Python\b", line) or re.search(r"\bR or Python\b", line):
                report.add(path, i, "R before Python", line.strip()[:70])
    return


def main() -> int:
    """--standard a runs the STE checks alone, which is what gates the lint target."""
    which = "ab"
    for arg in sys.argv[1:]:
        if arg.startswith("--standard"):
            which = arg.split("=", 1)[1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1]
    report = Report()
    if "a" in which:
        scan_python_messages(report)
        scan_js_messages(report)
    if "b" in which:
        scan_prose(report)
    return report.show()


if __name__ == "__main__":
    sys.exit(main())
