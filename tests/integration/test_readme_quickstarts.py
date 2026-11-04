"""The rates and vol quickstarts in README.md run and print what their comments say."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def _block(heading: str) -> str:
    text = README.read_text(encoding="utf-8")
    begin = text.index("```python\n", text.index(heading)) + 10
    return text[begin : text.index("\n```", begin)]


@pytest.mark.parametrize(
    "heading", ["### 9. Interest Rates", "### 10. Volatility Surfaces & Models"]
)
def test_quickstart_prints_its_comments(heading: str) -> None:
    code = _block(heading)
    expected = [m.group(1).strip() for m in re.finditer(r"^print\(.*\)\s+# (.+)$", code, re.M)]
    assert len(expected) == 4
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, heading, "exec"), {})
    assert out.getvalue().splitlines() == expected
