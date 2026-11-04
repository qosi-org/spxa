"""The spxa.rates example scripts run and reproduce their reference values."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_curve_example(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / "rates_curve_example.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert out.count("max residual < 1e-12: True") == 5
    assert "10Y payer at the quoted 2.81% prices to zero: True" in out
    assert "turn from 2024-12-31 to 2025-01-02" in out
    assert "O/N forward from 2024-12-31:  +24.8 bp" in out
    assert "agree to second order (< 1e-6): True" in out
    assert "nodes before 10Y unmoved: True" in out
    assert "nan" not in out and ": False" not in out
