"""The spxa.vol example scripts run and reproduce their reference values."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run(name: str, capsys: pytest.CaptureFixture[str]) -> str:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out


def test_svi_surface_example(capsys: pytest.CaptureFixture[str]) -> None:
    out = _run("vol_svi_surface_example.py", capsys)
    assert "min g = -0.032864 at k = 0.879  -> butterfly-free: False" in out
    assert "constrained refit: rmse 0.00231, butterfly-free: True" in out
    assert "jump-wings projection (keeps ATM level and skew): True" in out
    assert out.count("calendar-free True") == 3
    assert "butterfly-free True" in out
    assert "prices non-decreasing in t on 40 expiries: True" in out
    assert "mass 1.0000000000   E[S/F] 1.0000000000" in out
    assert "Monte Carlo within 4 standard errors: True" in out
    assert "nan" not in out
