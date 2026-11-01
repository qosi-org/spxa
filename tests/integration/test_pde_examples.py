"""The spxa.pde example scripts run and reproduce their reference values."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run(name: str, capsys: pytest.CaptureFixture[str]) -> str:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out


def test_black_scholes_example(capsys):
    out = _run("pde_black_scholes_example.py", capsys)
    assert "4.4862" in out and "Reiner-Rubinstein 8.1388" in out
    assert "Richardson" in out


def test_heston_adi_example(capsys):
    out = _run("pde_heston_adi_example.py", capsys)
    assert "vs Lewis 8.894869" in out
    assert "Margrabe 12.211951" in out


def test_levy_and_density_example(capsys):
    out = _run("pde_levy_and_density_example.py", capsys)
    assert "nan" not in out
    assert "mass 1.0000000000" in out
