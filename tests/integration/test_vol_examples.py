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


def test_surface_build_example(capsys: pytest.CaptureFixture[str]) -> None:
    out = _run("vol_surface_build_example.py", capsys)
    assert "parity forwards within 1 bp of the truth: True" in out
    assert out.count("  yes") == 5 and "calendar-free" in out
    assert "arbitrage-free: True" in out
    assert "worst implied-vol error below 1 vol point: True" in out
    assert "sticky moneyness larger under negative skew: True" in out
    assert "0.1y  0.0000  0.25y  0.0000" in out  # the book has no short-dated vega
    assert "buckets sum to the parallel vega: True" in out
    assert "reloaded surface identical: True" in out
    assert "E[S_T]/F - 1 within 4 standard errors: True" in out
    assert out.count("within 4 standard errors: True") == 2
    assert "nan" not in out and "False" not in out


def test_local_vol_example(capsys: pytest.CaptureFixture[str]) -> None:
    out = _run("vol_local_vol_example.py", capsys)
    assert "downside local vol above upside: True" in out
    assert "worst error 0.42 bp of vol at T=0.25, K=80" in out
    assert "local vol defined everywhere: True" in out
    assert "Monte Carlo calls within 4 standard errors of the surface: True" in out
    assert "density mass 1.00000000, E[S/F] 1.000000" in out
    assert "American 7.3705, European 7.1989" in out
    assert "= vanilla 9.1594: True" in out
    assert "nan" not in out and "False" not in out


def test_slv_example(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / "vol_slv_example.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "local vol DO(85): 8.1806" in out
    assert "0.0             1.25         1.0705    8.1849    5.7989" in out
    assert "1.0             3.46         1.2893    8.1029    6.3308" in out
    assert "UO(130): PDE 6.0128" in out
    assert "PDE within 3 standard errors: True" in out
    assert "max gap 1.38%" in out


def test_rough_example(capsys: pytest.CaptureFixture[str]) -> None:
    out = _run("vol_rough_example.py", capsys)
    assert "T=1.0   vols  23.23  20.24  17.38  14.96  13.63" in out
    assert "-> power -0.420 (H - 1/2 = -0.4)" in out
    assert "classical Heston skew stays bounded: True" in out
    for row in ("n=5   1.34", "n=20  0.46", "n=50  0.24"):
        assert row in out
    assert "H free: H = 0.0800" in out and "H = 1/2: RMSE 1.19 vol points" in out
    assert "vols  22.98  19.57  16.30  13.44  12.00" in out
    assert "conditional and plain agree within 3 SE: True" in out
    assert "sqrt E[VIX^2] 21.31, futures 18.04" in out
    assert "rough Bergomi VIX smile: 81.2 81.4 81.9" in out
    assert "Heston VIX smile:        94.8 86.4 76.0" in out
    assert "nan" not in out and "False" not in out


def test_mot_example(capsys: pytest.CaptureFixture[str]) -> None:
    out = _run("vol_mot_example.py", capsys)
    for row in ("k=0.9: [11.4047, 14.5295]", "k=1.0: [3.5508, 7.8021]", "k=1.1: [0.0837, 4.0360]"):
        assert row in out
    assert "upper bound 0.147050; vanillas cost 0.147050" in out
    assert "LP bounds [0.0507, 0.1389]" in out
    assert "(gap 3.0e-03," in out and "(gap 2.0e-04," in out
    assert "upper bound 0.0606, Heston price 0.0457" in out
    assert "hedge P&L: mean 0.0150, worst -0.0009" in out
    assert "  [0.0634, 0.0778]" in out
    assert "nan" not in out
