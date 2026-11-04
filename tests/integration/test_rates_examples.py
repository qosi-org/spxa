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


def test_vol_example(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / "rates_vol_example.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "caps reprice from stripped vols (< 1e-10): True" in out
    assert "SABR smile recovered (rmse < 1e-10): True" in out
    assert "Hagan density negative near zero: True" in out
    assert "PDE mass 1.000000000000, absorbed at zero 8.74%" in out
    assert "cube at 5Y x 10Y: alpha 0.0350, rho -0.2000" in out
    assert "smile lowers the adjustment (skewed right wing): True" in out
    assert "nan" not in out and ": False" not in out


def test_short_rate_example(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / "rates_short_rate_example.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "swaptions reprice (< 1e-10): True" in out
    assert "PDE 0.009495" in out
    assert "a 0.4873, sigma 0.00991, b 0.0406, eta 0.00806, rho -0.606" in out
    assert "max vol error 0.057 bp" in out
    assert "Hull-White  86.64  87.14  87.65" in out
    assert "CIR++       27.48  46.42  57.09" in out
    assert "within 4 standard errors: True" in out
    assert "nan" not in out and ": False" not in out


def test_hjm_cheyette_example(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / "rates_hjm_cheyette_example.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "closed form 0.018159" in out
    assert "MC agrees with Black at the scheme's variance: True" in out
    assert "sigma levels: 0.00994 0.00929 0.00842  beta 0.2360" in out
    assert "max vol error 1.77 bp" in out
    assert "PDE within 1.5 bp of the quotes: True" in out
    assert "European 0.003784, American 0.010735" in out
    assert "nan" not in out and ": False" not in out


def test_lmm_example(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / "rates_lmm_example.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "abcd 0.0500 0.0900 0.4400 0.1100" in out
    assert "caplets exact (< 1e-12): True" in out
    assert "correlation beta 0.1409, rho_inf 0.5178" in out
    assert "max error 3.8 bp" in out
    assert "Rebonato 131.18 bp, MC 131.20 +- 1.09 bp" in out
    assert "4.0Y x 6M ATM: forward-looking  20.40 bp, in arrears  20.68 bp" in out
    assert "K 3.02%: Hagan 20.67%, MC 20.75%" in out
    assert out.count("within 3 SE: True") == 2
    assert "prices within 3 SE + 1%: True" in out
    assert "nan" not in out and ": False" not in out
