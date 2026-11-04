"""JSON-safe serialisation of volatility surfaces and their inputs.

:func:`to_dict` turns a surface, a total-variance model, a forward or a
curve into nested dictionaries of strings, numbers and lists. Each
dictionary has a ``"type"`` tag, and :func:`from_dict` rebuilds the
object from it exactly: every float survives a JSON round trip because
:func:`json.dumps` writes the shortest repr that reads back to the same
double. :func:`dumps` and :func:`loads` wrap both steps.

Supported types:

* **Slices:** :class:`SVIRaw`, :class:`SVINatural`, :class:`SVIJumpWings`,
  :class:`ESSVISlice`.
* **Models:** :class:`SSVI` (with :class:`PowerLawPhi` or
  :class:`HestonLikePhi` and an :class:`ATMVarianceCurve`), :class:`ESSVI`,
  and :class:`PriceInterpolatedSurface` / :class:`VarianceInterpolatedSurface`
  of serialisable slices. Flat surfaces are covered too.
* **Market objects:** :class:`VolSurface`, an :class:`EquityForward`
  without discrete dividends, :class:`FlatCurve`, and
  :class:`PiecewiseCurve` (any scheme of :mod:`spxa.rates.interpolation`).

Callables (for example a lambda ``theta``) cannot be serialised and raise
``TypeError``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from typing import Any, Callable

import numpy as np

__all__ = ["to_dict", "from_dict", "dumps", "loads"]


def _date(d: Any) -> Any:
    return None if d is None else d.isoformat()


def _undate(s: Any) -> Any:
    return None if s is None else _dt.date.fromisoformat(s)


def _floats(a: Any) -> list[float]:
    return [float(x) for x in np.asarray(a, float).ravel()]


def _dc_name(dc: Any) -> str:
    return str(getattr(dc, "name", dc))


def _registry() -> dict[str, tuple[type, Callable[[Any], dict], Callable[[dict], Any]]]:
    from spxa.market.curves import EquityForward, FlatCurve
    from spxa.rates import interpolation as interp
    from spxa.vol.essvi import ESSVI, ESSVISlice
    from spxa.vol.interpolation import PriceInterpolatedSurface, VarianceInterpolatedSurface
    from spxa.vol.ssvi import SSVI, ATMVarianceCurve, HestonLikePhi, PowerLawPhi
    from spxa.vol.surface import VolSurface, _Flat
    from spxa.vol.svi import SVIJumpWings, SVINatural, SVIRaw

    def dc_fields(cls: type) -> tuple[Callable[[Any], dict], Callable[[dict], Any]]:
        names = [f.name for f in dataclasses.fields(cls)]
        return (
            lambda o: {n: float(getattr(o, n)) for n in names},
            lambda d: cls(**{n: d[n] for n in names}),
        )

    def curve_out(c: Any) -> dict:
        if isinstance(c, FlatCurve):
            return {
                "type": "FlatCurve",
                "rate": c.rate,
                "compounding": c.compounding,
                "reference_date": _date(c.reference_date),
                "day_count": _dc_name(c.day_count),
            }
        if isinstance(c, interp.PiecewiseCurve):
            return {
                "type": "PiecewiseCurve",
                "times": _floats(c.node_times),
                "discounts": _floats(c.node_discounts),
                "scheme": {"name": type(c.scheme).__name__, **vars(c.scheme)},
                "reference_date": _date(c.reference_date),
                "day_count": _dc_name(c.day_count),
            }
        raise TypeError(f"cannot serialise curve {type(c).__name__}")

    def curve_in(d: dict) -> Any:
        if d["type"] == "FlatCurve":
            return FlatCurve(
                d["rate"], d["compounding"], _undate(d["reference_date"]), d["day_count"]
            )
        sch = dict(d["scheme"])
        scheme = getattr(interp, sch.pop("name"))(**sch)
        return interp.PiecewiseCurve(
            d["times"], d["discounts"], scheme, _undate(d["reference_date"]), d["day_count"]
        )

    def fwd_out(f: EquityForward) -> dict:
        if f.dividends:
            raise TypeError("discrete dividends are not serialisable")
        repo = None if f.repo_curve is f.discount_curve else curve_out(f.repo_curve)
        div = None if f.dividend_curve is None else curve_out(f.dividend_curve)
        return {
            "spot": f.spot,
            "discount": curve_out(f.discount_curve),
            "dividend": div,
            "repo": repo,
        }

    def fwd_in(d: dict) -> EquityForward:
        div = None if d["dividend"] is None else curve_in(d["dividend"])
        repo = None if d["repo"] is None else curve_in(d["repo"])
        return EquityForward(d["spot"], curve_in(d["discount"]), div, repo)

    reg: dict[str, tuple[type, Callable[[Any], dict], Callable[[dict], Any]]] = {}
    for cls in (SVIRaw, SVINatural, SVIJumpWings, ESSVISlice, PowerLawPhi, HestonLikePhi):
        reg[cls.__name__] = (cls, *dc_fields(cls))
    reg["ATMVarianceCurve"] = (
        ATMVarianceCurve,
        lambda o: {"times": _floats(o.times), "thetas": _floats(o.thetas)},
        lambda d: ATMVarianceCurve(d["times"], d["thetas"]),
    )
    reg["SSVI"] = (
        SSVI,
        lambda o: {"rho": o.rho, "phi": to_dict(o.phi), "theta": to_dict(o.theta)},
        lambda d: SSVI(d["rho"], from_dict(d["phi"]), from_dict(d["theta"])),
    )
    reg["ESSVI"] = (
        ESSVI,
        lambda o: {
            "times": _floats(o.times),
            "slices": [to_dict(s) for s in o.slices],
            "long_theta_slope": o.long_theta_slope,
        },
        lambda d: ESSVI(d["times"], [from_dict(s) for s in d["slices"]], d["long_theta_slope"]),
    )
    reg["PriceInterpolatedSurface"] = (
        PriceInterpolatedSurface,
        lambda o: {"expiries": _floats(o.expiries), "slices": [to_dict(s) for s in o.slices]},
        lambda d: PriceInterpolatedSurface(d["expiries"], [from_dict(s) for s in d["slices"]]),
    )
    reg["VarianceInterpolatedSurface"] = (
        VarianceInterpolatedSurface,
        lambda o: {
            "expiries": _floats(o.expiries),
            "slices": [to_dict(s) for s in o.slices],
            "mode": o.mode,
        },
        lambda d: VarianceInterpolatedSurface(
            d["expiries"], [from_dict(s) for s in d["slices"]], d["mode"]
        ),
    )
    reg["Flat"] = (_Flat, lambda o: {"sigma": o.sigma}, lambda d: _Flat(d["sigma"]))
    reg["EquityForward"] = (EquityForward, fwd_out, fwd_in)
    reg["FlatCurve"] = (FlatCurve, lambda o: curve_out(o), curve_in)
    reg["PiecewiseCurve"] = (interp.PiecewiseCurve, lambda o: curve_out(o), curve_in)
    reg["VolSurface"] = (
        VolSurface,
        lambda o: {"model": to_dict(o.model), "forward": to_dict(o.forward_curve)},
        lambda d: VolSurface(from_dict(d["model"]), from_dict(d["forward"])),
    )
    return reg


def to_dict(obj: Any) -> dict[str, Any]:
    """Tagged, JSON-safe dictionary for a supported object."""
    for tag, (cls, out, _) in _registry().items():
        if type(obj) is cls:
            body = out(obj)
            body.pop("type", None)
            return {"type": tag, **body}
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def from_dict(d: dict[str, Any]) -> Any:
    """Rebuild an object from :func:`to_dict` output."""
    reg = _registry()
    tag = d.get("type")
    if tag not in reg:
        raise ValueError(f"unknown type tag {tag!r}")
    return reg[tag][2](d)


def dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(to_dict(obj), **kwargs)


def loads(s: str) -> Any:
    return from_dict(json.loads(s))
