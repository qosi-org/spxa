r"""Raw option quotes.

:class:`OptionQuote` is one listed option (strike, type, bid/ask);
:class:`OptionChain` collects the quotes of one expiry and turns them into
mid prices, out-of-the-money selections and implied volatilities.
:class:`OptionSurfaceQuotes` groups chains across expiries.

These containers hold *prices* as they come from an exchange. The
calibration layer (:mod:`spxa.calibration.surface`) works with implied-vol
slices; :meth:`OptionChain.to_market_slice` converts between the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np

from spxa.market.dates import DateLike, to_date
from spxa.market.daycount import DayCount
from spxa.market.implied import implied_black_volatility, implied_normal_volatility

__all__ = ["OptionChain", "OptionQuote", "OptionSurfaceQuotes"]


@dataclass(frozen=True)
class OptionQuote:
    """A single option quote.

    Parameters
    ----------
    strike : float
    option : {"call", "put"}
    bid, ask : float, optional
        Either may be missing (one-sided market); ``mid`` then falls back
        to ``last`` if given.
    last : float, optional
        Last traded price.
    volume, open_interest : float, optional
    """

    strike: float
    option: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None

    def __post_init__(self) -> None:
        o = str(self.option).lower()
        if o in ("c", "call"):
            object.__setattr__(self, "option", "call")
        elif o in ("p", "put"):
            object.__setattr__(self, "option", "put")
        else:
            raise ValueError(f"option must be 'call' or 'put', got {self.option!r}")
        for name in ("bid", "ask", "last"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v < 0):
                raise ValueError(f"{name} must be a non-negative finite number")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError(f"bid {self.bid} exceeds ask {self.ask} at strike {self.strike}")
        if not math.isfinite(self.strike):
            raise ValueError("strike must be finite")

    @property
    def is_call(self) -> bool:
        return self.option == "call"

    @property
    def two_sided(self) -> bool:
        """Both bid and ask present (a zero bid counts as quoted)."""
        return self.bid is not None and self.ask is not None

    @property
    def mid(self) -> Optional[float]:
        if self.two_sided:
            assert self.bid is not None and self.ask is not None
            return 0.5 * (self.bid + self.ask)
        return self.last

    @property
    def spread(self) -> Optional[float]:
        if not self.two_sided:
            return None
        assert self.bid is not None and self.ask is not None
        return self.ask - self.bid

    @property
    def relative_spread(self) -> Optional[float]:
        """``spread / mid`` (``inf`` for a zero mid)."""
        m, s = self.mid, self.spread
        if m is None or s is None:
            return None
        return s / m if m > 0 else math.inf


@dataclass
class OptionChain:
    """Quotes of one expiry.

    Parameters
    ----------
    expiry : float or date-like
        Time to expiry in years, or an expiry date (then ``valuation_date``
        is required and the time is ACT/365F unless ``day_count`` is given).
    quotes : sequence of OptionQuote
    forward, discount : float, optional
        Forward price and discount factor for this expiry, if known.
    spot : float, optional
    valuation_date : date-like, optional
    """

    expiry: Any
    quotes: list[OptionQuote]
    forward: Optional[float] = None
    discount: Optional[float] = None
    spot: Optional[float] = None
    valuation_date: Optional[DateLike] = None
    day_count: str = "ACT/365F"
    _T: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.quotes = sorted(self.quotes, key=lambda q: (q.strike, q.option))
        if isinstance(self.expiry, (int, float, np.floating)) and not isinstance(self.expiry, bool):
            self._T = float(self.expiry)
        else:
            if self.valuation_date is None:
                raise ValueError("an expiry date needs valuation_date")
            self._T = DayCount(self.day_count).year_fraction(
                to_date(self.valuation_date), to_date(self.expiry)
            )
        if self._T <= 0:
            raise ValueError("expiry must be after the valuation date")
        if self.forward is not None and self.forward <= 0:
            raise ValueError("forward must be positive")
        if self.discount is not None and not 0 < self.discount <= 10:
            raise ValueError("discount factor must be positive")
        seen = set()
        for q in self.quotes:
            key = (q.strike, q.option)
            if key in seen:
                raise ValueError(f"duplicate quote {key}")
            seen.add(key)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        expiry: Any,
        strikes: Sequence[float],
        call_bid: Optional[Sequence[float]] = None,
        call_ask: Optional[Sequence[float]] = None,
        put_bid: Optional[Sequence[float]] = None,
        put_ask: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> "OptionChain":
        """Build from strike-aligned bid/ask arrays (NaN marks a missing quote)."""
        quotes: list[OptionQuote] = []
        for opt, bids, asks in (("call", call_bid, call_ask), ("put", put_bid, put_ask)):
            if bids is None and asks is None:
                continue
            b = np.full(len(strikes), np.nan) if bids is None else np.asarray(bids, float)
            a = np.full(len(strikes), np.nan) if asks is None else np.asarray(asks, float)
            for k, bi, ai in zip(strikes, b, a):
                if math.isnan(bi) and math.isnan(ai):
                    continue
                quotes.append(
                    OptionQuote(
                        float(k),
                        opt,
                        None if math.isnan(bi) else float(bi),
                        None if math.isnan(ai) else float(ai),
                    )
                )
        return cls(expiry, quotes, **kwargs)

    # -- accessors ------------------------------------------------------
    @property
    def T(self) -> float:
        """Time to expiry in years."""
        return self._T

    def __len__(self) -> int:
        return len(self.quotes)

    def __iter__(self) -> Iterator[OptionQuote]:
        return iter(self.quotes)

    def calls(self) -> list[OptionQuote]:
        return [q for q in self.quotes if q.is_call]

    def puts(self) -> list[OptionQuote]:
        return [q for q in self.quotes if not q.is_call]

    @property
    def strikes(self) -> np.ndarray:
        """Sorted unique strikes."""
        return np.unique([q.strike for q in self.quotes])

    def mids(self, option: str = "call") -> tuple[np.ndarray, np.ndarray]:
        """``(strikes, mid prices)`` of the given type, skipping quotes without a mid."""
        sel = self.calls() if option == "call" else self.puts()
        pts = [(q.strike, q.mid) for q in sel if q.mid is not None]
        if not pts:
            return np.empty(0), np.empty(0)
        k, m = zip(*pts)
        return np.asarray(k, float), np.asarray(m, float)

    def pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Strikes with both a call and a put mid: ``(K, call_mid, put_mid)``."""
        kc, mc = self.mids("call")
        kp, mp_ = self.mids("put")
        common, ic, ip = np.intersect1d(kc, kp, return_indices=True)
        return common, mc[ic], mp_[ip]

    # -- filtering ------------------------------------------------------
    def filter(
        self,
        min_bid: Optional[float] = None,
        max_relative_spread: Optional[float] = None,
        two_sided: bool = False,
        strike_range: Optional[tuple[float, float]] = None,
        min_open_interest: Optional[float] = None,
    ) -> "OptionChain":
        """Copy keeping only quotes that pass every given criterion."""
        keep = []
        for q in self.quotes:
            if two_sided and not q.two_sided:
                continue
            if min_bid is not None and (q.bid is None or q.bid < min_bid):
                continue
            if max_relative_spread is not None:
                rs = q.relative_spread
                if rs is None or rs > max_relative_spread:
                    continue
            if strike_range is not None and not strike_range[0] <= q.strike <= strike_range[1]:
                continue
            if min_open_interest is not None and (
                q.open_interest is None or q.open_interest < min_open_interest
            ):
                continue
            keep.append(q)
        return replace(self, quotes=keep)

    def otm(self, forward: Optional[float] = None) -> "OptionChain":
        """Out-of-the-money quotes: puts with ``K < F`` and calls with ``K >= F``."""
        F = self._need_forward(forward)
        keep = [
            q
            for q in self.quotes
            if (q.is_call and q.strike >= F) or (not q.is_call and q.strike < F)
        ]
        return replace(self, quotes=keep)

    def _need_forward(self, forward: Optional[float]) -> float:
        F = forward if forward is not None else self.forward
        if F is None:
            raise ValueError("a forward is required (set chain.forward or pass forward=)")
        return float(F)

    def _need_discount(self, discount: Optional[float]) -> float:
        d = discount if discount is not None else self.discount
        return 1.0 if d is None else float(d)

    # -- implied volatilities -------------------------------------------
    def implied_vols(
        self,
        forward: Optional[float] = None,
        discount: Optional[float] = None,
        model: str = "black",
        which: str = "mid",
        otm_only: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implied volatilities of the chain.

        Parameters
        ----------
        model : {"black", "normal"}
        which : {"mid", "bid", "ask"}
        otm_only : bool
            Use only out-of-the-money quotes (one vol per strike).

        Returns
        -------
        strikes, vols, is_call : ndarray
            Quotes whose price violates the arbitrage bounds get NaN.
        """
        if model not in ("black", "normal"):
            raise ValueError("model must be 'black' or 'normal'")
        if which not in ("mid", "bid", "ask"):
            raise ValueError("which must be 'mid', 'bid' or 'ask'")
        F, df = self._need_forward(forward), self._need_discount(discount)
        chain = self.otm(F) if otm_only else self
        ks, prices, calls = [], [], []
        for q in chain.quotes:
            p = getattr(q, which)
            if p is None:
                continue
            ks.append(q.strike)
            prices.append(p)
            calls.append(q.is_call)
        k = np.asarray(ks, float)
        if k.size == 0:
            return k, np.empty(0), np.empty(0, dtype=bool)
        c = np.asarray(calls, bool)
        solver = implied_black_volatility if model == "black" else implied_normal_volatility
        vols = solver(np.asarray(prices), F, k, self.T, df, c)
        return k, np.atleast_1d(vols), c

    def to_market_slice(
        self, forward: Optional[float] = None, discount: Optional[float] = None
    ) -> Any:
        """Convert to :class:`spxa.calibration.surface.data.MarketSlice` (OTM Black vols)."""
        from spxa.calibration.surface.data import MarketSlice, QuoteType

        F, df = self._need_forward(forward), self._need_discount(discount)
        k, v, _ = self.implied_vols(F, df)
        ok = np.isfinite(v) & (v > 0)
        return MarketSlice(
            maturity=self.T,
            strikes=k[ok],
            quotes=v[ok],
            quote_type=QuoteType.IMPLIED_VOL,
            forward=F,
            discount=df,
        )


@dataclass
class OptionSurfaceQuotes:
    """Option chains for several expiries, sorted by time to expiry."""

    chains: list[OptionChain]
    spot: Optional[float] = None

    def __post_init__(self) -> None:
        self.chains = sorted(self.chains, key=lambda c: c.T)
        ts = [c.T for c in self.chains]
        if len(set(ts)) != len(ts):
            raise ValueError("duplicate expiries")

    @property
    def expiries(self) -> np.ndarray:
        return np.array([c.T for c in self.chains])

    def __len__(self) -> int:
        return len(self.chains)

    def __iter__(self) -> Iterator[OptionChain]:
        return iter(self.chains)

    def __getitem__(self, i: int) -> OptionChain:
        return self.chains[i]

    def filter(self, **criteria: Any) -> "OptionSurfaceQuotes":
        """Apply :meth:`OptionChain.filter` to every chain (empty chains are dropped)."""
        out = [c.filter(**criteria) for c in self.chains]
        return OptionSurfaceQuotes([c for c in out if len(c)], self.spot)

    @classmethod
    def from_records(
        cls, records: Iterable[dict[str, Any]], spot: Optional[float] = None, **chain_kwargs: Any
    ) -> "OptionSurfaceQuotes":
        """Build from dict records with keys ``expiry``, ``strike``, ``option``, ``bid``, ``ask``
        (optional ``last``, ``volume``, ``open_interest``)."""
        groups: dict[Any, list[OptionQuote]] = {}
        for r in records:
            q = OptionQuote(
                float(r["strike"]),
                r["option"],
                r.get("bid"),
                r.get("ask"),
                r.get("last"),
                r.get("volume"),
                r.get("open_interest"),
            )
            groups.setdefault(r["expiry"], []).append(q)
        return cls(
            [OptionChain(e, qs, spot=spot, **chain_kwargs) for e, qs in groups.items()], spot
        )
