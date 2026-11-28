"""Off-market ("whale") block-deal score from the PSX Off Market Transaction Summary.

W_ndm in [-1, 1]: a block larger than `volume_threshold_ratio` of the stock's 20-day average volume,
dealt at a premium (+) or discount (-) to the regular-market close, scaled by the block's size.

Only near-term settlements count. Deals settling weeks later (forward / financing deals, e.g. a trade
on 27-Nov settling 28-Dec) are priced with a carry/financing premium, which is not a bullish signal,
so they are reported separately and excluded from the score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MAX_SETTLEMENT_BDAYS = 3  # spot-like deals: settle within T+3 business days


def compute_whale_block_metric(
    ndm_volume: float,
    ndm_deal_rate: float,
    ready_close: float,
    vol_sma_20: float,
    volume_threshold_ratio: float = 0.05,
) -> dict[str, Any]:
    """Score one stock's aggregated off-market deals against its regular-market close."""
    if vol_sma_20 <= 0.0 or ready_close <= 0.0 or ndm_volume <= 0.0 or ndm_deal_rate <= 0.0:
        return {"ndm_volume": 0, "ndm_deal_rate": 0.0, "ndm_vol_pct_of_sma20": 0.0,
                "ndm_spread_pct": 0.0, "is_whale_block": False, "W_ndm": 0.0}

    vol_ratio = float(ndm_volume / vol_sma_20)
    spread_pct = float(((ndm_deal_rate - ready_close) / ready_close) * 100.0)
    is_whale = bool(vol_ratio > volume_threshold_ratio)
    if not is_whale or abs(spread_pct) < 1e-6:
        w_ndm = 0.0
    else:
        # +/-5% spread = full scale, amplified by block size above the 5% threshold (capped at 2.5x)
        size_factor = min(vol_ratio / volume_threshold_ratio, 2.5)
        w_ndm = float(np.clip((spread_pct / 5.0) * (0.5 + 0.25 * size_factor), -1.0, 1.0))
    return {"ndm_volume": int(round(ndm_volume)), "ndm_deal_rate": round(float(ndm_deal_rate), 4),
            "ndm_vol_pct_of_sma20": round(vol_ratio * 100.0, 2), "ndm_spread_pct": round(spread_pct, 3),
            "is_whale_block": is_whale, "W_ndm": round(w_ndm, 4)}


def settlement_bdays(trade_date: pd.Series, settlement_date: pd.Series) -> pd.Series:
    """Business days between trade and settlement (NaN if a date is unparseable)."""
    t = pd.to_datetime(trade_date, format="%d-%b-%y", errors="coerce")
    s = pd.to_datetime(settlement_date, format="%d-%b-%y", errors="coerce")
    ok = t.notna() & s.notna()
    out = pd.Series(np.nan, index=trade_date.index)
    out[ok] = np.busday_count(t[ok].values.astype("datetime64[D]"), s[ok].values.astype("datetime64[D]"))
    return out


def whale_scores(deals: pd.DataFrame, close: dict[str, float], vol20: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Per-symbol W_ndm from spot-like deals; forward deals are counted but excluded from the score."""
    out: dict[str, dict[str, Any]] = {}
    if deals.empty:
        return out
    d = deals.copy()
    if "settlement" in d and "date" in d:
        d["settle_bdays"] = settlement_bdays(d["date"], d["settlement"])
    else:
        d["settle_bdays"] = 0
    for sym, g in d.groupby("symbol"):
        spot = g[g["settle_bdays"].fillna(0) <= MAX_SETTLEMENT_BDAYS]
        forward = g.drop(spot.index)
        res: dict[str, Any]
        if not spot.empty and sym in close and sym in vol20:
            vwap = float((spot["volume"] * spot["rate"]).sum() / spot["volume"].sum())
            res = compute_whale_block_metric(float(spot["volume"].sum()), vwap, float(close[sym]), vol20[sym])
        else:
            res = compute_whale_block_metric(0, 0, 1, 1)
        res["deals"] = int(len(spot))
        res["forward_deals"] = int(len(forward))
        res["forward_volume"] = int(forward["volume"].sum()) if not forward.empty else 0
        out[sym] = res
    return out
