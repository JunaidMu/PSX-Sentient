"""Unit tests for rule-based baselines and the macro-lag leak fix."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.build_features import compute_features_and_labels
from src.models.baselines import predict_always_neutral, predict_momentum_rule


def test_always_neutral_rule() -> None:
    df = pd.DataFrame({"ret_5d": [0.1, -0.1, 0.0]})
    np.testing.assert_array_equal(predict_always_neutral(df), [1, 1, 1])


def test_momentum_rule_thresholds() -> None:
    df = pd.DataFrame({"ret_5d": [0.05, -0.05, 0.01, 0.02], "vol_20d": [0.02, 0.02, 0.02, 0.02]})
    np.testing.assert_array_equal(predict_momentum_rule(df), [2, 0, 1, 1])


def _one_symbol(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "OGDC",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
            "is_ex_dividend_gap": 0,
            "brent_close": 70.0 + np.arange(n) * 0.1,
            "usdpkr_close": 280.0 + np.arange(n) * 0.05,
            "dxy_close": 104.0 + np.arange(n) * 0.01,
        }
    )


def test_same_day_macro_close_does_not_affect_same_day_features() -> None:
    """Macro closes settle after the PSX close, so changing day-t macro must not change day-t features."""
    raw = _one_symbol()
    t = raw["date"].iloc[60]
    base = compute_features_and_labels(raw, drop_Incomplete=False)

    shocked = raw.copy()
    shocked.loc[shocked["date"] == t, ["brent_close", "usdpkr_close", "dxy_close"]] *= 1.5
    after = compute_features_and_labels(shocked, drop_Incomplete=False)

    macro_cols = ["brent_ret_1d", "brent_ret_5d", "usdpkr_ret_1d", "usdpkr_ret_5d", "dxy_ret_1d", "dxy_ret_5d"]
    row_t = base["date"] == t
    np.testing.assert_allclose(base.loc[row_t, macro_cols].to_numpy(), after.loc[row_t, macro_cols].to_numpy())
    # ...but the next day's features do see it
    row_next = base["date"] == raw["date"].iloc[61]
    assert not np.allclose(base.loc[row_next, "brent_ret_1d"], after.loc[row_next, "brent_ret_1d"])
