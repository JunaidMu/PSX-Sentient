"""Late-fusion decision rules: Conflict Veto Gate and blended conviction score.

Implements:
1. Conflict Veto Gate:
   If `(E_tech > 0.20 and S_t < -0.50)` OR `(E_tech < -0.20 and S_t > 0.50)`,
   set `veto_triggered = True` and force the final signal to `1` (`"NEUTRAL"`).
2. Blended Conviction Score:
   `C_t = 0.75 * E_tech + 0.25 * S_t`.
   If not vetoed:
   - Output `2` (`"UP"`) when `C_t > threshold` and `P_neutral < neutral_cap`
   - Output `0` (`"DOWN"`) when `C_t < -threshold` and `P_neutral < neutral_cap`
   The proposal's defaults are threshold 0.20 / neutral cap 0.50; the served threshold is tuned on the
   walk-forward validation years by src/models/signal_policy.py.
   - Otherwise output `1` (`"NEUTRAL"`).
Note: an earlier Test_2026 "ablation" in this module used simulated sentiment (template headlines assigned
from price patterns) and was removed; there is no historical news archive to evaluate fusion on yet.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


SIGNAL_LABELS: dict[int, str] = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


def apply_late_fusion_row(
    P_down: float,
    P_neutral: float,
    P_up: float,
    S_t: float,
    E_tech: float | None = None,
    threshold: float = 0.20,
    neutral_cap: float = 0.50,
) -> dict[str, Any]:
    """Apply Conflict Veto Gate and Blended Conviction thresholding to a single row."""
    p_d = float(P_down)
    p_n = float(P_neutral)
    p_u = float(P_up)
    e_tech = float(p_u - p_d) if E_tech is None else float(E_tech)
    s_t = float(np.clip(0.0 if pd.isna(S_t) else S_t, -1.0, 1.0))

    veto_triggered = bool(
        ((e_tech > 0.20) and (s_t < -0.50))
        or ((e_tech < -0.20) and (s_t > 0.50))
    )
    c_t = float(0.75 * e_tech + 0.25 * s_t)

    if veto_triggered:
        signal_class = 1
    elif (c_t > threshold) and (p_n < neutral_cap):
        signal_class = 2
    elif (c_t < -threshold) and (p_n < neutral_cap):
        signal_class = 0
    else:
        signal_class = 1

    return {
        "P_down": p_d,
        "P_neutral": p_n,
        "P_up": p_u,
        "E_tech": e_tech,
        "S_t": s_t,
        "C_t": c_t,
        "veto_triggered": veto_triggered,
        "fused_class": signal_class,
        "fused_signal": SIGNAL_LABELS[signal_class],
    }


def apply_conviction_tabular_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Conviction-Thresholded Tabular rule (`|E_tech| > 0.20` and `P_neutral < 0.50`, no NLP)."""
    out = df.copy()
    if "E_tech" not in out.columns:
        out["E_tech"] = out["P_up"] - out["P_down"]

    e_tech = out["E_tech"].to_numpy(dtype="float64")
    p_neut = out["P_neutral"].to_numpy(dtype="float64")

    conv_class = np.where(
        (e_tech > 0.20) & (p_neut < 0.50),
        2,
        np.where((e_tech < -0.20) & (p_neut < 0.50), 0, 1),
    ).astype("int64")

    out["conviction_tabular_class"] = conv_class
    out["conviction_tabular_signal"] = pd.Series(conv_class, index=out.index).map(SIGNAL_LABELS)
    return out


def apply_late_fusion_df(df: pd.DataFrame, sentiment_col: str = "S_t") -> pd.DataFrame:
    """Vectorized Late-Fusion & Conflict Veto Gate over a predictions DataFrame."""
    out = df.copy()
    if "E_tech" not in out.columns:
        out["E_tech"] = out["P_up"] - out["P_down"]

    if sentiment_col not in out.columns:
        out["S_t"] = 0.0
    else:
        out["S_t"] = out[sentiment_col].fillna(0.0).clip(lower=-1.0, upper=1.0).astype("float64")

    e_tech = out["E_tech"].to_numpy(dtype="float64")
    p_neut = out["P_neutral"].to_numpy(dtype="float64")
    s_t = out["S_t"].to_numpy(dtype="float64")

    veto_mask = ((e_tech > 0.20) & (s_t < -0.50)) | ((e_tech < -0.20) & (s_t > 0.50))
    c_t = 0.75 * e_tech + 0.25 * s_t

    raw_signal = np.where(
        (c_t > 0.20) & (p_neut < 0.50),
        2,
        np.where((c_t < -0.20) & (p_neut < 0.50), 0, 1),
    )
    fused_class = np.where(veto_mask, 1, raw_signal).astype("int64")

    out["C_t"] = c_t
    out["veto_triggered"] = veto_mask
    out["fused_class"] = fused_class
    out["fused_signal"] = pd.Series(fused_class, index=out.index).map(SIGNAL_LABELS)
    return out
