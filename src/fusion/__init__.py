"""Phase 5: Late-Fusion Engine, Conflict Veto Gate, and Tabular vs. Late-Fusion Ablation."""

from src.fusion.late_fusion import (
    ABLATION_METRICS_PATH,
    FUSED_PREDICTIONS_PATH,
    SIGNAL_LABELS,
    apply_conviction_tabular_df,
    apply_late_fusion_df,
    apply_late_fusion_row,
    build_test_2026_sentiment_series,
    run_phase5_ablation,
)

__all__ = [
    "ABLATION_METRICS_PATH",
    "FUSED_PREDICTIONS_PATH",
    "SIGNAL_LABELS",
    "apply_conviction_tabular_df",
    "apply_late_fusion_df",
    "apply_late_fusion_row",
    "build_test_2026_sentiment_series",
    "run_phase5_ablation",
]
