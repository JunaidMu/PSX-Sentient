"""Tests for the PSX daily-file parsers and the announcement-to-headline conversion."""

from __future__ import annotations

from pathlib import Path
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.premarket import announcement_headline, previous_trading_day
from src.data.psx_files import parse_market_summary, parse_off_market


def test_market_summary_zip_named_dot_z_is_parsed_and_futures_dropped(tmp_path) -> None:
    lis = (
        "27NOV2026|OGDC|0820|Oil & Gas Dev|310.38|311.55|309.05|310.02|1209498|310.35|||\n"
        "27NOV2026|OGDC-DECB|40|Oil & Gas Dev|314.77|315|311.6|312.27|2300500|313.1|||\n"
        "27NOV2026|HBL|0807|Habib Bank|304.10|305.70|303.32|304.12|765334|304.02|||\n"
    )
    path = tmp_path / "2026-11-27.Z"  # PSX serves a ZIP with a .Z extension
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("closing11.lis", lis)
    df = parse_market_summary(path)
    assert list(df["symbol"]) == ["OGDC", "HBL"]
    ogdc = df[df["symbol"] == "OGDC"].iloc[0]
    assert ogdc["close"] == 310.02 and ogdc["volume"] == 1209498 and str(ogdc["date"].date()) == "2026-11-27"


def test_off_market_csv_reads_deals_from_both_sections(tmp_path) -> None:
    csv = (
        " Off Market Transaction \n\nMEMBER , TO , MEMBER \n\n"
        "Date ,SETTLEMENT DATE ,MEMBER CODE,SYMBOL CODE,COMPANY ,TURNOVER ,RATE ,VALUES \n\n"
        "27-Nov-26,30-Nov-26,194,PPL,Pak Petroleum,200000,202.75,40550000\n\n"
        "CROSS ,TRANSACTIONS, BETWEEN, CLIENT TO ,CLIENT & FINANCIAL, INSTITUTIONS \n\n"
        "Date ,SETTLEMENT DATE ,MEMBER CODE,SYMBOL CODE,COMPANY ,TURNOVER ,RATE ,VALUES \n\n"
        "27-Nov-26,27-Nov-26,006,PSO,P.S.O.,10000,359.00,3590000\n"
    )
    path = tmp_path / "omts.csv"
    path.write_text(csv)
    df = parse_off_market(path)
    assert list(df["symbol"]) == ["PPL", "PSO"]
    assert df["volume"].tolist() == [200000, 10000] and df["rate"].tolist() == [202.75, 359.0]


def test_announcement_rows_become_factual_headlines() -> None:
    loss = announcement_headline("EXIDE", "Exide Pakistan Limited Half Year 30-09-2026 Unconsolidated - 40.516 (5.993) (0.77) - - - - -")
    assert loss.startswith("Exide Pakistan Limited reports half year after-tax loss of PKR 6.0 million")
    div = announcement_headline(
        "OGDC", "Oil & Gas Development Company Annual 30-06-2026 Unconsolidated 40%(F) (D) 120,500.2 85,300.1 19.83 AGM 20-Oct-26"
    )
    assert "after-tax profit of PKR 85,300.1 million" in div and "cash dividend of 40%" in div
    bc = announcement_headline("NATF", "National Foods Limited - - - - - - - 12-Dec-2026 12-Dec-2026 Book Closure for entitlemnt")
    assert bc == "National Foods Limited announces book closure"


def test_previous_trading_day_skips_weekends() -> None:
    from datetime import datetime

    from src.data.premarket import PKT

    assert previous_trading_day(datetime(2026, 11, 30, 9, 0, tzinfo=PKT)) == "2026-11-27"  # Monday -> Friday
    assert previous_trading_day(datetime(2026, 11, 26, 9, 0, tzinfo=PKT)) == "2026-11-25"
