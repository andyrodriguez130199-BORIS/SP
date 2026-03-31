#!/usr/bin/env python3
"""
sp500_extractor.py
==================
Constructs a Longitudinal Panel Dataset (Long format) for econometric modelling
of S&P 500 assets, covering fiscal years 2020-2024.

Output columns
--------------
Ticker | Año | Sector_GICS | Precio_Ajustado_12M | EPS_Diluido_Norm
       | Ratio_Deuda_Activos | Log_Activos_Totales | Dummy_Dividendos

Dependencies: pandas, numpy, yfinance, yahooquery, requests
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from yahooquery import Ticker as YQTicker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FISCAL_YEARS: list[int] = [2022, 2023, 2024]
OUTPUT_FILE: str = "sp500_panel_data.csv"

# GICS sector labels that must be excluded (case-insensitive substring match)
EXCLUDED_SECTOR_KEYWORDS: list[str] = ["financial", "utilities", "utility"]

# Minimum trading-day coverage required after filing date (out of 252 target)
MIN_TRADING_DAYS: int = 126

# Seconds to sleep between ticker requests (avoids API rate-limit bans)
REQUEST_DELAY: float = 0.5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# PHASE 1 – Universe construction and filtering
# ===========================================================================

def get_sp500_universe() -> tuple[list[str], dict[str, str]]:
    """
    Download the current S&P 500 constituent list from Wikipedia and return:

    * ``tickers``      – list of ticker symbols (dots replaced with hyphens
                         for yfinance compatibility, e.g. BRK.B → BRK-B).
    * ``sector_map``   – dict mapping each ticker to its GICS sector string.

    Survivorship-bias note
    ----------------------
    We intentionally retain only companies with *complete* data across the full
    2020-2024 window (enforced in Phase 2).  Companies that were delisted,
    went bankrupt, or entered the index after 2020 will have missing
    fundamental records and will be dropped at the NaN-filter stage.  This
    approximates survivorship-bias control without requiring a paid historical
    constituent database (CRSP/Compustat).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info("Downloading S&P 500 constituent list from Wikipedia …")
    try:
        # Disfrazamos la petición para que Wikipedia crea que somos Google Chrome
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        html_data = requests.get(url, headers=headers).text
        tables = pd.read_html(html_data, header=0)
        df = tables[0]
        # Normalise column names (Wikipedia occasionally changes them)
        df.columns = df.columns.str.strip()
        symbol_col = next(
            c for c in df.columns if c.lower() in ("symbol", "ticker")
        )
        sector_col = next(
            c for c in df.columns if "sector" in c.lower()
        )
        df[symbol_col] = df[symbol_col].str.strip().str.replace(".", "-", regex=False)
        tickers = df[symbol_col].tolist()
        sector_map = dict(zip(df[symbol_col], df[sector_col]))
        logger.info("Downloaded %d tickers.", len(tickers))
        return tickers, sector_map
    except Exception as exc:
        logger.error("Failed to download S&P 500 list: %s", exc)
        raise


def is_excluded_sector(sector: str | None) -> bool:
    """Return True if the sector falls in the excluded GICS categories
    (Financials → SIC 6000-6999, Utilities → SIC 4900-4999)."""
    if not sector:
        return True
    low = sector.lower()
    return any(kw in low for kw in EXCLUDED_SECTOR_KEYWORDS)


def apply_sector_filter(
    tickers: list[str], sector_map: dict[str, str]
) -> tuple[list[str], dict[str, str]]:
    """Remove tickers belonging to excluded sectors."""
    kept: list[str] = []
    kept_sectors: dict[str, str] = {}
    for t in tickers:
        sector = sector_map.get(t)
        if is_excluded_sector(sector):
            logger.debug("Excluded %s — sector: %s", t, sector)
            continue
        kept.append(t)
        kept_sectors[t] = sector
    logger.info(
        "After sector filter (Financials & Utilities removed): %d tickers.",
        len(kept),
    )
    return kept, kept_sectors


# ===========================================================================
# PHASE 2 – Variable extraction helpers
# ===========================================================================

def _fetch_financials(
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch annual income statement, balance sheet, and cash-flow statement
    for *ticker* via yahooquery.  Returns three DataFrames (may be empty on
    failure).
    """
    try:
        yq = YQTicker(ticker)
        income = yq.income_statement(frequency="a")
        balance = yq.balance_sheet(frequency="a")
        cashflow = yq.cash_flow(frequency="a")

        # yahooquery occasionally returns a dict on error
        income = income if isinstance(income, pd.DataFrame) else pd.DataFrame()
        balance = balance if isinstance(balance, pd.DataFrame) else pd.DataFrame()
        cashflow = cashflow if isinstance(cashflow, pd.DataFrame) else pd.DataFrame()

        # Ensure asOfDate is datetime
        for df in (income, balance, cashflow):
            if not df.empty and "asOfDate" in df.columns:
                df["asOfDate"] = pd.to_datetime(df["asOfDate"])

        return income, balance, cashflow
    except Exception as exc:
        logger.warning("Could not fetch financials for %s: %s", ticker, exc)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def _year_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return rows of *df* whose ``asOfDate`` falls in *year*."""
    if df.empty or "asOfDate" not in df.columns:
        return pd.DataFrame()
    return df[df["asOfDate"].dt.year == year].copy()


def _first_value(df: pd.DataFrame, *columns: str) -> float | None:
    """Return the first non-NaN value found across the candidate *columns*."""
    for col in columns:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.notna(val):
                return float(val)
    return None


def calculate_average_adjusted_price(
    ticker: str, filing_date: datetime, trading_days: int = 252
) -> float | None:
    """
    Download daily adjusted close prices for *ticker* starting at
    *filing_date* and spanning 252 trading days (≈ 12 months).  Returns the
    arithmetic mean of those prices.

    ``auto_adjust=True`` ensures prices are adjusted for splits and dividends,
    eliminating look-ahead bias from corporate actions.
    """
    try:
        # Fetch with a calendar buffer so we collect exactly 252 trading days
        end_date = filing_date + timedelta(days=trading_days + 60)
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(
            start=filing_date,
            end=end_date,
            auto_adjust=True,
            actions=False,
        )
        if hist.empty:
            return None
        hist = hist.head(trading_days)
        if len(hist) < MIN_TRADING_DAYS:
            logger.warning(
                "%s %s: only %d trading days available (need ≥ %d).",
                ticker, filing_date.date(), len(hist), MIN_TRADING_DAYS,
            )
            return None
        return float(hist["Close"].mean())
    except Exception as exc:
        logger.warning("Price fetch failed for %s (%s): %s", ticker, filing_date.date(), exc)
        return None


def extract_ticker_year(
    ticker: str,
    year: int,
    sector: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict | None:
    """
    Build one panel record for (*ticker*, *year*).  Returns ``None`` when
    insufficient data is available for this observation.
    """
    yi = _year_slice(income, year)
    yb = _year_slice(balance, year)
    yc = _year_slice(cashflow, year)

    if yi.empty and yb.empty:
        return None

    # ---- Filing date (asOfDate from 10-K) ----------------------------------
    filing_date: datetime | None = None
    if not yb.empty:
        filing_date = yb["asOfDate"].iloc[0]
    elif not yi.empty:
        filing_date = yi["asOfDate"].iloc[0]

    # ---- Variable 2: Adjusted price 12 M -----------------------------------
    price_12m: float | None = None
    if filing_date is not None:
        price_12m = calculate_average_adjusted_price(ticker, filing_date)

    # ---- Variable 3: EPS Diluted (excl. extraordinary items) ---------------
    eps: float | None = None
    if not yi.empty:
        eps = _first_value(
            yi,
            "DilutedEPS",
            "DilutedEPSFromContinuingOperations",
            "BasicEPS",
            "BasicEPSFromContinuingOperations",
        )

    # ---- Variable 4: Debt Ratio = Total Debt / Total Assets ----------------
    debt_ratio: float | None = None
    log_assets: float | None = None
    equity: float | None = None

    if not yb.empty:
        total_assets = _first_value(yb, "TotalAssets")
        short_debt = _first_value(
            yb,
            "CurrentDebt",
            "ShortTermDebt",
            "CurrentPortionOfLongTermDebt",
            "ShortLongTermDebtTotal",
        )
        long_debt = _first_value(
            yb,
            "LongTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermDebtAndCapitalLeaseObligation",
        )

        if total_assets is not None and total_assets > 0:
            total_debt = (short_debt or 0.0) + (long_debt or 0.0)
            debt_ratio = total_debt / total_assets

            # ---- Variable 5: Log Total Assets --------------------------------
            log_assets = float(np.log(abs(total_assets)))

        equity = _first_value(
            yb,
            "StockholdersEquity",
            "TotalStockholdersEquity",
            "CommonStockEquity",
            "TotalEquityGrossMinorityInterest",
        )

    # ---- Variable 6: Dividend Dummy ----------------------------------------
    dps_total: float = 0.0

    # Primary source: cash-flow statement (absolute value; outflows are negative)
    if not yc.empty:
        cf_div = _first_value(
            yc,
            "CashDividendsPaid",
            "DividendsPaid",
            "CommonStockDividendsPaid",
            "PaymentOfDividends",
        )
        if cf_div is not None:
            dps_total = abs(cf_div)

    # Fallback: yfinance dividend events for the calendar year
    if dps_total == 0.0:
        try:
            yf_t = yf.Ticker(ticker)
            divs = yf_t.dividends
            if not divs.empty:
                year_divs = divs[divs.index.year == year]
                if not year_divs.empty:
                    dps_total = float(year_divs.sum())
        except Exception:
            pass

    dummy_div = 1 if dps_total > 0.0 else 0

    return {
        "Ticker": ticker,
        "Año": year,
        "Sector_GICS": sector,
        "Precio_Ajustado_12M": price_12m,
        "EPS_Diluido_Norm": eps,
        "Ratio_Deuda_Activos": debt_ratio,
        "Log_Activos_Totales": log_assets,
        "Dummy_Dividendos": dummy_div,
        # Internal fields used for quality filtering (removed before export)
        "_equity": equity,
    }


# ===========================================================================
# PHASE 3 – Pipeline orchestration and export
# ===========================================================================

def run_pipeline() -> pd.DataFrame:
    """
    Execute all three phases and write the panel to ``sp500_panel_data.csv``.
    """
    # ------------------------------------------------------------------
    # Phase 1 — Universe
    # ------------------------------------------------------------------
    tickers, sector_map = get_sp500_universe()
    tickers, sector_map = apply_sector_filter(tickers, sector_map)

    # ------------------------------------------------------------------
    # Phase 2 — Data extraction
    # ------------------------------------------------------------------
    all_records: list[dict] = []
    skipped: list[str] = []

    for idx, ticker in enumerate(tickers, start=1):
        logger.info("[%d/%d] Processing %s …", idx, len(tickers), ticker)

        income, balance, cashflow = _fetch_financials(ticker)
        if income.empty and balance.empty:
            logger.warning("%s: no financial data — skipping.", ticker)
            skipped.append(ticker)
            continue

        ticker_records: list[dict] = []
        discard = False

        for year in FISCAL_YEARS:
            record = extract_ticker_year(
                ticker, year, sector_map.get(ticker, ""),
                income, balance, cashflow,
            )

            if record is None:
                logger.warning("%s %d: no data — discarding ticker.", ticker, year)
                discard = True
                break

            # Quality filter: negative equity in any year → discard entire ticker
            equity = record.pop("_equity", None)
            if equity is not None and equity < 0:
                logger.info(
                    "%s %d: negative equity (%.0f) — discarding ticker.",
                    ticker, year, equity,
                )
                discard = True
                break

            # Quality filter: NaN in any required fundamental variable
            required_fields = [
                "Precio_Ajustado_12M",
                "EPS_Diluido_Norm",
                "Ratio_Deuda_Activos",
                "Log_Activos_Totales",
            ]
            missing = [f for f in required_fields if record.get(f) is None]
            if missing:
                logger.warning(
                    "%s %d: missing fields %s — discarding ticker.", ticker, year, missing
                )
                discard = True
                break

            ticker_records.append(record)

        if discard or len(ticker_records) < len(FISCAL_YEARS):
            skipped.append(ticker)
        else:
            all_records.extend(ticker_records)

        time.sleep(REQUEST_DELAY)

    # ------------------------------------------------------------------
    # Phase 3 — Build panel and export
    # ------------------------------------------------------------------
    if not all_records:
        logger.error("No records collected — check API connectivity and ticker list.")
        return pd.DataFrame()

    PANEL_COLUMNS = [
        "Ticker",
        "Año",
        "Sector_GICS",
        "Precio_Ajustado_12M",
        "EPS_Diluido_Norm",
        "Ratio_Deuda_Activos",
        "Log_Activos_Totales",
        "Dummy_Dividendos",
    ]

    df = (
        pd.DataFrame(all_records)[PANEL_COLUMNS]
        .sort_values(["Ticker", "Año"])
        .reset_index(drop=True)
    )

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    logger.info("=" * 60)
    logger.info("Panel exported → %s", OUTPUT_FILE)
    logger.info("Shape: %s rows × %s columns", *df.shape)
    logger.info(
        "Tickers included: %d  |  Skipped/excluded: %d",
        df["Ticker"].nunique(),
        len(skipped),
    )
    logger.info("Sector distribution:\n%s", df.groupby("Sector_GICS")["Ticker"].nunique())
    logger.info("=" * 60)

    return df


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    run_pipeline()
