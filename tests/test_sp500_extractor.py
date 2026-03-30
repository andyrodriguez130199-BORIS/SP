"""
tests/test_sp500_extractor.py
Unit tests for the pure-logic helpers in sp500_extractor.py.
No network calls are made; financial DataFrames are built from fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from sp500_extractor import (
    _first_value,
    _year_slice,
    apply_sector_filter,
    extract_ticker_year,
    is_excluded_sector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_income() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asOfDate": pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31"]),
            "DilutedEPS": [2.0, 3.5, 4.1],
        }
    )


@pytest.fixture
def sample_balance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asOfDate": pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31"]),
            "TotalAssets": [800_000.0, 1_000_000.0, 1_200_000.0],
            "LongTermDebt": [100_000.0, 200_000.0, 250_000.0],
            "CurrentDebt": [20_000.0, 50_000.0, 60_000.0],
            "StockholdersEquity": [300_000.0, 400_000.0, 500_000.0],
        }
    )


@pytest.fixture
def sample_cashflow_div() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asOfDate": pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31"]),
            "CashDividendsPaid": [-5_000.0, -10_000.0, -12_000.0],
        }
    )


@pytest.fixture
def sample_cashflow_nodiv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asOfDate": pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31"]),
        }
    )


# ---------------------------------------------------------------------------
# is_excluded_sector
# ---------------------------------------------------------------------------

class TestIsExcludedSector:
    def test_financials_excluded(self):
        assert is_excluded_sector("Financials") is True

    def test_financial_services_excluded(self):
        assert is_excluded_sector("Financial Services") is True

    def test_utilities_excluded(self):
        assert is_excluded_sector("Utilities") is True

    def test_utility_excluded(self):
        assert is_excluded_sector("Utility") is True

    def test_none_excluded(self):
        assert is_excluded_sector(None) is True

    def test_empty_string_excluded(self):
        assert is_excluded_sector("") is True

    def test_information_technology_kept(self):
        assert is_excluded_sector("Information Technology") is False

    def test_health_care_kept(self):
        assert is_excluded_sector("Health Care") is False

    def test_energy_kept(self):
        assert is_excluded_sector("Energy") is False

    def test_real_estate_kept(self):
        assert is_excluded_sector("Real Estate") is False

    def test_communication_services_kept(self):
        assert is_excluded_sector("Communication Services") is False

    def test_consumer_discretionary_kept(self):
        assert is_excluded_sector("Consumer Discretionary") is False

    def test_industrials_kept(self):
        assert is_excluded_sector("Industrials") is False


# ---------------------------------------------------------------------------
# apply_sector_filter
# ---------------------------------------------------------------------------

class TestApplySectorFilter:
    def test_removes_financials_and_utilities(self):
        tickers = ["AAPL", "JPM", "NEE", "MSFT", "GS"]
        sectors = {
            "AAPL": "Information Technology",
            "JPM": "Financials",
            "NEE": "Utilities",
            "MSFT": "Information Technology",
            "GS": "Financial Services",
        }
        kept, kept_sec = apply_sector_filter(tickers, sectors)
        assert set(kept) == {"AAPL", "MSFT"}
        assert set(kept_sec.keys()) == {"AAPL", "MSFT"}

    def test_empty_input(self):
        kept, kept_sec = apply_sector_filter([], {})
        assert kept == []
        assert kept_sec == {}

    def test_all_kept_when_no_exclusion(self):
        tickers = ["AAPL", "MSFT", "AMZN"]
        sectors = {
            "AAPL": "Information Technology",
            "MSFT": "Information Technology",
            "AMZN": "Consumer Discretionary",
        }
        kept, _ = apply_sector_filter(tickers, sectors)
        assert set(kept) == set(tickers)


# ---------------------------------------------------------------------------
# _first_value
# ---------------------------------------------------------------------------

class TestFirstValue:
    def test_returns_first_non_nan(self):
        df = pd.DataFrame({"A": [np.nan], "B": [5.0], "C": [7.0]})
        assert _first_value(df, "A", "B", "C") == 5.0

    def test_returns_none_when_all_missing(self):
        df = pd.DataFrame({"A": [np.nan]})
        assert _first_value(df, "A") is None

    def test_returns_none_for_unknown_column(self):
        df = pd.DataFrame({"A": [1.0]})
        assert _first_value(df, "Z") is None

    def test_first_column_takes_priority(self):
        df = pd.DataFrame({"A": [1.0], "B": [2.0]})
        assert _first_value(df, "A", "B") == 1.0


# ---------------------------------------------------------------------------
# _year_slice
# ---------------------------------------------------------------------------

class TestYearSlice:
    def test_correct_year_returned(self):
        df = pd.DataFrame(
            {
                "asOfDate": pd.to_datetime(
                    ["2021-12-31", "2022-12-31", "2023-12-31"]
                ),
                "val": [10, 20, 30],
            }
        )
        sliced = _year_slice(df, 2022)
        assert len(sliced) == 1
        assert sliced["val"].iloc[0] == 20

    def test_empty_df_returns_empty(self):
        result = _year_slice(pd.DataFrame(), 2022)
        assert result.empty

    def test_missing_asofdate_returns_empty(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        result = _year_slice(df, 2022)
        assert result.empty

    def test_no_match_returns_empty(self):
        df = pd.DataFrame(
            {"asOfDate": pd.to_datetime(["2021-12-31"]), "val": [10]}
        )
        result = _year_slice(df, 2099)
        assert result.empty


# ---------------------------------------------------------------------------
# extract_ticker_year
# ---------------------------------------------------------------------------

class TestExtractTickerYear:
    def test_basic_record_structure(
        self, sample_income, sample_balance, sample_cashflow_div
    ):
        rec = extract_ticker_year(
            "TEST", 2022, "Information Technology",
            sample_income, sample_balance, sample_cashflow_div,
        )
        assert rec is not None
        expected_keys = {
            "Ticker", "Año", "Sector_GICS", "Precio_Ajustado_12M",
            "EPS_Diluido_Norm", "Ratio_Deuda_Activos", "Log_Activos_Totales",
            "Dummy_Dividendos", "_equity",
        }
        assert expected_keys.issubset(rec.keys())

    def test_ticker_and_year_set_correctly(
        self, sample_income, sample_balance, sample_cashflow_nodiv
    ):
        rec = extract_ticker_year(
            "AAPL", 2021, "Information Technology",
            sample_income, sample_balance, sample_cashflow_nodiv,
        )
        assert rec["Ticker"] == "AAPL"
        assert rec["Año"] == 2021
        assert rec["Sector_GICS"] == "Information Technology"

    def test_eps_extracted_correctly(
        self, sample_income, sample_balance, sample_cashflow_nodiv
    ):
        rec = extract_ticker_year(
            "TEST", 2022, "Health Care",
            sample_income, sample_balance, sample_cashflow_nodiv,
        )
        assert rec["EPS_Diluido_Norm"] == pytest.approx(3.5)

    def test_debt_ratio_calculated_correctly(
        self, sample_income, sample_balance, sample_cashflow_nodiv
    ):
        # 2022: TotalAssets=1_000_000, LongTermDebt=200_000, CurrentDebt=50_000
        # Ratio = 250_000 / 1_000_000 = 0.25
        rec = extract_ticker_year(
            "TEST", 2022, "Energy",
            sample_income, sample_balance, sample_cashflow_nodiv,
        )
        assert rec["Ratio_Deuda_Activos"] == pytest.approx(0.25)

    def test_log_assets_calculated_correctly(
        self, sample_income, sample_balance, sample_cashflow_nodiv
    ):
        rec = extract_ticker_year(
            "TEST", 2022, "Industrials",
            sample_income, sample_balance, sample_cashflow_nodiv,
        )
        assert rec["Log_Activos_Totales"] == pytest.approx(np.log(1_000_000))

    def test_dividend_dummy_one_when_dividends_paid(
        self, sample_income, sample_balance, sample_cashflow_div
    ):
        rec = extract_ticker_year(
            "TEST", 2022, "Materials",
            sample_income, sample_balance, sample_cashflow_div,
        )
        assert rec["Dummy_Dividendos"] == 1

    def test_dividend_dummy_zero_when_no_dividends(
        self, sample_income, sample_balance, sample_cashflow_nodiv
    ):
        rec = extract_ticker_year(
            "TEST", 2022, "Materials",
            sample_income, sample_balance, sample_cashflow_nodiv,
        )
        assert rec["Dummy_Dividendos"] == 0

    def test_negative_equity_detected(
        self, sample_income, sample_cashflow_nodiv
    ):
        balance_neg = pd.DataFrame(
            {
                "asOfDate": pd.to_datetime(["2022-12-31"]),
                "TotalAssets": [500_000.0],
                "LongTermDebt": [100_000.0],
                "CurrentDebt": [0.0],
                "StockholdersEquity": [-1_000.0],
            }
        )
        rec = extract_ticker_year(
            "TEST", 2022, "Consumer Staples",
            sample_income, balance_neg, sample_cashflow_nodiv,
        )
        assert rec is not None
        assert rec["_equity"] < 0

    def test_returns_none_for_missing_year(
        self, sample_income, sample_balance, sample_cashflow_nodiv
    ):
        rec = extract_ticker_year(
            "TEST", 2019, "Real Estate",
            sample_income, sample_balance, sample_cashflow_nodiv,
        )
        assert rec is None

    def test_returns_none_for_empty_dataframes(self):
        rec = extract_ticker_year(
            "TEST", 2022, "Industrials",
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        )
        assert rec is None
