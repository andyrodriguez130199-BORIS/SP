#!/usr/bin/env python3
"""
Extracción de panel S&P 500 (2022-2024) con metodología econométrica.

- Fuente financiera principal: FinancialModelingPrep (API key compatible con FinanceToolkit)
- Fuente de precios: FMP historical-price-full (adjClose)
- Resultado: panel listo para regresión con variables definidas en metodología.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from financetoolkit import Toolkit
except ImportError:  # pragma: no cover
    Toolkit = None


FMP_BASE = "https://financialmodelingprep.com/api/v3"


@dataclass
class StudyConfig:
    input_file: str
    output_file: str
    start_year: int = 2022
    end_year: int = 2024
    batch_sleep_seconds: float = 0.2
    profile_cache_sleep_seconds: float = 0.05
    constituents_by_year_file: Optional[str] = None


def _http_get_json(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fmp_url(endpoint: str, api_key: str, **params) -> str:
    query = {"apikey": api_key, **params}
    return f"{FMP_BASE}/{endpoint}?{urllib.parse.urlencode(query)}"


def read_tickers_from_input(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe archivo de entrada: {path}")

    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    if df.empty:
        raise ValueError("El archivo de entrada no contiene filas.")

    col_candidates = ["Ticker", "Symbol", "Simbolo", "Empresa"]
    ticker_col = next((c for c in col_candidates if c in df.columns), df.columns[0])

    df = df.copy()
    df["Ticker"] = (
        df[ticker_col].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    df = df[df["Ticker"].ne("")]
    return df


def read_constituents_by_year(path: str) -> Optional[Dict[int, set]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe archivo de constituyentes: {path}")

    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    required = {"Year", "Ticker"}
    if not required.issubset(df.columns):
        raise ValueError("El archivo de constituyentes debe incluir columnas 'Year' y 'Ticker'.")

    out: Dict[int, set] = {}
    for _, row in df.iterrows():
        y = int(row["Year"])
        t = str(row["Ticker"]).strip().upper().replace(".", "-")
        out.setdefault(y, set()).add(t)
    return out


def try_init_financetoolkit(tickers: Iterable[str], api_key: str) -> Optional[Toolkit]:
    if Toolkit is None:
        return None
    try:
        return Toolkit(tickers=list(tickers), api_key=api_key)
    except Exception:
        return None


def parse_sic(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(str(value).strip().split(".")[0])
    except Exception:
        return None


def is_excluded_sic(sic: Optional[int]) -> bool:
    if sic is None:
        return False
    return (4900 <= sic <= 4999) or (6000 <= sic <= 6999)


def fetch_company_profile_sic(ticker: str, api_key: str) -> Optional[int]:
    # En FMP, algunos perfiles incluyen sicCode o sic.
    url = _fmp_url(f"profile/{ticker}", api_key)
    data = _http_get_json(url)
    if not data:
        return None
    item = data[0]
    return parse_sic(item.get("sicCode") or item.get("sic"))


def fetch_income_statements(ticker: str, api_key: str, limit: int = 120) -> List[dict]:
    url = _fmp_url(f"income-statement/{ticker}", api_key, period="annual", limit=limit)
    data = _http_get_json(url)
    return data if isinstance(data, list) else []


def fetch_balance_sheets(ticker: str, api_key: str, limit: int = 120) -> List[dict]:
    url = _fmp_url(f"balance-sheet-statement/{ticker}", api_key, period="annual", limit=limit)
    data = _http_get_json(url)
    return data if isinstance(data, list) else []


def fetch_cashflows(ticker: str, api_key: str, limit: int = 120) -> List[dict]:
    url = _fmp_url(f"cash-flow-statement/{ticker}", api_key, period="annual", limit=limit)
    data = _http_get_json(url)
    return data if isinstance(data, list) else []


def fetch_daily_adjusted_prices(ticker: str, api_key: str, date_from: str, date_to: str) -> pd.DataFrame:
    url = _fmp_url(f"historical-price-full/{ticker}", api_key, **{"from": date_from, "to": date_to})
    data = _http_get_json(url)
    historical = data.get("historical", []) if isinstance(data, dict) else []
    if not historical:
        return pd.DataFrame(columns=["date", "adjClose", "close"])
    df = pd.DataFrame(historical)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def pick_value(record: dict, *keys, default=np.nan):
    for k in keys:
        if k in record and pd.notna(record.get(k)):
            return record.get(k)
    return default


def merge_financials_by_year(income: List[dict], balance: List[dict], cash: List[dict]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}

    def _ingest(rows: List[dict]):
        for r in rows:
            raw_date = r.get("calendarYear") or r.get("date")
            if raw_date is None:
                continue
            year = int(str(raw_date)[:4])
            out.setdefault(year, {}).update(r)

    _ingest(income)
    _ingest(balance)
    _ingest(cash)
    return out


def compute_forward_mean_price(prices: pd.DataFrame, report_date: pd.Timestamp) -> float:
    end = report_date + timedelta(days=365)
    mask = (prices["date"] > report_date) & (prices["date"] <= end)
    window = prices.loc[mask]
    if window.empty:
        return np.nan
    col = "adjClose" if "adjClose" in window.columns else "close"
    return float(pd.to_numeric(window[col], errors="coerce").dropna().mean())


def safe_log(x) -> float:
    try:
        x = float(x)
        return math.log(x) if x > 0 else np.nan
    except Exception:
        return np.nan


def build_panel(config: StudyConfig, api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    input_df = read_tickers_from_input(config.input_file)
    constituents_by_year = read_constituents_by_year(config.constituents_by_year_file)

    tickers = sorted(set(input_df["Ticker"].tolist()))
    toolkit_init_ok = try_init_financetoolkit(tickers, api_key) is not None

    errors: List[str] = []
    exclusions = []
    rows = []

    # SIC desde input si existe, si no, intento por API.
    input_sic_map = {}
    for col in ["SIC", "sic", "Sic", "SICCode", "sicCode"]:
        if col in input_df.columns:
            input_sic_map = dict(zip(input_df["Ticker"], input_df[col]))
            break

    for i, ticker in enumerate(tickers, start=1):
        try:
            sic = parse_sic(input_sic_map.get(ticker)) if input_sic_map else None
            if sic is None:
                sic = fetch_company_profile_sic(ticker, api_key)
                time.sleep(config.profile_cache_sleep_seconds)

            if is_excluded_sic(sic):
                exclusions.append({"Ticker": ticker, "Motivo": f"SIC excluido: {sic}"})
                continue

            income = fetch_income_statements(ticker, api_key)
            balance = fetch_balance_sheets(ticker, api_key)
            cash = fetch_cashflows(ticker, api_key)
            if not income and not balance:
                exclusions.append({"Ticker": ticker, "Motivo": "Sin estados financieros"})
                continue

            all_years = merge_financials_by_year(income, balance, cash)
            if not all_years:
                exclusions.append({"Ticker": ticker, "Motivo": "Sin años financieros válidos"})
                continue

            min_date = f"{config.start_year - 1}-01-01"
            max_date = f"{config.end_year + 2}-12-31"
            prices = fetch_daily_adjusted_prices(ticker, api_key, min_date, max_date)
            if prices.empty:
                exclusions.append({"Ticker": ticker, "Motivo": "Sin precios históricos"})
                continue

            for year in range(config.start_year, config.end_year + 1):
                if constituents_by_year is not None and ticker not in constituents_by_year.get(year, set()):
                    continue

                record = all_years.get(year)
                if not record:
                    continue

                report_date_raw = pick_value(record, "filingDate", "fillingDate", "acceptedDate", "date", default=None)
                if report_date_raw is None:
                    errors.append(f"{ticker}-{year}: sin fecha de publicación")
                    continue

                report_date = pd.to_datetime(str(report_date_raw)[:10], errors="coerce")
                if pd.isna(report_date):
                    errors.append(f"{ticker}-{year}: fecha inválida {report_date_raw}")
                    continue

                price_forward = compute_forward_mean_price(prices, report_date)

                total_assets = pick_value(record, "totalAssets", "TotalAssets")
                total_debt = pick_value(record, "totalDebt", "TotalDebt", default=np.nan)
                if pd.isna(total_debt):
                    long_debt = pick_value(record, "longTermDebt", "LongTermDebt", default=np.nan)
                    short_debt = pick_value(record, "shortTermDebt", "ShortTermDebt", "currentDebt", default=np.nan)
                    if pd.notna(long_debt) or pd.notna(short_debt):
                        long_debt = pd.to_numeric(long_debt, errors="coerce")
                        short_debt = pd.to_numeric(short_debt, errors="coerce")
                        total_debt = float(np.nan_to_num(long_debt, nan=0.0)) + float(
                            np.nan_to_num(short_debt, nan=0.0)
                        )

                eps = pick_value(
                    record,
                    "epsDilutedExcludingExtraordinaryItems",
                    "epsDiluted",
                    "epsdiluted",
                    "dilutedEPS",
                    "EPSDiluted",
                )

                dps = pick_value(
                    record,
                    "dividendPerShare",
                    "DividendPerShare",
                    "commonStockDividendsPerShare",
                    default=np.nan,
                )
                if pd.isna(dps):
                    cash_div = pick_value(record, "dividendsPaid", "cashDividendsPaid", "CashDividendsPaid", default=np.nan)
                    shares = pick_value(record, "weightedAverageShsOutDil", "weightedAverageShsOut", "sharesOutstanding", default=np.nan)
                    if pd.notna(cash_div) and pd.notna(shares) and float(shares) > 0:
                        dps = abs(float(cash_div)) / float(shares)

                endeudamiento = np.nan
                if pd.notna(total_assets) and float(total_assets) > 0 and pd.notna(total_debt):
                    endeudamiento = float(total_debt) / float(total_assets)

                rows.append(
                    {
                        "Ticker": ticker,
                        "Año": year,
                        "Fecha_Reporte": report_date.date(),
                        "Precio_Promedio_Forward": price_forward,
                        "EPS": pd.to_numeric(eps, errors="coerce"),
                        "Deuda_Total": pd.to_numeric(total_debt, errors="coerce"),
                        "Activos": pd.to_numeric(total_assets, errors="coerce"),
                        "Endeudamiento_Ratio": pd.to_numeric(endeudamiento, errors="coerce"),
                        "Tamano_Ln": safe_log(total_assets),
                        "DPS": pd.to_numeric(dps, errors="coerce"),
                        "Dummy_Dividendos": 1 if pd.notna(dps) and float(dps) > 0 else 0,
                        "SIC": sic,
                    }
                )

            if i % 25 == 0:
                print(f"Procesados: {i}/{len(tickers)}")
            time.sleep(config.batch_sleep_seconds)

        except Exception as exc:  # pragma: no cover
            msg = f"{ticker}: {exc}"
            errors.append(msg)
            exclusions.append({"Ticker": ticker, "Motivo": f"Error: {exc}"})

    panel = pd.DataFrame(rows)
    exclusions_df = pd.DataFrame(exclusions)

    if panel.empty:
        return panel, exclusions_df, errors, {
            "survivorship_bias_note": "No se generaron observaciones.",
            "constituents_source": "N/A",
        }

    # Limpieza mínima para modelo
    panel = panel.dropna(subset=["Precio_Promedio_Forward", "Activos", "Endeudamiento_Ratio"])
    panel["DPS"] = panel["DPS"].fillna(0)
    panel["Dummy_Dividendos"] = (panel["DPS"] > 0).astype(int)

    # Winsorización de razón de endeudamiento para robustez
    if not panel["Endeudamiento_Ratio"].dropna().empty:
        # Percentiles 1%-99% para limitar valores extremos de apalancamiento que distorsionan la regresión.
        p1, p99 = panel["Endeudamiento_Ratio"].quantile([0.01, 0.99])
        panel["Endeudamiento_Ratio"] = np.clip(panel["Endeudamiento_Ratio"], p1, p99)

    # Continuidad temporal 2022-2024 para reducir sesgo por datos incompletos
    required_years = set(range(config.start_year, config.end_year + 1))
    ticker_year_coverage = panel.groupby("Ticker")["Año"].apply(lambda ys: required_years.issubset(set(ys)))
    valid_tickers = ticker_year_coverage[ticker_year_coverage].index
    panel = panel[panel["Ticker"].isin(valid_tickers)].copy()

    panel = panel.sort_values(["Ticker", "Año"]).reset_index(drop=True)

    meta = {
        "survivorship_bias_note": (
            "Se utilizó archivo histórico de constituyentes por año."
            if constituents_by_year is not None
            else "No se reconstruyó composición histórica anual del S&P 500; se reconoce potencial sesgo de supervivencia."
        ),
        "constituents_source": config.constituents_by_year_file or "Lista de entrada actual",
        "toolkit_init_ok": str(toolkit_init_ok),
    }
    return panel, exclusions_df, errors, meta


def export_results(panel: pd.DataFrame, exclusions: pd.DataFrame, errors: List[str], meta: Dict[str, str], output: str):
    meta_df = pd.DataFrame(
        [
            {
                "Campo": "Nota_Sesgo_Supervivencia",
                "Valor": meta.get("survivorship_bias_note", ""),
            },
            {
                "Campo": "Fuente_Constituyentes",
                "Valor": meta.get("constituents_source", ""),
            },
            {
                "Campo": "Fecha_Extraccion",
                "Valor": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            {
                "Campo": "N_Observaciones",
                "Valor": len(panel),
            },
            {
                "Campo": "N_Empresas",
                "Valor": panel["Ticker"].nunique() if not panel.empty else 0,
            },
            {
                "Campo": "Toolkit_Init_OK",
                "Valor": meta.get("toolkit_init_ok", ""),
            },
        ]
    )

    errors_df = pd.DataFrame({"Error": errors}) if errors else pd.DataFrame(columns=["Error"])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name="panel", index=False)
        exclusions.to_excel(writer, sheet_name="exclusiones", index=False)
        errors_df.to_excel(writer, sheet_name="errores", index=False)
        meta_df.to_excel(writer, sheet_name="metadatos", index=False)


def parse_args() -> StudyConfig:
    parser = argparse.ArgumentParser(description="Extracción panel S&P 500 con FMP/FinanceToolkit.")
    parser.add_argument("--input", required=True, help="Ruta del archivo de empresas (xlsx/csv).")
    parser.add_argument("--output", default="Panel_Datos_Tesis_2022_2024.xlsx", help="Archivo de salida xlsx.")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--constituents-by-year",
        default=None,
        help="Archivo opcional (xlsx/csv) con columnas Year,Ticker para corregir sesgo de supervivencia.",
    )
    args = parser.parse_args()

    return StudyConfig(
        input_file=args.input,
        output_file=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        constituents_by_year_file=args.constituents_by_year,
    )


def main() -> int:
    cfg = parse_args()
    api_key = os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_KEY")
    if not api_key:
        print("ERROR: Define FMP_API_KEY o FINANCIAL_MODELING_PREP_KEY en variables de entorno.")
        return 1

    print("Iniciando extracción con metodología de panel...")
    panel, exclusions, errors, meta = build_panel(cfg, api_key)
    export_results(panel, exclusions, errors, meta, cfg.output_file)

    print(f"Salida: {cfg.output_file}")
    print(f"Observaciones válidas: {len(panel)}")
    print(f"Empresas válidas: {panel['Ticker'].nunique() if not panel.empty else 0}")
    print(f"Exclusiones: {len(exclusions)}")
    print(f"Errores: {len(errors)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
