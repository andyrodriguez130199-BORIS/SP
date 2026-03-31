# SP — S&P 500 Panel Data Extractor

Construye un panel de datos longitudinal (Long format) de activos del S&P 500
para los años fiscales **2020-2024**, listo para modelado econométrico con
efectos fijos por sector GICS.

## Columnas del panel de salida

| Columna | Descripción |
|---|---|
| `Ticker` | Símbolo bursátil |
| `Año` | Año fiscal (2020-2024) |
| `Sector_GICS` | Sector GICS (ej. "Information Technology") |
| `Precio_Ajustado_12M` | Media del precio ajustado en los 252 días hábiles desde el `asOfDate` del 10-K |
| `EPS_Diluido_Norm` | EPS Diluido (excl. partidas extraordinarias) |
| `Ratio_Deuda_Activos` | (Deuda CP + Deuda LP) / Activos Totales |
| `Log_Activos_Totales` | `log(Activos Totales)` |
| `Dummy_Dividendos` | 1 si pagó dividendos ese año, 0 si no |

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Opción A — Jupyter Notebook (recomendado)

Abre `sp500_panel.ipynb`. La primera celda instala las dependencias
directamente en el kernel activo con `%pip install -r requirements.txt`,
evitando el `ModuleNotFoundError`.

### Opción B — Script de línea de comandos

```bash
python sp500_extractor.py
```

El script exporta `sp500_panel_data.csv` en el directorio de trabajo.

## Filtros aplicados

- **Sector Financiero excluido** (SIC 6000-6999 → GICS "Financials")
- **Utilities excluido** (SIC 4900-4999 → GICS "Utilities")
- **Patrimonio negativo** → ticker descartado en cualquier año
- **Valores nulos** en variables requeridas → ticker descartado

## Tests

```bash
python -m pytest tests/
```
