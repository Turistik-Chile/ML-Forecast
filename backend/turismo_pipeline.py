# turismo_pipeline.py - VERSION CORREGIDA COMPLETA
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from data_pipeline import PreparedData, normalize_lag_list
from db_connection import get_connection
from exogena_loader import cargar_exogenas


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

SQL_TURISMO_TABLE = os.getenv("SQL_TURISMO_TABLE", "ia.ventas_turismo")
SQL_TURISMO_DATE_COLUMN = os.getenv("SQL_TURISMO_DATE_COLUMN", "servicedate")
SQL_TURISMO_CATEGORY_COLUMN = os.getenv("SQL_TURISMO_CATEGORY_COLUMN", "Category")
SQL_TURISMO_VALUE_COLUMN = os.getenv("SQL_TURISMO_VALUE_COLUMN", "n_pax")
TURISMO_MIN_TEST_DAYS = int(os.getenv("TURISMO_MIN_TEST_DAYS", "60"))
TURISMO_TEST_RATIO = float(os.getenv("TURISMO_TEST_RATIO", "0.2"))
TURISMO_COMPARISON_DAYS = int(os.getenv("TURISMO_COMPARISON_DAYS", "365"))
CATEGORY_ALL_TOKEN = "__all__"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.]+$")


# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------


def _sanitize_identifier(value: str) -> str:
    if not value or not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Identificador SQL no valido: {value!r}")
    return value


def _coerce_date(value: datetime | str | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _normalize_categories(categories: Sequence[str] | None) -> List[str]:
    if not categories:
        return []
    normalized = [
        category.strip()
        for category in categories
        if category.strip() and category.strip() != CATEGORY_ALL_TOKEN
    ]
    return list(dict.fromkeys(normalized))


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
    normalized = normalized.sort_index()
    if normalized.index.has_duplicates:
        normalized = normalized.groupby(normalized.index).sum()
    normalized["valor"] = (
        pd.to_numeric(normalized["valor"], errors="coerce").fillna(0.0).astype(float)
    )
    normalized.index.name = "fecha"
    return normalized


def _ensure_daily_range(
    df: pd.DataFrame, start_dt: datetime, end_dt: datetime
) -> pd.DataFrame:
    if df.empty or start_dt is None or end_dt is None:
        return df

    try:
        start_ts = pd.Timestamp(start_dt)
        end_ts = pd.Timestamp(end_dt)
    except (ValueError, OverflowError):
        return df

    full_index = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D")
    filled = df.reindex(full_index)
    filled.index.name = df.index.name or "fecha"
    filled["valor"] = filled["valor"].fillna(0.0)
    return _standardize_df(filled)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)

    day_of_year = idx.dayofyear.astype(int)
    days_in_year = np.where(idx.is_leap_year, 366, 365)

    day_of_month = idx.day.astype(int)
    days_in_month = idx.days_in_month.astype(int)

    iso = idx.isocalendar()
    week_of_year = iso.week.astype(int)
    iso_year = iso.year.astype(int)

    weeks_in_year_map = (
        pd.Series(week_of_year, index=idx)
        .groupby(iso_year)
        .transform("max")
        .astype(int)
        .values
    )

    month_of_year = idx.month.astype(int)
    months_in_year = 12

    out["sin_dayofyear"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    out["cos_dayofyear"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    out["sin_dayofmonth"] = np.sin(2 * np.pi * day_of_month / days_in_month)
    out["cos_dayofmonth"] = np.cos(2 * np.pi * day_of_month / days_in_month)

    out["sin_weekofyear"] = np.sin(2 * np.pi * week_of_year / weeks_in_year_map)
    out["cos_weekofyear"] = np.cos(2 * np.pi * week_of_year / weeks_in_year_map)

    out["sin_monthofyear"] = np.sin(2 * np.pi * month_of_year / months_in_year)
    out["cos_monthofyear"] = np.cos(2 * np.pi * month_of_year / months_in_year)

    return out


def _add_lag_features(
    df: pd.DataFrame, lags: Sequence[int], target_col: str = "valor"
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    return out


def _safe_date_range(start: datetime | None, end: datetime | None) -> pd.DatetimeIndex:
    if start is None or end is None:
        return pd.DatetimeIndex([])
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        return pd.DatetimeIndex([])
    return pd.date_range(start=start_ts, end=end_ts, freq="D")


def _load_exogenas_range(
    configuration_id: str | None, start: datetime, end: datetime
) -> pd.DataFrame:
    idx = _safe_date_range(start, end)
    if idx.empty:
        return pd.DataFrame(index=idx)
    config_str = str(configuration_id).strip() if configuration_id else ""
    if not config_str:
        return pd.DataFrame(index=idx)
    try:
        exog = cargar_exogenas(
            config_str, idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d")
        )
    except Exception as exc:
        logging.warning(
            "No se pudieron cargar exogenas para la configuracion %s: %s",
            config_str,
            exc,
        )
        return pd.DataFrame(index=idx)
    aligned = exog.reindex(idx).fillna(0)
    columns = list(aligned.columns)
    if columns:
        aligned.attrs["expected_exogenas"] = columns
    return aligned


def _merge_with_exogenas(base: pd.DataFrame, exogenas: pd.DataFrame) -> pd.DataFrame:
    result = base.copy()
    if exogenas.empty:
        return result
    joined = result.join(exogenas.reindex(result.index), how="left")
    exog_cols = list(exogenas.columns)
    if exog_cols:
        fill_map = {col: 0 for col in exog_cols}
        joined = joined.fillna(fill_map)
    return joined


def _safe_read_sql(query: str, conn, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pandas only supports SQLAlchemy connectable",
            category=UserWarning,
        )
        return pd.read_sql_query(query, conn, **kwargs)


def _get_end_date_limit() -> datetime:
    today = datetime.utcnow().date()
    limit = today - timedelta(days=TURISMO_COMPARISON_DAYS)
    return datetime.combine(limit, datetime.min.time())


def _sanitize_selection_dates(
    start_dt: datetime | None,
    end_dt: datetime | None,
    *,
    enforce_limit: bool = True,
    default_end: datetime | None = None,
) -> Tuple[datetime | None, datetime | None]:

    limit_dt = _get_end_date_limit()

    if enforce_limit:
        end_dt = end_dt or limit_dt
        if end_dt.date() > limit_dt.date():
            raise ValueError(
                f"La fecha final solo puede llegar hasta {limit_dt.date().isoformat()} "
                "para garantizar un ano completo de datos reales."
            )
    else:
        end_dt = end_dt or default_end

    if start_dt and end_dt and start_dt > end_dt:
        raise ValueError("La fecha inicial no puede ser posterior a la fecha final.")

    return start_dt, end_dt


# -------------------------------------------------------------------
# FETCH TURISMO RAW DATA
# -------------------------------------------------------------------


def fetch_turismo_series(
    *,
    categories: Sequence[str] | None,
    start_date: datetime | str | None,
    end_date: datetime | str | None,
    table_name: str,
    date_column: str,
    category_column: str,
    value_column: str,
) -> pd.DataFrame:

    table = _sanitize_identifier(table_name)
    date_col = _sanitize_identifier(date_column)
    cat_col = _sanitize_identifier(category_column)
    val_col = _sanitize_identifier(value_column)

    query = [
        f"SELECT {date_col} AS fecha, {cat_col} AS categoria, {val_col} AS valor",
        f"FROM {table}",
        "WHERE 1=1",
    ]
    params: List[Any] = []

    start_dt = _coerce_date(start_date)
    end_dt = _coerce_date(end_date)

    if start_dt:
        query.append(f"AND {date_col} >= ?")
        params.append(start_dt)
    if end_dt:
        query.append(f"AND {date_col} <= ?")
        params.append(end_dt)

    normalized_categories = _normalize_categories(categories)

    if normalized_categories:
        placeholders = ",".join("?" for _ in normalized_categories)
        query.append(f"AND {cat_col} IN ({placeholders})")
        params.extend(normalized_categories)

    query.append(f"ORDER BY {date_col}")
    sql = "\n".join(query)

    with get_connection() as conn:
        df = _safe_read_sql(sql, conn, params=params, parse_dates=["fecha"])

    if df.empty:
        raise ValueError("No se encontraron datos de turismo para los filtros dados.")

    grouped = (
        df.groupby("fecha", as_index=False)["valor"]
        .sum()
        .sort_values("fecha")
        .rename(columns={"valor": "valor"})
    )

    grouped["fecha"] = pd.to_datetime(grouped["fecha"])
    grouped = grouped.set_index("fecha")
    return _standardize_df(grouped)


# -------------------------------------------------------------------
# PIPELINE CONFIG
# -------------------------------------------------------------------


@dataclass
class TurismoPipelineConfig:
    categories: Sequence[str] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    lags: List[int] | None = None
    configuration_id: str | None = None
    forecast_only: bool = False
    forecast_end_date: datetime | None = None


# -------------------------------------------------------------------
# CORE PIPELINE
# -------------------------------------------------------------------


class TurismoPipeline:
    def __init__(self, config: TurismoPipelineConfig):
        self.config = config

    # ------------------------------------------------------------
    # PREPARE (corregido)
    # ------------------------------------------------------------
    def prepare(self) -> PreparedData:
        if self.config.forecast_only:
            return self._prepare_forecast_only()
        return self._prepare_training_with_comparison()

    # ------------------------------------------------------------
    # TRAINING + COMPARISON MODE (CORREGIDO)
    # ------------------------------------------------------------
    def _prepare_training_with_comparison(self) -> PreparedData:
        start_dt, cutoff_dt = _sanitize_selection_dates(
            self.config.start_date,
            self.config.end_date,
        )

        future_end = cutoff_dt + timedelta(days=TURISMO_COMPARISON_DAYS)

        df = fetch_turismo_series(
            categories=self.config.categories,
            start_date=start_dt,
            end_date=future_end,
            table_name=SQL_TURISMO_TABLE,
            date_column=SQL_TURISMO_DATE_COLUMN,
            category_column=SQL_TURISMO_CATEGORY_COLUMN,
            value_column=SQL_TURISMO_VALUE_COLUMN,
        )

        history = df.loc[df.index <= cutoff_dt].copy()
        if history.empty:
            raise ValueError("No hay datos de historia para las fechas solicitadas.")
        comparison_start = cutoff_dt + timedelta(days=1)
        comparison = df.loc[
            (df.index >= comparison_start) & (df.index <= future_end)
        ].copy()
        if comparison.empty:
            raise ValueError("No hay datos de comparacion para el periodo solicitado.")

        history = _ensure_daily_range(
            history, history.index.min(), history.index.max()
        )
        comparison = _ensure_daily_range(comparison, comparison_start, future_end)

        requested_lags = normalize_lag_list(self.config.lags)

        exog_start = history.index.min()
        exog_all = _load_exogenas_range(
            self.config.configuration_id,
            exog_start,
            future_end,
        )

        history_full = _merge_with_exogenas(
            _add_calendar_features(history), exog_all
        )
        comparison_full = _merge_with_exogenas(
            _add_calendar_features(comparison), exog_all
        )

        hist_lagged = _add_lag_features(history_full, requested_lags).dropna()
        if hist_lagged.empty:
            raise ValueError(
                "El historico no alcanza para generar las caracteristicas solicitadas."
            )

        X_train = hist_lagged.drop(columns=["valor"])
        y_train = hist_lagged["valor"]

        full_history = pd.concat([history_full, comparison_full]).sort_index()
        full_exogenas = exog_all.reindex(full_history.index).fillna(0)
        if full_exogenas.columns.tolist():
            full_exogenas.attrs["expected_exogenas"] = full_exogenas.columns.tolist()

        train_range = (
            history.index.min().strftime("%Y-%m-%d"),
            history.index.max().strftime("%Y-%m-%d"),
        )
        test_range = (
            comparison.index.min().strftime("%Y-%m-%d"),
            comparison.index.max().strftime("%Y-%m-%d"),
        )

        return PreparedData(
            X_train=X_train,
            y_train=y_train,
            feature_columns=X_train.columns.tolist(),
            history=history_full,
            df_real=comparison_full,
            train_range=train_range,
            test_range=test_range,
            lags=requested_lags,
            full_history=full_history,
            zero_padding_days=0,
            exogenas=full_exogenas,
            forecast_only=False,
            plot_comment="",
        )

    # ------------------------------------------------------------
    # FORECAST ONLY (CORREGIDO)
    # ------------------------------------------------------------
    def _prepare_forecast_only(self) -> PreparedData:
        forecast_end = self.config.forecast_end_date
        if forecast_end is None:
            raise ValueError("Debes indicar forecast_end_date.")

        start_dt = self.config.start_date
        history_end_dt = self.config.end_date

        df = fetch_turismo_series(
            categories=self.config.categories,
            start_date=start_dt,
            end_date=forecast_end,
            table_name=SQL_TURISMO_TABLE,
            date_column=SQL_TURISMO_DATE_COLUMN,
            category_column=SQL_TURISMO_CATEGORY_COLUMN,
            value_column=SQL_TURISMO_VALUE_COLUMN,
        )

        cutoff_ts = pd.Timestamp(history_end_dt) if history_end_dt else df.index.max()

        history = df.loc[df.index <= cutoff_ts].copy()
        if history.empty:
            raise ValueError("No hay datos de historia para el periodo solicitado.")
        history = _ensure_daily_range(
            history, history.index.min(), history.index.max()
        )

        forecast_start = history.index.max() + timedelta(days=1)
        forecast_index = pd.date_range(forecast_start, forecast_end, freq="D")
        if forecast_index.empty:
            raise ValueError("No hay dias disponibles para generar el forecast.")

        requested_lags = normalize_lag_list(self.config.lags)

        exog_start = history.index.min()
        exog_all = _load_exogenas_range(
            self.config.configuration_id,
            exog_start,
            forecast_end,
        )

        history_full = _merge_with_exogenas(
            _add_calendar_features(history), exog_all
        )

        forecast_base = pd.DataFrame(index=forecast_index, data={"valor": pd.NA})
        forecast_full = _merge_with_exogenas(
            _add_calendar_features(forecast_base), exog_all
        )

        hist_lagged = _add_lag_features(history_full, requested_lags).dropna()
        if hist_lagged.empty:
            raise ValueError(
                "No hay suficientes observaciones para los lags solicitados."
            )

        X_train = hist_lagged.drop(columns=["valor"])
        y_train = hist_lagged["valor"]

        full_history = pd.concat([history_full, forecast_full]).sort_index()
        full_exogenas = exog_all.reindex(full_history.index).fillna(0)
        if full_exogenas.columns.tolist():
            full_exogenas.attrs["expected_exogenas"] = full_exogenas.columns.tolist()

        return PreparedData(
            X_train=X_train,
            y_train=y_train,
            feature_columns=X_train.columns.tolist(),
            history=history_full,
            df_real=forecast_full,
            train_range=(
                history.index.min().strftime("%Y-%m-%d"),
                history.index.max().strftime("%Y-%m-%d"),
            ),
            test_range=(
                forecast_index.min().strftime("%Y-%m-%d"),
                forecast_index.max().strftime("%Y-%m-%d"),
            ),
            lags=requested_lags,
            plot_comment="",
            full_history=full_history,
            zero_padding_days=0,
            exogenas=full_exogenas,
            forecast_only=True,
        )

# -------------------------------------------------------------------
# PUBLIC BUILDER
# -------------------------------------------------------------------


def build_turismo_pipeline(
    *,
    categories: Sequence[str] | None = None,
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    forecast_end_date: datetime | str | None = None,
    forecast_only: bool = False,
    lags: List[int] | None = None,
    configuration_id: str | None = None,
    **_,
) -> TurismoPipeline:

    start_dt = _coerce_date(start_date)
    end_dt = _coerce_date(end_date)
    forecast_end_dt = _coerce_date(forecast_end_date)

    start_dt, end_dt = _sanitize_selection_dates(
        start_dt,
        end_dt,
        enforce_limit=not forecast_only,
        default_end=_get_end_date_limit() if not forecast_only else None,
    )

    config = TurismoPipelineConfig(
        categories=_normalize_categories(categories),
        start_date=start_dt,
        end_date=end_dt,
        lags=normalize_lag_list(lags),
        configuration_id=configuration_id,
        forecast_only=forecast_only,
        forecast_end_date=forecast_end_dt,
    )

    return TurismoPipeline(config)


def get_turismo_categories() -> List[str]:
    table = _sanitize_identifier(SQL_TURISMO_TABLE)
    category_col = _sanitize_identifier(SQL_TURISMO_CATEGORY_COLUMN)
    query = f"SELECT DISTINCT {category_col} AS categoria FROM {table} ORDER BY {category_col}"

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn)

    return sorted(
        [str(row["categoria"]) for _, row in df.iterrows() if row.get("categoria")]
    )


def get_turismo_date_bounds() -> Dict[str, str | None]:
    table = _sanitize_identifier(SQL_TURISMO_TABLE)
    date_col = _sanitize_identifier(SQL_TURISMO_DATE_COLUMN)
    query = f"SELECT MIN({date_col}) AS fecha_min, MAX({date_col}) AS fecha_max FROM {table}"

    with get_connection() as conn:
        result = _safe_read_sql(query, conn)

    if result.empty:
        return {"min": None, "max": None, "available_max": None}

    min_value = result.at[0, "fecha_min"]
    max_value = result.at[0, "fecha_max"]

    if pd.isna(min_value) or pd.isna(max_value):
        return {"min": None, "max": None, "available_max": None}

    min_date = pd.to_datetime(min_value).date()
    max_date = pd.to_datetime(max_value).date()
    allowed_max = _get_end_date_limit().date()

    bounded_max = min(max_date, allowed_max)

    return {
        "min": min_date.isoformat(),
        "max": bounded_max.isoformat(),
        "available_max": max_date.isoformat(),
    }
