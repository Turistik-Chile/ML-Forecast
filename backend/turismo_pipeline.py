from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import warnings

import pandas as pd

from data_pipeline import (
    PreparedData,
    align_exogenas_for_index,
    log_exogena_coverage,
    normalize_lag_list,
    pad_history_for_lags,
    preparar_features,
)
from db_connection import get_connection
from exogena_loader import cargar_exogenas

SQL_TURISMO_TABLE = os.getenv("SQL_TURISMO_TABLE", "ia.ventas_turismo")
SQL_TURISMO_DATE_COLUMN = os.getenv("SQL_TURISMO_DATE_COLUMN", "servicedate")
SQL_TURISMO_CATEGORY_COLUMN = os.getenv("SQL_TURISMO_CATEGORY_COLUMN", "Category")
SQL_TURISMO_VALUE_COLUMN = os.getenv("SQL_TURISMO_VALUE_COLUMN", "n_pax")
TURISMO_MIN_TEST_DAYS = int(os.getenv("TURISMO_MIN_TEST_DAYS", "60"))
TURISMO_TEST_RATIO = float(os.getenv("TURISMO_TEST_RATIO", "0.2"))
TURISMO_COMPARISON_DAYS = int(os.getenv("TURISMO_COMPARISON_DAYS", "365"))
CATEGORY_ALL_TOKEN = "__all__"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.]+$")


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

    min_supported = pd.Timestamp("1677-09-21")
    max_supported = pd.Timestamp("2262-04-11")
    if start_ts < min_supported or end_ts > max_supported:
        return df

    full_index = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D")
    filled = df.reindex(full_index)
    filled.index.name = df.index.name or "fecha"
    filled["valor"] = filled["valor"].fillna(0.0)
    return _standardize_df(filled)


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


@dataclass
class TurismoPipelineConfig:
    categories: Sequence[str] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    lags: List[int] | None = None
    table_name: str = SQL_TURISMO_TABLE
    date_column: str = SQL_TURISMO_DATE_COLUMN
    category_column: str = SQL_TURISMO_CATEGORY_COLUMN
    value_column: str = SQL_TURISMO_VALUE_COLUMN
    test_ratio: float = TURISMO_TEST_RATIO
    min_test_days: int = TURISMO_MIN_TEST_DAYS
    comparison_days: int = TURISMO_COMPARISON_DAYS
    configuration_id: str | None = None
    exogenas_df: pd.DataFrame | None = None
    forecast_only: bool = False
    forecast_end_date: datetime | None = None


class TurismoPipeline:
    def __init__(self, config: TurismoPipelineConfig):
        self.config = config

    def prepare(self) -> PreparedData:
        if self.config.forecast_only:
            return self._prepare_forecast_only()
        return self._prepare_with_comparison()

    def _maybe_load_exogenas(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        if not self.config.configuration_id:
            return self.config.exogenas_df
        if self.config.exogenas_df is not None:
            return self.config.exogenas_df
        try:
            return cargar_exogenas(
                self.config.configuration_id,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            logging.warning(
                "No se pudieron cargar las exogenas para %s: %s",
                self.config.configuration_id,
                exc,
            )
            return None

    def _prepare_with_comparison(self) -> PreparedData:
        start_dt, cutoff_dt = _sanitize_selection_dates(
            self.config.start_date,
            self.config.end_date,
        )
        comparison_days = self.config.comparison_days
        future_end = cutoff_dt + timedelta(days=comparison_days)

        df = fetch_turismo_series(
            categories=self.config.categories,
            start_date=start_dt,
            end_date=future_end,
            table_name=self.config.table_name,
            date_column=self.config.date_column,
            category_column=self.config.category_column,
            value_column=self.config.value_column,
        )
        if df.empty:
            raise ValueError("No se encontraron datos de turismo para los filtros dados.")
        df = _standardize_df(df)

        history = df.loc[df.index <= cutoff_dt].copy()
        comparison_start = cutoff_dt + timedelta(days=1)
        comparison = df.loc[(df.index >= comparison_start) & (df.index <= future_end)].copy()

        history = _standardize_df(history)
        comparison = _standardize_df(comparison)
        if not history.empty:
            history = _ensure_daily_range(
                history,
                history.index.min().to_pydatetime(),
                history.index.max().to_pydatetime(),
            )
        comparison = _ensure_daily_range(
            comparison,
            comparison_start,
            future_end,
        )

        if history.empty:
            raise ValueError(
                "El rango seleccionado no contiene datos historicos para entrenar el modelo."
            )
        expected_days = (future_end - comparison_start).days + 1
        if (
            comparison.empty
            or len(comparison) < expected_days
            or comparison.index.min() > comparison_start
            or comparison.index.max() < future_end
        ):
            raise ValueError(
                "No hay datos reales suficientes para realizar la comparativa del siguiente ano."
            )

        requested_lags = normalize_lag_list(self.config.lags)
        history = history.sort_index()
        padded_history, zero_padding_days = pad_history_for_lags(history, requested_lags)
        exogenas_df = self._maybe_load_exogenas(
            history.index.min().to_pydatetime(),
            future_end,
        )
        padded_history_exogenas = align_exogenas_for_index(exogenas_df, padded_history.index)
        X_train, y_train, _ = preparar_features(
            padded_history, requested_lags, padded_history_exogenas
        )

        train_range = (
            history.index.min().strftime("%Y-%m-%d"),
            history.index.max().strftime("%Y-%m-%d"),
        )
        test_range = (
            comparison.index.min().strftime("%Y-%m-%d"),
            comparison.index.max().strftime("%Y-%m-%d"),
        )

        plot_comment = build_plot_comment(
            categories=_normalize_categories(self.config.categories),
            train_range=train_range,
            test_range=test_range,
        )

        history_with_padding = padded_history[["valor"]].copy()
        if padded_history_exogenas is not None:
            history_with_padding = history_with_padding.join(padded_history_exogenas)

        comparison_exogenas = align_exogenas_for_index(exogenas_df, comparison.index)
        df_real_with_exog = comparison[["valor"]].copy()
        if comparison_exogenas is not None:
            df_real_with_exog = df_real_with_exog.join(comparison_exogenas)

        full_history = df[["valor"]].copy()
        full_history_exogenas = align_exogenas_for_index(exogenas_df, full_history.index)
        if full_history_exogenas is not None:
            full_history = full_history.join(full_history_exogenas)

        log_exogena_coverage(
            exogenas_df,
            history_with_padding,
            df_real_with_exog,
            context="prepare_turismo",
        )

        return PreparedData(
            X_train=X_train,
            y_train=y_train,
            feature_columns=X_train.columns.tolist(),
            history=history_with_padding,
            df_real=df_real_with_exog,
            train_range=train_range,
            test_range=test_range,
            lags=requested_lags,
            plot_comment=plot_comment,
            full_history=full_history if not full_history.empty else None,
            zero_padding_days=zero_padding_days,
            exogenas=full_history_exogenas,
            forecast_only=False,
        )

    def _prepare_forecast_only(self) -> PreparedData:
        forecast_end = self.config.forecast_end_date
        if forecast_end is None:
            raise ValueError("Debes indicar una fecha de limite para el forecast puro.")

        start_dt = self.config.start_date
        history_end_dt = self.config.end_date
        forecast_end_ts = pd.Timestamp(forecast_end)

        df = fetch_turismo_series(
            categories=self.config.categories,
            start_date=start_dt,
            end_date=forecast_end_ts,
            table_name=self.config.table_name,
            date_column=self.config.date_column,
            category_column=self.config.category_column,
            value_column=self.config.value_column,
        )
        if df.empty:
            raise ValueError("No se encontraron datos de turismo para los filtros dados.")
        df = _standardize_df(df)

        cutoff_ts = pd.Timestamp(history_end_dt) if history_end_dt else df.index.max()
        if cutoff_ts > df.index.max():
            cutoff_ts = df.index.max()

        history = df.loc[df.index <= cutoff_ts].copy()
        if history.empty:
            raise ValueError(
                "El rango seleccionado no contiene datos historicos para entrenar el modelo."
            )

        history = _ensure_daily_range(
            history,
            history.index.min().to_pydatetime(),
            history.index.max().to_pydatetime(),
        )

        forecast_start = history.index.max() + timedelta(days=1)
        if forecast_end_ts < forecast_start:
            raise ValueError("La fecha de limite debe ser posterior al ultimo dia con datos reales.")

        forecast_index = pd.date_range(forecast_start, forecast_end_ts, freq="D")
        requested_lags = normalize_lag_list(self.config.lags)
        history = history.sort_index()
        padded_history, zero_padding_days = pad_history_for_lags(history, requested_lags)

        exogenas_df = self._maybe_load_exogenas(
            history.index.min().to_pydatetime(),
            forecast_end_ts.to_pydatetime(),
        )
        padded_history_exogenas = align_exogenas_for_index(exogenas_df, padded_history.index)
        X_train, y_train, _ = preparar_features(
            padded_history, requested_lags, padded_history_exogenas
        )

        train_range = (
            history.index.min().strftime("%Y-%m-%d"),
            history.index.max().strftime("%Y-%m-%d"),
        )
        test_range = (
            forecast_index.min().strftime("%Y-%m-%d"),
            forecast_index.max().strftime("%Y-%m-%d"),
        )

        history_with_padding = padded_history[["valor"]].copy()
        if padded_history_exogenas is not None:
            history_with_padding = history_with_padding.join(padded_history_exogenas)

        forecast_real_placeholder = pd.DataFrame(index=forecast_index, data={"valor": pd.NA})
        forecast_exogenas = align_exogenas_for_index(exogenas_df, forecast_index)
        if forecast_exogenas is not None:
            forecast_real_placeholder = forecast_real_placeholder.join(forecast_exogenas)

        full_history = pd.concat([history_with_padding, forecast_real_placeholder]).sort_index()
        full_history_exogenas = align_exogenas_for_index(exogenas_df, full_history.index)

        log_exogena_coverage(
            exogenas_df, history_with_padding, forecast_real_placeholder, context="prepare_turismo_forecast_only"
        )

        plot_comment = build_forecast_only_comment(
            categories=_normalize_categories(self.config.categories),
            train_range=train_range,
            forecast_range=test_range,
        )

        return PreparedData(
            X_train=X_train,
            y_train=y_train,
            feature_columns=X_train.columns.tolist(),
            history=history_with_padding,
            df_real=forecast_real_placeholder,
            train_range=train_range,
            test_range=test_range,
            lags=requested_lags,
            plot_comment=plot_comment,
            full_history=full_history if not full_history.empty else None,
            zero_padding_days=zero_padding_days,
            exogenas=full_history_exogenas,
            forecast_only=True,
        )


def build_plot_comment(
    *, categories: Sequence[str], train_range: Tuple[str, str], test_range: Tuple[str, str]
) -> str:
    cat_label = ", ".join(categories) if categories else "Todas las categorias"
    return (
        f"Turismo | Categorias: {cat_label} | "
        f"Entrenamiento: {train_range[0]} - {train_range[1]} | "
        f"Test: {test_range[0]} - {test_range[1]}"
    )


def build_forecast_only_comment(
    *,
    categories: Sequence[str],
    train_range: Tuple[str, str],
    forecast_range: Tuple[str, str],
) -> str:
    cat_label = ", ".join(categories) if categories else "Todas las categorias"
    return (
        f"Turismo | Categorias: {cat_label} | "
        f"Entrenamiento: {train_range[0]} - {train_range[1]} | "
        f"Forecast: {forecast_range[0]} - {forecast_range[1]}"
    )


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


def build_turismo_pipeline(
    *,
    categories: Sequence[str] | None = None,
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    forecast_end_date: datetime | str | None = None,
    forecast_only: bool = False,
    lags: List[int] | None = None,
    test_ratio: float | None = None,
    min_test_days: int | None = None,
    configuration_id: str | None = None,
    **_: Any,
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

    comparison_days = TURISMO_COMPARISON_DAYS
    future_end = end_dt + timedelta(days=comparison_days) if end_dt else None
    exogenas_df = None
    if configuration_id:
        try:
            exog_start = start_dt
            exog_end = future_end if not forecast_only else forecast_end_dt
            if exog_start and exog_end:
                exogenas_df = cargar_exogenas(
                    configuration_id,
                    exog_start.strftime("%Y-%m-%d"),
                    exog_end.strftime("%Y-%m-%d"),
                )
        except Exception as exc:
            logging.warning(
                "No se pudieron cargar las exogenas para %s: %s",
                configuration_id,
                exc,
            )

    config = TurismoPipelineConfig(
        categories=_normalize_categories(categories),
        start_date=start_dt,
        end_date=end_dt,
        lags=normalize_lag_list(lags),
        test_ratio=test_ratio or TURISMO_TEST_RATIO,
        min_test_days=min_test_days or TURISMO_MIN_TEST_DAYS,
        comparison_days=TURISMO_COMPARISON_DAYS,
        configuration_id=configuration_id,
        exogenas_df=exogenas_df,
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

    categories = sorted(
        [
            str(row["categoria"])
            for _, row in df.iterrows()
            if row.get("categoria") not in (None, "")
        ]
    )
    return categories


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
    bounded_min = min_date
    bounded_max = min(max_date, allowed_max)

    if bounded_min > bounded_max:
        return {"min": None, "max": None, "available_max": None}

    return {
        "min": bounded_min.isoformat(),
        "max": bounded_max.isoformat(),
        "available_max": max_date.isoformat(),
    }


__all__ = [
    "TurismoPipeline",
    "TurismoPipelineConfig",
    "build_turismo_pipeline",
    "build_plot_comment",
    "build_forecast_only_comment",
    "fetch_turismo_series",
    "get_turismo_categories",
    "get_turismo_date_bounds",
    "CATEGORY_ALL_TOKEN",
    "TURISMO_COMPARISON_DAYS",
]
