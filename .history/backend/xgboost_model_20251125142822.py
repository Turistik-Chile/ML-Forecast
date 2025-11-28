from __future__ import annotations

from typing import List

import logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def entrenar_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
    )
    model.fit(X, y)
    return model


def _resolve_lag_value(hist: pd.DataFrame, lag_day: pd.Timestamp) -> float:
    if lag_day in hist.index:
        return float(hist.at[lag_day, "valor"])

    index = hist.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        return 0.0

    pos = index.searchsorted(lag_day)
    if pos > 0:
        prev_idx = index[pos - 1]
        return float(hist.at[prev_idx, "valor"])
    if pos < len(index):
        next_idx = index[pos]
        return float(hist.at[next_idx, "valor"])
    return float(hist.iloc[-1]["valor"])


def _get_exog_columns(
    history: pd.DataFrame, exogenas_df: pd.DataFrame | None = None
) -> List[str]:
    columns = [col for col in history.columns if col != "valor"]
    if exogenas_df is not None:
        for col in exogenas_df.columns:
            if col not in columns:
                columns.append(col)
    return columns


def _log_dataframe_preview(name: str, df: pd.DataFrame | None) -> None:
    """Loggea un preview de un DataFrame para depuracion."""
    if df is None:
        logging.info("%s: no se proporciono (None)", name)
        return
    logging.info(
        "%s shape=%s columns=%s\n%s",
        name,
        df.shape,
        df.columns.tolist(),
        df.head(5).to_string(),
    )


def _resolve_exog_value(
    date: pd.Timestamp,
    hist: pd.DataFrame,
    exogenas_df: pd.DataFrame | None,
    column: str,
) -> float:
    if column in hist.columns and date in hist.index:
        value = hist.at[date, column]
        if pd.notna(value):
            return float(value)
    if exogenas_df is not None and date in exogenas_df.index and column in exogenas_df.columns:
        value = exogenas_df.at[date, column]
        if pd.notna(value):
            return float(value)
    return 0.0


def _construir_fila(
    date: pd.Timestamp,
    hist: pd.DataFrame,
    lags: List[int],
    exogenas_df: pd.DataFrame | None,
    exog_columns: List[str],
) -> tuple[dict, dict]:
    week = date.isocalendar().week
    dow = date.weekday()
    day = date.day
    month = date.month

    features = {
        "dia_semana": dow,
        "is_weekend": int(dow >= 5),
        "day": day,
        "month": month,
        "weekofyear": int(week),
        "sin_month": np.sin(2 * np.pi * month / 12),
        "cos_month": np.cos(2 * np.pi * month / 12),
        "sin_week": np.sin(2 * np.pi * week / 52),
        "cos_week": np.cos(2 * np.pi * week / 52),
        "sin_day": np.sin(2 * np.pi * day / 31),
        "cos_day": np.cos(2 * np.pi * day / 31),
    }

    for lag in lags:
        lag_day = date - pd.Timedelta(days=lag)
        features[f"lag_{lag}"] = _resolve_lag_value(hist, lag_day)

    exog_values: dict[str, float] = {}
    for column in exog_columns:
        value = _resolve_exog_value(date, hist, exogenas_df, column)
        features[column] = value
        exog_values[column] = value

    return features, exog_values


def forecast_xgboost(
    model: XGBRegressor,
    feature_cols: List[str],
    history_df: pd.DataFrame,
    start: str,
    end: str,
    lags: List[int],
    exogenas_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    start_ts, end_ts = map(pd.to_datetime, (start, end))
    fechas = pd.date_range(start_ts, end_ts, freq="D")

    _log_dataframe_preview("XGBoost history input", history_df)
    _log_dataframe_preview("XGBoost exogenas input", exogenas_df)

    hist = history_df.copy()
    exog_columns = _get_exog_columns(hist, exogenas_df)
    for column in exog_columns:
        if column not in hist.columns:
            hist[column] = 0.0
    logging.info(
        "XGBoost exogenas detectadas: %s | feature_cols esperadas: %s",
        exog_columns,
        feature_cols,
    )
    preds = []

    for fecha in fechas:
        row, exog_values = _construir_fila(fecha, hist, lags, exogenas_df, exog_columns)
        X = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(X)[0])
        hist.loc[fecha, "valor"] = pred
        for column, value in exog_values.items():
            hist.at[fecha, column] = value
        preds.append((fecha, pred))

    return pd.DataFrame(preds, columns=["fecha", "prediccion"]).set_index("fecha")
