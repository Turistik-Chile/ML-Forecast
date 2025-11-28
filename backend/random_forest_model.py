from __future__ import annotations

from typing import List

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def entrenar_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _resolve_lag_value(hist: pd.DataFrame, lag_day: pd.Timestamp) -> float:
    """Resuelve el valor de lag buscando en hist y usando vecino cercano si falta."""
    if lag_day in hist.index:
        return float(hist.at[lag_day, "valor"])

    index = hist.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        return 0.0

    # Importante: searchsorted asume índice ordenado ascendente.
    # Aquí hist ya se ordena en forecast_random_forest, pero lo dejamos defensivo.
    index = index.sort_values()

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
    """Lista columnas exógenas presentes en history y exogenas_df."""
    columns = [col for col in history.columns if col != "valor"]
    if exogenas_df is not None:
        for col in exogenas_df.columns:
            if col not in columns:
                columns.append(col)
    return columns


def _log_dataframe_preview(name: str, df: pd.DataFrame | None) -> None:
    """Imprime informacion breve de un DataFrame para debug."""
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
    """Busca exógena en hist para esa fecha, si no en exogenas_df, si no 0."""
    if column in hist.columns and date in hist.index:
        value = hist.at[date, column]
        if pd.notna(value):
            return float(value)
    if (
        exogenas_df is not None
        and date in exogenas_df.index
        and column in exogenas_df.columns
    ):
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
    """
    Construye features calendario (afinadas) + lags + exógenas para una fecha.
    """
    # Variables calendario "afinadas"
    dow = int(date.weekday())  # 0=Lunes ... 6=Domingo (estable)
    is_weekend = int(dow >= 5)
    day = int(date.day)
    month = int(date.month)

    # ISO week puede ser 1..52 o 53
    week = int(date.isocalendar().week)
    weeks_in_year = 53  # para que el seno/cos tenga soporte de semana 53

    # Días reales del mes para trig de día
    days_in_month = int(date.days_in_month)

    features = {
        "dia_semana": dow,
        "is_weekend": is_weekend,
        "day": day,
        "month": month,
        "weekofyear": week,
        # Estacionalidad trig refinada
        "sin_month": np.sin(2 * np.pi * month / 12.0),
        "cos_month": np.cos(2 * np.pi * month / 12.0),
        "sin_week": np.sin(2 * np.pi * week / float(weeks_in_year)),
        "cos_week": np.cos(2 * np.pi * week / float(weeks_in_year)),
        "sin_day": np.sin(2 * np.pi * day / float(days_in_month)),
        "cos_day": np.cos(2 * np.pi * day / float(days_in_month)),
    }

    # Lags autoregresivos
    for lag in lags:
        lag_day = date - pd.Timedelta(days=lag)
        features[f"lag_{lag}"] = _resolve_lag_value(hist, lag_day)

    # Exógenas
    exog_values: dict[str, float] = {}
    for column in exog_columns:
        value = _resolve_exog_value(date, hist, exogenas_df, column)
        features[column] = value
        exog_values[column] = value

    return features, exog_values


def forecast_random_forest(
    model: RandomForestRegressor,
    feature_cols: List[str],
    history_df: pd.DataFrame,
    start: str,
    end: str,
    lags: List[int],
    exogenas_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Forecast diario autoregresivo usando RandomForest.
    - Construye features por fecha
    - Predice
    - Inserta predicción en hist para próximos lags
    """
    start_ts, end_ts = map(pd.to_datetime, (start, end))
    fechas = pd.date_range(start_ts, end_ts, freq="D")

    _log_dataframe_preview("RandomForest history input", history_df)
    _log_dataframe_preview("RandomForest exogenas input", exogenas_df)

    hist = history_df.copy()
    if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index)
    hist.index = hist.index.tz_localize(None)
    hist = hist.sort_index()

    exog_columns = _get_exog_columns(hist, exogenas_df)
    # Asegurar que hist tenga las exógenas como columnas (aunque sea 0)
    for column in exog_columns:
        if column not in hist.columns:
            hist[column] = 0.0
    logging.info(
        "RandomForest exogenas detectadas: %s | feature_cols esperadas: %s",
        exog_columns,
        feature_cols,
    )

    preds = []

    for fecha in fechas:
        row, exog_values = _construir_fila(fecha, hist, lags, exogenas_df, exog_columns)

        # Robustez: si falta alguna feature por diferencias de pipeline,
        # la llenamos con 0 sin reventar.
        X = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0.0)

        pred = float(model.predict(X)[0])

        # Autoregresivo: guardar predicción como valor future
        hist.loc[fecha, "valor"] = pred
        for column, value in exog_values.items():
            hist.at[fecha, column] = value

        preds.append((fecha, pred))

    return pd.DataFrame(preds, columns=["fecha", "prediccion"]).set_index("fecha")
