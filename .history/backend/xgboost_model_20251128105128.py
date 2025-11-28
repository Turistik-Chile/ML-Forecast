from __future__ import annotations

from typing import List

import logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def entrenar_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """
    Entrena un modelo XGBoost regresor sobre las features entregadas.

    IMPORTANTE:
    - Todas las columnas exógenas que quieras que el modelo use
      deben estar YA incluidas en X (por ejemplo, vía preparar_features).
    - X debe ser un DataFrame con nombres de columnas, para que
      el modelo guarde feature_names_in_ y podamos reutilizarlas en el forecast.
    """
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
    """
    Devuelve el valor de 'valor' para la fecha lag_day. Si no existe exactamente,
    usa el valor anterior o el siguiente más cercano. Si no hay información, retorna 0.0.
    """
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
    """
    Determina el conjunto de columnas exógenas a usar.

    - Incluye todas las columnas de history excepto 'valor'.
    - Agrega las columnas presentes en exogenas_df.
    """
    columns = [col for col in history.columns if col != "valor"]
    if exogenas_df is not None:
        for col in exogenas_df.columns:
            if col not in columns:
                columns.append(col)
    return columns


def _log_dataframe_preview(name: str, df: pd.DataFrame | None) -> None:
    """Loggea un preview de un DataFrame para depuración."""
    if df is None:
        logging.info("%s: no se proporcionó (None)", name)
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
    """
    Busca el valor de una columna exógena para 'date':
    1) Si existe en hist y no es NaN, se usa ese valor.
    2) Si existe en exogenas_df, se usa ese valor.
    3) En caso contrario, devuelve 0.0.
    """
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
    Construye el diccionario de features para una fecha dada:
    - Features de calendario
    - Lags de 'valor'
    - Columnas exógenas (exog_columns)

    Devuelve:
    - features: dict con TODAS las features
    - exog_values: dict con solo las columnas exógenas y sus valores
    """
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

    # Lags
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


def _resolve_effective_feature_cols(
    model: XGBRegressor, feature_cols: List[str]
) -> List[str]:
    """
    Determina el conjunto REAL de columnas a usar en el forecast.

    - Si el modelo tiene feature_names_in_ (entrenado con DataFrame), se usan esas.
    - Si no, se cae al parámetro feature_cols entregado por el pipeline.

    Esto asegura que el orden y el conjunto de columnas de predicción
    coincida con el entrenamiento, evitando que se pierdan exógenas
    que sí estaban en X_train.
    """
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        effective = list(model.feature_names_in_)
        logging.info(
            "XGBoost: usando feature_names_in_ del modelo como columnas efectivas: %s",
            effective,
        )
        return effective

    logging.warning(
        "XGBoost: el modelo no expone feature_names_in_; usando feature_cols del pipeline: %s",
        feature_cols,
    )
    return list(feature_cols)


def forecast_xgboost(
    model: XGBRegressor,
    feature_cols: List[str],
    history_df: pd.DataFrame,
    start: str,
    end: str,
    lags: List[int],
    exogenas_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Genera un forecast diario entre start y end (inclusive) usando un modelo XGBoost.

    - history_df: debe contener al menos la columna 'valor' y, idealmente,
      las exógenas alineadas al índice de fechas de entrenamiento.
    - exogenas_df: puede traer exógenas externas adicionales para el periodo de forecast.
    - feature_cols: columnas que el pipeline cree que se usaron para entrenar,
      pero se corrige internamente para calzar con model.feature_names_in_ si existe.

    Devuelve un DataFrame indexado por 'fecha' con columna 'prediccion'.
    """
    start_ts, end_ts = map(pd.to_datetime, (start, end))
    fechas = pd.date_range(start_ts, end_ts, freq="D")

    _log_dataframe_preview("XGBoost history input", history_df)
    _log_dataframe_preview("XGBoost exogenas input", exogenas_df)

    # Copiamos el histórico para ir extendiéndolo con las predicciones
    hist = history_df.copy()

    # Detectar columnas exógenas candidatas (las que no son 'valor')
    exog_columns = _get_exog_columns(hist, exogenas_df)

    # Columnas efectivas que el modelo realmente espera
    effective_feature_cols = _resolve_effective_feature_cols(model, feature_cols)

    # Avisar si hay exógenas que no forman parte de las features del modelo
    exog_not_in_model = [c for c in exog_columns if c not in effective_feature_cols]
    if exog_not_in_model:
        logging.warning(
            "XGBoost: columnas exógenas detectadas pero NO presentes en las features del modelo "
            "(no podrán afectar la predicción): %s",
            exog_not_in_model,
        )

    # Aseguramos que hist tenga todas las columnas exógenas (aunque sea rellenas con 0)
    for column in exog_columns:
        if column not in hist.columns:
            logging.info(
                "XGBoost: agregando columna exógena '%s' a history_df inicializada en 0.0",
                column,
            )
            hist[column] = 0.0

    logging.info(
        "XGBoost exógenas detectadas: %s | feature_cols pipeline: %s | feature_cols efectivas modelo: %s",
        exog_columns,
        feature_cols,
        effective_feature_cols,
    )

    preds = []

    for fecha in fechas:
        # Construimos todas las features (calendario + lags + exógenas)
        row, exog_values = _construir_fila(fecha, hist, lags, exogenas_df, exog_columns)

        # Armamos el DataFrame con el MISMO orden y columnas que el modelo espera
        try:
            X = pd.DataFrame([row])[effective_feature_cols]
        except KeyError as exc:
            missing = [c for c in effective_feature_cols if c not in row]
            logging.error(
                "XGBoost: error construyendo fila de features para %s. "
                "Faltan columnas: %s | Exception: %s",
                fecha,
                missing,
                exc,
            )
            # Rellenamos cualquier missing con 0.0 y volvemos a intentar
            for col in missing:
                row[col] = 0.0
            X = pd.DataFrame([row])[effective_feature_cols]

        # Predicción del modelo
        pred = float(model.predict(X)[0])

        # Extendemos el histórico con la predicción (para futuros lags)
        hist.loc[fecha, "valor"] = pred
        for column, value in exog_values.items():
            hist.at[fecha, column] = value

        preds.append((fecha, pred))

    return pd.DataFrame(preds, columns=["fecha", "prediccion"]).set_index("fecha")
