# random_forest_model.py REFACTORIZADO
import logging

import numpy as np
import pandas as pd
from data_pipeline import PreparedData
from sklearn.ensemble import RandomForestRegressor


def entrenar_random_forest(data: PreparedData):
    """Entrena el RF con log(1+y)."""
    X = data.X_train.astype(float)
    y_raw = data.y_train.astype(float)
    y_log = np.log1p(y_raw)

    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y_log)

    logging.info(
        "RF entrenado (filas=%d, columnas=%d, lags=%s)",
        len(X),
        len(X.columns),
        data.lags,
    )
    return model


def forecast_random_forest(model, data: PreparedData):
    """Pronostica iterativamente utilizando history + exogenas/lags."""
    future_df = data.df_real
    if future_df is None or future_df.empty:
        raise ValueError("No hay datos de pronóstico para generar predicciones.")

    history_vals = data.history["valor"].astype(float).copy()
    feature_columns = data.feature_columns or []
    lag_columns = [f"lag_{lag}" for lag in data.lags]

    preds = []
    for current_date in future_df.index:
        row = future_df.loc[current_date].copy()
        for lag, lag_col in zip(data.lags, lag_columns):
            lag_date = current_date - pd.Timedelta(days=lag)
            row[lag_col] = history_vals.get(lag_date, 0.0)
        row = row.reindex(feature_columns).astype(float).fillna(0.0)
        X_row = row.to_frame().T
        pred_log = model.predict(X_row)[0]
        pred_real = max(np.expm1(pred_log), 0.0)
        preds.append(pred_real)
        history_vals.loc[current_date] = pred_real

    return pd.DataFrame({"prediccion": preds}, index=future_df.index)
