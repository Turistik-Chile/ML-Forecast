# xgboost_model.py - CORREGIDO
import logging
import pandas as pd
from xgboost import XGBRegressor


def entrenar_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    logging.info("XGB entrenado con columnas: %s", list(X_train.columns))
    return model


def _build_feature_frame(
    idx: pd.DatetimeIndex,
    feature_columns: list[str],
    exogenas_full: pd.DataFrame | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=idx)
    if exogenas_full is not None:
        frame = frame.join(exogenas_full.reindex(idx).fillna(0))
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[feature_columns]


def forecast_xgboost(
    model,
    feature_columns,
    history_full,
    start,
    end,
    lags,
    exogenas_full,
):
    idx = pd.date_range(start, end, freq="D")
    Xf = _build_feature_frame(idx, feature_columns or [], exogenas_full)
    preds = model.predict(Xf)
    return pd.DataFrame({"prediccion": preds}, index=idx)
