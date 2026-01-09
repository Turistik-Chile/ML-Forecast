# sarimax_model.py REFACTORIZADO - EXOGENAS REALES
import logging
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def entrenar_sarimax_model(
    history_full: pd.DataFrame, exogenas_df: pd.DataFrame | None = None
):
    y = history_full["valor"]
    exog = history_full.drop(columns=["valor"], errors="ignore")
    if exog.empty and exogenas_df is not None:
        exog = exogenas_df.reindex(history_full.index).fillna(0)

    model = SARIMAX(
        y,
        exog=exog if not exog.empty else None,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    cols = list(exog.columns) if not exog.empty else []
    logging.info("SARIMAX entrenado con exogenas: %s", cols)
    return model


def forecast_sarimax_model(
    model, test_range: tuple[str, str], exogenas_df: pd.DataFrame | None = None
):
    start, end = test_range
    idx = pd.date_range(start, end, freq="D")
    exog_columns = [col for col in model.data.exog_names] if hasattr(model.data, "exog_names") else []
    exog_future: pd.DataFrame | None = None
    if exog_columns:
        if exogenas_df is not None:
            exog_future = exogenas_df.reindex(idx).fillna(0)
        else:
            exog_future = pd.DataFrame(index=idx)
        missing = [col for col in exog_columns if col not in exog_future.columns]
        for column in missing:
            exog_future[column] = 0.0
        exog_future = exog_future[exog_columns]

    fc = (
        model.get_forecast(steps=len(idx), exog=exog_future)
        if exog_future is not None
        else model.get_forecast(steps=len(idx))
    )
    pred = fc.predicted_mean
    return pd.DataFrame({"prediccion": pred.values}, index=idx)
