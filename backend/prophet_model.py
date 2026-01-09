# prophet_model.py (REFACTORIZADO - AHORA USA EXOGENAS CORRECTAMENTE)
import logging
import pandas as pd
from prophet import Prophet


def _history_to_prophet_df(df):
    out = df.reset_index().rename(columns={"fecha": "ds", "valor": "y"})
    return out


def entrenar_prophet_model(
    history: pd.DataFrame, exogenas_df: pd.DataFrame | None = None
) -> Prophet:
    df = _history_to_prophet_df(history)
    exog_cols = [c for c in history.columns if c != "valor"]

    if exogenas_df is not None and not exog_cols:
        exog_cols = list(exogenas_df.columns)
    if exogenas_df is not None:
        aligned_exog = exogenas_df.reindex(history.index).fillna(0)
        for col in aligned_exog.columns:
            df[col] = aligned_exog[col]

    model = Prophet(
        daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
    )

    for col in exog_cols:
        model.add_regressor(col)

    model.fit(df)

    logging.info("Prophet entrenado con exogenas: %s", exog_cols)
    return model


def forecast_prophet_model(
    model,
    history_full,
    test_range: tuple[str, str],
    exogenas_df: pd.DataFrame | None = None,
):
    start, end = test_range
    idx = pd.date_range(start, end, freq="D")
    future = pd.DataFrame({"ds": idx})

    exog_cols = [c for c in history_full.columns if c != "valor"]
    exog_future = (
        exogenas_df.reindex(idx).fillna(0) if exogenas_df is not None else pd.DataFrame(index=idx)
    )

    for col in exog_cols:
        if col in exog_future.columns:
            future[col] = exog_future[col].values
        else:
            future[col] = 0.0

    forecast = model.predict(future)
    preds = forecast.set_index("ds")["yhat"]

    return preds.to_frame("prediccion")
