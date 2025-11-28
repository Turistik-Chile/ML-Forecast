from __future__ import annotations

from typing import Tuple

import logging
import pandas as pd
from prophet import Prophet

from data_pipeline import align_exogenas_for_index


def _history_to_prophet_df(history: pd.DataFrame) -> pd.DataFrame:
    return history.rename(columns={"valor": "y"}).rename_axis("ds").reset_index()


def _log_dataframe_preview(name: str, df: pd.DataFrame | None) -> None:
    """Loggea una vista previa de un DataFrame."""
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


def entrenar_prophet_model(
    history: pd.DataFrame, exogenas_df: pd.DataFrame | None = None
) -> Prophet:
    _log_dataframe_preview("Prophet history input", history)
    train_df = _history_to_prophet_df(history)
    exog_cols = [col for col in history.columns if col != "valor"]
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    for col in exog_cols:
        model.add_regressor(col)
    model.fit(train_df)
    logging.info("Prophet exogenas detectadas: %s", exog_cols)
    return model


def forecast_prophet_model(
    model: Prophet,
    history: pd.DataFrame,
    test_range: Tuple[str, str],
    exogenas_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    start_ts, end_ts = map(pd.to_datetime, test_range)
    horizon = (end_ts - start_ts).days + 1
    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    _log_dataframe_preview("Prophet future raw", future)

    if "ds" not in future.columns:
        logging.error(
            "Prophet future está sin column 'ds'; columnas actuales=%s",
            future.columns.tolist(),
        )
        raise KeyError("ds")
    future = future.set_index("ds")

    exog_cols = [col for col in history.columns if col != "valor"]
    if exog_cols:
        exog_slice = align_exogenas_for_index(exogenas_df, future.index)
        if exog_slice is None:
            exog_slice = pd.DataFrame(
                0.0, index=future.index, columns=exog_cols
            )
        else:
            for col in exog_cols:
                if col not in exog_slice.columns:
                    exog_slice[col] = 0.0
        future = future.join(exog_slice)
    _log_dataframe_preview("Prophet future input", future)
    forecast = model.predict(future)
    preds = forecast.set_index("ds")["yhat"]
    preds = preds.loc[start_ts:end_ts]
    return preds.to_frame("prediccion")
