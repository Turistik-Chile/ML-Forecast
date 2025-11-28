from __future__ import annotations

from typing import Tuple

import logging
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

from data_pipeline import agregar_features_estaticas, align_exogenas_for_index


def _ensure_freq(series: pd.Series, default_freq: str = "D") -> pd.Series:
    """Ensure the time series carries a frequency for statsmodels."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("La serie de entrenamiento debe tener un indice datetime.")

    if series.index.freq is not None:
        return series

    freq = series.index.inferred_freq or default_freq

    if freq is None:
        raise ValueError("No se pudo inferir frecuencia de la serie historica.")

    return series.asfreq(freq)


def _build_exog_from_index(
    index: pd.DatetimeIndex, exogenas_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Generate deterministic seasonal features for SARIMAX exogenous inputs."""
    dummy_df = pd.DataFrame({"valor": 0.0}, index=index)
    features = agregar_features_estaticas(dummy_df)
    features = features.drop(columns=["valor"])
    if exogenas_df is not None:
        aligned = align_exogenas_for_index(exogenas_df, index)
        if aligned is not None:
            features = features.join(aligned)
    return features


def _log_dataframe_preview(name: str, df: pd.DataFrame | None) -> None:
    """Loggea una vista previa de un DataFrame."""
    if df is None or df.empty:
        logging.info("%s: no hay datos", name)
        return
    logging.info(
        "%s shape=%s columns=%s\n%s",
        name,
        df.shape,
        df.columns.tolist(),
        df.head(5).to_string(),
    )


def entrenar_sarimax_model(
    history: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
    exogenas_df: pd.DataFrame | None = None,
) -> SARIMAXResultsWrapper:
    _log_dataframe_preview("Sarimax history input", history)
    series = _ensure_freq(history["valor"])
    exog_source = history.drop(columns=["valor"], errors="ignore")
    _log_dataframe_preview("Sarimax exogenas source", exog_source)
    exog = _build_exog_from_index(series.index, exogenas_df=exog_source)
    _log_dataframe_preview("Sarimax exogenas features", exog)
    model = SARIMAX(
        series,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarimax_model(
    model: SARIMAXResultsWrapper,
    test_range: Tuple[str, str],
    exogenas_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    start_ts, end_ts = map(pd.to_datetime, test_range)
    steps = (end_ts - start_ts).days + 1
    index = pd.date_range(start_ts, end_ts, freq="D")
    exog = _build_exog_from_index(index, exogenas_df=exogenas_df)
    _log_dataframe_preview("Sarimax forecast exogenas", exog)
    forecast = model.get_forecast(steps=steps, exog=exog)
    preds = forecast.predicted_mean
    return pd.DataFrame({"prediccion": preds.values}, index=index)
