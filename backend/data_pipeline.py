from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_TRAIN_RANGE = ("1980-01-01", "2022-12-31")
DEFAULT_TEST_RANGE = ("2023-01-01", "2023-12-31")
DEFAULT_LAGS = [1]
MAX_ZERO_PADDING_DAYS = 30

OPEN_METEO_LOCATIONS = {
    "santiago": {
        "latitude": -33.4489,
        "longitude": -70.6693,
        "timezone": "America/Santiago",
    },
    "valparaiso": {
        "latitude": -33.0472,
        "longitude": -71.6127,
        "timezone": "America/Santiago",
    },
}


def normalize_lag_list(lags: Sequence[int] | None) -> List[int]:
    if not lags:
        return list(DEFAULT_LAGS)
    normalized: List[int] = []
    for lag in lags:
        if lag is None:
            continue
        try:
            lag_value = int(lag)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"El valor de lag '{lag}' no es un entero valido.") from exc
        if lag_value <= 0:
            continue
        if lag_value not in normalized:
            normalized.append(lag_value)
    normalized.sort()
    if not normalized:
        raise ValueError("Debes proporcionar al menos un lag entero positivo.")
    return normalized


def pad_history_for_lags(df: pd.DataFrame, lags: Sequence[int]) -> tuple[pd.DataFrame, int]:
    if df.empty:
        raise ValueError("No hay datos suficientes para generar el historico de entrenamiento.")
    max_lag = max(lags) if lags else 0
    required_length = max_lag + 1
    current_length = len(df)
    deficit = max(0, required_length - current_length)
    if deficit == 0:
        return df, 0
    if deficit > MAX_ZERO_PADDING_DAYS:
        raise ValueError(
            "El rango seleccionado no tiene suficientes dias previos para los lags solicitados. "
            f"Se necesitan {required_length} observaciones pero solo hay {current_length}. "
            f"No es posible rellenar mas de {MAX_ZERO_PADDING_DAYS} dias con valor cero."
        )
    start_date = df.index.min()
    pad_start = start_date - pd.Timedelta(days=deficit)
    pad_index = pd.date_range(start=pad_start, periods=deficit, freq="D")
    pad_df = pd.DataFrame({"valor": 0.0}, index=pad_index)
    padded = pd.concat([pad_df, df]).sort_index()
    return padded, deficit


@dataclass
class PipelineConfig:
    latitude: float
    longitude: float
    timezone: str
    train_range: Tuple[str, str]
    test_range: Tuple[str, str]
    lags: List[int]
    plot_comment: str | None = None


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    y_train: pd.Series
    feature_columns: List[str]
    history: pd.DataFrame
    df_real: pd.DataFrame | None
    train_range: Tuple[str, str]
    test_range: Tuple[str, str]
    lags: List[int]
    plot_comment: str | None = None
    full_history: pd.DataFrame | None = None
    zero_padding_days: int = 0
    exogenas: pd.DataFrame | None = None
    forecast_only: bool = False


class DataPipeline(Protocol):
    def prepare(self) -> PreparedData: ...


def obtener_json_openmeteo(
    latitude: float, longitude: float, timezone: str, start: str, end: str
) -> Dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max",
        "timezone": timezone,
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def preparar_dataframe(json_data: Dict) -> pd.DataFrame:
    daily = json_data.get("daily", {})
    df = pd.DataFrame(
        {"fecha": daily.get("time", []), "valor": daily.get("temperature_2m_max", [])}
    )

    if df.empty:
        raise ValueError("JSON sin datos.")

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha").sort_index()
    return df


def descargar_rango(
    latitude: float, longitude: float, timezone: str, start: str, end: str
) -> pd.DataFrame:
    return preparar_dataframe(
        obtener_json_openmeteo(latitude, longitude, timezone, start, end)
    )


def agregar_features_estaticas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index

    df["dia_semana"] = idx.weekday
    df["is_weekend"] = (df["dia_semana"] >= 5).astype(int)
    df["day"] = idx.day
    df["month"] = idx.month
    df["weekofyear"] = idx.isocalendar().week.astype(int)

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["sin_week"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["weekofyear"] / 52)

    df["sin_day"] = np.sin(2 * np.pi * df["day"] / 31)
    df["cos_day"] = np.cos(2 * np.pi * df["day"] / 31)

    return df


def agregar_lags(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["valor"].shift(lag)
    return df


def align_exogenas_for_index(
    exogenas_df: pd.DataFrame | None, index: pd.DatetimeIndex
) -> pd.DataFrame | None:
    if exogenas_df is None:
        return None
    aligned = exogenas_df.reindex(index)
    aligned.attrs.update(getattr(exogenas_df, "attrs", {}) or {})
    return aligned.fillna(0.0)


def log_exogena_coverage(
    exogenas_df: pd.DataFrame | None,
    history_df: pd.DataFrame | None,
    real_df: pd.DataFrame | None,
    context: str,
) -> List[str]:
    if exogenas_df is None:
        logging.info("%s: sin exogenas asociadas al pipeline.", context)
        return []

    exog_cols = list(exogenas_df.columns)
    missing_history = (
        [col for col in exog_cols if history_df is not None and col not in history_df.columns]
        if history_df is not None
        else exog_cols
    )
    missing_real = (
        [col for col in exog_cols if real_df is not None and col not in real_df.columns]
        if real_df is not None
        else exog_cols
    )

    logging.info(
        "%s: exogenas=%s | faltantes_history=%s | faltantes_real=%s",
        context,
        exog_cols,
        missing_history,
        missing_real,
    )
    if missing_history or missing_real:
        logging.warning(
            "%s: algun dataframe no contiene todas las exogenas esperadas.",
            context,
        )
    return exog_cols


def preparar_features(
    df: pd.DataFrame, lags: List[int], exogenas_df: pd.DataFrame | None = None
):
    df_feat = agregar_features_estaticas(df)
    if exogenas_df is not None:
        aligned_exogenas = align_exogenas_for_index(exogenas_df, df_feat.index)
        if aligned_exogenas is not None:
            df_feat = df_feat.join(aligned_exogenas)
    df_feat = agregar_lags(df_feat, lags)
    df_feat = df_feat.dropna()

    y = df_feat["valor"]
    X = df_feat.drop(columns=["valor"])
    return X, y, df_feat


def _build_prepared_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    lags: List[int],
    train_range: Tuple[str, str],
    test_range: Tuple[str, str],
    plot_comment: str | None = None,
    exogenas_df: pd.DataFrame | None = None,
) -> PreparedData:
    padded_train, zero_padding_days = pad_history_for_lags(df_train, lags)
    history_exogenas = align_exogenas_for_index(exogenas_df, padded_train.index)
    X_train, y_train, _ = preparar_features(padded_train, lags, history_exogenas)
    history = padded_train[["valor"]].copy()
    if history_exogenas is not None:
        history = history.join(history_exogenas)

    df_real = df_test[["valor"]].copy()
    test_exogenas = align_exogenas_for_index(exogenas_df, df_test.index)
    if test_exogenas is not None:
        df_real = df_real.join(test_exogenas)

    log_exogena_coverage(
        exogenas_df,
        history,
        df_real,
        context="prepare_open_meteo",
    )

    full_history = (
        pd.concat([history, df_real]).sort_index() if not history.empty else df_real
    )

    full_exogenas = (
        align_exogenas_for_index(exogenas_df, full_history.index)
        if exogenas_df is not None and not full_history.empty
        else None
    )

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        feature_columns=X_train.columns.tolist(),
        history=history,
        df_real=df_real,
        train_range=train_range,
        test_range=test_range,
        lags=lags,
        plot_comment=plot_comment,
        full_history=full_history if not full_history.empty else None,
        exogenas=full_exogenas,
        zero_padding_days=zero_padding_days,
    )


class OpenMeteoPipeline:
    def __init__(self, config: PipelineConfig, exogenas_df: pd.DataFrame | None = None):
        self.config = config
        self.exogenas_df = exogenas_df

    def prepare(self) -> PreparedData:
        train_start, train_end = self.config.train_range
        test_start, test_end = self.config.test_range

        df_train = descargar_rango(
            self.config.latitude,
            self.config.longitude,
            self.config.timezone,
            train_start,
            train_end,
        )
        df_test = descargar_rango(
            self.config.latitude,
            self.config.longitude,
            self.config.timezone,
            test_start,
            test_end,
        )
        return _build_prepared_data(
            df_train,
            df_test,
            self.config.lags,
            self.config.train_range,
            self.config.test_range,
            self.config.plot_comment,
            exogenas_df=self.exogenas_df,
        )


def obtener_datos_preparados(
    latitude: float,
    longitude: float,
    timezone: str,
    train_range: Tuple[str, str],
    test_range: Tuple[str, str],
    lags: List[int],
    plot_comment: str | None = None,
    exogenas_df: pd.DataFrame | None = None,
) -> PreparedData:
    df_train = descargar_rango(latitude, longitude, timezone, *train_range)
    df_test = descargar_rango(latitude, longitude, timezone, *test_range)
    return _build_prepared_data(
        df_train,
        df_test,
        lags,
        train_range,
        test_range,
        plot_comment,
        exogenas_df=exogenas_df,
    )


def build_open_meteo_pipeline(
    latitude: float,
    longitude: float,
    timezone: str = "America/Santiago",
    train_range: Tuple[str, str] | None = None,
    test_range: Tuple[str, str] | None = None,
    lags: List[int] | None = None,
    plot_comment: str | None = None,
    exogenas_df: pd.DataFrame | None = None,
) -> OpenMeteoPipeline:
    config = PipelineConfig(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        train_range=train_range or DEFAULT_TRAIN_RANGE,
        test_range=test_range or DEFAULT_TEST_RANGE,
        lags=normalize_lag_list(lags),
        plot_comment=plot_comment,
    )
    return OpenMeteoPipeline(config, exogenas_df=exogenas_df)


def get_open_meteo_location_names() -> List[str]:
    return sorted(OPEN_METEO_LOCATIONS.keys())


def _get_open_meteo_location(location: str) -> Dict[str, float | str]:
    try:
        return OPEN_METEO_LOCATIONS[location]
    except KeyError as exc:
        raise ValueError(f"Ubicacion '{location}' no esta soportada.") from exc


def build_open_meteo_pipeline_for_location(
    location: str,
    *,
    lags: List[int] | None = None,
    exogenas_df: pd.DataFrame | None = None,
) -> OpenMeteoPipeline:
    loc_cfg = _get_open_meteo_location(location)
    latitude = loc_cfg["latitude"]
    longitude = loc_cfg["longitude"]
    comment = (
        f"Zona: {location.title()} | Latitud: {latitude:.4f} | Longitud: {longitude:.4f}"
    )
    return build_open_meteo_pipeline(
        plot_comment=comment,
        lags=lags,
        **loc_cfg,
        exogenas_df=exogenas_df,
    )
