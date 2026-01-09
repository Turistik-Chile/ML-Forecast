from __future__ import annotations

import argparse
import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import (
    PreparedData,
    build_open_meteo_pipeline_for_location,
    get_open_meteo_location_names,
    log_exogena_coverage,
)
from random_forest_model import entrenar_random_forest, forecast_random_forest
from prophet_model import entrenar_prophet_model, forecast_prophet_model
from sarimax_model import entrenar_sarimax_model, forecast_sarimax_model
from xgboost_model import entrenar_xgboost, forecast_xgboost
from turismo_pipeline import (
    TURISMO_COMPARISON_DAYS,
    build_turismo_pipeline,
    get_turismo_categories,
    get_turismo_date_bounds,
)
from exogena_loader import cargar_exogenas
from configuracion_endpoints import (
    router as configuracion_router,
    listar_configuraciones_db,
)
from dataset_persistence import persist_dataset_to_db
from exogena_endpoints import router as exogena_router


# -----------------------------
# CONFIGURACION
# -----------------------------
LOG_DIR = Path("logs")
FLOW_LOG_DIR = LOG_DIR / "flow_reports"
LOCATION_CHOICES = get_open_meteo_location_names()
DEFAULT_BATCH_LOCATIONS = ("santiago", "valparaiso")
FRONTEND_ALLOWED_ORIGINS = ["http://localhost:5173"]


def cargar_exogenas_para_rango(
    configuracion_id: str | None, inicio: str, fin: str
) -> pd.DataFrame | None:
    if not configuracion_id:
        return None
    try:
        return cargar_exogenas(configuracion_id, inicio, fin)
    except Exception as exc:  # pragma: no cover - siempre compensamos la carga
        logging.warning(
            "No se pudo cargar las exogenas para %s (%s - %s): %s",
            configuracion_id,
            inicio,
            fin,
            exc,
        )
        return None


def build_open_meteo_pipeline_entry(
    *,
    location: str | None = None,
    lags: List[int] | None = None,
    configuration_id: str | None = None,
    **_: Any,
):
    if not location:
        raise ValueError("El pipeline 'open_meteo' requiere especificar una ubicacion.")
    pipeline = build_open_meteo_pipeline_for_location(location, lags=lags)
    if configuration_id:
        start = pipeline.config.train_range[0]
        end = pipeline.config.test_range[1]
        pipeline.exogenas_df = cargar_exogenas_para_rango(configuration_id, start, end)
    return pipeline


PIPELINE_REGISTRY: Dict[str, Callable[..., Any]] = {
    "open_meteo": build_open_meteo_pipeline_entry,
    "turismo": build_turismo_pipeline,
}

ModelTrainFn = Callable[[PreparedData], Any]
ModelForecastFn = Callable[[Any, PreparedData], Any]


class ModelHandlers(TypedDict):
    train: ModelTrainFn
    forecast: ModelForecastFn


MODEL_REGISTRY: Dict[str, ModelHandlers] = {
    "random_forest": {
        "train": lambda data: entrenar_random_forest(data),
        "forecast": lambda model, data: forecast_random_forest(model, data),
    },
    "prophet": {
        "train": lambda data: entrenar_prophet_model(data.history, data.exogenas),
        "forecast": lambda model, data: forecast_prophet_model(
            model,
            data.history,
            data.test_range,
            data.exogenas,
        ),
    },
    "sarimax": {
        "train": lambda data: entrenar_sarimax_model(
            data.history, exogenas_df=data.exogenas
        ),
        "forecast": lambda model, data: forecast_sarimax_model(
            model,
            data.test_range,
            exogenas_df=data.exogenas,
        ),
    },
    "xgboost": {
        "train": lambda data: entrenar_xgboost(data.X_train, data.y_train),
        "forecast": lambda model, data: forecast_xgboost(
            model,
            data.feature_columns,
            data.history,
            data.test_range[0],
            data.test_range[1],
            data.lags,
            data.exogenas,
        ),
    },
}


class ForecastRequest(BaseModel):
    pipeline: str = Field(
        default="open_meteo", description="Pipeline de ingesta de datos a utilizar."
    )
    location: str | None = Field(
        default=None,
        description="Ubicacion soportada por el pipeline seleccionado (solo si aplica).",
    )
    model: str = Field(
        default="random_forest", description="Modelo registrado para el pronostico."
    )
    start_date: datetime | None = Field(
        default=None, description="Fecha inicial (inclusive) para filtrar los datos."
    )
    end_date: datetime | None = Field(
        default=None, description="Fecha final (inclusive) para filtrar los datos."
    )
    categories: List[str] | None = Field(
        default=None,
        description="Listado de categorias para el pipeline de turismo. Lista vacia o None incluye todas.",
    )
    lags: List[int] | None = Field(
        default=None,
        description="Conjunto de lags (en dias) utilizados para construir las caracteristicas.",
    )
    configuration_id: str | None = Field(
        default=None,
        description="Configuracion con exogenas a incorporar en los modelos.",
    )
    forecast_only: bool = Field(
        default=False,
        description="Si es True, usa todo el historico disponible y solo genera prediccion hasta la fecha limite.",
    )
    forecast_end_date: datetime | None = Field(
        default=None,
        description="Fecha final para el forecast puro (obligatoria si forecast_only es True).",
    )

    def build_pipeline_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.location:
            kwargs["location"] = self.location
        if self.start_date:
            kwargs["start_date"] = self.start_date
        if self.end_date:
            kwargs["end_date"] = self.end_date
        if self.categories is not None:
            kwargs["categories"] = self.categories
        normalized_lags = self.normalized_lags()
        if normalized_lags is not None:
            kwargs["lags"] = normalized_lags
        if self.configuration_id:
            kwargs["configuration_id"] = self.configuration_id
        if self.forecast_only:
            kwargs["forecast_only"] = True
        if self.forecast_end_date:
            kwargs["forecast_end_date"] = self.forecast_end_date
        return kwargs

    def normalized_lags(self) -> List[int] | None:
        if not self.lags:
            return None
        normalized = sorted(
            {lag for lag in self.lags if isinstance(lag, int) and lag > 0}
        )
        if not normalized:
            raise ValueError("Los lags deben ser enteros positivos.")
        return normalized


class ForecastPoint(BaseModel):
    fecha: datetime
    valor_real: float | None = None
    prediccion: float
    error: float | None = None
    abs_error: float | None = None


class ForecastResponse(BaseModel):
    pipeline: str
    location: str
    model: str
    lags: List[int]
    zero_padding_days: int
    mae: float | None = None
    rmse: float | None = None
    forecast_only: bool = False
    timestamp: str
    plot_path: str
    plot_image: str | None = None
    dataset_table: str | None = None
    forecast: List[ForecastPoint]


def resolve_location_label(
    pipeline_name: str, pipeline_kwargs: Dict[str, Any] | None
) -> str:
    kwargs = pipeline_kwargs or {}
    if pipeline_name == "turismo":
        categories = kwargs.get("categories") or []
        categories = [cat for cat in categories if cat]
        if not categories:
            return "Todas las categorias"
        return ", ".join(categories)
    return kwargs.get("location") or "-"


# -----------------------------
# EVALUACION
# -----------------------------
def comparar(pred_df, real_df):
    df = real_df.rename(columns={"valor": "valor_real"})
    merged = pred_df.join(df, how="inner")
    mae = mean_absolute_error(merged["valor_real"], merged["prediccion"])
    rmse = np.sqrt(mean_squared_error(merged["valor_real"], merged["prediccion"]))
    merged["error"] = merged["prediccion"] - merged["valor_real"]
    merged["abs_error"] = merged["error"].abs()
    return merged, mae, rmse


# -----------------------------
# LOG + PLOTS
# -----------------------------
def _format_lag_label(lags):
    return ", ".join(str(lag) for lag in lags)


def _normalize_to_dataframe(
    series_or_df: pd.Series | pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Return a DataFrame view for either a Series or DataFrame (None preserved)."""
    if series_or_df is None:
        return None
    if isinstance(series_or_df, pd.Series):
        df = series_or_df.to_frame()
        df.columns = [col or "valor" for col in df.columns]
        return df
    return series_or_df


def _render_dataframe_section(name: str, df: pd.DataFrame | None) -> str:
    """Build an HTML section describing a DataFrame."""
    if df is None:
        return f"<section><h2>{name}</h2><p>None</p></section>"
    try:
        html_table = df.head(200).to_html(index=True, justify="left")
    except Exception as exc:  # pragma: no cover - defensivo ante tablas inusuales
        html_table = f"<p>No se pudo renderizar la tabla: {exc}</p>"
    return f"<section><h2>{name} (shape={df.shape})</h2>" f"{html_table}</section>"


def _dump_dataset_csv(writer, label: str, df: pd.DataFrame | None) -> None:
    writer.write(f"# {label}")
    if df is None or df.empty:
        writer.write(" None\n\n")
        return
    writer.write(f" shape={df.shape}\n")
    df.to_csv(writer)
    writer.write("\n")


def _get_dump_datasets(data: PreparedData) -> list[tuple[str, pd.DataFrame | None]]:
    return [
        ("full_history", _normalize_to_dataframe(data.full_history)),
        ("history", _normalize_to_dataframe(data.history)),
        ("real_values", _normalize_to_dataframe(data.df_real)),
        ("X_train", _normalize_to_dataframe(data.X_train)),
        ("y_train", _normalize_to_dataframe(data.y_train)),
        ("exogenas", _normalize_to_dataframe(data.exogenas)),
    ]


def _dump_prepared_dataframes_csv(
    model_name: str, datasets: list[tuple[str, pd.DataFrame | None]], timestamp: str
) -> None:
    dump_dir = LOG_DIR / "dataframes"
    dump_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dump_dir / f"{model_name}_{timestamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        for label, df in datasets:
            _dump_dataset_csv(csvfile, label, df)
    logging.info("Se guardaron las versiones CSV para %s en %s", model_name, csv_path)


def _dump_prepared_dataframes(model_name: str, data: PreparedData) -> None:
    """Persiste las tablas clave del Pipeline en un archivo HTML para revisión."""
    dump_dir = LOG_DIR / "dataframes"
    dump_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    datasets = _get_dump_datasets(data)
    sections = []
    for label, df in datasets:
        sections.append(_render_dataframe_section(label, df))

    content = (
        "<html><head><meta charset='utf-8'><title>Pipeline DataFrames</title></head>"
        "<body>"
        f"<h1>{model_name} - {timestamp}</h1>" + "".join(sections) + "</body></html>"
    )

    path = dump_dir / f"{model_name}_{timestamp}.html"
    path.write_text(content, encoding="utf-8")
    logging.info("Se guardaron tablas de entrenamiento para %s en %s", model_name, path)
    _dump_prepared_dataframes_csv(model_name, datasets, timestamp)


def _df_stats(df: pd.DataFrame | None) -> str:
    if df is None:
        return "None"
    shape = f"shape={df.shape}"
    if df.empty:
        return f"empty ({shape})"
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
        return f"{shape} | range={df.index.min().date()} -> {df.index.max().date()}"
    return shape


def _build_flow_report(
    pipeline_name: str,
    model_name: str,
    data: PreparedData,
    exog_columns: list[str],
    expected_from_loader: list[str],
) -> tuple[str, list[str]]:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    exog_reference = expected_from_loader or exog_columns
    history_cols = [
        c
        for c in (data.history.columns.tolist() if data.history is not None else [])
        if c != "valor"
    ]
    real_cols = (
        [
            c
            for c in (data.df_real.columns.tolist() if data.df_real is not None else [])
            if c != "valor"
        ]
        if data.df_real is not None
        else []
    )
    feature_cols = list(data.feature_columns or [])

    exog_index = data.exogenas.index if data.exogenas is not None else None
    history_index = data.history.index if data.history is not None else None
    real_index = data.df_real.index if data.df_real is not None else None

    def _missing_coverage(target_index: pd.Index | None) -> tuple[int, str] | None:
        if exog_index is None or target_index is None:
            return None
        target_dt = pd.DatetimeIndex(target_index)
        exog_dt = pd.DatetimeIndex(exog_index)
        missing = target_dt.difference(exog_dt)
        if missing.empty:
            return None
        return len(missing), f"{missing.min().date()} -> {missing.max().date()}"

    missing_hist_cov = _missing_coverage(history_index)
    missing_real_cov = _missing_coverage(real_index)

    missing_history = [c for c in exog_reference if c not in history_cols]
    missing_features = [c for c in exog_reference if c not in feature_cols]
    missing_real = (
        [c for c in exog_reference if c not in real_cols] if exog_reference else []
    )

    warnings: list[str] = []
    if not exog_reference:
        warnings.append(
            "Sin exogenas detectadas/esperadas; los modelos correran sin variables externas."
        )
    if missing_history:
        warnings.append(f"Exogenas ausentes en history: {missing_history}")
    if missing_features:
        warnings.append(
            f"Exogenas no incluidas en feature_columns (no impactaran el modelo): {missing_features}"
        )
    if missing_real and data.df_real is not None:
        warnings.append(f"Exogenas ausentes en df_real/comparacion: {missing_real}")
    if missing_hist_cov:
        warnings.append(
            f"Cobertura incompleta de exogenas en history: faltan {missing_hist_cov[0]} dias "
            f"({missing_hist_cov[1]}), se rellenan con 0."
        )
    if missing_real_cov:
        warnings.append(
            f"Cobertura incompleta de exogenas en df_real/forecast: faltan {missing_real_cov[0]} dias "
            f"({missing_real_cov[1]}), se rellenan con 0."
        )

    status = "OK" if not warnings else "WARN"
    lines = [
        f"# Flow diagnostics :: {pipeline_name} / {model_name}",
        f"- timestamp: {timestamp}",
        f"- status: {status}",
        f"- lags: {data.lags} | zero_padding_days: {data.zero_padding_days} | forecast_only: {data.forecast_only}",
        "- datasets:",
        f"  * history: {_df_stats(data.history)}",
        f"  * df_real: {_df_stats(data.df_real)}",
        f"  * X_train: {_df_stats(data.X_train)}",
        f"  * exogenas: {_df_stats(data.exogenas)}",
        "- exogenas:",
        f"  * esperadas (loader o detectadas): {exog_reference or 'None'}",
        f"  * en history: {history_cols or 'None'}",
        f"  * en feature_columns: {feature_cols or 'None'}",
        f"  * en df_real: {real_cols or 'None'}",
        f"  * cobertura exogenas: range={_df_stats(data.exogenas)}",
    ]
    if warnings:
        lines.append("- warnings:")
        for w in warnings:
            lines.append(f"  * {w}")
    else:
        lines.append("- warnings: None")
    report = "\n".join(lines)
    return report, warnings


def _write_flow_report(pipeline_name: str, model_name: str, report: str) -> Path:
    FLOW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{pipeline_name}_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    )
    path = FLOW_LOG_DIR / filename
    path.write_text(report, encoding="utf-8")
    return path


def graficar(
    df,
    timestamp,
    pipeline_name,
    model_name,
    lags,
    mae,
    rmse,
    comment: str | None = None,
    forecast_only: bool = False,
):
    LOG_DIR.mkdir(exist_ok=True)
    filename = f"{pipeline_name}_{model_name}_forecast_{timestamp}.png"
    path = LOG_DIR / filename

    plt.figure(figsize=(12, 5))
    has_real = "valor_real" in df and df["valor_real"].notna().any()
    if has_real:
        plt.plot(df.index, df["valor_real"], label="Real")
    plt.plot(df.index, df["prediccion"], label="Prediccion")
    mae_label = f"{mae:.2f}" if mae is not None else "-"
    rmse_label = f"{rmse:.2f}" if rmse is not None else "-"
    title = (
        f"Flujo llamado: {pipeline_name} - Modelo: {model_name} | "
        f"Lags: {_format_lag_label(lags)} | MAE: {mae_label} | RMSE: {rmse_label}"
    )
    if forecast_only:
        title = f"{title} | Forecast puro"
    plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=40, ha="right")
    plt.legend()
    if comment:
        plt.figtext(0.5, 0.01, comment, ha="center", fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _encode_plot_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# -----------------------------
# MAIN + ORQUESTACION
# -----------------------------
def build_pipeline(name: str, **pipeline_kwargs: Any):
    try:
        pipeline_factory = PIPELINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Pipeline '{name}' no esta registrado.") from exc
    try:
        return pipeline_factory(**pipeline_kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Argumentos invalidos para el pipeline '{name}': {exc}"
        ) from exc


def get_model_handlers(name: str) -> Dict[str, Callable[..., object]]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Modelo '{name}' no esta registrado.") from exc


def ejecutar_forecast(
    pipeline_name: str,
    model_name: str,
    *,
    pipeline_kwargs: Dict[str, Any] | None = None,
    location_label: str | None = None,
    forecast_only: bool = False,
) -> ForecastResponse:
    pipeline = build_pipeline(pipeline_name, **(pipeline_kwargs or {}))
    data = pipeline.prepare()
    exog_columns = log_exogena_coverage(
        data.exogenas,
        data.history,
        data.df_real,
        context=f"{pipeline_name}/{model_name}:pre-train",
    )

    expected_from_loader = (
        data.exogenas.attrs.get("expected_exogenas")
        if data.exogenas is not None
        else []
    )
    if expected_from_loader:
        missing_expected = [
            col for col in expected_from_loader if col not in exog_columns
        ]
        if missing_expected:
            logging.warning(
                "%s: faltan columnas esperadas desde loader: %s",
                pipeline_name,
                missing_expected,
            )
        else:
            logging.info(
                "%s: todas las exogenas esperadas presentes antes de entrenar.",
                pipeline_name,
            )

    dataset_table = None
    try:
        options = pipeline_kwargs or {}
        categories = options.get("categories")
        dataset_table = persist_dataset_to_db(
            data.history,
            pipeline_name=pipeline_name,
            model_name=model_name,
            configuration_id=options.get("configuration_id"),
            categories=categories,
            start_date=data.train_range[0] if data.train_range else None,
            end_date=data.train_range[1] if data.train_range else None,
        )
    except Exception as exc:
        logging.warning("No se pudo persistir el dataset: %s", exc)

    report, flow_warnings = _build_flow_report(
        pipeline_name,
        model_name,
        data,
        exog_columns,
        expected_from_loader,
    )
    report_path = _write_flow_report(pipeline_name, model_name, report)
    logging.info("Flow diagnostics guardado en %s", report_path)
    for warn in flow_warnings:
        logging.warning("%s", warn)

    _dump_prepared_dataframes(model_name, data)
    model_handlers = get_model_handlers(model_name)
    trainer = model_handlers["train"]
    forecaster = model_handlers["forecast"]
    logging.info(
        "Entrenando modelo %s sobre pipeline %s | History cols: %s | Exogenas validadas: %s",
        model_name,
        pipeline_name,
        data.history.columns.tolist(),
        exog_columns or "sin exogenas",
    )

    modelo = trainer(data)
    pred_df = forecaster(modelo, data)

    has_real = (
        data.df_real is not None
        and not data.df_real.empty
        and "valor" in data.df_real.columns
        and data.df_real["valor"].notna().any()
    )
    is_forecast_only = forecast_only or data.forecast_only or not has_real

    if not is_forecast_only and data.df_real is not None:
        comparacion, mae, rmse = comparar(pred_df, data.df_real)
    else:
        comparacion = pred_df.copy()
        comparacion["valor_real"] = np.nan
        comparacion["error"] = np.nan
        comparacion["abs_error"] = np.nan
        mae = None
        rmse = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = graficar(
        comparacion,
        timestamp,
        pipeline_name,
        model_name,
        data.lags,
        mae,
        rmse,
        comment=data.plot_comment,
        forecast_only=is_forecast_only,
    )

    def _clean_number(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if isinstance(value, float) and np.isnan(value):
                return None
        except Exception:
            pass
        return float(value)

    forecast_rows = [
        ForecastPoint(
            fecha=index.to_pydatetime(),
            valor_real=_clean_number(row.get("valor_real")),
            prediccion=float(row["prediccion"]),
            error=_clean_number(row.get("error")),
            abs_error=_clean_number(row.get("abs_error")),
        )
        for index, row in comparacion.iterrows()
    ]

    resolved_label = location_label or resolve_location_label(
        pipeline_name, pipeline_kwargs
    )

    return ForecastResponse(
        pipeline=pipeline_name,
        location=resolved_label,
        model=model_name,
        lags=list(data.lags),
        zero_padding_days=int(data.zero_padding_days),
        mae=float(mae) if mae is not None else None,
        rmse=float(rmse) if rmse is not None else None,
        forecast_only=is_forecast_only,
        timestamp=timestamp,
        plot_path=str(plot_path),
        plot_image=_encode_plot_base64(plot_path),
        dataset_table=dataset_table,
        forecast=forecast_rows,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecucion de pipelines de pronostico."
    )
    parser.add_argument(
        "--pipeline",
        default="open_meteo",
        choices=sorted(PIPELINE_REGISTRY.keys()),
        help="Nombre del pipeline de datos a utilizar.",
    )
    parser.add_argument(
        "--location",
        default=LOCATION_CHOICES[0] if LOCATION_CHOICES else "",
        choices=LOCATION_CHOICES,
        help="Ubicacion de Chile a consultar (open_meteo).",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Modelo a entrenar para el pronostico.",
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default=None,
        help="Fecha inicial (YYYY-MM-DD) para pipelines basados en base de datos.",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        default=None,
        help="Fecha final (YYYY-MM-DD) para pipelines basados en base de datos.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Categorias para el pipeline turismo. Dejar vacio para usar todas.",
    )
    parser.add_argument(
        "--lags",
        nargs="*",
        type=int,
        default=None,
        help="Lista de lags en dias para construir las caracteristicas. Dejar vacio para el valor por defecto.",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Ejecuta un forecast puro usando todo el historico y sin comparar contra datos reales.",
    )
    parser.add_argument(
        "--forecast-end-date",
        dest="forecast_end_date",
        default=None,
        help="Fecha final (YYYY-MM-DD) para el forecast puro.",
    )
    return parser.parse_args()


def main(
    pipeline_name: str,
    model_name: str,
    *,
    pipeline_kwargs: Dict[str, Any] | None = None,
    location_label: str | None = None,
    forecast_only: bool = False,
):
    resultado = ejecutar_forecast(
        pipeline_name,
        model_name,
        pipeline_kwargs=pipeline_kwargs,
        location_label=location_label,
        forecast_only=forecast_only,
    )

    print(f"Contexto evaluado: {resultado.location}")
    print(f"Pipeline: {resultado.pipeline} | Modelo: {resultado.model}")
    mae_label = f"{resultado.mae:.3f}" if resultado.mae is not None else "-"
    rmse_label = f"{resultado.rmse:.3f}" if resultado.rmse is not None else "-"
    print(f"MAE  = {mae_label}")
    print(f"RMSE = {rmse_label}")
    print(f"Grafico generado: {resultado.plot_path}")
    if resultado.dataset_table:
        print(f"Dataset persistido: {resultado.dataset_table}")
    return resultado


def run_all_flows(
    pipeline_name: str = "open_meteo",
    locations: tuple[str, ...] = DEFAULT_BATCH_LOCATIONS,
    forecast_only: bool = False,
) -> None:
    """Ejecuta todos los modelos registrados para cada ubicacion dada."""
    for location in locations:
        for model_name in MODEL_REGISTRY.keys():
            print("=" * 60)
            print(f"Ejecutando {pipeline_name} | {location} | {model_name}")
            main(
                pipeline_name,
                model_name,
                pipeline_kwargs={"location": location},
                location_label=location,
                forecast_only=forecast_only,
            )


app = FastAPI(
    title="Pronostico Climatico API",
    description="Servicio FastAPI para ejecutar pipelines y modelos de pronostico.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(configuracion_router)
app.include_router(exogena_router)


@app.get("/health")
def healthcheck():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/options")
def options():
    turismo_categories: List[str] = []
    turismo_dates: Dict[str, str | None] = {"min": None, "max": None}
    configuraciones: List[Dict[str, Any]] = []
    try:
        turismo_categories = get_turismo_categories()
    except Exception as exc:  # pragma: no cover - logging auxiliar
        logging.warning("No fue posible obtener las categorias de turismo: %s", exc)
    try:
        turismo_dates = get_turismo_date_bounds()
    except Exception as exc:  # pragma: no cover - logging auxiliar
        logging.warning(
            "No fue posible obtener el rango disponible de turismo: %s", exc
        )
    try:
        configuraciones = listar_configuraciones_db()
    except Exception as exc:  # pragma: no cover - logging auxiliar
        logging.warning("No fue posible obtener las configuraciones: %s", exc)
    return {
        "pipelines": sorted(PIPELINE_REGISTRY.keys()),
        "locations": LOCATION_CHOICES,
        "models": sorted(MODEL_REGISTRY.keys()),
        "configurations": configuraciones,
        "turismo": {
            "categories": turismo_categories,
            "date_bounds": turismo_dates,
            "min_gap_days": TURISMO_COMPARISON_DAYS,
        },
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast_endpoint(request: ForecastRequest):
    try:
        if request.forecast_only and not request.forecast_end_date:
            raise ValueError(
                "Debes indicar 'forecast_end_date' cuando 'forecast_only' es True."
            )
        pipeline_kwargs = request.build_pipeline_kwargs()
        label = resolve_location_label(request.pipeline, pipeline_kwargs)
        return ejecutar_forecast(
            request.pipeline,
            request.model,
            pipeline_kwargs=pipeline_kwargs,
            location_label=label,
            forecast_only=request.forecast_only,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - fallback para errores inesperados
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    # Para cambiar a correr todos los flujos, comentar las dos  siguientes y quitar el comentario a la llamada de run_all_flows
    args = parse_args()
    pipeline_kwargs: Dict[str, Any] = {}
    if args.location:
        pipeline_kwargs["location"] = args.location
    if args.start_date:
        pipeline_kwargs["start_date"] = args.start_date
    if args.end_date:
        pipeline_kwargs["end_date"] = args.end_date
    if args.categories is not None:
        pipeline_kwargs["categories"] = args.categories
    if args.lags:
        pipeline_kwargs["lags"] = sorted(
            {lag for lag in args.lags if lag is not None and lag > 0}
        )
    if args.forecast_only:
        pipeline_kwargs["forecast_only"] = True
    if args.forecast_end_date:
        pipeline_kwargs["forecast_end_date"] = args.forecast_end_date

    main(
        args.pipeline,
        args.model,
        pipeline_kwargs=pipeline_kwargs,
        location_label=resolve_location_label(args.pipeline, pipeline_kwargs),
        forecast_only=args.forecast_only,
    )

    # run_all_flows()
