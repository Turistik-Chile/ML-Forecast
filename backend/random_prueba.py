# turismo_pipeline.py  (versión Random Forest)
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence, List, Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from db_connection import get_connection  # archivo aparte


# -------------------------------------------------------------------
# CONFIG PIPELINE
# -------------------------------------------------------------------
@dataclass
class TurismoRFConfig:
    configuration_id: int = 4
    categories: Sequence[str] | None = ("nieve",)

    start_date: datetime | str | None = None
    end_date: datetime | str | None = None  # límite train

    forecast_days: int = 365  # 1 año futuro

    # lags para RF (muy importantes)
    lags: Tuple[int, ...] = (1, 7, 14, 28)

    # RF params base
    n_estimators: int = 800
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = -1

    model_name: str = "random_forest"


# -------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------
class TurismoRFPipeline:
    def __init__(self, config: TurismoRFConfig):
        self.cfg = config
        logging.basicConfig(level=logging.INFO)

        self.table = "ia.ventas_turismo"
        self.date_col = "servicedate"
        self.cat_col = "Category"
        self.val_col = "n_pax"

        self.sql_exog = """
            SELECT d.id, d.nombre, r.fecha_inicio, r.fecha_fin
            FROM ia.variable_exogena_dummie d
            LEFT JOIN ia.rango_fechas_dummies r ON r.exogena_id = d.id
            WHERE d.configuracion_id = ? AND d.is_active = 1
        """

    # ---------------------------
    # utils para persistencia
    # ---------------------------
    def _sanitize_sql_name(self, s: str, max_len: int = 80) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_]", "", s)
        return s[:max_len] if len(s) > max_len else s

    def _sql_type_from_series(self, ser: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(ser):
            return "DATETIME"
        if pd.api.types.is_bool_dtype(ser):
            return "BIT"
        if pd.api.types.is_integer_dtype(ser):
            return "BIGINT"
        if pd.api.types.is_float_dtype(ser):
            return "FLOAT"
        return "NVARCHAR(255)"

    def persist_dataset_to_db(self, df: pd.DataFrame, *, schema: str = "ia") -> str:
        if df is None or df.empty:
            raise ValueError("El dataframe viene vacío, no se puede persistir.")

        data = df.copy()

        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(columns={"index": "fecha"})
        if "fecha" not in data.columns:
            raise ValueError("No encuentro columna 'fecha' tras reset_index.")

        start_str = self._sanitize_sql_name(
            pd.to_datetime(self.cfg.start_date).date() if self.cfg.start_date else "na"
        )
        end_str = self._sanitize_sql_name(
            pd.to_datetime(self.cfg.end_date).date() if self.cfg.end_date else "na"
        )
        cat_str = "_".join(
            self._sanitize_sql_name(c) for c in (self.cfg.categories or ["all"])
        )
        model_str = self._sanitize_sql_name(self.cfg.model_name)

        table_base = f"dataset_{model_str}_cfg{self.cfg.configuration_id}_{cat_str}_{start_str}_{end_str}"
        table_base = self._sanitize_sql_name(table_base, max_len=110)
        full_table = f"{schema}.{table_base}"

        col_map: Dict[str, str] = {}
        used = set()
        for col in data.columns:
            safe = self._sanitize_sql_name(col, max_len=60) or "col"
            base = safe
            i = 1
            while safe in used:
                safe = f"{base}_{i}"
                i += 1
            used.add(safe)
            col_map[col] = safe

        data_sql = data.rename(columns=col_map)

        col_defs = []
        for orig_col, safe_col in col_map.items():
            sql_type = self._sql_type_from_series(data[orig_col])
            col_defs.append(f"[{safe_col}] {sql_type}")

        drop_sql = f"""
        IF OBJECT_ID('{full_table}', 'U') IS NOT NULL
            DROP TABLE {full_table};
        """

        create_sql = f"""
        CREATE TABLE {full_table} (
            {", ".join(col_defs)}
        );
        """

        safe_cols = list(col_map.values())
        placeholders = ", ".join("?" for _ in safe_cols)
        insert_sql = f"""
        INSERT INTO {full_table} ({", ".join(f"[{c}]" for c in safe_cols)})
        VALUES ({placeholders});
        """

        values = [
            tuple(row) for row in data_sql[safe_cols].itertuples(index=False, name=None)
        ]

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(drop_sql)
            cur.execute(create_sql)
            cur.fast_executemany = True
            cur.executemany(insert_sql, values)
            conn.commit()

        logging.info(f"Tabla creada y cargada: {full_table} ({len(values)} filas)")
        return full_table

    # ---------------------------
    # 1) FETCH target
    # ---------------------------
    def fetch_series(self) -> pd.DataFrame:
        if self.cfg.end_date is None:
            raise ValueError("Debes indicar end_date (límite de entrenamiento).")

        start_dt = pd.to_datetime(self.cfg.start_date) if self.cfg.start_date else None
        end_dt = pd.to_datetime(self.cfg.end_date)

        query = [
            f"SELECT {self.date_col} AS fecha, {self.cat_col} AS categoria, {self.val_col} AS valor",
            f"FROM {self.table}",
            "WHERE 1=1",
        ]
        params: List[Any] = []

        if start_dt is not None:
            query.append(f"AND {self.date_col} >= ?")
            params.append(start_dt)

        query.append(f"AND {self.date_col} <= ?")
        params.append(end_dt)

        if self.cfg.categories:
            placeholders = ",".join("?" for _ in self.cfg.categories)
            query.append(f"AND {self.cat_col} IN ({placeholders})")
            params.extend(list(self.cfg.categories))

        query.append(f"ORDER BY {self.date_col}")
        sql = "\n".join(query)

        with get_connection() as conn:
            raw = pd.read_sql_query(sql, conn, params=params)

        if raw.empty:
            raise ValueError("La consulta no devolvió datos para esos filtros.")

        raw["fecha"] = pd.to_datetime(raw["fecha"], errors="coerce")
        raw = raw.dropna(subset=["fecha"])
        if raw.empty:
            raise ValueError("Los datos devueltos no tienen fechas válidas.")

        grouped = (
            raw.groupby("fecha", as_index=False)["valor"]
            .sum()
            .sort_values("fecha")
            .set_index("fecha")
        )

        grouped.index = pd.to_datetime(grouped.index).tz_localize(None)
        grouped["valor"] = pd.to_numeric(grouped["valor"], errors="coerce").fillna(0.0)
        grouped = grouped.sort_index()

        full_idx = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
        grouped = grouped.reindex(full_idx).fillna(0.0)
        grouped.index.name = "fecha"

        return grouped

    # ---------------------------
    # 2) Features circulares (corregidas)
    # ---------------------------
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        idx = out.index

        day_of_year = idx.dayofyear.astype(int)
        days_in_year = np.where(idx.is_leap_year, 366, 365)

        day_of_month = idx.day.astype(int)
        days_in_month = idx.days_in_month.astype(int)

        iso = idx.isocalendar()
        week_of_year = iso.week.astype(int)
        iso_year = iso.year.astype(int)

        weeks_in_year_map = (
            pd.Series(week_of_year, index=idx)
            .groupby(iso_year)
            .transform("max")
            .astype(int)
            .values
        )

        month_of_year = idx.month.astype(int)
        months_in_year = 12

        out["sin_dayofyear"] = np.sin(2 * np.pi * day_of_year / days_in_year)
        out["cos_dayofyear"] = np.cos(2 * np.pi * day_of_year / days_in_year)

        out["sin_dayofmonth"] = np.sin(2 * np.pi * day_of_month / days_in_month)
        out["cos_dayofmonth"] = np.cos(2 * np.pi * day_of_month / days_in_month)

        out["sin_weekofyear"] = np.sin(2 * np.pi * week_of_year / weeks_in_year_map)
        out["cos_weekofyear"] = np.cos(2 * np.pi * week_of_year / weeks_in_year_map)

        out["sin_monthofyear"] = np.sin(2 * np.pi * month_of_year / months_in_year)
        out["cos_monthofyear"] = np.cos(2 * np.pi * month_of_year / months_in_year)

        return out

    # ---------------------------
    # 3) Exógenas dummies por rangos
    # ---------------------------
    def load_exog_dummies(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        full_index = pd.date_range(start_dt, end_dt, freq="D")
        exog = pd.DataFrame(index=full_index)

        with get_connection() as conn:
            rows = pd.read_sql_query(
                self.sql_exog, conn, params=[self.cfg.configuration_id]
            )

        if rows.empty:
            return exog.fillna(0)

        rows["fecha_inicio"] = pd.to_datetime(rows["fecha_inicio"], errors="coerce")
        rows["fecha_fin"] = pd.to_datetime(rows["fecha_fin"], errors="coerce")

        for _, r in rows.iterrows():
            exog_id = r["id"]
            nombre = str(r["nombre"]).strip().replace(" ", "_")
            col = f"dummie_{exog_id}_{nombre}"

            if col not in exog.columns:
                exog[col] = 0

            fi, ff = r["fecha_inicio"], r["fecha_fin"]
            if pd.isna(fi) or pd.isna(ff):
                continue

            mask = (exog.index >= fi) & (exog.index <= ff)
            exog.loc[mask, col] = 1

        return exog.fillna(0)

    # ---------------------------
    # 4) Arma history + futuro 1 año
    # ---------------------------
    def build_history_future(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        history = self.fetch_series()
        history = self.add_calendar_features(history)

        end_dt = pd.to_datetime(self.cfg.end_date)
        future_start = end_dt + timedelta(days=1)
        future_end = end_dt + timedelta(days=self.cfg.forecast_days)
        future_idx = pd.date_range(future_start, future_end, freq="D")

        future = pd.DataFrame(index=future_idx, data={"valor": np.nan})
        future = self.add_calendar_features(future)

        exog_all = self.load_exog_dummies(history.index.min(), future_end)

        history_full = history.join(exog_all, how="left").fillna(0)
        future_full = future.join(exog_all, how="left").fillna(0)

        return history_full, future_full

    # ---------------------------
    # 5) Crear lags para RF
    # ---------------------------
    def add_lag_features(
        self, df: pd.DataFrame, target_col: str = "valor"
    ) -> pd.DataFrame:
        out = df.copy()
        for lag in self.cfg.lags:
            out[f"lag_{lag}"] = out[target_col].shift(lag)
        return out

    # ---------------------------
    # 6) Entrenar RF y forecast iterativo 1 año
    # ---------------------------
    def fit_forecast(self, history_full: pd.DataFrame, future_full: pd.DataFrame):
        # lags sobre history
        hist_lagged = self.add_lag_features(history_full, "valor").dropna()

        y_train_raw = hist_lagged["valor"].astype(float)
        y_train_log = np.log1p(y_train_raw)

        feature_cols = [c for c in hist_lagged.columns if c != "valor"]

        X_train = hist_lagged[feature_cols].astype(float)

        rf = RandomForestRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        rf.fit(X_train, y_train_log)

        # forecast iterativo
        max_lag = max(self.cfg.lags)
        history_vals = history_full["valor"].astype(float).copy()

        preds_log = []
        future_index = future_full.index

        for t in range(len(future_full)):
            current_date = future_index[t]

            # construir fila de features para este día
            row = future_full.loc[current_date:current_date].copy()

            # agregar lags desde history_vals (que se va extendiendo con preds)
            for lag in self.cfg.lags:
                lag_date = current_date - timedelta(days=lag)
                row[f"lag_{lag}"] = (
                    history_vals.loc[lag_date]
                    if lag_date in history_vals.index
                    else 0.0
                )

            X_row = row[feature_cols].astype(float).values
            pred_log = rf.predict(X_row)[0]
            preds_log.append(pred_log)

            # volver a escala real para alimentar próximos lags
            pred_real = max(np.expm1(pred_log), 0.0)
            history_vals.loc[current_date] = pred_real

        forecast = pd.Series(np.expm1(preds_log), index=future_index).clip(lower=0)

        return y_train_raw, forecast, feature_cols, rf

    # ---------------------------
    # 7) Plot
    # ---------------------------
    def plot(self, y_train: pd.Series, forecast: pd.Series):
        plt.figure(figsize=(13, 6))
        plt.plot(y_train.index, y_train.values, label="History (train)")
        plt.plot(forecast.index, forecast.values, label="Forecast 1 año")

        plt.axvline(
            y_train.index.max(), linestyle="--", alpha=0.6, label="Cutoff train"
        )
        plt.title(
            f"RF Turismo - {self.cfg.categories or 'all'} "
            f"(cfg {self.cfg.configuration_id}) - 1 año futuro"
        )
        plt.xlabel("Fecha")
        plt.ylabel("Ventas")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # Orquestador único
    # ---------------------------
    def run(self):
        history_full, future_full = self.build_history_future()

        # Persistimos history para validar dataset real
        table_name = self.persist_dataset_to_db(history_full)

        y_train, forecast, feature_cols, model = self.fit_forecast(
            history_full, future_full
        )

        print("Tabla persistida:", table_name)
        print("Features usadas:", feature_cols[:20], "...")  # preview
        print("Forecast head:")
        print(forecast.head())

        self.plot(y_train, forecast)

        return {
            "history_full": history_full,
            "future_full": future_full,
            "dataset_table": table_name,
            "y_train": y_train,
            "forecast": forecast,
            "feature_cols": feature_cols,
            "model": model,
        }


# -------------------------------------------------------------------
# Ejecución directa
# -------------------------------------------------------------------
if __name__ == "__main__":
    cfg = TurismoRFConfig(
        configuration_id=2,
        categories=("nieve",),
        start_date="2023-12-18",
        end_date="2025-11-28",
        forecast_days=365,
        lags=(1, 7, 14, 28, 56),
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        model_name="random_forest",
    )

    pipeline = TurismoRFPipeline(cfg)
    results = pipeline.run()
