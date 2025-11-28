import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List, Dict

import requests
import pyodbc

from db_connection import get_connection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------
# APIs Open-Meteo
# -------------------------
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_LAT = float(os.getenv("NIEVE_LAT", "-33.35"))
DEFAULT_LON = float(os.getenv("NIEVE_LON", "-70.25"))

TIMEZONE = "America/Santiago"
MODEL_ARCHIVE = "era5_land"
FORECAST_DAYS_MAX = 16

CONFIG_NOMBRE = "Nieve test forecast"
EXOGENA_NOMBRE = "nieve_caida_dia"

FIXED_START_DATE = date(2024, 6, 3)
TODAY_OVERRIDE = date(2025, 11, 25)  # tu "HOY"


# =======================================================
# UTILIDADES DB (DETECTAR ESQUEMA REAL)
# =======================================================
def get_table_columns(conn: pyodbc.Connection, schema: str, table: str) -> List[str]:
    q = """
    SELECT LOWER(COLUMN_NAME)
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
    """
    with conn.cursor() as cur:
        cur.execute(q, (schema, table))
        return [r[0] for r in cur.fetchall()]


def pick_pk_column(columns: List[str]) -> str:
    """
    Tu BD a veces usa 'id' y a veces 'code_id' como PK.
    """
    if "id" in columns:
        return "id"
    if "code_id" in columns:
        return "code_id"
    raise ValueError(f"No pude detectar PK. Columnas: {columns}")


def ensure_row_by_nombre(
    conn: pyodbc.Connection,
    schema: str,
    table: str,
    nombre: str,
    extra_insert_cols: Optional[Dict[str, object]] = None,
) -> object:
    """
    Asegura existencia de fila en {schema}.{table} con 'nombre'.
    Devuelve PK (id o code_id).
    """
    cols = get_table_columns(conn, schema, table)
    pk = pick_pk_column(cols)

    with conn.cursor() as cur:
        # Buscar existente
        cur.execute(f"SELECT {pk} FROM {schema}.{table} WHERE nombre = ?", (nombre,))
        row = cur.fetchone()
        if row:
            logging.info(f"{schema}.{table} ya existe: {nombre} ({pk}={row[0]})")
            return row[0]

        # Insertar nuevo
        insert_cols = ["nombre"]
        insert_vals = [nombre]

        if extra_insert_cols:
            for k, v in extra_insert_cols.items():
                if k.lower() in cols:
                    insert_cols.append(k)
                    insert_vals.append(v)

        cols_sql = ", ".join(insert_cols)
        params_sql = ", ".join(["?"] * len(insert_cols))

        # OUTPUT inserted.<pk> para recuperar PK sin asumir IDENTITY
        sql = f"""
        INSERT INTO {schema}.{table} ({cols_sql})
        OUTPUT inserted.{pk}
        VALUES ({params_sql})
        """

        cur.execute(sql, insert_vals)
        new_pk = cur.fetchone()[0]
        conn.commit()

        logging.info(f"{schema}.{table} creado: {nombre} ({pk}={new_pk})")
        return new_pk


def fetch_existing_dates(conn: pyodbc.Connection, exogena_id: object) -> set:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT fecha FROM ia.fecha_valor_exogena WHERE exogena_id = ?",
            (exogena_id,),
        )
        return {r[0] for r in cur.fetchall()}


def insert_fecha_valor_bulk(
    conn: pyodbc.Connection, exogena_id: object, items: List[Tuple[date, int]]
):
    if not items:
        logging.info("No hay nuevos registros para insertar en fecha_valor_exogena.")
        return

    with conn.cursor() as cur:
        cur.fast_executemany = True
        cur.executemany(
            """
            INSERT INTO ia.fecha_valor_exogena (exogena_id, fecha, valor)
            VALUES (?, ?, ?)
            """,
            [(exogena_id, f, v) for f, v in items],
        )
    conn.commit()
    logging.info(f"Insertados {len(items)} registros en ia.fecha_valor_exogena.")


# =======================================================
# API HELPERS
# =======================================================
def fetch_snow_archive_daily(
    lat: float, lon: float, start_date: date, end_date: date
) -> List[Tuple[date, float]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "snowfall_sum",
        "timezone": TIMEZONE,
        "models": MODEL_ARCHIVE,
    }
    logging.info(f"Archive API params: {params}")
    r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    times = data.get("daily", {}).get("time", [])
    snow = data.get("daily", {}).get("snowfall_sum", [])

    out = []
    for t, s in zip(times, snow):
        d = datetime.strptime(t, "%Y-%m-%d").date()
        out.append((d, float(s) if s is not None else 0.0))
    return out


def fetch_snow_forecast_daily(
    lat: float, lon: float, forecast_days: int
) -> List[Tuple[date, float]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "snowfall_sum",
        "timezone": TIMEZONE,
        "forecast_days": forecast_days,
    }
    logging.info(f"Forecast API params: {params}")
    r = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    times = data.get("daily", {}).get("time", [])
    snow = data.get("daily", {}).get("snowfall_sum", [])

    out = []
    for t, s in zip(times, snow):
        d = datetime.strptime(t, "%Y-%m-%d").date()
        out.append((d, float(s) if s is not None else 0.0))
    return out


# =======================================================
# MAIN
# =======================================================
def run_pipeline(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    start_date: Optional[date] = None,
    today: Optional[date] = None,
):
    if start_date is None:
        start_date = FIXED_START_DATE
    if today is None:
        today = TODAY_OVERRIDE

    target_end = start_date + timedelta(days=365)

    logging.info(f"Start fijo: {start_date}")
    logging.info(f"Hoy: {today}")
    logging.info(f"Target 1 año adelante: {target_end}")

    conn = get_connection()
    try:
        # 1) Configuración (tabla real puede tener id o code_id)
        config_pk = ensure_row_by_nombre(
            conn,
            "ia",
            "configuracion",
            CONFIG_NOMBRE,
            extra_insert_cols={"comentarios": None},
        )

        # 2) Variable exógena NORMAL (ia.variable_exogena)
        exog_normal_pk = ensure_row_by_nombre(
            conn,
            "ia",
            "variable_exogena",
            EXOGENA_NOMBRE,
            extra_insert_cols={"is_active": 1, "configuracion_id": config_pk},
        )

        # 3) Dummie espejo (porque FK fecha_valor_exogena apunta a dummie)
        exog_dummie_pk = ensure_row_by_nombre(
            conn,
            "ia",
            "variable_exogena_dummie",
            EXOGENA_NOMBRE,
            extra_insert_cols={"is_active": 1, "configuracion_id": config_pk},
        )

        existing = fetch_existing_dates(conn, exog_dummie_pk)

        # A) Histórico: start -> min(hoy, target_end)
        hist_end = min(today, target_end)
        hist_rows = fetch_snow_archive_daily(lat, lon, start_date, hist_end)

        to_insert_hist = []
        for d, snow_mm in hist_rows:
            if d in existing:
                continue
            to_insert_hist.append((d, int(round(snow_mm))))

        insert_fecha_valor_bulk(conn, exog_dummie_pk, to_insert_hist)
        existing.update([d for d, _ in to_insert_hist])

        # B) Futuro real si existe (máx 16 días)
        if today < target_end:
            fc_days = min(FORECAST_DAYS_MAX, (target_end - today).days)
            if fc_days > 0:
                fc_rows = fetch_snow_forecast_daily(lat, lon, fc_days)

                to_insert_fc = []
                for d, snow_mm in fc_rows:
                    if d <= today or d in existing:
                        continue
                    to_insert_fc.append((d, int(round(snow_mm))))

                insert_fecha_valor_bulk(conn, exog_dummie_pk, to_insert_fc)

        logging.info(
            f"OK -> configuracion_pk={config_pk}, "
            f"exog_normal_pk={exog_normal_pk}, exog_dummie_pk={exog_dummie_pk}"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    run_pipeline()
