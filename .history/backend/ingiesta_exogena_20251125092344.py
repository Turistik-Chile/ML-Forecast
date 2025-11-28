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
TODAY_OVERRIDE = date(2025, 11, 25)  # tu "HOY" fijo

SCHEMA = "ia"


# =======================================================
# UTILIDADES DB (SCHEMA SIEMPRE ia)
# =======================================================
def get_table_columns(conn: pyodbc.Connection, table: str) -> List[str]:
    q = """
    SELECT LOWER(COLUMN_NAME)
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
    """
    with conn.cursor() as cur:
        cur.execute(q, (SCHEMA, table))
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
    table: str,
    nombre: str,
    extra_insert_cols: Optional[Dict[str, object]] = None,
) -> object:
    """
    Asegura existencia de fila en ia.{table} con 'nombre'.
    Devuelve PK (id o code_id).

    Si PK es code_id (nvarchar NOT NULL), mete cualquier string único.
    """
    cols = get_table_columns(conn, table)
    pk = pick_pk_column(cols)

    with conn.cursor() as cur:
        # 1) Buscar existente
        cur.execute(f"SELECT {pk} FROM {SCHEMA}.{table} WHERE nombre = ?", (nombre,))
        row = cur.fetchone()
        if row:
            logging.info(f"{SCHEMA}.{table} ya existe: {nombre} ({pk}={row[0]})")
            return row[0]

        # 2) Armar INSERT
        insert_cols = ["nombre"]
        insert_vals = [nombre]

        # ✅ si PK es code_id, hay que incluirlo sí o sí
        new_code_id = None
        if pk == "code_id":
            new_code_id = f"local_{table}_{int(datetime.now().timestamp())}"
            insert_cols.insert(0, "code_id")
            insert_vals.insert(0, new_code_id)

        # extras (is_active, configuracion_id, comentarios, etc.)
        if extra_insert_cols:
            for k, v in extra_insert_cols.items():
                if k.lower() in cols:
                    insert_cols.append(k)
                    insert_vals.append(v)

        cols_sql = ", ".join(insert_cols)
        params_sql = ", ".join(["?"] * len(insert_cols))

        sql = f"INSERT INTO {SCHEMA}.{table} ({cols_sql}) VALUES ({params_sql})"

        cur.execute(sql, insert_vals)
        conn.commit()

        # 3) devolver PK insertada
        if pk == "code_id":
            logging.info(f"{SCHEMA}.{table} creado: {nombre} (code_id={new_code_id})")
            return new_code_id
        else:
            cur.execute("SELECT SCOPE_IDENTITY()")
            new_id = cur.fetchone()[0]
            logging.info(f"{SCHEMA}.{table} creado: {nombre} (id={new_id})")
            return new_id


def fetch_existing_dates(conn: pyodbc.Connection, exogena_id: object) -> set:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT fecha FROM {SCHEMA}.fecha_valor_exogena WHERE exogena_id = ?",
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
            f"""
            INSERT INTO {SCHEMA}.fecha_valor_exogena (exogena_id, fecha, valor)
            VALUES (?, ?, ?)
            """,
            [(exogena_id, f, v) for f, v in items],
        )
    conn.commit()
    logging.info(f"Insertados {len(items)} registros en {SCHEMA}.fecha_valor_exogena.")


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
    # start fijo
    if start_date is None:
        start_date = FIXED_START_DATE
    # hoy fijo
    if today is None:
        today = TODAY_OVERRIDE

    target_end = start_date + timedelta(days=365)

    logging.info(f"Start fijo: {start_date}")
    logging.info(f"Hoy: {today}")
    logging.info(f"Target 1 año adelante: {target_end}")

    conn = get_connection()
    try:
        # 1) Configuración
        config_pk = ensure_row_by_nombre(
            conn,
            "configuracion",
            CONFIG_NOMBRE,
            extra_insert_cols={"comentarios": None},
        )

        # 2) Variable exógena NORMAL
        exog_normal_pk = ensure_row_by_nombre(
            conn,
            "variable_exogena",
            EXOGENA_NOMBRE,
            extra_insert_cols={"is_active": 1, "configuracion_id": config_pk},
        )

        # 3) Variable espejo DUMMIE para FK de fecha_valor_exogena
        exog_dummie_pk = ensure_row_by_nombre(
            conn,
            "variable_exogena_dummie",
            EXOGENA_NOMBRE,
            extra_insert_cols={"is_active": 1, "configuracion_id": config_pk},
        )

        existing = fetch_existing_dates(conn, exog_dummie_pk)

        # A) Histórico: start -> hoy (o target si fuera menor)
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
