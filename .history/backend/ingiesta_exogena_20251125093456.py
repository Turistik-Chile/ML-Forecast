import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List, Dict
from uuid import uuid4

import requests
import pyodbc

from db_connection import get_connection

# -------------------------------------------------------
# LOGGING
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------------------------------------
# Open-Meteo endpoints
# -------------------------------------------------------
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Coordenadas default (puedes moverlas por env)
DEFAULT_LAT = float(os.getenv("NIEVE_LAT", "-33.35"))  # aprox Valle Nevado
DEFAULT_LON = float(os.getenv("NIEVE_LON", "-70.25"))

TIMEZONE = "America/Santiago"

# OJO:
# ERA5-Land no siempre trae snowfall (puede venir todo 0).
# ERA5 sí trae el set completo para precip/snowfall.
MODEL_ARCHIVE = os.getenv("NIEVE_MODEL_ARCHIVE", "era5").lower()

FORECAST_DAYS_MAX = 16  # Open-Meteo forecast max ~16 días

CONFIG_NOMBRE = "Nieve test forecast"
EXOGENA_NOMBRE = "nieve_caida_dia"

FIXED_START_DATE = date(2024, 6, 3)
TODAY_OVERRIDE = date(2025, 11, 25)  # tu "HOY" explícito


# =======================================================
# DB HELPERS (schema fijo ia)
# =======================================================
def get_table_columns(conn: pyodbc.Connection, table: str) -> List[str]:
    q = """
    SELECT LOWER(COLUMN_NAME)
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'ia' AND TABLE_NAME = ?
    """
    with conn.cursor() as cur:
        cur.execute(q, (table,))
        return [r[0] for r in cur.fetchall()]


def ensure_configuracion(conn: pyodbc.Connection, nombre: str) -> int:
    cols = get_table_columns(conn, "configuracion")

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM ia.configuracion WHERE nombre = ?", (nombre,))
        row = cur.fetchone()
        if row:
            logging.info(f"Configuración ya existe: {nombre} (id={row.id})")
            return int(row.id)

        insert_cols = ["nombre"]
        insert_vals = [nombre]

        if "comentarios" in cols:
            insert_cols.append("comentarios")
            insert_vals.append(None)

        # si existe code_id y es NOT NULL, le ponemos cualquier string
        if "code_id" in cols:
            insert_cols.append("code_id")
            insert_vals.append(f"cfg_{uuid4().hex}")

        cols_sql = ", ".join(insert_cols)
        params_sql = ", ".join(["?"] * len(insert_cols))

        cur.execute(
            f"""
            INSERT INTO ia.configuracion ({cols_sql})
            VALUES ({params_sql})
            """,
            insert_vals,
        )
        conn.commit()
        config_id = cur.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]
        logging.info(f"Configuración creada: {nombre} (id={config_id})")
        return int(config_id)


def ensure_variable_exogena_normal(
    conn: pyodbc.Connection, nombre: str, configuracion_id: int
) -> int:
    cols = get_table_columns(conn, "variable_exogena")

    with conn.cursor() as cur:
        cur.execute("SELECT id FROM ia.variable_exogena WHERE nombre = ?", (nombre,))
        row = cur.fetchone()
        if row:
            logging.info(f"Variable exógena NORMAL ya existe: {nombre} (id={row.id})")
            return int(row.id)

        insert_cols = ["nombre", "is_active", "configuracion_id"]
        insert_vals = [nombre, 1, configuracion_id]

        if "code_id" in cols:
            insert_cols.append("code_id")
            insert_vals.append(f"exg_{uuid4().hex}")

        cols_sql = ", ".join(insert_cols)
        params_sql = ", ".join(["?"] * len(insert_cols))

        cur.execute(
            f"""
            INSERT INTO ia.variable_exogena ({cols_sql})
            VALUES ({params_sql})
            """,
            insert_vals,
        )
        conn.commit()
        exog_id = cur.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]
        logging.info(f"Variable exógena NORMAL creada: {nombre} (id={exog_id})")
        return int(exog_id)


def ensure_variable_exogena_dummie(
    conn: pyodbc.Connection, nombre: str, configuracion_id: int
) -> int:
    cols = get_table_columns(conn, "variable_exogena_dummie")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM ia.variable_exogena_dummie WHERE nombre = ?", (nombre,)
        )
        row = cur.fetchone()
        if row:
            logging.info(f"Variable exógena DUMMIE ya existe: {nombre} (id={row.id})")
            return int(row.id)

        insert_cols = ["nombre", "is_active", "configuracion_id"]
        insert_vals = [nombre, 1, configuracion_id]

        if "code_id" in cols:
            insert_cols.append("code_id")
            insert_vals.append(f"exgd_{uuid4().hex}")

        cols_sql = ", ".join(insert_cols)
        params_sql = ", ".join(["?"] * len(insert_cols))

        cur.execute(
            f"""
            INSERT INTO ia.variable_exogena_dummie ({cols_sql})
            VALUES ({params_sql})
            """,
            insert_vals,
        )
        conn.commit()
        exog_id = cur.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]
        logging.info(f"Variable exógena DUMMIE creada: {nombre} (id={exog_id})")
        return int(exog_id)


def fetch_existing_dates(conn: pyodbc.Connection, exogena_dummie_id: int) -> set:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT fecha FROM ia.fecha_valor_exogena WHERE exogena_id = ?",
            (exogena_dummie_id,),
        )
        return {r[0] for r in cur.fetchall()}


def insert_fecha_valor_bulk(
    conn: pyodbc.Connection, exogena_dummie_id: int, items: List[Tuple[date, int]]
):
    if not items:
        logging.info("No hay nuevos registros para insertar.")
        return

    with conn.cursor() as cur:
        cur.fast_executemany = True
        cur.executemany(
            """
            INSERT INTO ia.fecha_valor_exogena (exogena_id, fecha, valor)
            VALUES (?, ?, ?)
            """,
            [(exogena_dummie_id, f, v) for (f, v) in items],
        )
    conn.commit()
    logging.info(f"Insertados {len(items)} registros en ia.fecha_valor_exogena.")


# =======================================================
# API HELPERS
# =======================================================
def _scale_snow_to_mm(
    data_json: dict, snow_values: List[Optional[float]], unit_key: str
) -> List[float]:
    """
    Convierte snowfall a mm según hourly_units/daily_units.
    """
    unit = (
        data_json.get("hourly_units", {}).get(unit_key)
        or data_json.get("daily_units", {}).get(unit_key)
        or ""
    ).lower()

    if unit in ["m", "meter", "meters"]:
        factor = 1000.0
    elif unit in ["cm", "centimeter", "centimeters"]:
        factor = 10.0
    elif unit in ["mm", "millimeter", "millimeters"]:
        factor = 1.0
    else:
        factor = 1.0  # fallback

    logging.info(f"Unidad {unit_key}='{unit}' -> factor_to_mm={factor}")

    out = []
    for s in snow_values:
        val = float(s) if s is not None else 0.0
        out.append(val * factor)
    return out


def _aggregate_hourly_to_daily(
    times: List[str], vals_mm: List[float]
) -> List[Tuple[date, float]]:
    """
    Agrupa series horarias a sumas diarias.
    Espera timestamps tipo 'YYYY-MM-DDTHH:MM'
    """
    by_day: Dict[date, float] = {}
    for t, v in zip(times, vals_mm):
        d = datetime.strptime(t, "%Y-%m-%dT%H:%M").date()
        by_day[d] = by_day.get(d, 0.0) + (v or 0.0)
    return sorted(by_day.items())


def _all_zero_or_none(vals: List[Optional[float]]) -> bool:
    if not vals:
        return True
    for v in vals:
        if v is None:
            continue
        if float(v) != 0.0:
            return False
    return True


def fetch_snow_archive_daily(
    lat: float, lon: float, start_date: date, end_date: date
) -> List[Tuple[date, float]]:
    """
    Históricos: pedir hourly=snowfall y sumar por día.
    Fallback: si snowfall viene todo 0/None, pedir snowfall_water_equivalent
    y convertir a nieve con factor 7 (1mm agua ~ 7mm nieve).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "snowfall",
        "timezone": TIMEZONE,
        "models": MODEL_ARCHIVE,
    }
    logging.info(f"Archive API params: {params}")
    r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    snow_raw = hourly.get("snowfall", [])

    logging.info(f"Archive hourly keys: {list(hourly.keys())}")
    logging.info(f"Archive sample snowfall raw: {snow_raw[:5]}")

    # Fallback si snowfall no sirve
    if _all_zero_or_none(snow_raw):
        logging.warning(
            "Archive snowfall vino vacío / todo 0. Reintentando con "
            "'snowfall_water_equivalent' y factor 7."
        )
        params_fb = dict(params)
        params_fb["hourly"] = "snowfall_water_equivalent"
        r_fb = requests.get(OPEN_METEO_ARCHIVE_URL, params=params_fb, timeout=60)
        r_fb.raise_for_status()
        data_fb = r_fb.json()
        hourly_fb = data_fb.get("hourly", {})
        times = hourly_fb.get("time", [])
        swe_raw = hourly_fb.get("snowfall_water_equivalent", [])

        logging.info(f"Archive FB hourly keys: {list(hourly_fb.keys())}")
        logging.info(f"Archive FB sample SWE raw: {swe_raw[:5]}")

        swe_mm = _scale_snow_to_mm(data_fb, swe_raw, "snowfall_water_equivalent")
        snow_mm = [v * 7.0 for v in swe_mm]  # convertir agua->nieve
        return _aggregate_hourly_to_daily(times, snow_mm)

    snow_mm = _scale_snow_to_mm(data, snow_raw, "snowfall")
    return _aggregate_hourly_to_daily(times, snow_mm)


def fetch_snow_forecast_daily(
    lat: float, lon: float, forecast_days: int
) -> List[Tuple[date, float]]:
    """
    Forecast: pedir hourly=snowfall y sumar por día.
    Fallback igual que en histórico.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "snowfall",
        "timezone": TIMEZONE,
        "forecast_days": forecast_days,
    }
    logging.info(f"Forecast API params: {params}")
    r = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    snow_raw = hourly.get("snowfall", [])

    logging.info(f"Forecast hourly keys: {list(hourly.keys())}")
    logging.info(f"Forecast sample snowfall raw: {snow_raw[:5]}")

    if _all_zero_or_none(snow_raw):
        logging.warning(
            "Forecast snowfall vino vacío / todo 0. Reintentando con "
            "'snowfall_water_equivalent' y factor 7."
        )
        params_fb = dict(params)
        params_fb["hourly"] = "snowfall_water_equivalent"
        r_fb = requests.get(OPEN_METEO_FORECAST_URL, params=params_fb, timeout=60)
        r_fb.raise_for_status()
        data_fb = r_fb.json()
        hourly_fb = data_fb.get("hourly", {})
        times = hourly_fb.get("time", [])
        swe_raw = hourly_fb.get("snowfall_water_equivalent", [])

        logging.info(f"Forecast FB hourly keys: {list(hourly_fb.keys())}")
        logging.info(f"Forecast FB sample SWE raw: {swe_raw[:5]}")

        swe_mm = _scale_snow_to_mm(data_fb, swe_raw, "snowfall_water_equivalent")
        snow_mm = [v * 7.0 for v in swe_mm]
        return _aggregate_hourly_to_daily(times, snow_mm)

    snow_mm = _scale_snow_to_mm(data, snow_raw, "snowfall")
    return _aggregate_hourly_to_daily(times, snow_mm)


# =======================================================
# MAIN PIPELINE
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
    logging.info(f"Hoy usado: {today}")
    logging.info(f"Target 1 año: {target_end}")
    logging.info(f"Coords: lat={lat}, lon={lon}")
    logging.info(f"Modelo histórico usado: {MODEL_ARCHIVE}")

    conn = get_connection()
    try:
        # 1) Configuración
        config_id = ensure_configuracion(conn, CONFIG_NOMBRE)

        # 2) Variable exógena NORMAL
        exog_normal_id = ensure_variable_exogena_normal(conn, EXOGENA_NOMBRE, config_id)

        # 3) Variable espejo DUMMIE (FK actual)
        exog_dummie_id = ensure_variable_exogena_dummie(conn, EXOGENA_NOMBRE, config_id)

        existing = fetch_existing_dates(conn, exog_dummie_id)

        # A) HISTÓRICO: start -> min(hoy, target_end)
        hist_end = min(today, target_end)
        hist_rows = fetch_snow_archive_daily(lat, lon, start_date, hist_end)

        to_insert_hist = []
        for d, snow_mm in hist_rows:
            if d in existing:
                continue
            # guardamos en mm redondeado a int
            to_insert_hist.append((d, int(round(snow_mm))))

        insert_fecha_valor_bulk(conn, exog_dummie_id, to_insert_hist)
        existing.update([d for d, _ in to_insert_hist])

        # B) FUTURO REAL: mañana -> forecast max
        if today < target_end:
            fc_days = min(FORECAST_DAYS_MAX, (target_end - today).days)
            if fc_days > 0:
                fc_rows = fetch_snow_forecast_daily(lat, lon, fc_days)

                to_insert_fc = []
                for d, snow_mm in fc_rows:
                    if d <= today or d in existing:
                        continue
                    to_insert_fc.append((d, int(round(snow_mm))))

                insert_fecha_valor_bulk(conn, exog_dummie_id, to_insert_fc)

        logging.info(
            f"Pipeline OK. config_id={config_id}, "
            f"exog_normal_id={exog_normal_id}, exog_dummie_id={exog_dummie_id}"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    run_pipeline()
