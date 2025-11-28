import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List

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

# Coordenadas (ajustables por env)
DEFAULT_LAT = float(os.getenv("NIEVE_LAT", "-33.35"))  # aprox Valle Nevado
DEFAULT_LON = float(os.getenv("NIEVE_LON", "-70.25"))

TIMEZONE = "America/Santiago"
MODEL_ARCHIVE = "era5_land"
FORECAST_DAYS_MAX = (
    16  # límite del endpoint forecast :contentReference[oaicite:2]{index=2}
)

CONFIG_NOMBRE = "Nieve test forecast"
EXOGENA_NOMBRE = "nieve_caida_dia"

# START fijo requerido
FIXED_START_DATE = date(2024, 6, 3)

# "HOY" explícito según tú
# (si quieres que sea dinámico, reemplaza por date.today())
TODAY_OVERRIDE = date(2025, 11, 25)


# -------------------------------------------------------
# DB HELPERS
# -------------------------------------------------------
def ensure_configuracion(conn: pyodbc.Connection, nombre: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM ia.configuracion WHERE nombre = ?", (nombre,))
        row = cur.fetchone()
        if row:
            logging.info(f"Configuración ya existe: {nombre} (id={row.id})")
            return row.id

        cur.execute(
            "INSERT INTO ia.configuracion (nombre, comentarios) VALUES (?, ?)",
            (nombre, None),
        )
        conn.commit()
        config_id = cur.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]
        logging.info(f"Configuración creada: {nombre} (id={config_id})")
        return int(config_id)


def ensure_variable_exogena_normal(
    conn: pyodbc.Connection, nombre: str, configuracion_id: int
) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM ia.variable_exogena WHERE nombre = ?", (nombre,))
        row = cur.fetchone()
        if row:
            logging.info(f"Variable exógena NORMAL ya existe: {nombre} (id={row.id})")
            return row.id

        cur.execute(
            """
            INSERT INTO ia.variable_exogena (nombre, is_active, configuracion_id)
            VALUES (?, 1, ?)
            """,
            (nombre, configuracion_id),
        )
        conn.commit()
        exog_id = cur.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]
        logging.info(f"Variable exógena NORMAL creada: {nombre} (id={exog_id})")
        return int(exog_id)


def ensure_variable_exogena_dummie(
    conn: pyodbc.Connection, nombre: str, configuracion_id: int
) -> int:
    """
    Necesario porque ia.fecha_valor_exogena FK apunta a ia.variable_exogena_dummie(id)
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM ia.variable_exogena_dummie WHERE nombre = ?", (nombre,)
        )
        row = cur.fetchone()
        if row:
            logging.info(f"Variable exógena DUMMIE ya existe: {nombre} (id={row.id})")
            return row.id

        cur.execute(
            """
            INSERT INTO ia.variable_exogena_dummie (nombre, is_active, configuracion_id)
            VALUES (?, 1, ?)
            """,
            (nombre, configuracion_id),
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


# -------------------------------------------------------
# API HELPERS
# -------------------------------------------------------
def fetch_snow_archive_daily(
    lat: float, lon: float, start_date: date, end_date: date
) -> List[Tuple[date, float]]:
    """
    Histórico (reanálisis) desde Archive API.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "snowfall_sum",
        "timezone": TIMEZONE,
        "models": MODEL_ARCHIVE,
    }
    logging.info(f"Archive API: {params}")
    r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    times = data.get("daily", {}).get("time", [])
    snow = data.get("daily", {}).get("snowfall_sum", [])

    out = []
    for t, s in zip(times, snow):
        d = datetime.strptime(t, "%Y-%m-%d").date()
        s_val = float(s) if s is not None else 0.0
        out.append((d, s_val))
    return out


def fetch_snow_forecast_daily(
    lat: float, lon: float, forecast_days: int
) -> List[Tuple[date, float]]:
    """
    Futuro real desde Forecast API (máx ~16 días).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "snowfall_sum",
        "timezone": TIMEZONE,
        "forecast_days": forecast_days,
    }
    logging.info(f"Forecast API: {params}")
    r = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    times = data.get("daily", {}).get("time", [])
    snow = data.get("daily", {}).get("snowfall_sum", [])

    out = []
    for t, s in zip(times, snow):
        d = datetime.strptime(t, "%Y-%m-%d").date()
        s_val = float(s) if s is not None else 0.0
        out.append((d, s_val))
    return out


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def run_pipeline(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    start_date: Optional[date] = None,
    today: Optional[date] = None,
):
    """
    - start fijo 2024-06-03
    - target_end = start + 365 días
    - inserta histórico hasta hoy
    - intenta insertar futuro real hasta forecast max (≈16 días)
    - NO rellena días sin data futura
    """
    if start_date is None:
        start_date = FIXED_START_DATE

    if today is None:
        today = TODAY_OVERRIDE  # tu "HOY" explícito

    target_end = start_date + timedelta(days=365)

    logging.info(f"Start fijo: {start_date}")
    logging.info(f"Hoy usado: {today}")
    logging.info(f"Target 1 año: {target_end}")
    logging.info(f"Coords: lat={lat}, lon={lon}")

    conn = get_connection()
    try:
        # 1) Configuración
        config_id = ensure_configuracion(conn, CONFIG_NOMBRE)

        # 2) Variable exógena NORMAL
        exog_normal_id = ensure_variable_exogena_normal(conn, EXOGENA_NOMBRE, config_id)

        # 3) Variable espejo DUMMIE para FK actual
        exog_dummie_id = ensure_variable_exogena_dummie(conn, EXOGENA_NOMBRE, config_id)

        existing = fetch_existing_dates(conn, exog_dummie_id)

        # -------------------------
        # A) HISTÓRICO: start -> hoy
        # -------------------------
        hist_end = min(today, target_end)
        hist_rows = fetch_snow_archive_daily(lat, lon, start_date, hist_end)

        to_insert = []
        for d, snow_mm in hist_rows:
            if d in existing:
                continue
            to_insert.append((d, int(round(snow_mm))))

        insert_fecha_valor_bulk(conn, exog_dummie_id, to_insert)
        existing.update([d for d, _ in to_insert])

        # ----------------------------------------
        # B) FUTURO REAL: mañana -> forecast max
        # ----------------------------------------
        if today < target_end:
            # API forecast solo da hasta ~16 días :contentReference[oaicite:3]{index=3}
            fc_days = min(FORECAST_DAYS_MAX, (target_end - today).days)
            if fc_days > 0:
                fc_rows = fetch_snow_forecast_daily(lat, lon, fc_days)

                to_insert_fc = []
                for d, snow_mm in fc_rows:
                    if d <= today:
                        continue  # evita duplicar el día de hoy si viniera
                    if d in existing:
                        continue
                    to_insert_fc.append((d, int(round(snow_mm))))

                insert_fecha_valor_bulk(conn, exog_dummie_id, to_insert_fc)
            else:
                logging.info("Target futuro alcanzado; no se solicita forecast.")
        else:
            logging.info(
                "Target (1 año) ya está en el pasado; no hay futuro que cargar."
            )

        logging.info(
            f"Pipeline OK. config_id={config_id}, "
            f"exog_normal_id={exog_normal_id}, exog_dummie_id={exog_dummie_id}"
        )

        # Si corriges la FK a variable_exogena, cambia exog_dummie_id por exog_normal_id arriba.

    finally:
        conn.close()


if __name__ == "__main__":
    run_pipeline()
