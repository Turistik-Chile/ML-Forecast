import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List, Dict

import requests
import matplotlib.pyplot as plt

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

DEFAULT_LAT = float(os.getenv("NIEVE_LAT", "-33.35"))  # aprox Valle Nevado
DEFAULT_LON = float(os.getenv("NIEVE_LON", "-70.25"))

TIMEZONE = "America/Santiago"
MODEL_ARCHIVE = os.getenv("NIEVE_MODEL_ARCHIVE", "era5").lower()
FORECAST_DAYS_MAX = 16

FIXED_START_DATE = date(2024, 6, 3)
TODAY_OVERRIDE = date(2025, 11, 25)


# =======================================================
# API HELPERS
# =======================================================
def _scale_snow_to_mm(
    data_json: dict, snow_values: List[Optional[float]], unit_key: str
) -> List[float]:
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
        factor = 1.0

    logging.info(f"Unidad {unit_key}='{unit}' -> factor_to_mm={factor}")
    return [(float(s) if s is not None else 0.0) * factor for s in snow_values]


def _aggregate_hourly_to_daily(
    times: List[str], vals_mm: List[float]
) -> List[Tuple[date, float]]:
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

    if _all_zero_or_none(snow_raw):
        logging.warning(
            "Archive snowfall vacío / todo 0. Reintentando con "
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

        swe_mm = _scale_snow_to_mm(data_fb, swe_raw, "snowfall_water_equivalent")
        snow_mm = [v * 7.0 for v in swe_mm]
        return _aggregate_hourly_to_daily(times, snow_mm)

    snow_mm = _scale_snow_to_mm(data, snow_raw, "snowfall")
    return _aggregate_hourly_to_daily(times, snow_mm)


def fetch_snow_forecast_daily(
    lat: float, lon: float, forecast_days: int
) -> List[Tuple[date, float]]:
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
            "Forecast snowfall vacío / todo 0. Reintentando con "
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

        swe_mm = _scale_snow_to_mm(data_fb, swe_raw, "snowfall_water_equivalent")
        snow_mm = [v * 7.0 for v in swe_mm]
        return _aggregate_hourly_to_daily(times, snow_mm)

    snow_mm = _scale_snow_to_mm(data, snow_raw, "snowfall")
    return _aggregate_hourly_to_daily(times, snow_mm)


# =======================================================
# PLOTTING
# =======================================================
def plot_snow_series(
    hist_rows: List[Tuple[date, float]],
    fc_rows: List[Tuple[date, float]],
    today: date,
    title: str = "Nieve caída diaria (histórico + forecast)",
):
    # Separar a listas
    hist_dates = [d for d, _ in hist_rows]
    hist_vals = [v for _, v in hist_rows]

    fc_dates = [d for d, _ in fc_rows if d > today]
    fc_vals = [v for d, v in fc_rows if d > today]

    plt.figure(figsize=(14, 6))
    # Histórico como barras sólidas
    if hist_dates:
        plt.bar(hist_dates, hist_vals, label="Histórico (archive)", alpha=0.8)

    # Forecast como barras con borde (o línea si prefieres)
    if fc_dates:
        plt.bar(fc_dates, fc_vals, label="Forecast", alpha=0.5)

    # Línea vertical de "hoy"
    plt.axvline(today, linestyle="--", linewidth=1, label=f"Hoy {today}")

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Nieve (mm/día)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =======================================================
# MAIN
# =======================================================
def run_pipeline_plot_only(
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
    hist_end = min(today, target_end)

    logging.info(f"Start fijo: {start_date}")
    logging.info(f"Hoy usado: {today}")
    logging.info(f"Target 1 año: {target_end}")
    logging.info(f"Coords: lat={lat}, lon={lon}")
    logging.info(f"Modelo histórico usado: {MODEL_ARCHIVE}")

    # A) HISTÓRICO
    hist_rows = fetch_snow_archive_daily(lat, lon, start_date, hist_end)

    # B) FORECAST (mañana → max forecast)
    fc_rows: List[Tuple[date, float]] = []
    if today < target_end:
        fc_days = min(FORECAST_DAYS_MAX, (target_end - today).days)
        if fc_days > 0:
            fc_rows = fetch_snow_forecast_daily(lat, lon, fc_days)

    # Plot combinado
    plot_snow_series(hist_rows, fc_rows, today)


if __name__ == "__main__":
    run_pipeline_plot_only()
