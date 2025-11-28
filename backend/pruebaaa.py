import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple, Dict

import requests
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_LAT = float(os.getenv("NIEVE_LAT", "-33.35"))
DEFAULT_LON = float(os.getenv("NIEVE_LON", "-70.25"))
TIMEZONE = "America/Santiago"
MODEL_ARCHIVE = os.getenv("NIEVE_MODEL_ARCHIVE", "era5").lower()

FORECAST_DAYS_MAX = 16
FIXED_START_DATE = date(2024, 6, 3)
TODAY_OVERRIDE = date(2025, 11, 25)

# Factor configurable si hay que usar SWE
SNOW_RATIO_FROM_SWE = float(os.getenv("NIEVE_SWE_RATIO", "7.0"))


def _all_zero_or_none(vals: List[Optional[float]]) -> bool:
    if not vals:
        return True
    for v in vals:
        if v is None:
            continue
        if float(v) != 0.0:
            return False
    return True


def _scale_daily_to_mm(
    data_json: dict, values: List[Optional[float]], unit_key: str
) -> List[float]:
    """
    Escala a mm según daily_units y/o precipitation_unit=mm.
    Dejamos esto por seguridad por si algún modelo devuelve cm o m.
    """
    unit = (data_json.get("daily_units", {}).get(unit_key) or "").lower()

    if unit in ["m", "meter", "meters"]:
        factor = 1000.0
    elif unit in ["cm", "centimeter", "centimeters"]:
        factor = 10.0
    elif unit in ["mm", "millimeter", "millimeters", ""]:
        factor = 1.0
    else:
        factor = 1.0

    logging.info(f"Unidad diaria {unit_key}='{unit}' -> factor_to_mm={factor}")
    return [(float(v) if v is not None else 0.0) * factor for v in values]


def fetch_daily_snowfall(
    url: str,
    lat: float,
    lon: float,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    forecast_days: Optional[int] = None,
    models: Optional[str] = None,
    use_swe_fallback: bool = True,
) -> List[Tuple[date, float]]:
    """
    Consulta diaria directa a Open-Meteo:
    - primero snowfall_sum
    - si no hay/está todo 0 y use_swe_fallback=True -> usa SWE * ratio
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": TIMEZONE,
        "daily": "snowfall_sum",
        "precipitation_unit": "mm",  # fuerza unidad estándar
    }
    if start_date and end_date:
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()
    if forecast_days is not None:
        params["forecast_days"] = forecast_days
    if models:
        params["models"] = models

    logging.info(f"Daily API params ({url}): {params}")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily", {})
    times = daily.get("time", [])
    snow_raw = daily.get("snowfall_sum", [])

    if not times:
        logging.warning("No llegó daily.time. Respuesta extraña.")
        return []

    # Escalar a mm si hace falta
    snow_mm = _scale_daily_to_mm(data, snow_raw, "snowfall_sum")

    # Fallback uniforme a SWE si snowfall viene vacío/0
    if use_swe_fallback and _all_zero_or_none(snow_raw):
        logging.warning(
            "snowfall_sum vacío / todo 0. "
            f"Usando snowfall_water_equivalent_sum * {SNOW_RATIO_FROM_SWE}"
        )
        params_fb = dict(params)
        params_fb["daily"] = "snowfall_water_equivalent_sum"
        r_fb = requests.get(url, params=params_fb, timeout=60)
        r_fb.raise_for_status()
        data_fb = r_fb.json()

        daily_fb = data_fb.get("daily", {})
        times = daily_fb.get("time", times)
        swe_raw = daily_fb.get("snowfall_water_equivalent_sum", [])

        swe_mm = _scale_daily_to_mm(data_fb, swe_raw, "snowfall_water_equivalent_sum")
        snow_mm = [v * SNOW_RATIO_FROM_SWE for v in swe_mm]

    out: List[Tuple[date, float]] = []
    for t, v in zip(times, snow_mm):
        d = datetime.strptime(t, "%Y-%m-%d").date()
        out.append((d, float(v)))

    return out


def plot_snow_series(
    hist_rows, fc_rows, today, title="Nieve diaria (histórico + forecast)"
):
    hist_dates = [d for d, _ in hist_rows]
    hist_vals = [v for _, v in hist_rows]

    fc_dates = [d for d, _ in fc_rows if d > today]
    fc_vals = [v for d, v in fc_rows if d > today]

    plt.figure(figsize=(14, 6))
    if hist_dates:
        plt.bar(hist_dates, hist_vals, label="Histórico (daily)", alpha=0.85)
    if fc_dates:
        plt.bar(fc_dates, fc_vals, label="Forecast (daily)", alpha=0.5)

    all_dates = hist_dates + fc_dates
    if all_dates and min(all_dates) <= today <= max(all_dates):
        plt.axvline(today, linestyle="--", linewidth=1, label=f"Hoy {today}")

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Nieve (mm/día)")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    # Histórico diario directo
    hist_rows = fetch_daily_snowfall(
        url=OPEN_METEO_ARCHIVE_URL,
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=hist_end,
        models=MODEL_ARCHIVE,
        use_swe_fallback=True,  # igual para ambos
    )

    # Forecast diario directo
    fc_rows: List[Tuple[date, float]] = []
    if today < target_end:
        fc_days = min(FORECAST_DAYS_MAX, (target_end - today).days)
        if fc_days > 0:
            fc_rows = fetch_daily_snowfall(
                url=OPEN_METEO_FORECAST_URL,
                lat=lat,
                lon=lon,
                forecast_days=fc_days,
                use_swe_fallback=True,  # misma lógica
            )

    plot_snow_series(hist_rows, fc_rows, today)


if __name__ == "__main__":
    run_pipeline_plot_only()
