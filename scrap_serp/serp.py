from serpapi import GoogleSearch
import db_connection
import os
import time
from datetime import datetime, timedelta, timezone

BASE_PARAMS = {
    "engine": "google_flights",
    "arrival_id": "SCL",
    "currency": "CLP",
    "hl": "es",
    "type": 2,
    "api_key": "f8b5926e9f3674c6cdf272f26a0946814c9c1c471f04bf90fada0d82b234c061",
}

AEROPUERTOS_BRASIL = ["GRU", "GIG", "BSB"]

NUM_DAYS = 365
START_DATE = datetime.now().date() + timedelta(days=1)  # próximos 365 desde mañana

BATCH_DAYS = 14  # tamaño del lote (días). Ej: 7, 14, 30
SLEEP_BETWEEN_REQUESTS = 1  # ajusta si necesitas rate limit
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0  # se multiplica por intento


# -----------------------------
# Logging helper
# -----------------------------
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# -----------------------------
# Helpers (parsing / typing)
# -----------------------------
def none_if_empty(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    return x


def epoch_to_timestamp(timestamp: int) -> datetime:
    try:
        return datetime.fromtimestamp(timestamp)
    except Exception:
        return datetime(1970, 1, 1)


def datetime_format(date_str: str) -> datetime:
    try:
        # SerpAPI suele traer "... UTC"
        fecha = datetime.strptime(date_str.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
        return fecha
    except Exception:
        return datetime(1970, 1, 1)


def parse_dt_or_none(s):
    """
    Convierte strings típicos de SerpAPI a datetime naive (sin tz).
    Si venía con tz, lo normaliza a UTC naive.
    Retorna None si viene vacío o no parsea.
    """
    s = none_if_empty(s)
    if s is None:
        return None

    if isinstance(s, datetime):
        if s.tzinfo is not None:
            return s.astimezone(timezone.utc).replace(tzinfo=None)
        return s

    txt = str(s).strip().replace(" UTC", "")

    # 1) ISO (con/sin offset)
    try:
        iso = txt.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    # 2) Fallbacks típicos
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(txt, fmt)
        except Exception:
            continue

    return None


def parse_date_or_none(s):
    """
    Convierte 'YYYY-mm-dd' a date (datetime.date).
    Retorna None si vacío o inválido.
    """
    s = none_if_empty(s)
    if s is None:
        return None

    # Ya es date (pero no datetime)
    if (
        hasattr(s, "year")
        and hasattr(s, "month")
        and hasattr(s, "day")
        and not isinstance(s, datetime)
    ):
        return s

    txt = str(s).strip()
    try:
        return datetime.strptime(txt, "%Y-%m-%d").date()
    except Exception:
        return None


# -----------------------------
# SerpAPI
# -----------------------------
def serp_request(params: dict, ctx: str) -> dict:
    """Request con retry + logs."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            log(f"{ctx} | SERP attempt {attempt}/{MAX_RETRIES} | requesting...")
            search = GoogleSearch(params)
            data = search.get_dict()
            dt = time.time() - t0

            meta = data.get("search_metadata", {}) or {}
            sid = meta.get("id", "")
            status = meta.get("status", "")

            log(f"{ctx} | SERP OK in {dt:.2f}s | status={status} | search_id={sid}")
            return data

        except Exception as e:
            last_err = e
            backoff = RETRY_BACKOFF_SECONDS * attempt
            log(
                f"{ctx} | SERP ERROR attempt {attempt} -> {e} | sleeping {backoff:.1f}s"
            )
            time.sleep(backoff)

    log(f"{ctx} | SERP FAILED after {MAX_RETRIES} retries | last_err={last_err}")
    raise last_err


def chunk_dates(start_date, num_days, chunk_size):
    """Genera rangos de fechas en chunks."""
    end_date = start_date + timedelta(days=num_days)
    current = start_date
    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_size), end_date)
        yield current, chunk_end
        current = chunk_end


# -----------------------------
# SQL QUERIES
# -----------------------------
SQL_CONSULTAS = """
INSERT INTO ia.consultas_serp (
    search_id,
    status,
    json_endpoint,
    created_at,
    processed_at,
    google_flights_url,
    raw_html_file,
    prettify_html_file,
    total_time_taken,
    engine,
    departure_id,
    arrival_id,
    outbound_date,
    currency,
    lowest_price,
    price_level,
    typical_price_low,
    typical_price_high
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
"""

SQL_VUELOS_ESCALA = """
INSERT INTO ia.vuelos_serp (
    search_id,
    total_duration,
    carbon_emissions_this,
    carbon_emissions_typical,
    carbon_emissions_diff_percent,
    price,
    type,
    airline_logo,
    flight_number,
    airplane,
    airline,
    departure_id,
    departure_name,
    departure_date,
    arrival_id,
    arrival_name,
    arrival_date,
    intermedia,
    clase_vuelo,
    legroom
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

SQL_VUELOS_DIRECTO = """
INSERT INTO ia.vuelos_serp (
    search_id,
    total_duration,
    carbon_emissions_this,
    carbon_emissions_typical,
    carbon_emissions_diff_percent,
    price,
    type,
    airline_logo,
    flight_number,
    airplane,
    airline,
    departure_id,
    departure_name,
    departure_date,
    arrival_id,
    arrival_name,
    arrival_date,
    clase_vuelo,
    legroom
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

SQL_PRECIOS_DIA = """
INSERT INTO ia.precios_dia_serp (search_id, date, price)
VALUES (?,?,?)
"""


# -----------------------------
# Scraper
# -----------------------------
def run_scrap_next_365_days():
    log("START script")

    log("DB connecting...")
    conexion = db_connection.get_connection()
    log("DB connected OK")

    cursor = conexion.cursor()
    cursor.fast_executemany = True
    log("DB cursor OK (fast_executemany=True)")

    total_requests = 0
    last_ctx = None  # para reportar en error

    try:
        for chunk_start, chunk_end in chunk_dates(START_DATE, NUM_DAYS, BATCH_DAYS):
            # Buffers para batching de inserts
            consultas_rows = []
            vuelos_escala_rows = []
            vuelos_directo_rows = []
            precios_rows = []

            log(
                f"=== Batch START: {chunk_start} -> {chunk_end - timedelta(days=1)} ==="
            )

            day = chunk_start
            while day < chunk_end:
                for aeropuerto in AEROPUERTOS_BRASIL:
                    ctx = f"[day={day.strftime('%Y-%m-%d')} dep={aeropuerto}]"
                    last_ctx = ctx

                    params = dict(BASE_PARAMS)
                    params.update(
                        {
                            "departure_id": aeropuerto,
                            "outbound_date": day.strftime("%Y-%m-%d"),
                        }
                    )

                    # ---- SERP ----
                    json_serp = serp_request(params, ctx)
                    total_requests += 1

                    # (rate limit)
                    if SLEEP_BETWEEN_REQUESTS:
                        log(f"{ctx} | sleep {SLEEP_BETWEEN_REQUESTS}s")
                        time.sleep(SLEEP_BETWEEN_REQUESTS)

                    # ---- Parse ----
                    search_metadata = json_serp.get("search_metadata", {}) or {}
                    search_parameters = json_serp.get("search_parameters", {}) or {}
                    price_insights = json_serp.get("price_insights", {}) or {}

                    search_id = search_metadata.get("id", "")
                    status = search_metadata.get("status", "")
                    json_endpoint = search_metadata.get("json_endpoint", "")
                    created_at = datetime_format(
                        search_metadata.get("created_at", "1970-01-01 00:00:00")
                    )
                    processed_at = datetime_format(
                        search_metadata.get("processed_at", "1970-01-01 00:00:00")
                    )
                    google_flights_url = search_metadata.get("google_flights_url", "")
                    raw_html_url = search_metadata.get("raw_html_file", "")
                    prettify_html_url = search_metadata.get("prettify_html_file", "")
                    total_time_taken = search_metadata.get("total_time_taken", 0.0)

                    engine = search_parameters.get("engine", "")
                    departure_id = search_parameters.get("departure_id", "")
                    arrival_id = search_parameters.get("arrival_id", "")
                    outbound_date = parse_date_or_none(
                        search_parameters.get("outbound_date", "")
                    )
                    currency = search_parameters.get("currency", "")

                    lowest_price = price_insights.get("lowest_price", 0) or 0
                    price_level = price_insights.get("price_level", "") or ""
                    typical_range = price_insights.get(
                        "typical_price_range", [0, 0]
                    ) or [
                        0,
                        0,
                    ]
                    typical_price_low = (
                        typical_range[0] if len(typical_range) > 0 else 0
                    )
                    typical_price_high = (
                        typical_range[1] if len(typical_range) > 1 else 0
                    )

                    # ORDEN: igual al SQL_CONSULTAS (18 cols)
                    consultas_rows.append(
                        (
                            search_id,
                            status,
                            json_endpoint,
                            created_at,
                            processed_at,
                            google_flights_url,
                            raw_html_url,
                            prettify_html_url,
                            total_time_taken,
                            engine,
                            departure_id,
                            arrival_id,
                            outbound_date,
                            currency,
                            lowest_price,
                            price_level,
                            typical_price_low,
                            typical_price_high,
                        )
                    )

                    # ---- Flights ----
                    added_escala = 0
                    added_directo = 0

                    for category in ("best_flights", "other_flights"):
                        for option in json_serp.get(category, []) or []:
                            flights = option.get("flights", []) or []
                            if not flights:
                                continue

                            precio = option.get("price", 0) or 0
                            tipo_vuelo = option.get("type", "") or ""
                            logo_aerolinea = option.get("airline_logo", "") or ""
                            duracion_total = option.get("total_duration", 0) or 0

                            emissions = option.get("carbon_emissions", {}) or {}
                            emission_this = emissions.get("this_flight", 0) or 0
                            emission_normal = (
                                emissions.get("typical_for_this_route", 0) or 0
                            )
                            percent_diff_emissions = (
                                emissions.get("difference_percent", 0) or 0
                            )

                            if len(flights) >= 2:
                                first = flights[0]
                                last = flights[-1]

                                aeropuerto_salida = (
                                    first.get("departure_airport", {}) or {}
                                ).get("name", "")
                                aeropuerto_salida_id = (
                                    first.get("departure_airport", {}) or {}
                                ).get("id", "")
                                aeropuerto_salida_fecha = parse_dt_or_none(
                                    (first.get("departure_airport", {}) or {}).get(
                                        "time", ""
                                    )
                                )

                                aeropuerto_llegada = (
                                    last.get("arrival_airport", {}) or {}
                                ).get("name", "")
                                aeropuerto_llegada_id = (
                                    last.get("arrival_airport", {}) or {}
                                ).get("id", "")
                                aeropuerto_llegada_fecha = parse_dt_or_none(
                                    (last.get("arrival_airport", {}) or {}).get(
                                        "time", ""
                                    )
                                )

                                parada_intermedia_id = (
                                    first.get("arrival_airport", {}) or {}
                                ).get("id", "")

                                aeronave = first.get("airplane", "") or ""
                                aerolinea = first.get("airline", "") or ""
                                clase_vuelo = first.get("travel_class", "") or ""
                                codigo_vuelo = first.get("flight_number", "") or ""
                                legroom = first.get("legroom", "") or ""

                                # ORDEN: igual al SQL_VUELOS_ESCALA (20 cols)
                                vuelos_escala_rows.append(
                                    (
                                        search_id,
                                        duracion_total,
                                        emission_this,
                                        emission_normal,
                                        percent_diff_emissions,
                                        precio,
                                        tipo_vuelo,
                                        logo_aerolinea,
                                        codigo_vuelo,
                                        aeronave,
                                        aerolinea,
                                        aeropuerto_salida_id,
                                        aeropuerto_salida,
                                        aeropuerto_salida_fecha,  # datetime|None
                                        aeropuerto_llegada_id,
                                        aeropuerto_llegada,
                                        aeropuerto_llegada_fecha,  # datetime|None
                                        parada_intermedia_id,
                                        clase_vuelo,
                                        legroom,
                                    )
                                )
                                added_escala += 1
                            else:
                                vuelo = flights[0]

                                aeropuerto_salida = (
                                    vuelo.get("departure_airport", {}) or {}
                                ).get("name", "")
                                aeropuerto_salida_id = (
                                    vuelo.get("departure_airport", {}) or {}
                                ).get("id", "")
                                aeropuerto_salida_fecha = parse_dt_or_none(
                                    (vuelo.get("departure_airport", {}) or {}).get(
                                        "time", ""
                                    )
                                )

                                aeropuerto_llegada = (
                                    vuelo.get("arrival_airport", {}) or {}
                                ).get("name", "")
                                aeropuerto_llegada_id = (
                                    vuelo.get("arrival_airport", {}) or {}
                                ).get("id", "")
                                aeropuerto_llegada_fecha = parse_dt_or_none(
                                    (vuelo.get("arrival_airport", {}) or {}).get(
                                        "time", ""
                                    )
                                )

                                aeronave = vuelo.get("airplane", "") or ""
                                aerolinea = vuelo.get("airline", "") or ""
                                clase_vuelo = vuelo.get("travel_class", "") or ""
                                codigo_vuelo = vuelo.get("flight_number", "") or ""
                                legroom = vuelo.get("legroom", "") or ""

                                # ORDEN: igual al SQL_VUELOS_DIRECTO (19 cols)
                                vuelos_directo_rows.append(
                                    (
                                        search_id,
                                        duracion_total,
                                        emission_this,
                                        emission_normal,
                                        percent_diff_emissions,
                                        precio,
                                        tipo_vuelo,
                                        logo_aerolinea,
                                        codigo_vuelo,
                                        aeronave,
                                        aerolinea,
                                        aeropuerto_salida_id,
                                        aeropuerto_salida,
                                        aeropuerto_salida_fecha,  # datetime|None
                                        aeropuerto_llegada_id,
                                        aeropuerto_llegada,
                                        aeropuerto_llegada_fecha,  # datetime|None
                                        clase_vuelo,
                                        legroom,
                                    )
                                )
                                added_directo += 1

                    # ---- Price history ----
                    added_precios = 0
                    for item in price_insights.get("price_history", []) or []:
                        # item: [epoch, price]
                        if not item or len(item) < 2:
                            continue
                        fecha = epoch_to_timestamp(item[0])
                        price = item[1] or 0
                        precios_rows.append((search_id, fecha, price))
                        added_precios += 1

                    log(
                        f"{ctx} | buffered: consultas={len(consultas_rows)} "
                        f"vuelos_escala(+{added_escala})={len(vuelos_escala_rows)} "
                        f"vuelos_directo(+{added_directo})={len(vuelos_directo_rows)} "
                        f"precios(+{added_precios})={len(precios_rows)} | total_requests={total_requests}"
                    )

                day += timedelta(days=1)

            # -----------------------------
            # DB inserts (batch)
            # -----------------------------
            log(
                f"Batch INSERT START | consultas={len(consultas_rows)} "
                f"vuelos_escala={len(vuelos_escala_rows)} vuelos_directo={len(vuelos_directo_rows)} "
                f"precios={len(precios_rows)}"
            )

            if consultas_rows:
                t0 = time.time()
                log("DB executemany SQL_CONSULTAS START")
                cursor.executemany(SQL_CONSULTAS, consultas_rows)
                log(f"DB executemany SQL_CONSULTAS OK in {time.time()-t0:.2f}s")

            if vuelos_escala_rows:
                t0 = time.time()
                log("DB executemany SQL_VUELOS_ESCALA START")
                cursor.executemany(SQL_VUELOS_ESCALA, vuelos_escala_rows)
                log(f"DB executemany SQL_VUELOS_ESCALA OK in {time.time()-t0:.2f}s")

            if vuelos_directo_rows:
                t0 = time.time()
                log("DB executemany SQL_VUELOS_DIRECTO START")
                cursor.executemany(SQL_VUELOS_DIRECTO, vuelos_directo_rows)
                log(f"DB executemany SQL_VUELOS_DIRECTO OK in {time.time()-t0:.2f}s")

            if precios_rows:
                t0 = time.time()
                log("DB executemany SQL_PRECIOS_DIA START")
                cursor.executemany(SQL_PRECIOS_DIA, precios_rows)
                log(f"DB executemany SQL_PRECIOS_DIA OK in {time.time()-t0:.2f}s")

            t0 = time.time()
            log("DB commit START")
            conexion.commit()
            log(f"DB commit OK in {time.time()-t0:.2f}s")

            log(
                f"=== Batch OK === requests_acum={total_requests} | "
                f"consultas={len(consultas_rows)} vuelos_escala={len(vuelos_escala_rows)} "
                f"vuelos_directo={len(vuelos_directo_rows)} precios={len(precios_rows)}"
            )

    except Exception as e:
        log(f"!!! ERROR !!! last_ctx={last_ctx} err={e}")
        try:
            log("DB rollback START")
            conexion.rollback()
            log("DB rollback OK")
        except Exception as rb_e:
            log(f"DB rollback FAILED err={rb_e}")
        raise
    finally:
        try:
            log("DB close START")
            conexion.close()
            log("DB close OK")
        except Exception as close_e:
            log(f"DB close FAILED err={close_e}")


if __name__ == "__main__":
    run_scrap_next_365_days()
