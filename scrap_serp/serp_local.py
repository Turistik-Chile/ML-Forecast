import asyncio
import logging
import pyodbc
import time
from datetime import datetime, timedelta
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
import db_connection

# === Cargar variables de entorno ===
load_dotenv()
serp_key = os.getenv("SERP_KEY")

if not serp_key:
    raise EnvironmentError("❌ No se encontró SERP_KEY en el archivo .env")

# === Configuración de logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# === Utilidades ===
def epoch_to_timestamp(ts):
    return datetime.fromtimestamp(ts)


def datetime_format(date):
    return datetime.strptime(date.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")


def get_db_connection_with_retry(max_retries=3, wait_seconds=5):
    """Intenta reconectarse a la base de datos si hay errores."""
    for intento in range(1, max_retries + 1):
        try:
            conexion = db_connection.get_connection()
            logging.info("✅ Conexión a base de datos establecida correctamente.")
            return conexion
        except pyodbc.Error as e:
            logging.warning(
                f"⚠️ Error de conexión (intento {intento}/{max_retries}): {e}"
            )
            if intento < max_retries:
                time.sleep(wait_seconds)
            else:
                logging.error("❌ No se pudo conectar tras varios intentos.")
                raise


def save_to_csv(consultas, vuelos, precios, fecha_periodo):
    """Guarda los datos en CSV en caso de error con la BD."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if consultas:
            df_c = pd.DataFrame(
                consultas,
                columns=[
                    "search_id",
                    "status",
                    "json_endpoint",
                    "created_at",
                    "processed_at",
                    "google_flights_url",
                    "raw_html_file",
                    "prettify_html_file",
                    "total_time_taken",
                    "engine",
                    "departure_id",
                    "arrival_id",
                    "outbound_date",
                    "currency",
                    "lowest_price",
                    "price_level",
                    "typical_price_low",
                    "typical_price_high",
                ],
            )
            df_c.to_csv(f"consultas_serp_{fecha_periodo}_{timestamp}.csv", index=False)
            logging.info(
                f"📄 Backup creado: consultas_serp_{fecha_periodo}_{timestamp}.csv"
            )

        if vuelos:
            df_v = pd.DataFrame(
                vuelos,
                columns=[
                    "search_id",
                    "total_duration",
                    "carbon_emissions_this",
                    "carbon_emissions_typical",
                    "carbon_emissions_diff_percent",
                    "price",
                    "type",
                    "airline_logo",
                    "flight_number",
                    "airplane",
                    "airline",
                    "departure_id",
                    "departure_name",
                    "departure_date",
                    "arrival_id",
                    "arrival_name",
                    "arrival_date",
                    "clase_vuelo",
                    "legroom",
                ],
            )
            df_v.to_csv(f"vuelos_serp_{fecha_periodo}_{timestamp}.csv", index=False)
            logging.info(
                f"📄 Backup creado: vuelos_serp_{fecha_periodo}_{timestamp}.csv"
            )

        if precios:
            df_p = pd.DataFrame(precios, columns=["search_id", "date", "price"])
            df_p.to_csv(f"precios_serp_{fecha_periodo}_{timestamp}.csv", index=False)
            logging.info(
                f"📄 Backup creado: precios_serp_{fecha_periodo}_{timestamp}.csv"
            )

    except Exception as e:
        logging.error(f"❌ Error guardando CSV de respaldo: {e}")


# === Parámetros de ejecución ===
junio_1_2026 = datetime(2026, 6, 1)
agosto_1_2026 = datetime(2026, 8, 1)
aeropuertos_brasil = ["GRU", "GIG"]


# === Función de búsqueda ===
async def fetch_search(params, sem):
    """Ejecuta una búsqueda SERP con límite de concurrencia."""
    async with sem:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: GoogleSearch(params).get_dict()
            )
        except Exception as e:
            logging.error(
                f"Error en búsqueda ({params.get('departure_id')}, {params.get('outbound_date')}): {e}"
            )
            return None


# === Procesamiento ===
async def procesar_periodo(conexion, fecha_inicio, dias):
    cursor = conexion.cursor()
    base_params = {
        "engine": "google_flights",
        "arrival_id": "SCL",
        "currency": "CLP",
        "hl": "es",
        "type": 2,
        "api_key": serp_key,
    }

    tareas = []
    sem = asyncio.Semaphore(10)
    for i in range(dias):
        dia_salida = (fecha_inicio + timedelta(days=i)).strftime("%Y-%m-%d")
        for aeropuerto in aeropuertos_brasil:
            params = base_params.copy()
            params.update({"departure_id": aeropuerto, "outbound_date": dia_salida})
            tareas.append(fetch_search(params, sem))

    logging.info(f"🚀 Ejecutando {len(tareas)} búsquedas concurrentes...")
    resultados_serp = await asyncio.gather(*tareas)
    resultados_serp = [r for r in resultados_serp if r]
    logging.info(f"✅ Recibidos {len(resultados_serp)} resultados válidos")

    consultas, vuelos, precios = [], [], []

    for json_serp in resultados_serp:
        try:
            meta = json_serp.get("search_metadata", {})
            params = json_serp.get("search_parameters", {})
            price_insights = json_serp.get("price_insights", {})
            logging.info(meta.get("id", ""))
            consultas.append(
                (
                    meta.get("id", ""),
                    meta.get("status", ""),
                    meta.get("json_endpoint", ""),
                    datetime_format(meta.get("created_at", "1970-01-01 00:00:00 UTC")),
                    datetime_format(
                        meta.get("processed_at", "1970-01-01 00:00:00 UTC")
                    ),
                    meta.get("google_flights_url", ""),
                    meta.get("raw_html_file", ""),
                    meta.get("prettify_html_file", ""),
                    meta.get("total_time_taken", 0),
                    params.get("engine", ""),
                    params.get("departure_id", ""),
                    params.get("arrival_id", ""),
                    params.get("outbound_date", ""),
                    params.get("currency", ""),
                    price_insights.get("lowest_price", 0),
                    price_insights.get("price_level", ""),
                    price_insights.get("typical_price_range", [0, 0])[0],
                    price_insights.get("typical_price_range", [0, 0])[1],
                )
            )

            for ph in price_insights.get("price_history", []):
                ts = ph[0] if len(ph) > 0 else 0
                price = ph[1] if len(ph) > 1 else 0
                precios.append((meta.get("id", ""), epoch_to_timestamp(ts), price))

            for category in ["best_flights", "other_flights"]:
                for option in json_serp.get(category, []):
                    for vuelo in option.get("flights", []):
                        vuelos.append(
                            (
                                meta.get("id", ""),
                                option.get("total_duration", ""),
                                option.get("carbon_emissions", {}).get(
                                    "this_flight", 0
                                ),
                                option.get("carbon_emissions", {}).get(
                                    "typical_for_this_route", 0
                                ),
                                option.get("carbon_emissions", {}).get(
                                    "difference_percent", 0
                                ),
                                option.get("price", 0),
                                option.get("type", ""),
                                option.get("airline_logo", ""),
                                vuelo.get("flight_number", ""),
                                vuelo.get("airplane", ""),
                                vuelo.get("airline", ""),
                                vuelo.get("departure_airport", {}).get("id", ""),
                                vuelo.get("departure_airport", {}).get("name", ""),
                                vuelo.get("departure_airport", {}).get("time", ""),
                                vuelo.get("arrival_airport", {}).get("id", ""),
                                vuelo.get("arrival_airport", {}).get("name", ""),
                                vuelo.get("arrival_airport", {}).get("time", ""),
                                vuelo.get("travel_class", ""),
                                vuelo.get("legroom", ""),
                            )
                        )
        except Exception as e:
            logging.error(f"Error procesando resultado SERP: {e}")

    try:
        if consultas:
            cursor.executemany(
                """
                INSERT INTO ia.consultas_serp (
                    search_id, status, json_endpoint, created_at, processed_at,
                    google_flights_url, raw_html_file, prettify_html_file,
                    total_time_taken, engine, departure_id, arrival_id,
                    outbound_date, currency, lowest_price, price_level,
                    typical_price_low, typical_price_high
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                consultas,
            )

        if vuelos:
            cursor.executemany(
                """
                INSERT INTO ia.vuelos_serp (
                    search_id, total_duration, carbon_emissions_this,
                    carbon_emissions_typical, carbon_emissions_diff_percent, price,
                    type, airline_logo, flight_number, airplane, airline,
                    departure_id, departure_name, departure_date,
                    arrival_id, arrival_name, arrival_date,
                    clase_vuelo, legroom
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                vuelos,
            )

        if precios:
            cursor.executemany(
                """
                INSERT INTO ia.precios_dia_serp (search_id, date, price)
                VALUES (?,?,?)
            """,
                precios,
            )

        conexion.commit()
        logging.info(
            f"💾 Insertados {len(consultas)} consultas, {len(vuelos)} vuelos y {len(precios)} precios "
            f"para {fecha_inicio.strftime('%Y-%m')}"
        )

    except pyodbc.Error as e:
        logging.error(f"❌ Error al insertar datos: {e}")
        conexion.rollback()
        save_to_csv(consultas, vuelos, precios, fecha_inicio.strftime("%Y-%m"))


# === Main ===
async def main():
    conexion = get_db_connection_with_retry()
    try:
        await asyncio.gather(
            procesar_periodo(conexion, junio_1_2026, 30),
            procesar_periodo(conexion, agosto_1_2026, 31),
        )
        logging.info("✅ Ejecución completada con éxito.")
    except Exception as e:
        logging.exception(f"Error general durante ejecución: {e}")
    finally:
        conexion.close()
        logging.info("🔒 Conexión cerrada correctamente.")


if __name__ == "__main__":
    asyncio.run(main())
