# Flujo del pipeline de turismo

Este documento describe qué ocurre cuando se activa el pipeline de turismo en el backend, qué módulos se llaman y qué artefactos se generan antes de devolver un pronóstico. El punto de entrada puede ser el CLI (`main.py`) o el endpoint `/forecast` de FastAPI; ambos convergen en `ejecutar_forecast` y, finalmente, en `TurismoPipeline`.

## 1. Punto de entrada y configuración

- **CLI/API** (`main.py`): el flag `--pipeline turismo` o `ForecastRequest.pipeline = "turismo"` inicia el flujo. El request agrupa parámetros (categorías, rango de fechas, lags, `configuration_id`) y llama a `build_pipeline`, que consulta el registro `PIPELINE_REGISTRY` y ejecuta `build_turismo_pipeline`.
- **`build_turismo_pipeline`**: normaliza fechas (`_coerce_date`, `_sanitize_selection_dates`), firma los tokens SQL permitidos y valida que la fecha final no supere `TURISMO_COMPARISON_DAYS` antes de hoy. También solicita las exógenas asociadas a `configuration_id` usando `cargar_exogenas`, capturando advertencias si fallan.
- El resultado es una instancia `TurismoPipelineConfig`, que alimenta a `TurismoPipeline`. Esta clase carga la serie, la divide entre historia y comparación y prepara los datos finales.

## 2. Extracción de datos de turismo

- **`fetch_turismo_series`** se encarga de la SQL real. Sanitiza identificadores SQL, arma parámetros (`start_date`, `end_date`, categorías filtradas) y ejecuta la consulta con `db_connection.get_connection`. El resultado se agrupa por fecha, se convierte a datetime y se estandariza con `_standardize_df`.
- `_standardize_df` limpia duplicados, ordena el índice, fuerza la columna `valor` a float y nombra el índice como `fecha`. `_ensure_daily_range` garantiza que no haya huecos entre el primer y último día, rellenando con ceros cuando falta información.

## 3. Construcción de history/test y validaciones

- `TurismoPipeline.prepare` fija un `cutoff_dt` y extiende el rango hasta `future_end = cutoff_dt + comparison_days` (1 año por defecto). Divide la tabla completa en historia (`df.index <= cutoff_dt`) y comparación (`cutoff_dt+1` a `future_end`).
- Se valida que la historia tenga datos y que la comparación cubra todos los días esperados; en caso contrario se lanza `ValueError`, lo que se refleja como error 400 en la API.
- La historia se rellena hacia atrás con `_ensure_daily_range` para soportar lags, y el muestreo de comparación también se asegura completo.

## 4. Preparación de características

- `normalize_lag_list` ordena los lags solicitados, evita duplicados y exige enteros positivos.
- `pad_history_for_lags` agrega ceros al inicio si el histórico corto no alcanza el máximo lag. Limita el relleno a `MAX_ZERO_PADDING_DAYS` (30) y devuelve el número de días añadidos (`zero_padding_days`).
- `align_exogenas_for_index` sincroniza las exógenas (si existen) con los índices de history, comparación y serie completa. Conserva atributos como `expected_exogenas`.
- `preparar_features` añade columnas de fecha (`dia_semana`, `month`, `sin_month`, etc.), junta las exógenas alineadas y calcula los lags (`lag_{n}`). El método elimina filas con `NaN` generados por los desplazamientos y separa `X_train`/`y_train`.
- La función `log_exogena_coverage` reporta qué columnas de exógenas están presentes y alerta si faltan en history o comparison; se llama dos veces: desde `TurismoPipeline.prepare` (contexto `prepare_turismo`) y desde `ejecutar_forecast` antes del entrenamiento (`{pipeline}/{modelo}:pre-train`).

## 5. Datos preparados devueltos

- `PreparedData` empaqueta:
  - `X_train`, `y_train`, `feature_columns`: las características ya filtradas por `preparar_features`.
  - `history`: valores con padding y columnas exógenas alineadas (base para entrenamiento y gráficas).
  - `df_real`: valores reales del periodo de comparación (con exógenas si hay).
  - `train_range`/`test_range`: tuplas con start/end en `YYYY-MM-DD` para logging y comentarios.
  - `full_history`/`exogenas`: historial completo con exógenas, útil para que algunos modelos aprovechen todo el rango.
  - `zero_padding_days` y `plot_comment`.
- `build_plot_comment` genera el texto que se incrusta en la gráfica (“Turismo | Categorías: … | Entrenamiento: … | Test: …”).

## 6. Entrenamiento y pronóstico

- `ejecutar_forecast` invoca:
  1. `pipeline.prepare()` para obtener `PreparedData`.
  2. `log_exogena_coverage` para verificar (`data.exogenas`, `data.history`, `data.df_real`).
  3. `_dump_prepared_dataframes` para persistir tablas clave en `logs/dataframes/{modelo}_{timestamp}.html` y una copia CSV con `_dump_prepared_dataframes_csv`. Estos archivos incluyen `full_history`, `history`, `real_values`, `X_train`, `y_train`, `exogenas`.
  4. Obtener el `trainer` y `forecaster` desde `MODEL_REGISTRY` y entrenar/pronosticar.
  5. `comparar` (MAE, RMSE) y `graficar`, que renderiza PNG en `logs/{pipeline}_{modelo}_forecast_{timestamp}.png` e incorpora el `plot_comment`.
  6. Construir `ForecastResponse` con la lista de `ForecastPoint`, frecuencias, métricas, rutas de artefactos y la imagen en base64 (`plot_image` vía `_encode_plot_base64`).
- Los modelos disponibles (`random_forest`, `prophet`, `sarimax`, `xgboost`) consumen distintos subconjuntos de `PreparedData`: algunos usan `X_train`/`y_train`, otros sólo `history` y `exogenas`.

## 7. Exógenas y configuraciones

- `cargar_exogenas` (en `exogena_loader.py`) conecta a la misma base de datos y:
  - Resuelve el ID de configuración (`ia.configuracion`) aceptando `code_id`, `cfg_` o el ID numérico.
  - Carga variables normales y dummies activas.
  - Construye un DataFrame diario entre `inicio` y `fin`, llenándolo con valores existentes y rangos de dummies (1 cuando la fecha cae en un rango). Ceros por defecto al resto.
  - Guarda en `df.attrs["expected_exogenas"]` la lista esperada de columnas para permitir la verificación posterior.
- El loader se usa en `build_turismo_pipeline` si `configuration_id` está presente, y también en `main.cargar_exogenas_para_rango` cuando un pipeline OpenMeteo necesita exógenas para entrenar/validar.

## 8. Endpoints auxiliares

- `/options` consulta `get_turismo_categories` y `get_turismo_date_bounds` para alimentar el frontend:
  - `get_turismo_categories`: consulta `SELECT DISTINCT Category FROM ia.ventas_turismo` y retorna las categorías ordenadas.
  - `get_turismo_date_bounds`: obtiene `MIN/MAX` de la columna de fecha, limita el máximo a `today - TURISMO_COMPARISON_DAYS` y normaliza a `YYYY-MM-DD`.
- También se listan configuraciones (`configuracion_endpoints.listar_configuraciones_db`) para que el usuario elija `configuration_id`.

## 9. Artefactos generados

- Archivos de logs:
  - `logs/dataframes/` contiene los HTML y CSV resumidos de cada ejecución.
  - `logs/{pipeline}_{modelo}_forecast_{timestamp}.png` con la comparación real vs. pronóstico.
- Las métricas (MAE, RMSE) y `plot_path` se devuelven en `ForecastResponse`; la imagen en base64 está lista para ser embebida.

Con esto, el pipeline de turismo queda completamente trazado desde el request inicial hasta la entrega del pronóstico, pasando por la sanitización SQL, la incorporación de exógenas, la preparación de los datasets y el entrenamiento/pronóstico con cualquiera de los modelos registrados.


## Modo forecast puro

- Se activa enviando `forecast_only=true` y `forecast_end_date` en el endpoint `/forecast` (o via CLI).
- El pipeline usa todo el historico disponible (sin forzar el gap de comparacion) y fija `test_range` desde el ultimo dia real + 1 hasta la fecha limite.
- No se calculan MAE/RMSE porque no hay valores reales; la respuesta incluye `forecast_only=true` y las columnas `valor_real/error` vienen en `null`.
- Las exogenas se cargan para todo el rango historico + horizonte solicitado, de modo que los modelos autoregresivos tengan insumos futuros.
