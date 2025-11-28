# API Backend (FastAPI)

Base URL local: `http://localhost:8000`

## Salud
- `GET /health`  
  Responde `{"status": "ok", "timestamp": "<ISO>"}`.

## Opciones iniciales
- `GET /options`  
  Devuelve listas de pipelines, modelos, locations y configuraciones disponibles.  
  Respuesta (ejemplo abreviado):
  ```json
  {
    "pipelines": ["open_meteo", "turismo"],
    "locations": ["santiago", "valparaiso"],
    "models": ["random_forest", "prophet", "sarimax", "xgboost"],
    "configurations": [{"code_id": "<hash>", "nombre": "<hash>", "comentarios": null}],
    "turismo": {
      "categories": ["cat1", "cat2"],
      "date_bounds": {"min": "2022-01-01", "max": "2024-12-31", "available_max": "2025-12-31"},
      "min_gap_days": 365
    }
  }
  ```

## Forecast
- `POST /forecast`    Ejecuta un pipeline + modelo. Si se incluye `configuration_id`, se cargan todas las exogenas (normales + dummies) para el rango usado por el pipeline y se pasan a los modelos.    Parametros extra:
  - `forecast_only` (bool, opcional): usa todo el historico y solo devuelve la prediccion hasta `forecast_end_date` (sin MAE/RMSE).
  - `forecast_end_date` (YYYY-MM-DD, requerido si `forecast_only` es true): fecha limite hasta donde proyectar.    Cuerpo (ejemplo `open_meteo`):
  ```json
  {
    "pipeline": "open_meteo",
    "location": "santiago",
    "model": "random_forest",
    "lags": [1, 7],
    "configuration_id": "cfg_1234"  // o code_id sin prefijo
  }
  ```
  Cuerpo (ejemplo `turismo`):
  ```json
  {
    "pipeline": "turismo",
    "model": "prophet",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "categories": ["ski", "hotel"],
    "lags": [7, 14],
    "configuration_id": "cfg_1234",
    "forecast_only": true,
    "forecast_end_date": "2025-12-31"
  }
  ```
  Respuesta: metricas (o null en forecast puro), flag `forecast_only`, ruta del grafico y arreglo `forecast` con filas `{fecha, valor_real?, prediccion, error?, abs_error?}`.

## Configuraciones (exógenas)
Las configuraciones generan un `code_id` (hash) con el que luego se consultan y se usan en `/forecast`.

- `GET /configuracion/list`  
  Lista `[{code_id, nombre, comentarios, id}]`.

- `GET /configuracion/{code_id}`  
  Devuelve detalle completo, incluidas exógenas.

- `POST /configuracion/create`  
  Crea configuración y sus exógenas. Cuerpo:
  ```json
  {
    "nombre": "Mi configuración",
    "comentarios": null,
    "exogenas_dummies": [
      {
        "nombre": "feriado",
        "rangos": [
          { "inicio": "2024-01-01", "fin": "2024-01-02" },
          { "inicio": "2024-02-10", "fin": "2024-02-15" }
        ]
      }
    ],
    "exogenas_normales": [
      {
        "nombre": "tipo_cambio",
        "valores": [
          { "fecha": "2024-01-01", "valor": 900 },
          { "fecha": "2024-01-02", "valor": 905 }
        ]
      }
    ]
  }
  ```
  Respuesta: `{ "status": "created" | "exists", "code_id": "<hash>" }`.

- `PUT /configuracion/{code_id}`  
  Reemplaza totalmente las exógenas de la configuración con el mismo cuerpo que `create`.

- `DELETE /configuracion/{code_id}`  
  Elimina la configuración y todas sus exógenas.

- `POST /configuracion/export`  
  Recibe `{"configuraciones": [...]}` para exportar/respaldar (solo registra el JSON).

## Exógenas puntuales
Pensadas para uso interno/auxiliar; también respetan `code_id`.

- `POST /exogena/create`  
  Cuerpo: `{"nombre": "mi_var", "configuracion_id": "<code_id>", "is_dummie": true}`.

- `POST /exogena/rango`  
  Cuerpo: `{"exogena_id": 1, "fecha_inicio": "2024-01-01", "fecha_fin": "2024-01-10"}`; activa la dummie en ese rango (todo fuera del rango queda en 0).

## Notas sobre exógenas
- Dummies: por defecto 0; cada rango declarado se marca 1 entre `inicio` y `fin` inclusive. Se combinan varios rangos de la misma dummie con OR.
- Normales: se cargan por fecha exacta; si falta una fecha en el rango solicitado, se rellena con 0.
- El `configuration_id` aceptado en `/forecast` puede ser el `code_id` exacto, con prefijo `cfg_`, o el `id` entero de la tabla `ia.configuracion`.
