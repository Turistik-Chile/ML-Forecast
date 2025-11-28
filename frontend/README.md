# Frontend de pronosticos

Interfaz construida con [React 18](https://react.dev/), [TypeScript](https://www.typescriptlang.org/), [Vite](https://vitejs.dev/) y [Chart.js](https://www.chartjs.org/) para consumir la API de pronosticos definida en `main.py`.

## Requisitos basicos

- Node.js 18 o superior
- Backend FastAPI corriendo (por defecto en `http://localhost:8000`)

## Scripts disponibles

```bash
cd frontend
npm install
npm run dev      # Modo desarrollo (http://localhost:5173)
npm run build    # Genera artefactos de produccion
npm run preview  # Sirve el build generado
```

Configura la URL del backend creando `frontend/.env`:

```
VITE_API_URL=http://localhost:8000
```

## Contrato del API

La interfaz consume dos endpoints principales expuestos por FastAPI.

### `GET /options`

Respuesta:

```json
{
  "pipelines": ["open_meteo"],
  "locations": ["santiago", "valparaiso"],
  "models": ["prophet", "random_forest", "sarimax", "xgboost"]
}
```

### `POST /forecast`

Solicitud:

```json
{
  "pipeline": "open_meteo",
  "location": "santiago",
  "model": "random_forest"
}
```

Respuesta:

```json
{
  "pipeline": "open_meteo",
  "location": "santiago",
  "model": "random_forest",
  "lags": [1, 2, 3, 7, 30, 90, 180, 365],
  "mae": 1.23,
  "rmse": 1.56,
  "timestamp": "20231118_120000",
  "plot_path": "logs/open_meteo_random_forest_forecast_20231118_120000.png",
  "plot_image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "forecast": [
    {
      "fecha": "2023-01-01T00:00:00",
      "valor_real": 25.1,
      "prediccion": 24.7,
      "error": -0.4,
      "abs_error": 0.4
    }
  ]
}
```

- `plot_image` es la imagen PNG codificada en Base64. Para renderizarla: `data:image/png;base64,<plot_image>`.
- `forecast` contiene la serie diaria comparando valores reales y predichos, junto con el error absoluto por registro.

La aplicacion React usa estos datos para:

1. Poblar los selects con `/options`.
2. Crear un grafico comparativo a partir de `forecast`.
3. Mostrar indicadores globales (`mae`, `rmse`, `lags`) y tabla con las ultimas observaciones.

Con esto queda documentado el flujo completo entre el cliente React y la API.
