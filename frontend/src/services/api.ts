import type {
  ConfigDetail,
  ConfigSummary,
  ExogenaDummy,
  ExogenaNormal,
  ForecastResponse,
  FormValues,
  OptionsResponse,
} from "../types";

const API_BASE_URL = "http://127.0.0.1:8000";

const fetchJSON = async <T,>(
  path: string,
  options: RequestInit = {}
): Promise<T> => {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      (error as { detail?: string }).detail ?? "Error inesperado consultando el API."
    );
  }
  return response.json() as Promise<T>;
};

const buildForecastPayload = (values: FormValues, configurationId?: string) => ({
  pipeline: values.pipeline,
  location: values.pipeline === "turismo" ? null : values.location || null,
  model: values.model,
  start_date:
    values.pipeline === "turismo" && values.startDate ? values.startDate : null,
  end_date:
    values.pipeline === "turismo" && values.endDate ? values.endDate : null,
  categories: values.pipeline === "turismo" ? values.categories : null,
  lags: values.lags.length > 0 ? values.lags : null,
  configuration_id: configurationId ?? values.configurationId ?? null,
  forecast_only: values.forecastOnly,
  forecast_end_date: values.forecastOnly ? values.forecastEndDate || null : null,
});

export const getOptions = () => fetchJSON<OptionsResponse>("/options");

export const generateForecast = (
  values: FormValues,
  configurationId?: string
) =>
  fetchJSON<ForecastResponse>("/forecast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildForecastPayload(values, configurationId)),
  });

export const listConfigurations = () =>
  fetchJSON<ConfigSummary[]>("/configuracion/list");

export const getConfiguration = (codeId: string) =>
  fetchJSON<ConfigDetail>(`/configuracion/${codeId}`);

export const createConfiguration = (payload: {
  nombre: string;
  comentarios: string | null;
  exogenas_dummies: ExogenaDummy[];
  exogenas_normales: ExogenaNormal[];
}) =>
  fetchJSON<{ status: string; code_id: string }>("/configuracion/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const updateConfiguration = (
  codeId: string,
  payload: {
    nombre: string;
    comentarios: string | null;
    exogenas_dummies: ExogenaDummy[];
    exogenas_normales: ExogenaNormal[];
  }
) =>
  fetchJSON(`/configuracion/${codeId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const deleteConfiguration = (codeId: string) =>
  fetchJSON(`/configuracion/${codeId}`, {
    method: "DELETE",
  });

export const exportConfigurations = (
  configs: ConfigSummary[] | ConfigDetail[]
) =>
  fetchJSON("/configuracion/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ configuraciones: configs }),
  });
