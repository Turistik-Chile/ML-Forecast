export type TurismoMetadata = {
  categories: string[];
  date_bounds: {
    min: string | null;
    max: string | null;
    available_max?: string | null;
  };
  min_gap_days: number;
};

export type OptionsResponse = {
  pipelines: string[];
  locations: string[];
  models: string[];
  configurations?: ConfigSummary[];
  turismo?: TurismoMetadata;
};

export type FormValues = {
  pipeline: string;
  location: string;
  model: string;
  startDate: string;
  endDate: string;
  categories: string[];
  lags: number[];
  configurationId?: string;
  forecastOnly: boolean;
  forecastEndDate: string;
};

export type StatusState = {
  message: string;
  type: "info" | "error";
};

export type ExogenaRange = {
  inicio: string;
  fin: string;
};

export type ExogenaValue = {
  fecha: string;
  valor: number | string;
};

export type ExogenaDummy = {
  id?: number;
  nombre: string;
  rangos: ExogenaRange[];
};

export type ExogenaNormal = {
  id?: number;
  nombre: string;
  valores: ExogenaValue[];
};

export type ConfigSummary = {
  code_id: string;
  nombre: string;
  comentarios: string | null;
};

export type ConfigDetail = ConfigSummary & {
  exogenas_dummies: ExogenaDummy[];
  exogenas_normales: ExogenaNormal[];
};

export type ForecastPoint = {
  fecha: string;
  valor_real: number | null;
  prediccion: number;
  error: number | null;
  abs_error: number | null;
};

export type ForecastResponse = {
  pipeline: string;
  location: string;
  model: string;
  lags: number[];
  zero_padding_days: number;
  mae: number | null;
  rmse: number | null;
  forecast_only: boolean;
  timestamp: string;
  plot_path: string;
  plot_image: string | null;
  forecast: ForecastPoint[];
};
