import { useEffect, useState, type FormEvent } from "react";
import { motion } from "framer-motion";
import PageHeader from "../components/PageHeader";
import ForecastForm from "../components/ForecastForm";
import StatusMessage from "../components/StatusMessage";
import MetricsSection from "../components/MetricsSection";
import ForecastChart from "../components/ForecastChart";
import ObservationsTable from "../components/ObservationsTable";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { AlertTriangle } from "lucide-react";
import { generateForecast, getOptions } from "../services/api";
import type {
  ForecastResponse,
  FormValues,
  OptionsResponse,
  StatusState,
} from "../types";

const DEFAULT_OPTIONS: OptionsResponse = {
  pipelines: [],
  locations: [],
  models: [],
  turismo: {
    categories: [],
    date_bounds: { min: null, max: null },
    min_gap_days: 365,
  },
};

const DEFAULT_LAGS = [1];

const clampDate = (
  value: string,
  min?: string | null,
  max?: string | null
): string => {
  if (!value) return "";
  let result = value;
  if (min && result < min) result = min;
  if (max && result > max) result = max;
  return result;
};

const ForecastPage = () => {
  const [options, setOptions] = useState<OptionsResponse>(DEFAULT_OPTIONS);
  const [formValues, setFormValues] = useState<FormValues>({
    pipeline: "",
    location: "",
    model: "",
    startDate: "",
    endDate: "",
    categories: [],
    lags: DEFAULT_LAGS,
  });
  const [status, setStatus] = useState<StatusState>({ message: "", type: "info" });
  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);

  const showStatus = (message: string, type: StatusState["type"] = "info") =>
    setStatus({ message, type });

  useEffect(() => {
    const loadOptions = async () => {
      showStatus("Cargando configuraciones disponibles...");
      try {
        const data = await getOptions();
        setOptions(data);
        setFormValues((prev) => ({
          pipeline: prev.pipeline || data.pipelines[0] || "",
          location: prev.location || data.locations[0] || "",
          model: prev.model || data.models[0] || "",
          startDate: prev.startDate || data.turismo?.date_bounds?.min || "",
          endDate: prev.endDate || data.turismo?.date_bounds?.max || "",
          categories: prev.categories.length ? prev.categories : [],
          lags: prev.lags.length ? prev.lags : DEFAULT_LAGS,
        }));
        showStatus("");
      } catch (error) {
        console.error(error);
        const message =
          error instanceof Error
            ? error.message
            : "No fue posible cargar las opciones.";
        showStatus(message, "error");
      }
    };

    loadOptions();
  }, []);

  const handleFieldChange = (
    name: keyof FormValues,
    value: string | string[]
  ) => {
    setFormValues((prev) => {
      if (name === "startDate" || name === "endDate") {
        const bounds = options.turismo?.date_bounds;
        const minBound = bounds?.min;
        const maxBound = bounds?.max;
        const nextValue = clampDate(value as string, minBound, maxBound);
        let startDate = name === "startDate" ? nextValue : prev.startDate;
        let endDate = name === "endDate" ? nextValue : prev.endDate;
        if (startDate && endDate && startDate > endDate) {
          if (name === "startDate") {
            endDate = startDate;
          } else {
            startDate = endDate;
          }
        }
        return { ...prev, startDate, endDate };
      }
      if (name === "pipeline") {
        const nextPipeline = value as string;
        const isTurismo = nextPipeline === "turismo";
        const bounds = options.turismo?.date_bounds;
        return {
          ...prev,
          pipeline: nextPipeline,
          startDate: isTurismo
            ? prev.startDate
              ? clampDate(prev.startDate, bounds?.min, bounds?.max)
              : bounds?.min || ""
            : "",
          endDate: isTurismo
            ? prev.endDate
              ? clampDate(prev.endDate, bounds?.min, bounds?.max)
              : bounds?.max || ""
            : "",
        };
      }
      if (name === "categories") {
        const normalized = (Array.isArray(value) ? value : [value]).filter(
          (item) => item && item !== "__all__"
        );
        return { ...prev, categories: normalized };
      }
      return { ...prev, [name]: value as string };
    });
  };

  const handleLagAdd = (lag: number) => {
    setFormValues((prev) => {
      if (prev.lags.includes(lag)) return prev;
      return { ...prev, lags: [...prev.lags, lag].sort((a, b) => a - b) };
    });
  };

  const handleLagUpdate = (index: number, lag: number) => {
    setFormValues((prev) => {
      if (prev.lags.some((value, idx) => idx !== index && value === lag)) return prev;
      const next = prev.lags.slice();
      next[index] = lag;
      next.sort((a, b) => a - b);
      return { ...prev, lags: next };
    });
  };

  const handleLagDelete = (index: number) => {
    setFormValues((prev) => {
      if (prev.lags.length <= 1) return prev;
      const next = prev.lags.filter((_, idx) => idx !== index);
      return { ...prev, lags: next };
    });
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!formValues.pipeline) {
      showStatus("Selecciona un pipeline válido.", "error");
      return;
    }
    if (formValues.pipeline === "open_meteo" && !formValues.location) {
      showStatus("Selecciona una ubicación para Open Meteo.", "error");
      return;
    }
    if (
      formValues.pipeline === "turismo" &&
      (!formValues.startDate || !formValues.endDate)
    ) {
      showStatus("Debes definir el rango de fechas para turismo.", "error");
      return;
    }
    if (!formValues.lags.length) {
      showStatus("Debes agregar al menos un lag válido.", "error");
      return;
    }
    setLoading(true);
    showStatus("Generando pronóstico...");
    try {
      const data = await generateForecast(formValues);
      setForecast(data);
      showStatus("Pronóstico generado correctamente.");
    } catch (error) {
      console.error(error);
      const message =
        error instanceof Error
          ? error.message
          : "No fue posible generar el pronóstico.";
      showStatus(message, "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.section
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <PageHeader
        title="PrediBúnker, predicciones con amor"
        subtitle="Experimenta con pipelines y modelos de series temporales en segundos."
      />
      <ForecastForm
        options={options}
        values={formValues}
        loading={loading}
        onFieldChange={handleFieldChange}
        onLagAdd={handleLagAdd}
        onLagUpdate={handleLagUpdate}
        onLagDelete={handleLagDelete}
        onSubmit={handleSubmit}
      />
      <StatusMessage message={status.message} type={status.type} />
      {forecast && forecast.zero_padding_days > 0 ? (
        <Alert className="border-destructive/40 bg-destructive/10 text-foreground">
          <AlertTitle className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-destructive" />
            Datos rellenados
          </AlertTitle>
          <AlertDescription>
            Se rellenaron {forecast.zero_padding_days}{" "}
            {forecast.zero_padding_days === 1 ? "día" : "días"} sin datos previos con valor 0.
            Amplía el rango o reduce los lags si quieres evitarlo.
          </AlertDescription>
        </Alert>
      ) : null}
      {forecast ? (
        <>
          <MetricsSection mae={forecast.mae} rmse={forecast.rmse} lags={forecast.lags} />
          {forecast.forecast?.length ? <ForecastChart data={forecast.forecast} /> : null}
          {forecast.forecast?.length ? (
            <ObservationsTable rows={forecast.forecast} maxRows={10} />
          ) : null}
        </>
      ) : null}
    </motion.section>
  );
};

export default ForecastPage;
