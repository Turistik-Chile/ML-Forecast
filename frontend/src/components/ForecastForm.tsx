import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Badge } from "./ui/badge";
import { Alert, AlertDescription } from "./ui/alert";
import { cn } from "../lib/utils";
import type { ConfigSummary, FormValues, OptionsResponse } from "../types";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronUp,
  Edit2,
  Plus,
  Trash2,
  X,
} from "lucide-react";

type ForecastFormProps = {
  options: OptionsResponse;
  configOptions: ConfigSummary[];
  values: FormValues;
  loading: boolean;
  onFieldChange: (name: keyof FormValues, value: string | string[] | boolean) => void;
  onLagAdd: (lag: number) => void;
  onLagUpdate: (index: number, lag: number) => void;
  onLagDelete: (index: number) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
};

const MS_PER_DAY = 24 * 60 * 60 * 1000;
const MAX_ZERO_PADDING_DAYS = 30;

type StepperInputProps = {
  value: string;
  onValueChange: (next: string) => void;
  min?: number;
  className?: string;
  inputClassName?: string;
  placeholder?: string;
  disabled?: boolean;
};

const StepperInput = ({
  value,
  onValueChange,
  min = 1,
  className,
  inputClassName,
  placeholder,
  disabled,
}: StepperInputProps) => {
  const numericValue = Number(value);
  const current = Number.isFinite(numericValue) && numericValue > 0 ? numericValue : min;

  const adjust = (delta: number) => {
    const next = Math.max(min, current + delta);
    onValueChange(String(next));
  };

  return (
    <div className={cn("relative", className)}>
      <Input
        type="number"
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        min={min}
        placeholder={placeholder}
        disabled={disabled}
        className={cn("pr-12", inputClassName)}
      />
      <div className="pointer-events-none absolute inset-y-1 right-1 flex flex-col overflow-hidden rounded-md border border-border bg-muted text-muted-foreground">
        <button
          type="button"
          onClick={() => adjust(1)}
          className="pointer-events-auto flex h-5 w-7 items-center justify-center border-b border-border text-[10px] hover:bg-accent hover:text-accent-foreground"
          aria-label="Incrementar"
          disabled={disabled}
        >
          <ChevronUp className="h-3 w-3" />
        </button>
        <button
          type="button"
          onClick={() => adjust(-1)}
          className="pointer-events-auto flex h-5 w-7 items-center justify-center text-[10px] hover:bg-accent hover:text-accent-foreground"
          aria-label="Disminuir"
          disabled={disabled}
        >
          <ChevronDown className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
};

const ForecastForm = ({
  options,
  configOptions,
  values,
  loading,
  onFieldChange,
  onLagAdd,
  onLagUpdate,
  onLagDelete,
  onSubmit,
}: ForecastFormProps) => {
  const isTurismo = values.pipeline === "turismo";
  const turismoCategories = options.turismo?.categories ?? [];
  const turismoBounds = options.turismo?.date_bounds;
  const minDate = turismoBounds?.min || "";
  const maxDate = turismoBounds?.max || "";
  const availableMaxDate = turismoBounds?.available_max || maxDate || "";
  const minGapDays = options.turismo?.min_gap_days ?? 365;

  const [lagInput, setLagInput] = useState("");
  const [lagError, setLagError] = useState<string | null>(null);
  const [editingLagIndex, setEditingLagIndex] = useState<number | null>(null);
  const [editingLagValue, setEditingLagValue] = useState("");

  const historySpanDays = useMemo(() => {
    if (!isTurismo || !values.startDate || !values.endDate) return null;
    const start = new Date(values.startDate);
    const end = new Date(values.endDate);
    const diff = end.getTime() - start.getTime();
    if (!Number.isFinite(diff) || diff < 0) return null;
    const days = Math.floor(diff / MS_PER_DAY) + 1;
    return days > 0 ? days : null;
  }, [isTurismo, values.startDate, values.endDate]);

  const maxSelectedLag = values.lags.length ? Math.max(...values.lags) : 0;
  const approxPaddingNeeded =
    historySpanDays !== null ? Math.max(0, maxSelectedLag - historySpanDays) : 0;
  const paddingWarningActive = isTurismo && approxPaddingNeeded > 0;

  const parseLagValue = (value: string): number => {
    const lag = Number(value);
    if (!Number.isFinite(lag) || !Number.isInteger(lag) || lag <= 0) {
      throw new Error("El lag debe ser un entero positivo.");
    }
    return lag;
  };

  const handleLagAdd = () => {
    try {
      const lag = parseLagValue(lagInput);
      if (values.lags.includes(lag)) {
        setLagError("Ese lag ya existe.");
        return;
      }
      const exceedsHistory = historySpanDays !== null && lag > historySpanDays;
      if (exceedsHistory && lag > MAX_ZERO_PADDING_DAYS) {
        const missingDays = historySpanDays === null ? 0 : lag - historySpanDays;
        setLagError(
          `No hay historial suficiente para un lag de ${lag} días (faltan aprox. ${missingDays}). Solo se permite rellenar hasta ${MAX_ZERO_PADDING_DAYS} días.`
        );
        return;
      }
      onLagAdd(lag);
      setLagInput("");
      setLagError(null);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "No fue posible agregar el lag.";
      setLagError(message);
    }
  };

  const handleLagEdit = () => {
    if (editingLagIndex === null) return;
    try {
      const nextLag = parseLagValue(editingLagValue);
      const duplicated = values.lags.some(
        (value, index) => index !== editingLagIndex && value === nextLag
      );
      if (duplicated) {
        setLagError("Ese lag ya existe.");
        return;
      }
      const exceedsHistory = historySpanDays !== null && nextLag > historySpanDays;
      if (exceedsHistory && nextLag > MAX_ZERO_PADDING_DAYS) {
        const missingDays = historySpanDays === null ? 0 : nextLag - historySpanDays;
        setLagError(
          `No hay historial suficiente para un lag de ${nextLag} días (faltan aprox. ${missingDays}). Solo se permite rellenar hasta ${MAX_ZERO_PADDING_DAYS} días.`
        );
        return;
      }
      onLagUpdate(editingLagIndex, nextLag);
      setLagError(null);
      setEditingLagIndex(null);
      setEditingLagValue("");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "No fue posible editar el lag.";
      setLagError(message);
    }
  };

  const handleCategoryToggle = (category: string) => {
    const current = new Set(values.categories);
    if (current.has(category)) {
      current.delete(category);
    } else {
      current.add(category);
    }
    onFieldChange("categories", Array.from(current));
  };

  const renderLagItem = (lag: number, index: number) => {
    const isEditing = editingLagIndex === index;
    return (
      <li
        key={`${lag}-${index}`}
        className="flex items-center justify-between rounded-lg bg-secondary/70 px-4 py-2"
      >
        {isEditing ? (
          <div className="flex w-full items-center gap-2">
            <StepperInput
              value={editingLagValue}
              onValueChange={(val) => setEditingLagValue(val)}
              min={1}
              className="w-24"
              inputClassName="h-9"
            />
            <Button type="button" size="icon" variant="ghost" onClick={handleLagEdit}>
              <Check className="h-4 w-4" />
            </Button>
            <Button
              type="button"
              size="icon"
              variant="ghost"
              onClick={() => {
                setLagError(null);
                setEditingLagIndex(null);
                setEditingLagValue("");
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <>
            <p className="text-sm font-semibold text-foreground">
              Lag de {lag} día{lag === 1 ? "" : "s"}
            </p>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="icon"
                variant="ghost"
                onClick={() => {
                  setEditingLagIndex(index);
                  setEditingLagValue(String(lag));
                }}
              >
                <Edit2 className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                size="icon"
                variant="ghost"
                disabled={values.lags.length <= 1}
                onClick={() => {
                  if (values.lags.length <= 1) {
                    setLagError("Debes mantener al menos un lag.");
                    return;
                  }
                  onLagDelete(index);
                }}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </>
        )}
      </li>
    );
  };

  return (
    <motion.form
      onSubmit={onSubmit}
      className="mt-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <Card className="border-none bg-background/90 shadow-2xl backdrop-blur">
        <CardHeader className="space-y-2">
          <CardTitle>Configura tu pronóstico</CardTitle>
          <CardDescription>
            Selecciona el pipeline, modelo y rango temporal, luego personaliza lags y
            categorías antes de solicitar la predicción.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-8">
          <div className="grid gap-6 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Pipeline</Label>
              <Select
                value={values.pipeline || undefined}
                onValueChange={(val) => onFieldChange("pipeline", val)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Selecciona un pipeline" />
                </SelectTrigger>
                <SelectContent>
                  {options.pipelines.map((pipeline) => (
                    <SelectItem key={pipeline} value={pipeline}>
                      {pipeline}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Modelo</Label>
              <Select
                value={values.model || undefined}
                onValueChange={(val) => onFieldChange("model", val)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Selecciona un modelo" />
                </SelectTrigger>
                <SelectContent>
                  {options.models.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Configuración</Label>
              <Select
                value={values.configurationId || undefined}
                onValueChange={(val) => onFieldChange("configurationId", val)}
                disabled={!configOptions.length}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Selecciona una configuración" />
                </SelectTrigger>
                <SelectContent>
                  {configOptions.length ? (
                    configOptions.map((config) => (
                      <SelectItem key={config.code_id} value={config.code_id}>
                        {config.nombre}
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem disabled value="__no-config">
                      Sin configuraciones disponibles
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Ubicación</Label>
              {isTurismo ? (
                <Input value="No aplica" disabled />
              ) : (
                <Select
                  value={values.location || undefined}
                  onValueChange={(val) => onFieldChange("location", val)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Selecciona una ubicación" />
                  </SelectTrigger>
                  <SelectContent>
                    {options.locations.map((location) => (
                      <SelectItem key={location} value={location}>
                        {location}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>
          </div>

          {isTurismo ? (
            <div className="space-y-5 rounded-lg border border-dashed border-muted p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <Label className="text-base">Modo de uso</Label>
                  <p className="text-sm text-muted-foreground">
                    Activa forecast puro para usar todo el historico disponible y proyectar hasta la fecha limite.
                  </p>
                </div>
                <label className="flex items-center gap-2 text-sm font-semibold">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-primary"
                    checked={values.forecastOnly}
                    onChange={(event) => onFieldChange("forecastOnly", event.target.checked)}
                  />
                  Forecast puro
                </label>
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Fecha de inicio</Label>
                  <Input
                    type="date"
                    value={values.startDate}
                    min={minDate || undefined}
                    max={values.endDate || (values.forecastOnly ? availableMaxDate : maxDate) || undefined}
                    onChange={(event) => onFieldChange("startDate", event.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Fecha de termino</Label>
                  <Input
                    type="date"
                    value={values.endDate}
                    min={values.startDate || minDate || undefined}
                    max={(values.forecastOnly ? availableMaxDate : maxDate) || undefined}
                    onChange={(event) => onFieldChange("endDate", event.target.value)}
                  />
                </div>
              </div>
              {values.forecastOnly ? (
                <div className="space-y-2">
                  <Label>Fecha limite para el forecast</Label>
                  <Input
                    type="date"
                    value={values.forecastEndDate}
                    min={values.endDate || availableMaxDate || minDate || undefined}
                    onChange={(event) => onFieldChange("forecastEndDate", event.target.value)}
                  />
                  <p className="text-sm text-muted-foreground">
                    El resultado solo incluira la prediccion hasta esta fecha (sin comparacion contra datos reales).
                  </p>
                </div>
              ) : null}
            </div>
          ) : null}


          {isTurismo ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Categorías</Label>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => onFieldChange("categories", [])}
                >
                  Todas
                </Button>
              </div>
              {turismoCategories.length ? (
                <ScrollArea className="h-32 rounded-lg border border-dashed border-muted p-2">
                  <div className="flex flex-wrap gap-2">
                    {turismoCategories.map((category) => {
                      const isSelected = values.categories.includes(category);
                      return (
                        <Badge
                          key={category}
                          variant={isSelected ? "default" : "outline"}
                          className="cursor-pointer"
                          onClick={() => handleCategoryToggle(category)}
                        >
                          {category}
                        </Badge>
                      );
                    })}
                  </div>
                </ScrollArea>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No hay categorías disponibles.
                </p>
              )}
              <p className="text-sm text-muted-foreground">
                {values.forecastOnly
                  ? `Se usara todo el historico disponible (max: ${
                      availableMaxDate
                        ? new Date(availableMaxDate).toLocaleDateString("es-CL")
                        : "sin datos"
                    }) y solo se devolvera la prediccion.`
                  : `La fecha de termino debe estar al menos ${minGapDays} dias antes de hoy. Maximo permitido: ${
                      maxDate ? new Date(maxDate).toLocaleDateString("es-CL") : "sin datos"
                    }.`}
              </p>
            </div>
          ) : null}

          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <Label className="text-base">Lags (días)</Label>
                <p className="text-sm text-muted-foreground">
                  Define los retrasos que alimentarán al modelo autoregresivo.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <StepperInput
                  value={lagInput}
                  onValueChange={(val) => {
                    setLagError(null);
                    setLagInput(val);
                  }}
                  min={1}
                  className="w-32"
                  placeholder="Ej: 7"
                />
                <Button type="button" onClick={handleLagAdd} variant="secondary">
                  <Plus className="mr-2 h-4 w-4" />
                  Agregar
                </Button>
              </div>
            </div>
            {lagError ? (
              <p className="text-sm font-semibold text-destructive">{lagError}</p>
            ) : (
              <p className="text-sm text-muted-foreground">
                Solo enteros positivos. Si el historial es menor que el lag, se rellenará con 0
                hasta {MAX_ZERO_PADDING_DAYS} días.
              </p>
            )}

            {paddingWarningActive && historySpanDays !== null ? (
              <Alert className="flex items-start gap-3 border-destructive/40 bg-destructive/10 text-foreground">
                <AlertTriangle className="mt-1 h-4 w-4 flex-none text-destructive" />
                <AlertDescription>
                  El rango seleccionado aporta aprox. {historySpanDays} días y tu mayor lag requiere{" "}
                  {maxSelectedLag}. Se rellenarán cerca de {approxPaddingNeeded} día(s) con valor 0
                  (máximo {MAX_ZERO_PADDING_DAYS}). Ajusta el rango o los lags si no es deseado.
                </AlertDescription>
              </Alert>
            ) : null}

            <ScrollArea className="h-48 rounded-lg border border-dashed border-muted p-3">
              {values.lags.length ? (
                <ul className="space-y-3">
                  {values.lags.map((lag, index) => renderLagItem(lag, index))}
                </ul>
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                  No hay lags configurados.
                </div>
              )}
            </ScrollArea>
          </div>

          <Button type="submit" className="w-full" disabled={loading || !values.pipeline}>
            {loading ? "Generando..." : "Generar pronóstico"}
          </Button>
        </CardContent>
      </Card>
    </motion.form>
  );
};

export default ForecastForm;
