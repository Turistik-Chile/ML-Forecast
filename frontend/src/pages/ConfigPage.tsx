import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import StatusMessage from "../components/StatusMessage";
import { ConfirmDialog } from "../components/ui/dialog";
import {
  createConfiguration,
  deleteConfiguration,
  exportConfigurations,
  generateForecast,
  getConfiguration,
  getOptions,
  listConfigurations,
  updateConfiguration,
} from "../services/api";
import ForecastChart from "../components/ForecastChart";
import MetricsSection from "../components/MetricsSection";
import ObservationsTable from "../components/ObservationsTable";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { Badge } from "../components/ui/badge";
import { ScrollArea } from "../components/ui/scroll-area";
import type {
  ConfigDetail,
  ConfigSummary,
  ExogenaDummy,
  ExogenaNormal,
  ForecastResponse,
  OptionsResponse,
  StatusState,
} from "../types";
import { cn } from "../lib/utils";
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

type ViewMode = "list" | "form" | "detail";

const emptyFormState = {
  nombre: "",
  comentarios: "",
  exogenas_dummies: [] as ExogenaDummy[],
  exogenas_normales: [] as ExogenaNormal[],
};

const DEFAULT_REPORT_LAGS = [1];

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

const ConfigPage = () => {
  const [mode, setMode] = useState<ViewMode>("list");
  const [status, setStatus] = useState<StatusState>({ message: "", type: "info" });
  const [configs, setConfigs] = useState<ConfigSummary[]>([]);
  const [selectedDetail, setSelectedDetail] = useState<ConfigDetail | null>(null);
  const [formState, setFormState] = useState(emptyFormState);
  const [isEditing, setIsEditing] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [options, setOptions] = useState<OptionsResponse>({
    pipelines: [],
    locations: [],
    models: [],
  });
  const [reportPipeline, setReportPipeline] = useState("");
  const [reportModel, setReportModel] = useState("");
  const [reportLocation, setReportLocation] = useState("");
  const [reportStartDate, setReportStartDate] = useState("");
  const [reportEndDate, setReportEndDate] = useState("");
  const [reportCategories, setReportCategories] = useState<string[]>([]);
  const [reportLags, setReportLags] = useState<number[]>(DEFAULT_REPORT_LAGS);
  const [lagInput, setLagInput] = useState("");
  const [lagError, setLagError] = useState<string | null>(null);
  const [editingLagIndex, setEditingLagIndex] = useState<number | null>(null);
  const [editingLagValue, setEditingLagValue] = useState("");
  const [reportConfigId, setReportConfigId] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [isReportLoading, setIsReportLoading] = useState(false);

  const loadConfigs = async () => {
    try {
      const data = await listConfigurations();
      setConfigs(data);
      setReportConfigId((prev) => prev || data[0]?.code_id || null);
    } catch (error) {
      setStatus({
        message:
          error instanceof Error ? error.message : "No fue posible cargar las configuraciones.",
        type: "error",
      });
    }
  };

  const loadOptions = async () => {
    try {
      const data = await getOptions();
      setOptions(data);
      setReportPipeline((prev) => prev || data.pipelines[0] || "");
      setReportModel((prev) => prev || data.models[0] || "");
      setReportLocation((prev) => prev || data.locations[0] || "");
      setReportStartDate((prev) => prev || data.turismo?.date_bounds?.min || "");
      setReportEndDate((prev) => prev || data.turismo?.date_bounds?.max || "");

      const configsFromOptions = data.configurations ?? [];
      if (configsFromOptions.length) {
        setConfigs(configsFromOptions);
        setReportConfigId((prev) => prev || configsFromOptions[0]?.code_id || null);
      } else {
        await loadConfigs();
      }
    } catch (error) {
      setStatus({
        message: error instanceof Error ? error.message : "No fue posible cargar los pipelines.",
        type: "error",
      });
      await loadConfigs();
    }
  };

  useEffect(() => {
    loadOptions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!reportConfigId && configs.length) {
      setReportConfigId(configs[0].code_id);
    }
  }, [configs, reportConfigId]);

  const resetForm = () => {
    setFormState(emptyFormState);
    setIsEditing(false);
    setEditingId(null);
  };

  const handleAddExogena = () => {
    setFormState((prev) => ({
      ...prev,
      exogenas_dummies: [...prev.exogenas_dummies, { nombre: "", rangos: [] }],
    }));
  };

  const handleExogenaName = (index: number, value: string) => {
    setFormState((prev) => {
      const next = [...prev.exogenas_dummies];
      next[index] = { ...next[index], nombre: value };
      return { ...prev, exogenas_dummies: next };
    });
  };

  const handleAddRange = (exoIndex: number) => {
    setFormState((prev) => {
      const next = [...prev.exogenas_dummies];
      next[exoIndex] = {
        ...next[exoIndex],
        rangos: [...next[exoIndex].rangos, { inicio: "", fin: "" }],
      };
      return { ...prev, exogenas_dummies: next };
    });
  };

  const handleRangeUpdate = (
    exoIndex: number,
    rangeIndex: number,
    field: "inicio" | "fin",
    value: string
  ) => {
    setFormState((prev) => {
      const next = [...prev.exogenas_dummies];
      const ranges = [...next[exoIndex].rangos];
      ranges[rangeIndex] = { ...ranges[rangeIndex], [field]: value };
      next[exoIndex] = { ...next[exoIndex], rangos: ranges };
      return { ...prev, exogenas_dummies: next };
    });
  };

  const handleDeleteRange = (exoIndex: number, rangeIndex: number) => {
    setFormState((prev) => {
      const next = [...prev.exogenas_dummies];
      next[exoIndex] = {
        ...next[exoIndex],
        rangos: next[exoIndex].rangos.filter((_, idx) => idx !== rangeIndex),
      };
      return { ...prev, exogenas_dummies: next };
    });
  };

  const handleRemoveExogena = (index: number) => {
    setFormState((prev) => {
      const next = prev.exogenas_dummies.filter((_, idx) => idx !== index);
      return { ...prev, exogenas_dummies: next };
    });
  };

  const handleAddNormalExogena = () => {
    setFormState((prev) => ({
      ...prev,
      exogenas_normales: [
        ...prev.exogenas_normales,
        { nombre: "", valores: [{ fecha: "", valor: "" }] },
      ],
    }));
  };

  const handleNormalNameChange = (index: number, value: string) => {
    setFormState((prev) => {
      const next = [...prev.exogenas_normales];
      next[index] = { ...next[index], nombre: value };
      return { ...prev, exogenas_normales: next };
    });
  };

  const handleAddNormalValue = (exoIndex: number) => {
    setFormState((prev) => {
      const next = [...prev.exogenas_normales];
      const target = next[exoIndex];
      next[exoIndex] = {
        ...target,
        valores: [...target.valores, { fecha: "", valor: "" }],
      };
      return { ...prev, exogenas_normales: next };
    });
  };

  const handleNormalValueChange = (
    exoIndex: number,
    valueIndex: number,
    field: "fecha" | "valor",
    value: string
  ) => {
    setFormState((prev) => {
      const next = [...prev.exogenas_normales];
      const target = next[exoIndex];
      const valores = [...target.valores];
      valores[valueIndex] = { ...valores[valueIndex], [field]: value };
      next[exoIndex] = { ...target, valores };
      return { ...prev, exogenas_normales: next };
    });
  };

  const handleRemoveNormalValue = (exoIndex: number, valueIndex: number) => {
    setFormState((prev) => {
      const next = [...prev.exogenas_normales];
      const target = next[exoIndex];
      next[exoIndex] = {
        ...target,
        valores: target.valores.filter((_, idx) => idx !== valueIndex),
      };
      return { ...prev, exogenas_normales: next };
    });
  };

  const handleRemoveNormalExogena = (index: number) => {
    setFormState((prev) => {
      const next = prev.exogenas_normales.filter((_, idx) => idx !== index);
      return { ...prev, exogenas_normales: next };
    });
  };

  const handleSaveConfig = async (event: FormEvent) => {
    event.preventDefault();
    if (!formState.nombre.trim()) {
      setStatus({ message: "El nombre es requerido.", type: "error" });
      return;
    }

    for (const exo of formState.exogenas_normales) {
      if (!exo.nombre.trim()) {
        setStatus({
          message: "Cada exógena normal requiere un nombre.",
          type: "error",
        });
        return;
      }
      if (!exo.valores.length) {
        setStatus({
          message: `La exógena "${exo.nombre}" debe tener al menos un valor.`,
          type: "error",
        });
        return;
      }
      for (const valor of exo.valores) {
        if (!valor.fecha.trim()) {
          setStatus({
            message: `Todos los valores de "${exo.nombre}" deben incluir una fecha.`,
            type: "error",
          });
          return;
        }
        const numeric = Number(valor.valor);
        if (!Number.isFinite(numeric)) {
          setStatus({
            message: `El valor para ${valor.fecha} en "${exo.nombre}" no es válido.`,
            type: "error",
          });
          return;
        }
      }
    }

    const payload = {
      nombre: formState.nombre,
      comentarios: formState.comentarios || null,
      exogenas_dummies: formState.exogenas_dummies.map((exo) => ({
        nombre: exo.nombre,
        rangos: exo.rangos,
      })),
      exogenas_normales: formState.exogenas_normales.map((exo) => ({
        nombre: exo.nombre,
        valores: exo.valores.map((valor) => ({
          fecha: valor.fecha,
          valor: Number(valor.valor),
        })),
      })),
    };

    try {
      setIsSubmitting(true);
      if (isEditing && editingId) {
        await updateConfiguration(editingId, payload);
        setStatus({ message: "Configuración actualizada.", type: "info" });
      } else {
        await createConfiguration(payload);
        setStatus({ message: "Configuración creada.", type: "info" });
      }
      resetForm();
      await loadConfigs();
      setMode("list");
    } catch (error) {
      setStatus({
        message:
          error instanceof Error
            ? error.message
            : "No fue posible guardar la configuración.",
        type: "error",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleEdit = async (codeId: string) => {
    try {
      const detail = await getConfiguration(codeId);
      setSelectedDetail(detail);
      setFormState({
        nombre: detail.nombre,
        comentarios: detail.comentarios || "",
        exogenas_dummies: detail.exogenas_dummies,
        exogenas_normales: detail.exogenas_normales || [],
      });
      setIsEditing(true);
      setEditingId(codeId);
      setMode("form");
    } catch (error) {
      setStatus({
        message:
          error instanceof Error ? error.message : "No fue posible obtener la configuración.",
        type: "error",
      });
    }
  };

  const handleView = async (codeId: string) => {
    try {
      const detail = await getConfiguration(codeId);
      setSelectedDetail(detail);
      setMode("detail");
    } catch (error) {
      setStatus({
        message:
          error instanceof Error ? error.message : "No fue posible obtener la configuración.",
        type: "error",
      });
    }
  };

  const handleDelete = (codeId: string) => {
    setDeleteTarget(codeId);
    setShowDeleteConfirm(true);
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteConfiguration(deleteTarget);
      setStatus({ message: "Configuración eliminada.", type: "info" });
      await loadConfigs();
      setShowDeleteConfirm(false);
      setDeleteTarget(null);
    } catch (error) {
      setStatus({
        message:
          error instanceof Error ? error.message : "No fue posible eliminar la configuración.",
        type: "error",
      });
    }
  };

  const handleExport = async () => {
    if (!configs.length) {
      setStatus({ message: "No hay configuraciones para exportar.", type: "error" });
      return;
    }

    try {
      await exportConfigurations(configs);
      setStatus({ message: "JSON enviado correctamente.", type: "info" });
    } catch (error) {
      setStatus({
        message:
          error instanceof Error
            ? error.message
            : "No fue posible enviar el JSON de configuraciones.",
        type: "error",
      });
    }
  };

  const parseLagValue = (value: string) => {
    const next = Number(value);
    if (!Number.isFinite(next) || !Number.isInteger(next) || next <= 0) {
      throw new Error("El lag debe ser un entero positivo.");
    }
    return next;
  };

  const handleLagAdd = () => {
    try {
      const lag = parseLagValue(lagInput);
      if (reportLags.includes(lag)) {
        setLagError("Ese lag ya existe.");
        return;
      }
      setReportLags((prev) => [...prev, lag].sort((a, b) => a - b));
      setLagInput("");
      setLagError(null);
    } catch (error) {
      setLagError(error instanceof Error ? error.message : "No fue posible agregar el lag.");
    }
  };

  const handleLagUpdate = () => {
    if (editingLagIndex === null) return;
    try {
      const nextLag = parseLagValue(editingLagValue);
      if (
        reportLags.some((value, idx) => idx !== editingLagIndex && value === nextLag)
      ) {
        setLagError("Ese lag ya existe.");
        return;
      }
      setReportLags((prev) => {
        const next = prev.slice();
        next[editingLagIndex] = nextLag;
        return next.sort((a, b) => a - b);
      });
      setLagError(null);
      setEditingLagIndex(null);
      setEditingLagValue("");
    } catch (error) {
      setLagError(error instanceof Error ? error.message : "No fue posible editar el lag.");
    }
  };

  const handleLagDelete = (index: number) => {
    if (reportLags.length <= 1) {
      setLagError("Debes mantener al menos un lag.");
      return;
    }
    setLagError(null);
    setReportLags((prev) => prev.filter((_, idx) => idx !== index));
  };

  const handleCategoryToggle = (category: string) => {
    setReportCategories((prev) => {
      if (prev.includes(category)) {
        return prev.filter((item) => item !== category);
      }
      return [...prev, category];
    });
  };

  const handleCategoryClear = () => {
    setReportCategories([]);
  };

  const handlePipelineChange = (value: string) => {
    setReportPipeline(value);
    if (value !== "turismo") {
      handleCategoryClear();
    }
  };

  const handleGenerateReport = async (configId?: string) => {
    if (!reportPipeline || !reportModel) {
      setStatus({ message: "Selecciona pipeline y modelo.", type: "error" });
      return;
    }
    const targetConfigId = configId ?? reportConfigId;
    if (!targetConfigId) {
      setStatus({ message: "Selecciona una configuración.", type: "error" });
      return;
    }

    if (reportPipeline === "open_meteo" && !reportLocation) {
      setStatus({ message: "Selecciona una ubicación para Open Meteo.", type: "error" });
      return;
    }

    if (reportPipeline === "turismo" && (!reportStartDate || !reportEndDate)) {
      setStatus({
        message: "Define el rango de fechas para turismo.",
        type: "error",
      });
      return;
    }

    if (!reportLags.length) {
      setStatus({ message: "Define al menos un lag válido.", type: "error" });
      return;
    }

    const payload = {
      pipeline: reportPipeline,
      location: reportPipeline === "turismo" ? "" : reportLocation,
      model: reportModel,
      startDate: reportPipeline === "turismo" ? reportStartDate : "",
      endDate: reportPipeline === "turismo" ? reportEndDate : "",
      categories: reportPipeline === "turismo" ? reportCategories : [],
      lags: reportLags,
      configurationId: targetConfigId,
    };

    try {
      setIsReportLoading(true);
      setForecast(null);
      setStatus({ message: "Generando reporte...", type: "info" });
      const data = await generateForecast(payload, targetConfigId);
      setForecast(data);
      setStatus({ message: "Reporte enviado correctamente.", type: "info" });
    } catch (error) {
      setStatus({
        message:
          error instanceof Error ? error.message : "No fue posible generar el reporte.",
        type: "error",
      });
    } finally {
      setIsReportLoading(false);
    }
  };

  const LagControls = () => (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Label className="text-base">Lags (días)</Label>
          <p className="text-sm text-muted-foreground">
            Define los retrasos que alimentarán al modelo.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StepperInput
            value={lagInput}
            onValueChange={(val) => {
              setLagInput(val);
              setLagError(null);
            }}
            min={1}
            className="w-32"
            placeholder="Ej: 7"
          />
          <Button type="button" variant="secondary" onClick={handleLagAdd} className="flex items-center gap-1">
            <Plus className="h-4 w-4" />
            Agregar
          </Button>
        </div>
      </div>
      {lagError ? (
        <p className="text-sm font-semibold text-destructive">{lagError}</p>
      ) : (
        <p className="text-sm text-muted-foreground">
          Solo enteros positivos. Mantén al menos un lag.
        </p>
      )}
      <ul className="space-y-2">
        {reportLags.map((lag, index) => {
          const isEditing = editingLagIndex === index;
          return (
            <li
              key={`${lag}-${index}`}
              className="flex items-center justify-between rounded-lg bg-secondary/70 px-4 py-2"
            >
              {isEditing ? (
                <div className="flex items-center gap-2">
                  <StepperInput
                    value={editingLagValue}
                    onValueChange={(val) => setEditingLagValue(val)}
                    min={1}
                    className="w-24"
                    placeholder="Lag"
                  />
                  <Button type="button" size="icon" variant="ghost" onClick={handleLagUpdate}>
                    <Check className="h-4 w-4" />
                  </Button>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    onClick={() => {
                      setEditingLagIndex(null);
                      setLagError(null);
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
                      disabled={reportLags.length <= 1}
                      onClick={() => handleLagDelete(index)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );

  const pipelineItems = useMemo(() => {
    return options.pipelines;
  }, [options.pipelines]);

  const modelItems = useMemo(() => {
    return options.models;
  }, [options.models]);

  const locationItems = useMemo(() => {
    return options.locations;
  }, [options.locations]);

  const turismoCategories = useMemo(() => {
    return options.turismo?.categories ?? [];
  }, [options.turismo?.categories]);

  if (mode === "form") {
    return (
      <div className="p-8 max-w-3xl space-y-6 mx-auto">
        <h1 className="text-3xl font-bold">{isEditing ? "Editar configuración" : "Nueva configuración"}</h1>
        <form onSubmit={handleSaveConfig} className="space-y-6">
          <Card className="space-y-4">
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Nombre</Label>
                <Input
                  value={formState.nombre}
                  onChange={(event) =>
                    setFormState((prev) => ({ ...prev, nombre: event.target.value }))
                  }
                  required
                />
              </div>
              <div className="space-y-2">
                <Label>Comentarios</Label>
                <Input
                  value={formState.comentarios}
                  onChange={(event) =>
                    setFormState((prev) => ({ ...prev, comentarios: event.target.value }))
                  }
                />
              </div>
            </CardContent>
          </Card>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Exógenas Dummies</h2>
              <Button type="button" variant="outline" onClick={handleAddExogena}>
                + Nueva exógena
              </Button>
            </div>
            {formState.exogenas_dummies.map((exo, idx) => (
              <Card key={`${exo.nombre}-${idx}`} className="space-y-4 border">
                <CardContent className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-1 space-y-2">
                      <Label>Nombre de la exógena</Label>
                      <Input
                        value={exo.nombre}
                        onChange={(event) => handleExogenaName(idx, event.target.value)}
                      />
                    </div>
                    <Button variant="destructive" onClick={() => handleRemoveExogena(idx)}>
                      Eliminar
                    </Button>
                  </div>
                  <div className="space-y-3">
                    {exo.rangos.map((rango, rIdx) => (
                      <div key={rIdx} className="grid gap-3 md:grid-cols-3">
                        <div>
                          <Label>Inicio</Label>
                          <Input
                            type="date"
                            value={rango.inicio}
                            onChange={(event) =>
                              handleRangeUpdate(idx, rIdx, "inicio", event.target.value)
                            }
                          />
                        </div>
                        <div>
                          <Label>Fin</Label>
                          <Input
                            type="date"
                            value={rango.fin}
                            onChange={(event) =>
                              handleRangeUpdate(idx, rIdx, "fin", event.target.value)
                            }
                          />
                        </div>
                        <Button variant="destructive" onClick={() => handleDeleteRange(idx, rIdx)}>
                          Borrar rango
                        </Button>
                      </div>
                    ))}
                    <Button type="button" variant="secondary" onClick={() => handleAddRange(idx)}>
                      + Agregar rango
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Exógenas normales</h2>
              <Button type="button" variant="outline" onClick={handleAddNormalExogena}>
                + Nueva exógena
              </Button>
            </div>
            {formState.exogenas_normales.map((exo, idx) => (
              <Card key={`${exo.nombre}-${idx}`} className="space-y-4 border">
                <CardContent className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-1 space-y-2">
                      <Label>Nombre de la exógena</Label>
                      <Input value={exo.nombre} onChange={(event) => handleNormalNameChange(idx, event.target.value)} />
                    </div>
                    <Button variant="destructive" onClick={() => handleRemoveNormalExogena(idx)}>
                      Eliminar
                    </Button>
                  </div>
                  <div className="space-y-3">
                    {exo.valores.map((valor, valueIdx) => (
                      <div key={`${valor.fecha}-${valueIdx}`} className="grid gap-3 md:grid-cols-3">
                        <div>
                          <Label>Fecha</Label>
                          <Input
                            type="date"
                            value={valor.fecha}
                            onChange={(event) =>
                              handleNormalValueChange(idx, valueIdx, "fecha", event.target.value)
                            }
                          />
                        </div>
                        <div>
                          <Label>Valor</Label>
                          <Input
                            type="number"
                            step="0.01"
                            value={valor.valor}
                            onChange={(event) =>
                              handleNormalValueChange(idx, valueIdx, "valor", event.target.value)
                            }
                          />
                        </div>
                        <Button
                          variant="destructive"
                          onClick={() => handleRemoveNormalValue(idx, valueIdx)}
                        >
                          Eliminar valor
                        </Button>
                      </div>
                    ))}
                    <Button
                      type="button"
                      variant="secondary"
                      onClick={() => handleAddNormalValue(idx)}
                    >
                      + Agregar valor
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="flex gap-3">
            <Button type="submit" disabled={isSubmitting}>
              {isEditing ? "Actualizar configuración" : "Guardar configuración"}
            </Button>
            <Button type="button" variant="ghost" onClick={() => { resetForm(); setMode("list"); }}>
              Cancelar
            </Button>
          </div>
        </form>
      </div>
    );
  }

  if (mode === "detail" && selectedDetail) {
    return (
      <div className="p-8 max-w-3xl space-y-6 mx-auto">
        <Card className="space-y-4">
          <CardHeader>
            <CardTitle className="text-3xl font-bold">{selectedDetail.nombre}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p>
              <strong>Nombre:</strong> {selectedDetail.nombre}
            </p>
            <p>
              <strong>Comentarios:</strong> {selectedDetail.comentarios || "Sin comentarios"}
            </p>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Exógenas Dummies</h3>
              {selectedDetail.exogenas_dummies.length ? (
                selectedDetail.exogenas_dummies.map((exo) => (
                  <div key={exo.nombre} className="space-y-1 rounded-lg border border-dashed p-3">
                    <p className="font-semibold">{exo.nombre}</p>
                    <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
                      {exo.rangos.map((rango, idx) => (
                        <span key={`${rango.inicio}-${idx}`} className="rounded-full border px-3 py-1">
                          {rango.inicio} → {rango.fin}
                        </span>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">No hay exógenas registradas.</p>
              )}
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Exógenas normales</h3>
              {selectedDetail.exogenas_normales.length ? (
                selectedDetail.exogenas_normales.map((exo) => (
                  <div key={exo.nombre} className="space-y-1 rounded-lg border border-dashed p-3">
                    <p className="font-semibold">{exo.nombre}</p>
                    <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
                      {exo.valores.map((valor, idx) => (
                        <span key={`${valor.fecha}-${idx}`} className="rounded-full border px-3 py-1">
                          {valor.fecha}: {valor.valor}
                        </span>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">
                  No hay exógenas normales registradas.
                </p>
              )}
            </div>
          </CardContent>
          <CardFooter className="flex justify-end gap-3">
            <Button variant="outline" onClick={() => setMode("list")}>
              Volver
            </Button>
            <Button onClick={() => handleEdit(selectedDetail.code_id)}>Editar</Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-6">
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Configuraciones</h1>
          <Button onClick={() => { resetForm(); setMode("form"); }}>Nueva configuración</Button>
        </div>
        <StatusMessage message={status.message} type={status.type} />
      </div>

      <Card className="border border-border bg-background/80 shadow-lg">
        <CardHeader>
          <CardTitle>Generar reportes desde configuraciones</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Pipeline</Label>
              <Select
                value={reportPipeline || undefined}
                onValueChange={handlePipelineChange}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Selecciona un pipeline" />
                </SelectTrigger>
                <SelectContent>
                  {pipelineItems.map((pipeline) => (
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
                value={reportModel || undefined}
                onValueChange={setReportModel}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Selecciona un modelo" />
                </SelectTrigger>
                <SelectContent>
                  {modelItems.map((model) => (
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
                value={reportConfigId || undefined}
                onValueChange={(value) => setReportConfigId(value || null)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Selecciona una configuración" />
                </SelectTrigger>
                <SelectContent>
                  {configs.map((config) => (
                    <SelectItem key={config.code_id} value={config.code_id}>
                      {config.nombre}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {reportPipeline === "open_meteo" ? (
            <>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Ubicación</Label>
                  <Select
                    value={reportLocation || undefined}
                    onValueChange={(value) => setReportLocation(value || "")}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Selecciona una ubicación" />
                    </SelectTrigger>
                    <SelectContent>
                      {locationItems.map((location) => (
                        <SelectItem key={location} value={location}>
                          {location}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <LagControls />
            </>
          ) : (
            <>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <Label>Inicio</Label>
                  <Input
                    type="date"
                    value={reportStartDate}
                    onChange={(event) => setReportStartDate(event.target.value)}
                  />
                </div>
                <div>
                  <Label>Fin</Label>
                  <Input
                    type="date"
                    value={reportEndDate}
                    onChange={(event) => setReportEndDate(event.target.value)}
                  />
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Categorías</Label>
                  <Button type="button" variant="ghost" size="sm" onClick={handleCategoryClear}>
                    Todas
                  </Button>
                </div>
                {turismoCategories.length ? (
                  <ScrollArea className="h-32 rounded-lg border border-dashed border-muted p-2">
                    <div className="flex flex-wrap gap-2">
                      {turismoCategories.map((category) => {
                        const isSelected = reportCategories.includes(category);
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
              </div>
              <LagControls />
            </>
          )}
        </CardContent>
        <CardFooter className="flex flex-wrap gap-3">
          <Button onClick={() => handleGenerateReport()} disabled={isReportLoading}>
            {isReportLoading ? "Generando..." : "Generar reportes"}
          </Button>
          <Button variant="outline" onClick={handleExport}>
            Enviar JSON de configuraciones
          </Button>
        </CardFooter>
      </Card>

      {forecast ? (
        <div className="space-y-6">
          {forecast.zero_padding_days > 0 ? (
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
          <MetricsSection mae={forecast.mae} rmse={forecast.rmse} lags={forecast.lags} />
          {forecast.forecast?.length ? <ForecastChart data={forecast.forecast} /> : null}
          {forecast.forecast?.length ? (
            <ObservationsTable rows={forecast.forecast} maxRows={10} />
          ) : null}
        </div>
      ) : null}

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {configs.map((config) => (
          <Card key={config.code_id} className="space-y-3 border">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">{config.nombre}</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-1">
            <p>
              <strong>Nombre:</strong> {config.nombre}
            </p>
            <p>
              <strong>Comentarios:</strong> {config.comentarios || "Sin comentarios"}
            </p>
          </CardContent>
            <CardFooter className="flex flex-wrap gap-2">
              <Button variant="secondary" onClick={() => handleView(config.code_id)}>
                Ver detalle
              </Button>
              <Button variant="outline" onClick={() => handleEdit(config.code_id)}>
                Editar
              </Button>
              <Button variant="destructive" onClick={() => handleDelete(config.code_id)}>
                Eliminar
              </Button>
              <Button variant="ghost" onClick={() => handleGenerateReport(config.code_id)}>
                Enviar configuración
              </Button>
            </CardFooter>
        </Card>
      ))}
      </div>

      <ConfirmDialog
        open={showDeleteConfirm}
        title="Eliminar configuración"
        description="¿Estás seguro? Esto eliminará la configuración y sus exógenas."
        onClose={() => setShowDeleteConfirm(false)}
        onConfirm={confirmDelete}
      />
    </div>
  );
};

export default ConfigPage;
