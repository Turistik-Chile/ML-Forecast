import { useMemo, useState } from "react";
import type { ChangeEvent, FormEvent, KeyboardEvent } from "react";

import type { FormValues, OptionsResponse } from "../types";



type ForecastFormProps = {

  options: OptionsResponse;

  values: FormValues;

  loading: boolean;

  onFieldChange: (name: keyof FormValues, value: string | string[]) => void;

  onLagAdd: (lag: number) => void;

  onLagUpdate: (index: number, lag: number) => void;

  onLagDelete: (index: number) => void;

  onSubmit: (event: FormEvent<HTMLFormElement>) => void;

};



const MS_PER_DAY = 24 * 60 * 60 * 1000;
const MAX_ZERO_PADDING_DAYS = 30;

const ForecastForm = ({

  options,

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
  const minGapDays = options.turismo?.min_gap_days ?? 365;

  const selectedCategories =

    values.categories.length > 0 ? values.categories : ["__all__"];

  const [lagInput, setLagInput] = useState("");

  const [lagError, setLagError] = useState<string | null>(null);

  const [editingLagIndex, setEditingLagIndex] = useState<number | null>(null);

  const [editingLagValue, setEditingLagValue] = useState("");
  const historySpanDays = useMemo(() => {
    if (!isTurismo || !values.startDate || !values.endDate) {
      return null;
    }
    const start = new Date(values.startDate);
    const end = new Date(values.endDate);
    const diff = end.getTime() - start.getTime();
    if (!Number.isFinite(diff) || diff < 0) {
      return null;
    }
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

  const handleLagInputChange = (event: ChangeEvent<HTMLInputElement>) => {

    setLagInput(event.target.value);

    if (lagError) {

      setLagError(null);

    }

  };

  const handleLagInputKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {

    if (event.key === "Enter") {

      event.preventDefault();

      handleAddLag();

    }

  };

  const handleAddLag = () => {

    try {

      const lag = parseLagValue(lagInput);

      if (values.lags.includes(lag)) {

        setLagError("Ese lag ya existe.");

        return;

      }

      const exceedsHistory =
        historySpanDays !== null && lag > historySpanDays;
      if (exceedsHistory && lag > MAX_ZERO_PADDING_DAYS) {
        const missingDays = historySpanDays === null ? 0 : lag - historySpanDays;
        setLagError(
          `No hay historial suficiente para un lag de ${lag} dias (faltan aprox. ${missingDays} dias). ` +
            `Cuando no existen datos previos solo puedes agregar lags de hasta ${MAX_ZERO_PADDING_DAYS} dias.`
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

  const startLagEdit = (index: number, currentValue: number) => {

    setEditingLagIndex(index);

    setEditingLagValue(String(currentValue));

    setLagError(null);

  };

  const handleEditingLagChange = (event: ChangeEvent<HTMLInputElement>) => {

    setEditingLagValue(event.target.value);

    if (lagError) {

      setLagError(null);

    }

  };

  const confirmLagEdit = () => {

    if (editingLagIndex === null) {

      return;

    }

    try {

      const nextLag = parseLagValue(editingLagValue);

      const duplicated = values.lags.some(

        (value, index) => index !== editingLagIndex && value === nextLag

      );

      if (duplicated) {

        setLagError("Ese lag ya existe.");

        return;

      }

      const exceedsHistory =
        historySpanDays !== null && nextLag > historySpanDays;
      if (exceedsHistory && nextLag > MAX_ZERO_PADDING_DAYS) {
        const missingDays =
          historySpanDays === null ? 0 : nextLag - historySpanDays;
        setLagError(
          `No hay historial suficiente para un lag de ${nextLag} dias (faltan aprox. ${missingDays} dias). ` +
            `Cuando no existen datos previos solo puedes usar lags de hasta ${MAX_ZERO_PADDING_DAYS} dias.`
        );
        return;
      }

      onLagUpdate(editingLagIndex, nextLag);

      setEditingLagIndex(null);

      setEditingLagValue("");

      setLagError(null);

    } catch (error) {

      const message =

        error instanceof Error ? error.message : "No fue posible editar el lag.";

      setLagError(message);

    }

  };

  const cancelLagEdit = () => {

    setEditingLagIndex(null);

    setEditingLagValue("");

    if (lagError) {

      setLagError(null);

    }

  };

  const handleLagDeleteClick = (index: number) => {

    if (values.lags.length <= 1) {

      setLagError("Debes mantener al menos un lag.");

      return;

    }

    if (editingLagIndex === index) {

      cancelLagEdit();

    }

    setLagError(null);

    onLagDelete(index);

  };


  const handleSelectChange = (event: ChangeEvent<HTMLSelectElement>) => {

    const { name, value, multiple, selectedOptions } = event.target;

    if (multiple) {

      const picks = Array.from(selectedOptions).map((option) => option.value);

      if (picks.includes("__all__")) {

        onFieldChange("categories", []);

      } else {

        onFieldChange(name as keyof FormValues, picks);

      }

      return;

    }

    onFieldChange(name as keyof FormValues, value);

  };



  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {

    const { name, value } = event.target;

    onFieldChange(name as keyof FormValues, value);

  };



  return (

    <form onSubmit={onSubmit}>

      <label>

        Pipeline

        <select name="pipeline" value={values.pipeline} onChange={handleSelectChange}>

          {options.pipelines.map((item) => (

            <option key={item} value={item}>

              {item}

            </option>

          ))}

        </select>

      </label>

      <label>

        Ubicacion

        <select

          name="location"

          value={values.location}

          onChange={handleSelectChange}

          disabled={isTurismo}

        >

          {options.locations.map((item) => (

            <option key={item} value={item}>

              {item}

            </option>

          ))}

        </select>

      </label>

      <label>

        Modelo

        <select name="model" value={values.model} onChange={handleSelectChange}>

          {options.models.map((item) => (

            <option key={item} value={item}>

              {item}

            </option>

          ))}

        </select>

      </label>

      <fieldset className="lags-fieldset">

        <legend>Lags (dias)</legend>

        <div className="lag-input-row">

          <input

            type="number"

            min={1}

            value={lagInput}

            onChange={handleLagInputChange}

            onKeyDown={handleLagInputKeyDown}

            placeholder="Agregar lag (max 30)"

          />

          <button

            type="button"

            className="icon-button add-button"

            onClick={handleAddLag}

            aria-label="Agregar lag"

          >

            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">

              <path

                d="M12 5v14M5 12h14"

                stroke="currentColor"

                strokeWidth="2"

                strokeLinecap="round"

                strokeLinejoin="round"

              />

            </svg>

          </button>

        </div>

        {lagError ? (
          <small className="lag-error">{lagError}</small>
        ) : (
          <small>
            Solo lags enteros positivos. Si el rango no cubre el lag solicitado,
            solo se permiten hasta {MAX_ZERO_PADDING_DAYS} dias con relleno en 0.
          </small>
        )}

        {paddingWarningActive && historySpanDays !== null ? (
          <div className="lag-warning">
            ⚠ El rango seleccionado aporta aprox. {historySpanDays} dias de
            historial y tu mayor lag requiere hasta {maxSelectedLag} dias.
            Se rellenaran cerca de {approxPaddingNeeded} dia(s) con valor 0
            (maximo {MAX_ZERO_PADDING_DAYS}). Considera ampliar el rango o
            reducir los lags para evitarlo.
          </div>
        ) : null}

        <div className="lag-list-container">

          {values.lags.length ? (

            <ul className="lag-list">

              {values.lags.map((lag, index) => (

                <li key={`${lag}-${index}`} className="lag-item">

                  {editingLagIndex === index ? (

                    <div className="lag-edit-row">

                      <input

                        type="number"

                        min={1}

                        value={editingLagValue}

                        onChange={handleEditingLagChange}

                      />

                      <button

                        type="button"

                        className="icon-button confirm-button"

                        onClick={confirmLagEdit}

                        aria-label="Guardar lag"

                      >

                        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">

                          <path

                            d="M5 13l4 4L19 7"

                            stroke="currentColor"

                            strokeWidth="2"

                            strokeLinecap="round"

                            strokeLinejoin="round"

                          />

                        </svg>

                      </button>

                      <button

                        type="button"

                        className="icon-button cancel-button"

                        onClick={cancelLagEdit}

                        aria-label="Cancelar edicion"

                      >

                        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">

                          <path

                            d="M6 6l12 12M6 18L18 6"

                            stroke="currentColor"

                            strokeWidth="2"

                            strokeLinecap="round"

                            strokeLinejoin="round"

                          />

                        </svg>

                      </button>

                    </div>

                  ) : (

                    <>

                      <span className="lag-pill">{lag} dia(s)</span>

                      <div className="lag-actions">

                        <button

                          type="button"

                          className="icon-button"

                          onClick={() => startLagEdit(index, lag)}

                          aria-label={`Editar lag ${lag}`}

                        >

                          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">

                            <path

                              d="M4 20h4l10-10-4-4L4 16v4z"

                              stroke="currentColor"

                              strokeWidth="2"

                              strokeLinecap="round"

                              strokeLinejoin="round"

                              fill="none"

                            />

                            <path

                              d="M12 6l4 4"

                              stroke="currentColor"

                              strokeWidth="2"

                              strokeLinecap="round"

                              strokeLinejoin="round"

                            />

                          </svg>

                        </button>

                        <button

                          type="button"

                          className="icon-button delete-button"

                          onClick={() => handleLagDeleteClick(index)}

                          aria-label={`Eliminar lag ${lag}`}

                          disabled={values.lags.length <= 1}

                        >

                          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">

                            <path

                              d="M6 7h12M10 7V5h4v2m2 0v12a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V7h12z"

                              stroke="currentColor"

                              strokeWidth="2"

                              strokeLinecap="round"

                              strokeLinejoin="round"

                              fill="none"

                            />

                          </svg>

                        </button>

                      </div>

                    </>

                  )}

                </li>

              ))}

            </ul>

          ) : (

            <div className="lag-empty">No hay lags configurados.</div>

          )}

        </div>

      </fieldset>

      {isTurismo ? (

        <>

          <label>

            Fecha de inicio

            <input

              type="date"

              name="startDate"

              value={values.startDate}

              onChange={handleInputChange}

              min={minDate || undefined}

              max={values.endDate || maxDate || undefined}

            />

          </label>

          <label>

            Fecha de termino

            <input

              type="date"

              name="endDate"

              value={values.endDate}

              onChange={handleInputChange}

              min={values.startDate || minDate || undefined}

              max={maxDate || undefined}

            />

          </label>

          <label>

            Categorias

            {turismoCategories.length > 0 ? (

              <select

                name="categories"

                multiple

                size={Math.min(6, turismoCategories.length + 1)}

                value={selectedCategories}

                onChange={handleSelectChange}

              >

                <option value="__all__">Todas las categorias</option>

                {turismoCategories.map((item) => (

                  <option key={item} value={item}>

                    {item}

                  </option>

                ))}

              </select>

            ) : (

              <small>No hay categorias disponibles.</small>

            )}

          </label>

          <small>
            La fecha de termino debe estar al menos {minGapDays} dias antes de hoy.
            Maximo permitido:{" "}
            {maxDate ? new Date(maxDate).toLocaleDateString("es-CL") : "sin datos"}.
          </small>

          <small>

            Mantén presionadas las teclas Control/Command para seleccionar múltiples

            categorías. Deja la selección vacía para incluir todas.

          </small>

        </>

      ) : null}

      <button type="submit" disabled={loading || !values.pipeline}>

        {loading ? "Generando..." : "Generar pronostico"}

      </button>

    </form>

  );

};



export default ForecastForm;

