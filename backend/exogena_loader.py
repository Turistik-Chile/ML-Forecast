from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

from db_connection import get_connection


def _to_date(value: Any) -> date | None:
    """Normalize different representations of a date into datetime.date."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.to_datetime(value)
    except (ValueError, TypeError):
        return None
    if parsed is pd.NaT:
        return None
    return parsed.date()


def _resolve_config_id(cur, config_identifier: str) -> int:
    """
    Devuelve el ID entero de ia.configuracion a partir de un identificador.
    Acepta:
      - code_id exacto
      - code_id con prefijo 'cfg_' (se quita y se vuelve a probar)
      - id numerico directo
    """
    base = str(config_identifier).strip()
    candidates = [base]

    if base.startswith("cfg_"):
        candidates.append(base[len("cfg_") :])

    lower = base.lower()
    if lower != base:
        candidates.append(lower)

    if base.isdigit():
        candidates.append(int(base))

    for candidate in candidates:
        if isinstance(candidate, int):
            cur.execute("SELECT id FROM ia.configuracion WHERE id = ?", (candidate,))
        else:
            cur.execute(
                "SELECT id FROM ia.configuracion WHERE LOWER(code_id) = LOWER(?)",
                (candidate,),
            )
        row = cur.fetchone()
        if row:
            return row[0]

    raise ValueError(f"Configuracion no encontrada para identificador={config_identifier}")


def _build_dummy_series(
    index: pd.DatetimeIndex, ranges: Iterable[tuple[date | None, date | None]]
) -> pd.Series:
    """
    Construye una serie de 0/1 para un conjunto de rangos.
    Por defecto todo es 0 y se activa (1) en cada rango valido.
    """
    mask = pd.Series(False, index=index)
    for start, end in ranges:
        start_date = _to_date(start)
        end_date = _to_date(end)
        if start_date is None or end_date is None:
            continue
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        mask |= (index.date >= start_date) & (index.date <= end_date)
    return mask.astype(int)


def cargar_exogenas(config_id: str, inicio: str, fin: str) -> pd.DataFrame:
    """
    Retorna un DataFrame con todas las exogenas (normales + dummies)
    entre la fecha inicio y fin.
    """
    with get_connection() as conn:
        cur = conn.cursor()

        configuracion_db_id = _resolve_config_id(cur, config_id)

        cur.execute(
            """
            SELECT d.id, d.nombre, r.fecha_inicio, r.fecha_fin
            FROM ia.variable_exogena_dummie d
            LEFT JOIN ia.rango_fechas_dummies r ON r.exogena_id = d.id
            WHERE d.configuracion_id = ? AND d.is_active = 1
        """,
            (configuracion_db_id,),
        )
        dummies = cur.fetchall()

        cur.execute(
            """
            SELECT v.id, v.nombre, f.fecha, f.valor
            FROM ia.variable_exogena v
            LEFT JOIN ia.fecha_valor_exogena f ON f.exogena_id = v.id
            WHERE v.configuracion_id = ? AND v.is_active = 1
        """,
            (configuracion_db_id,),
        )
        normales = cur.fetchall()

    dummy_names = [row[1] for row in dummies]
    normal_names = [row[1] for row in normales]
    overlap = sorted(set(dummy_names) & set(normal_names))
    logging.info(
        "Exogenas cargadas para config %s | dummies=%s | normales=%s",
        config_id,
        dummy_names,
        normal_names,
    )
    if overlap:
        logging.warning(
            "Nombres duplicados entre dummies y normales: %s. Se usaran las columnas tal como vienen.",
            overlap,
        )

    idx = pd.date_range(inicio, fin, freq="D")
    df = pd.DataFrame(index=idx)

    dummy_ranges: dict[str, list[tuple[date | None, date | None]]] = {}
    for _, nombre, f_ini, f_fin in dummies:
        dummy_ranges.setdefault(nombre, []).append((f_ini, f_fin))

    for nombre, ranges in dummy_ranges.items():
        df[nombre] = _build_dummy_series(df.index, ranges)

    temp: dict[str, dict[date, Any]] = {}
    for _, nombre, fecha, valor in normales:
        if nombre not in temp:
            temp[nombre] = {}
        fecha_key = _to_date(fecha)
        if fecha_key is None:
            continue
        temp[nombre][fecha_key] = valor

    for nombre, valores in temp.items():
        df[nombre] = df.index.map(lambda d: valores.get(_to_date(d), 0))

    df = df.fillna(0)
    expected_cols = list(dict.fromkeys(list(dummy_ranges.keys()) + normal_names))
    missing_cols = [col for col in expected_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_cols]
    logging.info(
        "DataFrame exogenas construido (%d filas) | columnas=%s | faltantes=%s | extra=%s",
        len(df),
        df.columns.tolist(),
        missing_cols,
        extra_cols,
    )

    df.attrs["expected_exogenas"] = expected_cols
    return df
