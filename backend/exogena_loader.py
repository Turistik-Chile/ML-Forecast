# exogena_loader.py (REFACTORIZADO)
from __future__ import annotations
import logging
from datetime import date, datetime
from typing import Any, Iterable, Dict
import pandas as pd
from db_connection import get_connection


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _resolve_config_id(cur, config_identifier: str) -> int:
    base = str(config_identifier).strip()
    cur.execute(
        "SELECT id FROM ia.configuracion WHERE LOWER(code_id)=LOWER(?)", (base,)
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Config no encontrada: {config_identifier}")
    return row[0]


def _build_dummy_series(index, ranges):
    mask = pd.Series(0, index=index)
    for start, end in ranges:
        s = _to_date(start)
        e = _to_date(end)
        if s and e:
            mask |= ((index.date >= s) & (index.date <= e)).astype(int)
    return mask


def cargar_exogenas(config_id: str, inicio: str, fin: str) -> pd.DataFrame:
    with get_connection() as conn:
        cur = conn.cursor()

        config_db_id = _resolve_config_id(cur, config_id)

        cur.execute(
            """
            SELECT d.id, d.nombre, r.fecha_inicio, r.fecha_fin
            FROM ia.variable_exogena_dummie d
            LEFT JOIN ia.rango_fechas_dummies r ON r.exogena_id = d.id
            WHERE d.configuracion_id = ? AND d.is_active = 1
        """,
            (config_db_id,),
        )
        dummies = cur.fetchall()

        cur.execute(
            """
            SELECT v.id, v.nombre, f.fecha, f.valor
            FROM ia.variable_exogena v
            LEFT JOIN ia.fecha_valor_exogena f ON f.exogena_id = v.id
            WHERE v.configuracion_id = ? AND v.is_active = 1
        """,
            (config_db_id,),
        )
        normales = cur.fetchall()

    idx = pd.date_range(inicio, fin, freq="D")
    df = pd.DataFrame(index=idx)

    # Dummies
    dummy_ranges = {}
    for _, nombre, f_ini, f_fin in dummies:
        dummy_ranges.setdefault(nombre, []).append((f_ini, f_fin))

    for nombre, ranges in dummy_ranges.items():
        df[nombre] = _build_dummy_series(df.index, ranges)

    # Normales
    temp = {}
    for _, nombre, fecha, valor in normales:
        d = _to_date(fecha)
        if d:
            temp.setdefault(nombre, {})[d] = valor

    for nombre, valores in temp.items():
        df[nombre] = df.index.map(lambda d: valores.get(d.date(), 0))

    df = df.fillna(0)
    logging.info("Exogenas cargadas: %s", list(df.columns))
    return df
