from __future__ import annotations

import logging
import re
from typing import Sequence

import pandas as pd

from db_connection import get_connection


_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]+$")


def _sanitize_sql_name(value: str, max_len: int = 80) -> str:
    text = value or ""
    text = text.lower().strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    if not text:
        return "na"
    return text[:max_len]


def _sql_type_from_series(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DATETIME"
    if pd.api.types.is_bool_dtype(series):
        return "BIT"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "FLOAT"
    return "NVARCHAR(255)"


def _prepare_table_name(
    pipeline_name: str,
    model_name: str,
    configuration_id: str | None,
    categories: Sequence[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> str:
    cfg_token = _sanitize_sql_name(str(configuration_id)) if configuration_id else "nocfg"
    cat_list = [
        _sanitize_sql_name(str(cat)) for cat in (categories or []) if str(cat).strip()
    ]
    cat_token = "_".join(cat_list) if cat_list else "all"
    start_token = _sanitize_sql_name(start_date or "na")
    end_token = _sanitize_sql_name(end_date or "na")
    base = (
        f"dataset_{pipeline_name}_{model_name}_cfg{cfg_token}_{cat_token}"
        f"_{start_token}_{end_token}"
    )
    return _sanitize_sql_name(base, max_len=110)


def persist_dataset_to_db(
    df: pd.DataFrame,
    *,
    pipeline_name: str,
    model_name: str,
    configuration_id: str | None = None,
    categories: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    schema: str = "ia",
) -> str:
    if df is None or df.empty:
        raise ValueError("No hay datos para persistir.")

    data = df.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index().rename(columns={"index": "fecha"})
    if "fecha" not in data.columns:
        raise ValueError("El dataframe debe contener una columna 'fecha'.")

    table_name = _prepare_table_name(
        pipeline_name,
        model_name,
        configuration_id,
        categories,
        start_date,
        end_date,
    )
    full_table = f"{schema}.{table_name}"

    col_map: dict[str, str] = {}
    used: set[str] = set()
    for column in data.columns:
        safe = _sanitize_sql_name(column, max_len=60)
        if not safe:
            safe = "col"
        base = safe
        counter = 1
        while safe in used:
            safe = f"{base}_{counter}"
            counter += 1
        used.add(safe)
        col_map[column] = safe

    data_sql = data.rename(columns=col_map)

    col_defs = [
        f"[{safe}] {_sql_type_from_series(data[col])}"
        for col, safe in col_map.items()
    ]

    drop_sql = f"""
    IF OBJECT_ID('{full_table}', 'U') IS NOT NULL
        DROP TABLE {full_table};
    """
    create_sql = f"""
    CREATE TABLE {full_table} (
        {', '.join(col_defs)}
    );
    """

    safe_cols = list(col_map.values())
    placeholders = ", ".join("?" for _ in safe_cols)
    insert_sql = f"""
    INSERT INTO {full_table} ({', '.join(f'[{c}]' for c in safe_cols)})
    VALUES ({placeholders});
    """

    values = [
        tuple(row)
        for row in data_sql[safe_cols].itertuples(index=False, name=None)
    ]

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(drop_sql)
        cur.execute(create_sql)
        cur.fast_executemany = True
        cur.executemany(insert_sql, values)
        conn.commit()

    logging.info("Dataset persistido en %s (%d filas).", full_table, len(values))
    logging.info("Columnas del dataset: %s", list(data.columns))
    return full_table
