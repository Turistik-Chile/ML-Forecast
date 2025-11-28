import hashlib
import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from db_connection import get_connection

router = APIRouter(prefix="/configuracion", tags=["configuracion"])


# =====================================================
# MODELOS Pydantic
# =====================================================
class RangoFecha(BaseModel):
    inicio: str
    fin: str


class ExogenaDummy(BaseModel):
    nombre: str
    rangos: List[RangoFecha]


class ValorFecha(BaseModel):
    fecha: str
    valor: float


class ExogenaNormal(BaseModel):
    nombre: str
    valores: List[ValorFecha]


class ConfigRequest(BaseModel):
    nombre: str
    comentarios: str | None = None
    exogenas_dummies: List[ExogenaDummy] = Field(default_factory=list)
    exogenas_normales: List[ExogenaNormal] = Field(default_factory=list)


class ConfigExportRequest(BaseModel):
    configuraciones: List[ConfigRequest]


# =====================================================
# HELPERS
# =====================================================
def listar_configuraciones_db():
    """
    Lee las configuraciones disponibles de la base de datos.
    Se reutiliza desde distintos controladores (por ejemplo, /options).

    Nota: el esquema actual solo almacena code_id en ia.configuracion,
    por lo que devolvemos code_id como identificador y nombre de despliegue.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, code_id
            FROM ia.configuracion
            ORDER BY code_id ASC
            """
        )
        rows = cur.fetchall()
    return [
        {"code_id": r[1], "nombre": r[1], "comentarios": None, "id": r[0]}
        for r in rows
    ]


def _validate_config_request(req: ConfigRequest):
    if not req.nombre.strip():
        raise HTTPException(
            status_code=400, detail="El nombre de la configuracion no puede estar vacio"
        )

    for exo in req.exogenas_dummies:
        if not exo.nombre.strip():
            raise HTTPException(status_code=400, detail="Una exogena tiene nombre vacio")

    for exo in req.exogenas_normales:
        if not exo.nombre.strip():
            raise HTTPException(status_code=400, detail="Una exogena normal tiene nombre vacio")
        if not exo.valores:
            raise HTTPException(
                status_code=400, detail="Cada exogena normal debe tener al menos un valor"
            )
        for valor in exo.valores:
            if not valor.fecha.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Cada valor de una exogena normal debe incluir una fecha",
                )


def _clear_config_exogenas(cur, configuracion_db_id: int):
    cur.execute(
        """
        DELETE FROM ia.rango_fechas_dummies
        WHERE exogena_id IN (
            SELECT id FROM ia.variable_exogena_dummie WHERE configuracion_id = ?
        )
        """,
        (configuracion_db_id,),
    )
    cur.execute(
        "DELETE FROM ia.variable_exogena_dummie WHERE configuracion_id = ?",
        (configuracion_db_id,),
    )
    cur.execute(
        """
        DELETE FROM ia.fecha_valor_exogena
        WHERE exogena_id IN (
            SELECT id FROM ia.variable_exogena WHERE configuracion_id = ?
        )
        """,
        (configuracion_db_id,),
    )
    cur.execute(
        "DELETE FROM ia.variable_exogena WHERE configuracion_id = ?",
        (configuracion_db_id,),
    )


def _insert_exogenas_dummies(cur, exogenas: List[ExogenaDummy], configuracion_db_id: int):
    for exo in exogenas:
        cur.execute(
            """
            INSERT INTO ia.variable_exogena_dummie (nombre, configuracion_id, is_active)
            OUTPUT INSERTED.id
            VALUES (?, ?, 1)
            """,
            (exo.nombre, configuracion_db_id),
        )
        exo_id = cur.fetchone()[0]
        for rango in exo.rangos:
            cur.execute(
                """
                INSERT INTO ia.rango_fechas_dummies (exogena_id, fecha_inicio, fecha_fin)
                VALUES (?, ?, ?)
                """,
                (exo_id, rango.inicio, rango.fin),
            )


def _insert_exogenas_normales(cur, exogenas: List[ExogenaNormal], configuracion_db_id: int):
    for exo in exogenas:
        cur.execute(
            """
            INSERT INTO ia.variable_exogena (nombre, configuracion_id, is_active)
            OUTPUT INSERTED.id
            VALUES (?, ?, 1)
            """,
            (exo.nombre, configuracion_db_id),
        )
        exo_id = cur.fetchone()[0]
        for valor in exo.valores:
            cur.execute(
                """
                INSERT INTO ia.fecha_valor_exogena (exogena_id, fecha, valor)
                VALUES (?, ?, ?)
                """,
                (exo_id, valor.fecha, valor.valor),
            )


# =====================================================
# GENERAR CODE ID
# =====================================================
def generar_code_id(req: ConfigRequest) -> str:
    normalized_nombre = req.nombre.strip().lower()
    dummy_names = sorted(
        e.nombre.strip().lower() for e in req.exogenas_dummies if e.nombre.strip()
    )
    normal_names = sorted(
        e.nombre.strip().lower() for e in req.exogenas_normales if e.nombre.strip()
    )
    components = dummy_names + normal_names
    base = normalized_nombre + "|" + "|".join(components)
    return hashlib.sha256(base.encode()).hexdigest()


# =====================================================
# POST /configuracion/create
# =====================================================
@router.post("/create")
def crear_configuracion(req: ConfigRequest):
    _validate_config_request(req)

    code_id = generar_code_id(req)

    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("SELECT code_id FROM ia.configuracion WHERE code_id = ?", (code_id,))
        if cur.fetchone():
            return {"status": "exists", "code_id": code_id}

        cur.execute(
            """
            INSERT INTO ia.configuracion (code_id)
            OUTPUT INSERTED.id
            VALUES (?)
            """,
            (code_id,),
        )
        configuracion_db_id = cur.fetchone()[0]

        _insert_exogenas_dummies(cur, req.exogenas_dummies, configuracion_db_id)
        _insert_exogenas_normales(cur, req.exogenas_normales, configuracion_db_id)

        conn.commit()

    return {"status": "created", "code_id": code_id}


@router.post("/export")
def exportar_configuraciones(req: ConfigExportRequest):
    logging.info("Exportando %s configuraciones desde el frontend", len(req.configuraciones))
    return {"status": "received", "count": len(req.configuraciones)}


# =====================================================
# GET /configuracion/list
# =====================================================
@router.get("/list")
def listar_configuraciones():
    return listar_configuraciones_db()


# =====================================================
# GET /configuracion/{code_id}
# =====================================================
@router.get("/{code_id}")
def obtener_config(code_id: str):
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, code_id
            FROM ia.configuracion
            WHERE code_id = ?
            """,
            (code_id,),
        )
        cfg = cur.fetchone()

        if not cfg:
            raise HTTPException(status_code=404, detail="Configuracion no encontrada")

        cur.execute(
            """
            SELECT id, nombre
            FROM ia.variable_exogena_dummie
            WHERE configuracion_id = ?
            """,
            (cfg[0],),
        )
        exo_rows = cur.fetchall()

        exogenas = []
        for exo_id, exo_nombre in exo_rows:
            cur.execute(
                """
                SELECT fecha_inicio, fecha_fin
                FROM ia.rango_fechas_dummies
                WHERE exogena_id = ?
                """,
                (exo_id,),
            )

            rangos = [{"inicio": r[0], "fin": r[1]} for r in cur.fetchall()]

            exogenas.append({"id": exo_id, "nombre": exo_nombre, "rangos": rangos})

        cur.execute(
            """
            SELECT id, nombre
            FROM ia.variable_exogena
            WHERE configuracion_id = ?
            """,
            (cfg[0],),
        )
        normales_rows = cur.fetchall()
        exogenas_normales = []
        for normal_id, normal_nombre in normales_rows:
            cur.execute(
                """
                SELECT fecha, valor
                FROM ia.fecha_valor_exogena
                WHERE exogena_id = ?
                ORDER BY fecha ASC
                """,
                (normal_id,),
            )
            valores = [{"fecha": row[0], "valor": row[1]} for row in cur.fetchall()]
            exogenas_normales.append(
                {"id": normal_id, "nombre": normal_nombre, "valores": valores}
            )

    return {
        "code_id": cfg[1],
        "nombre": cfg[1],
        "comentarios": None,
        "exogenas_dummies": exogenas,
        "exogenas_normales": exogenas_normales,
    }


@router.put("/{code_id}")
def actualizar_configuracion(code_id: str, req: ConfigRequest):
    _validate_config_request(req)

    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("SELECT id FROM ia.configuracion WHERE code_id = ?", (code_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Configuracion no encontrada")
        configuracion_db_id = row[0]

        _clear_config_exogenas(cur, configuracion_db_id)
        _insert_exogenas_dummies(cur, req.exogenas_dummies, configuracion_db_id)
        _insert_exogenas_normales(cur, req.exogenas_normales, configuracion_db_id)

        conn.commit()

    return {"status": "updated", "code_id": code_id}


@router.delete("/{code_id}")
def eliminar_configuracion(code_id: str):
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("SELECT id FROM ia.configuracion WHERE code_id = ?", (code_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Configuracion no encontrada")
        configuracion_db_id = row[0]

        _clear_config_exogenas(cur, configuracion_db_id)
        cur.execute("DELETE FROM ia.configuracion WHERE code_id = ?", (code_id,))
        conn.commit()

    return {"status": "deleted", "code_id": code_id}
