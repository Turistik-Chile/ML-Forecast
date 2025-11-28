# exogena_endpoints.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db_connection import get_connection

router = APIRouter(prefix="/exogena", tags=["exogenas"])


# ---------------------------------------
# CREAR VARIABLE EXÓGENA
# ---------------------------------------
class ExogenaRequest(BaseModel):
    nombre: str
    configuracion_id: str
    is_dummie: bool = False


@router.post("/create")
def crear_exogena(req: ExogenaRequest):
    """
    Crea una variable ex?gena (normal o dummie).
    """
    tabla = "ia.variable_exogena_dummie" if req.is_dummie else "ia.variable_exogena"

    with get_connection() as conn:
        cur = conn.cursor()

        # Validar configuraci?n
        cur.execute(
            "SELECT id FROM ia.configuracion WHERE code_id = ?", (req.configuracion_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Configuraci?n no encontrada")
        configuracion_db_id = row[0]

        # Insertar ex?gena
        cur.execute(
            f"""
            INSERT INTO {tabla} (nombre, configuracion_id)
            OUTPUT inserted.id
            VALUES (?, ?)
        """,
            (req.nombre, configuracion_db_id),
        )

        new_id = cur.fetchone()[0]
        conn.commit()

    return {"status": "created", "exogena_id": new_id}

# ---------------------------------------
# CREAR RANGO DE FECHAS PARA DUMMIES
# ---------------------------------------
class RangoRequest(BaseModel):
    exogena_id: int
    fecha_inicio: str
    fecha_fin: str


@router.post("/rango")
def crear_rango(req: RangoRequest):
    """
    Crea un rango activo para una variable exógena tipo dummie.
    """
    with get_connection() as conn:
        cur = conn.cursor()

        # Validar dummy
        cur.execute(
            "SELECT 1 FROM ia.variable_exogena_dummie WHERE id = ?", req.exogena_id
        )
        if not cur.fetchone():
            raise HTTPException(404, "Exógena dummie no encontrada")

        # Insertar rango
        cur.execute(
            """
            INSERT INTO ia.rango_fechas_dummies (exogena_id, fecha_inicio, fecha_fin)
            VALUES (?, ?, ?)
        """,
            (req.exogena_id, req.fecha_inicio, req.fecha_fin),
        )

        conn.commit()

    return {"status": "created"}
