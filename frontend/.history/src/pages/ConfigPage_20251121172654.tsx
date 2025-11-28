// FULL UPDATED VERSION – Beautiful UI with delete range included
import React, { useEffect, useState } from "react";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter
} from "../components/ui/card";

// ==========================
// TYPES
// ==========================
type ExogenaRango = {
  inicio: string;
  fin: string;
};

type Exogena = {
  nombre: string;
  rangos: ExogenaRango[];
};

type Configuracion = {
  code_id: string;
  nombre: string;
  comentarios: string | null;
};

export default function ConfigPage() {
  const [mode, setMode] = useState<"list" | "new" | "detail">("list");

  const [configs, setConfigs] = useState<Configuracion[]>([]);
  const [selected, setSelected] = useState<Configuracion | null>(null);

  const [nombre, setNombre] = useState("");
  const [comentarios, setComentarios] = useState("");

  const [exogenas, setExogenas] = useState<Exogena[]>([]);

  // ==========================
  // LOAD LIST
  // ==========================
  const loadConfigs = async () => {
    const res = await fetch("http://localhost:8000/configuracion/list");
    const data = await res.json();
    setConfigs(data);
  };

  const loadDetail = async (id: string) => {
    const res = await fetch("http://localhost:8000/configuracion/" + id);
    const d = await res.json();
    setSelected(d);
    setMode("detail");
  };

  useEffect(() => {
    loadConfigs();
  }, []);

  // ==========================
  // ADD EXOGENA
  // ==========================
  const agregarExogena = () => {
    setExogenas([...exogenas, { nombre: "", rangos: [] }]);
  };

  const actualizarExogenaNombre = (idx: number, value: string) => {
    const copia = [...exogenas];
    copia[idx].nombre = value;
    setExogenas(copia);
  };

  // ==========================
  // RANGOS
  // ==========================
  const agregarRango = (exoIndex: number) => {
    const copia = [...exogenas];
    copia[exoIndex].rangos.push({ inicio: "", fin: "" });
    setExogenas(copia);
  };

  const actualizarRango = (
    exoIndex: number,
    rangoIndex: number,
    field: "inicio" | "fin",
    value: string
  ) => {
    const copia = [...exogenas];
    copia[exoIndex].rangos[rangoIndex][field] = value;
    setExogenas(copia);
  };

  const borrarRango = (exoIndex: number, rangoIndex: number) => {
    const copia = [...exogenas];
    copia[exoIndex].rangos.splice(rangoIndex, 1);
    setExogenas(copia);
  };

  const borrarExogena = (exoIndex: number) => {
    const copia = [...exogenas];
    copia.splice(exoIndex, 1);
    setExogenas(copia);
  };

  // ==========================
  // CREATE CONFIG
  // ==========================
  const crear = async (e: any) => {
    e.preventDefault();

    await fetch("http://localhost:8000/configuracion/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nombre,
        comentarios,
        exogenas_dummies: exogenas
      })
    });

    await loadConfigs();
    setMode("list");
  };

  // ==========================
  // LIST MODE
  // ==========================
  if (mode === "list") {
    return (
      <div className="p-8 space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold">Configuraciones</h1>
          <Button onClick={() => setMode("new")}>Nueva Configuración</Button>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 pt-4">
          {configs.map((c) => (
            <Card
              key={c.code_id}
              className="hover:shadow-lg cursor-pointer transition border"
              onClick={() => loadDetail(c.code_id)}
            >
              <CardHeader>
                <CardTitle className="text-lg font-semibold">{c.nombre}</CardTitle>
              </CardHeader>

              <CardContent className="text-sm space-y-1">
                <p><strong>ID:</strong> {c.code_id}</p>
                <p className="line-clamp-2"><strong>Comentarios:</strong> {c.comentarios}</p>
              </CardContent>

              <CardFooter>
                <Button className="w-full" variant="secondary">Ver Detalle</Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  // ==========================
  // NEW MODE
  // ==========================
  if (mode === "new") {
    return (
      <div className="p-8 max-w-3xl space-y-6 mx-auto">
        <h1 className="text-3xl font-bold">Nueva Configuración</h1>

        <Card className="shadow-sm">
          <CardContent className="pt-4 space-y-6">
            <form onSubmit={crear} className="space-y-6">
              <div className="space-y-2">
                <Label>Nombre</Label>
                <Input value={nombre} onChange={(e) => setNombre(e.target.value)} />
              </div>

              <div className="space-y-2">
                <Label>Comentarios</Label>
                <Input value={comentarios} onChange={(e) => setComentarios(e.target.value)} />
              </div>

              <h2 className="text-xl font-semibold pt-4">Exógenas Dummies</h2>

              {exogenas.map((exo, idx) => (
                <Card key={idx} className="p-4 mt-4 border shadow-sm">
                  <div className="flex justify-between items-center">
                    <div className="w-full space-y-2">
                      <Label>Nombre exógena</Label>
                      <Input
                        value={exo.nombre}
                        onChange={(e) => actualizarExogenaNombre(idx, e.target.value)}
                      />
                    </div>

                    <Button
                      type="button"
                      variant="destructive"
                      className="ml-4 h-10"
                      onClick={() => borrarExogena(idx)}
                    >
                      Eliminar
                    </Button>
                  </div>

                  <h3 className="mt-4 font-medium">Rangos de fechas</h3>

                  <div className="space-y-4 mt-2">
                    {exo.rangos.map((r, rIdx) => (
                      <div key={rIdx} className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
                        <div>
                          <Label>Inicio</Label>
                          <Input
                            type="date"
                            value={r.inicio}
                            onChange={(e) => actualizarRango(idx, rIdx, "inicio", e.target.value)}
                          />
                        </div>
                        <div>
                          <Label>Fin</Label>
                          <Input
                            type="date"
                            value={r.fin}
                            onChange={(e) => actualizarRango(idx, rIdx, "fin", e.target.value)}
                          />
                        </div>

                        <Button type="button" variant="destructive" onClick={() => borrarRango(idx, rIdx)}>
                          Borrar
                        </Button>
                      </div>
                    ))}
                  </div>

                  <Button
                    type="button"
                    className="mt-4"
                    variant="secondary"
                    onClick={() => agregarRango(idx)}
                  >
                    + Agregar rango
                  </Button>
                </Card>
              ))}

              <Button type="button" variant="outline" onClick={agregarExogena}>
                + Nueva Exógena Dummy
              </Button>

              <Button type="submit" className="w-full mt-6">
                Guardar Configuración
              </Button>
            </form>
          </CardContent>
        </Card>

        <Button variant="ghost" onClick={() => setMode("list")}>Volver</Button>
      </div>
    );
  }

  // ==========================
  // DETAIL MODE
  // ==========================
  if (mode === "detail" && selected) {
    return (
      <div className="p-8 max-w-3xl space-y-6 mx-auto">
        <h1 className="text-3xl font-bold">Detalle Configuración</h1>

        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle>{selected.nombre}</CardTitle>
          </CardHeader>

          <CardContent className="text-sm space-y-2">
            <p><strong>ID:</strong> {selected.code_id}</p>
            <p><strong>Comentarios:</strong> {selected.comentarios}</p>
          </CardContent>

          <CardFooter>
            <Button variant="outline" className="w-full" onClick={() => setMode("list")}>Volver</Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return null;
}