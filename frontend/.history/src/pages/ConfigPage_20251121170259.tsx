import React, { useEffect, useState } from "react";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter
} from "../components/ui/card";

type Configuracion = {
  code_id: string;
  nombre: string;
  comentarios: string | null;
};

export default function ConfigPage() {
  const [mode, setMode] =
    useState<"list" | "new" | "detail">("list");

  const [configs, setConfigs] = useState<Configuracion[]>([]);
  const [selected, setSelected] = useState<Configuracion | null>(null);

  const [nombre, setNombre] = useState("");
  const [comentarios, setComentarios] = useState("");
  const [variables, setVariables] = useState("");

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

  const crear = async (e: any) => {
    e.preventDefault();
    await fetch("http://localhost:8000/configuracion/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nombre,
        comentarios,
        variables: variables.split(",").map((v) => v.trim())
      })
    });

    await loadConfigs();
    setMode("list");
  };

  // ================== LIST ==================
  if (mode === "list") {
    return (
      <div className="p-8 space-y-6">
        <h1 className="text-3xl font-bold">Configuraciones</h1>

        <Button onClick={() => setMode("new")}>Nueva Configuración</Button>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 pt-4">
          {configs.map((c) => (
            <Card
              key={c.code_id}
              className="hover:shadow-lg cursor-pointer transition"
              onClick={() => loadDetail(c.code_id)}
            >
              <CardHeader>
                <CardTitle>{c.nombre}</CardTitle>
              </CardHeader>

              <CardContent>
                <p><strong>ID:</strong> {c.code_id}</p>
                <p><strong>Comentarios:</strong> {c.comentarios}</p>
              </CardContent>

              <CardFooter>
                <Button className="w-full">Ver Detalle</Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  // ================== NEW ==================
  if (mode === "new") {
    return (
      <div className="p-8 max-w-xl space-y-6">
        <h1 className="text-3xl font-bold">Nueva Configuración</h1>

        <Card>
          <CardContent className="pt-4 space-y-4">
            <form onSubmit={crear}>
              <label className="text-sm">Nombre</label>
              <Input value={nombre} onChange={(e) => setNombre(e.target.value)} />

              <label className="text-sm pt-2">Comentarios</label>
              <Input value={comentarios}
                     onChange={(e) => setComentarios(e.target.value)} />

              <label className="text-sm pt-2">Variables (excursion, citytour...)</label>
              <Input value={variables}
                     onChange={(e) => setVariables(e.target.value)} />

              <Button type="submit" className="w-full mt-4">Guardar</Button>
            </form>
          </CardContent>
        </Card>

        <Button variant="outline" onClick={() => setMode("list")}>Volver</Button>
      </div>
    );
  }

  // ================== DETAIL ==================
  if (mode === "detail" && selected) {
    return (
      <div className="p-8 max-w-3xl space-y-6">
        <h1 className="text-3xl font-bold">Detalle Configuración</h1>

        <Card>
          <CardHeader>
            <CardTitle>{selected.nombre}</CardTitle>
          </CardHeader>

          <CardContent>
            <p><strong>ID:</strong> {selected.code_id}</p>
            <p><strong>Comentarios:</strong> {selected.comentarios}</p>
          </CardContent>

          <CardFooter>
            <Button variant="outline" className="w-full"
                    onClick={() => setMode("list")}>
              Volver
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return null;
}
