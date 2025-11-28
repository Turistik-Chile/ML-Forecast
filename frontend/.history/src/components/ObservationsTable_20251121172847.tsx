import type { ForecastPoint } from "../types";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { cn } from "../lib/utils";

type ObservationsTableProps = {
  rows: ForecastPoint[];
  maxRows?: number;
};

const ObservationsTable = ({ rows, maxRows = 10 }: ObservationsTableProps) => {
  const startIndex = Math.max(rows.length - maxRows, 0);
  const limitedRows = rows.slice(startIndex);

  return (
    <Card className="border-none bg-background/80 shadow-xl">
      <CardHeader>
        <CardTitle>Últimas observaciones</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-80 overflow-auto">
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 bg-background/90 backdrop-blur">
              <tr className="text-xs uppercase tracking-wide text-muted-foreground">
                <th className="px-4 py-2">Fecha</th>
                <th className="px-4 py-2">Real</th>
                <th className="px-4 py-2">Predicción</th>
                <th className="px-4 py-2">Error</th>
              </tr>
            </thead>
            <tbody>
              {limitedRows.map((row) => (
                <tr
                  key={`${row.fecha}-${row.prediccion}`}
                  className="border-b border-muted/40 last:border-none"
                >
                  <td className="px-4 py-2">
                    {new Date(row.fecha).toLocaleDateString("es-CL")}
                  </td>
                  <td className="px-4 py-2 font-semibold text-foreground">
                    {row.valor_real.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 font-semibold text-foreground">
                    {row.prediccion.toFixed(2)}
                  </td>
                  <td className="px-4 py-2">
                    <span
                      className={cn(
                        "font-medium",
                        row.error >= 0 ? "text-destructive" : "text-primary"
                      )}
                    >
                      {row.error.toFixed(2)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};

export default ObservationsTable;
