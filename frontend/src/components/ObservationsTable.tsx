import type { ForecastPoint } from "../types";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { cn } from "../lib/utils";

type ObservationsTableProps = {
  rows: ForecastPoint[];
  maxRows?: number;
  forecastOnly?: boolean;
};

const formatNumber = (value: number | null) =>
  value === null || value === undefined ? "-" : value.toFixed(2);

const ObservationsTable = ({ rows, maxRows = 10, forecastOnly = false }: ObservationsTableProps) => {
  const startIndex = Math.max(rows.length - maxRows, 0);
  const limitedRows = rows.slice(startIndex);

  return (
    <Card className="border-none bg-background/80 shadow-xl">
      <CardHeader>
        <CardTitle>Ultimas observaciones</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-80 overflow-auto">
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 bg-background/90 backdrop-blur">
              <tr className="text-xs uppercase tracking-wide text-muted-foreground">
                <th className="px-4 py-2">Fecha</th>
                {!forecastOnly ? <th className="px-4 py-2">Real</th> : null}
                <th className="px-4 py-2">Prediccion</th>
                {!forecastOnly ? <th className="px-4 py-2">Error</th> : null}
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
                  {!forecastOnly ? (
                    <td className="px-4 py-2 font-semibold text-foreground">
                      {formatNumber(row.valor_real)}
                    </td>
                  ) : null}
                  <td className="px-4 py-2 font-semibold text-foreground">
                    {formatNumber(row.prediccion)}
                  </td>
                  {!forecastOnly ? (
                    <td className="px-4 py-2">
                      <span
                        className={cn(
                          "font-medium",
                          row.error !== null && row.error !== undefined && row.error >= 0
                            ? "text-destructive"
                            : "text-primary"
                        )}
                      >
                        {formatNumber(row.error)}
                      </span>
                    </td>
                  ) : null}
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
