import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastPoint } from "../types";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import "./ForecastChart.css";

type ForecastChartProps = {
  data: ForecastPoint[];
  forecastOnly?: boolean;
};

const ForecastChart = ({ data, forecastOnly = false }: ForecastChartProps) => {
  const chartData = useMemo(
    () =>
      data.map((point) => {
        const date = new Date(point.fecha);
        return {
          fecha: date.getTime(),
          fechaLabel: date.toLocaleDateString("es-CL", {
            month: "short",
            day: "numeric",
          }),
          valorReal: point.valor_real,
          prediccion: point.prediccion,
        };
      }),
    [data]
  );

  const hasReal = chartData.some(
    (item) => item.valorReal !== null && item.valorReal !== undefined
  );

  return (
    <Card className="border-none bg-background/80 shadow-xl">
      <CardHeader>
        <CardTitle>{forecastOnly ? "Forecast proyectado" : "Comparacion de valores"}</CardTitle>
      </CardHeader>
      <CardContent className="h-[360px]">
        <div className="chart-container h-full w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ left: 12, right: 12, top: 10 }}>
              <defs>
                <linearGradient id="colorReal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--chart-primary)" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="var(--chart-primary)" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorPred" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--chart-accent)" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="var(--chart-accent)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="4 4" stroke="var(--chart-muted)" />
              <XAxis
                dataKey="fecha"
                type="number"
                domain={["dataMin", "dataMax"]}
                stroke="var(--chart-muted-foreground)"
                tickLine={false}
                axisLine={false}
                minTickGap={30}
                tickFormatter={(value) =>
                  new Date(value).toLocaleDateString("es-CL", {
                    month: "short",
                    day: "numeric",
                  })
                }
              />
              <YAxis
                stroke="var(--chart-muted-foreground)"
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--chart-bg)",
                  borderColor: "var(--chart-muted)",
                  color: "var(--chart-fg)",
                  borderRadius: 12,
                }}
                labelFormatter={(value) =>
                  new Date(value).toLocaleDateString("es-CL", {
                    year: "numeric",
                    month: "short",
                    day: "numeric",
                  })
                }
                cursor={{ stroke: "var(--chart-muted)", strokeWidth: 2 }}
              />
              <Legend />
              {hasReal ? (
                <Area
                  type="monotone"
                  dataKey="valorReal"
                  name="Valor real"
                  stroke="var(--chart-primary)"
                  strokeWidth={2}
                  fill="url(#colorReal)"
                />
              ) : null}
              <Area
                type="monotone"
                dataKey="prediccion"
                name={forecastOnly ? "Prediccion (forecast)" : "Prediccion"}
                stroke="var(--chart-accent)"
                strokeWidth={2}
                fill="url(#colorPred)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default ForecastChart;
