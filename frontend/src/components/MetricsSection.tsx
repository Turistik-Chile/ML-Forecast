import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

type MetricsSectionProps = {
  mae: number | null;
  rmse: number | null;
  lags: number[];
  forecastOnly?: boolean;
};

const formatNumber = (value: number | null) =>
  value === null || value === undefined
    ? "-"
    : new Intl.NumberFormat("es-CL", { maximumFractionDigits: 3 }).format(value);

const MetricsSection = ({ mae, rmse, lags, forecastOnly = false }: MetricsSectionProps) => {
  const metricsAvailable = mae !== null && mae !== undefined && rmse !== null && rmse !== undefined;
  const helperText = forecastOnly && !metricsAvailable ? "No disponible en forecast puro" : undefined;

  return (
    <Card className="border-none bg-background/80 shadow-xl">
      <CardHeader>
        <CardTitle>Indicadores</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard
            label="MAE"
            value={formatNumber(mae)}
            helper={helperText || "Error absoluto medio"}
          />
          <MetricCard
            label="RMSE"
            value={formatNumber(rmse)}
            helper={helperText || "Raiz del error cuadratico"}
          />
          <div className="rounded-lg bg-secondary/60 p-4">
            <p className="text-sm font-medium text-muted-foreground">Lags usados</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {lags.map((lag) => (
                <Badge key={lag} variant="outline">
                  {lag} dia{lag === 1 ? "" : "s"}
                </Badge>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const MetricCard = ({
  label,
  value,
  helper,
}: {
  label: string;
  value: string;
  helper?: string;
}) => (
  <div className="rounded-lg bg-secondary/60 p-4">
    <p className="text-sm font-medium text-muted-foreground">{label}</p>
    <p className="mt-2 text-3xl font-semibold text-foreground">{value}</p>
    {helper ? <p className="mt-1 text-xs text-muted-foreground">{helper}</p> : null}
  </div>
);

export default MetricsSection;
