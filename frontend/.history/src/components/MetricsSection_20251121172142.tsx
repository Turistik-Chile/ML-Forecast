import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

type MetricsSectionProps = {
  mae: number;
  rmse: number;
  lags: number[];
};

const formatNumber = (value: number) =>
  new Intl.NumberFormat("es-CL", { maximumFractionDigits: 3 }).format(value);

const MetricsSection = ({ mae, rmse, lags }: MetricsSectionProps) => (
  <Card className="border-none bg-background/80 shadow-xl">
    <CardHeader>
      <CardTitle>Indicadores</CardTitle>
    </CardHeader>
    <CardContent>
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="MAE" value={formatNumber(mae)} helper="Error absoluto medio" />
        <MetricCard label="RMSE" value={formatNumber(rmse)} helper="Raíz del error cuadrático" />
        <div className="rounded-lg bg-secondary/60 p-4">
          <p className="text-sm font-medium text-muted-foreground">Lags usados</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {lags.map((lag) => (
              <Badge key={lag} variant="outline">
                {lag} día{lag === 1 ? "" : "s"}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </CardContent>
  </Card>
);

const MetricCard = ({
  label,
  value,
  helper,
}: {
  label: string;
  value: string;
  helper: string;
}) => (
  <div className="rounded-lg bg-secondary/60 p-4">
    <p className="text-sm font-medium text-muted-foreground">{label}</p>
    <p className="mt-2 text-3xl font-semibold text-foreground">{value}</p>
    <p className="mt-1 text-xs text-muted-foreground">{helper}</p>
  </div>
);

export default MetricsSection;