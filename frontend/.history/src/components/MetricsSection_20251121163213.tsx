type MetricsSectionProps = {
  mae: number;
  rmse: number;
  lags: number[];
};

const MetricsSection = ({ mae, rmse, lags }: MetricsSectionProps) => (
  <section className="metrics">
    <h2>Indicadores</h2>
    <div className="metrics-grid">
      <div className="metric">
        <span>MAE</span>
        <strong>{mae.toFixed(3)}</strong>
      </div>
      <div className="metric">
        <span>RMSE</span>
        <strong>{rmse.toFixed(3)}</strong>
      </div>
      <div className="metric">
        <span>Lags</span>
        <strong>{lags.join(", ")}</strong>
      </div>
    </div>
  </section>
);

export default MetricsSection;
