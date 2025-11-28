import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import type { Chart as ChartInstance } from "chart.js";
import type { ForecastPoint } from "../types";

type ForecastChartProps = {
  data: ForecastPoint[];
};

const ForecastChart = ({ data }: ForecastChartProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<ChartInstance | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const labels = data.map((point) => point.fecha);
    const reales = data.map((point) => point.valor_real);
    const predicciones = data.map((point) => point.prediccion);
    const monthFormatter = new Intl.DateTimeFormat("es-CL", {
      month: "short",
      year: "numeric",
    });
    const dateFormatter = new Intl.DateTimeFormat("es-CL");

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    chartRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Valor real",
            data: reales,
            borderColor: "#2563eb",
            backgroundColor: "rgba(37, 99, 235, 0.1)",
            fill: true,
            tension: 0.2,
          },
          {
            label: "Prediccion",
            data: predicciones,
            borderColor: "#f97316",
            backgroundColor: "rgba(249, 115, 22, 0.1)",
            fill: true,
            tension: 0.2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: "index",
        },
        plugins: {
          legend: {
            display: true,
            position: "bottom",
          },
          tooltip: {
            callbacks: {
              title: (items) => {
                if (!items.length) {
                  return "";
                }
                const originalLabel = labels[items[0].dataIndex];
                return dateFormatter.format(new Date(originalLabel));
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              padding: 8,
              callback(value, index) {
                const currentDate = new Date(labels[index]);
                const previousDate =
                  index > 0 ? new Date(labels[index - 1]) : null;
                if (
                  !previousDate ||
                  currentDate.getMonth() !== previousDate.getMonth() ||
                  currentDate.getFullYear() !== previousDate.getFullYear()
                ) {
                  return monthFormatter.format(currentDate);
                }
                return "";
              },
            },
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
    };
  }, [data]);

  return (
    <section className="chart-card">
      <h2>Comparacion de valores</h2>
      <div className="chart-container">
        <canvas ref={canvasRef} />
      </div>
    </section>
  );
};

export default ForecastChart;
