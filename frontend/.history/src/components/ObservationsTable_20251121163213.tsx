import type { ForecastPoint } from "../types";

type ObservationsTableProps = {
  rows: ForecastPoint[];
  maxRows?: number;
};

const ObservationsTable = ({ rows, maxRows = 10 }: ObservationsTableProps) => {
  const startIndex = Math.max(rows.length - maxRows, 0);
  const limitedRows = rows.slice(startIndex);

  return (
    <section className="table-card">
      <h2>Ultimas observaciones</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Fecha</th>
              <th>Real</th>
              <th>Prediccion</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody>
            {limitedRows.map((row) => (
              <tr key={`${row.fecha}-${row.prediccion}`}>
                <td>{new Date(row.fecha).toLocaleDateString("es-CL")}</td>
                <td>{row.valor_real.toFixed(2)}</td>
                <td>{row.prediccion.toFixed(2)}</td>
                <td>{row.error.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default ObservationsTable;
