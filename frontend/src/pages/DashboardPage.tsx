import { modelResults, taskProgress } from '../data/evalResults'

function formatPct(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(2)}%`
}

function DashboardPage() {
  return (
    <section className="dashboard-page">
      <div className="dashboard-intro">
        <h1>Evaluation dashboard</h1>
        <p>
          Snapshot transcribed from{' '}
          <code>docs/planning/nextSteps.md</code> — not a live read of{' '}
          <code>artifacts/</code>, which stays gitignored and empty outside
          Colab/Drive. Swap <code>src/data/evalResults.ts</code> for a real
          artifact/API reader once results are wired up.
        </p>
      </div>

      <h2>Task progress</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Task</th>
              <th>State</th>
              <th>Decision</th>
            </tr>
          </thead>
          <tbody>
            {taskProgress.map((task) => (
              <tr key={task.task}>
                <td>{task.task}</td>
                <td>
                  <span className={`state-badge ${task.state}`}>{task.label}</span>
                </td>
                <td>{task.decision}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Model &amp; fusion ablations</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>Clean</th>
              <th>Robustness</th>
              <th>Locked 50/50</th>
              <th>Status</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {modelResults.map((row) => (
              <tr key={row.component}>
                <td>{row.component}</td>
                <td>{formatPct(row.clean)}</td>
                <td>{formatPct(row.robustness)}</td>
                <td className="locked-cell">{formatPct(row.locked)}</td>
                <td>
                  <span className={`status-badge ${row.status}`}>{row.status}</span>
                </td>
                <td className="note-cell">{row.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default DashboardPage
