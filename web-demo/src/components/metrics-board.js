export class MetricsBoard {
  constructor(container) {
    this.container = container;
  }

  renderSingle(summary, frame, maxFrame, runName) {
    const aliveCount = frame.agents.filter((agent) => agent.alive).length;
    const activeTargets = frame.targets.filter((target) => target.active).length;

    this.container.innerHTML = `
      <h3 class="section-title">Mission Metrics</h3>
      <div class="metric-card">
        <div class="metric-label">Current Strategy</div>
        <div class="metric-value">${runName}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Coverage</div>
        <div class="metric-value">${(summary.coverage * 100).toFixed(1)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Completion Rate</div>
        <div class="metric-value">${(summary.taskCompletionRate * 100).toFixed(1)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Response Time</div>
        <div class="metric-value">${summary.avgResponseTime.toFixed(2)} s</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Survival Rate</div>
        <div class="metric-value">${(summary.survivalRate * 100).toFixed(1)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Link Degree</div>
        <div class="metric-value">${summary.avgLinkDegree.toFixed(2)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Realtime State</div>
        <div class="metric-label">Frame: ${frame.t}/${maxFrame}</div>
        <div class="metric-label">Alive Agents: ${aliveCount}</div>
        <div class="metric-label">Active Targets: ${activeTargets}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Briefing Notes</div>
        <div class="metric-label">高覆盖率 + 短响应时延：协同效率更稳定。</div>
        <div class="metric-label">链路度下降 + 生存率下降：通信与失效压力增大。</div>
      </div>
    `;
  }

  renderCompare(primary, secondary, frameIndex, maxFrame) {
    const primaryAlive = primary.frame.agents.filter((agent) => agent.alive).length;
    const secondaryAlive = secondary.frame.agents.filter((agent) => agent.alive).length;
    const primaryActiveTargets = primary.frame.targets.filter((target) => target.active).length;
    const secondaryActiveTargets = secondary.frame.targets.filter((target) => target.active).length;

    this.container.innerHTML = `
      <h3 class="section-title">Mission Metrics</h3>
      <div class="metric-card">
        <div class="metric-label">Synchronized Frame</div>
        <div class="metric-value">${frameIndex} / ${maxFrame}</div>
      </div>
      ${this.compareCard("Left View", primary.runName, primary.summary, primaryAlive, primaryActiveTargets)}
      ${this.compareCard("Right View", secondary.runName, secondary.summary, secondaryAlive, secondaryActiveTargets)}
    `;
  }

  compareCard(title, runName, summary, aliveCount, activeTargets) {
    return `
      <div class="metric-card">
        <div class="metric-label">${title}</div>
        <div class="metric-value">${runName}</div>
        <div class="metric-label">Completion ${(summary.taskCompletionRate * 100).toFixed(1)}%</div>
        <div class="metric-label">Coverage ${(summary.coverage * 100).toFixed(1)}%</div>
        <div class="metric-label">Response ${summary.avgResponseTime.toFixed(2)} s</div>
        <div class="metric-label">Alive Agents ${aliveCount}</div>
        <div class="metric-label">Remaining Targets ${activeTargets}</div>
      </div>
    `;
  }

  render(summary, frame, maxFrame, runName) {
    this.renderSingle(summary, frame, maxFrame, runName);
  }
}
