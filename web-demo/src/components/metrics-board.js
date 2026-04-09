export class MetricsBoard {
  constructor(container) {
    this.container = container;
  }

  renderSingle(summary, frame, maxFrame, runName) {
    const aliveCount = frame.agents.filter((agent) => agent.alive).length;
    const activeTargets = frame.targets.filter((target) => target.active).length;

    this.container.innerHTML = `
      <h3 class="section-title">Metrics Board</h3>
      <div class="metric-card">
        <div class="metric-label">当前策略</div>
        <div class="metric-value">${runName}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">覆盖率</div>
        <div class="metric-value">${(summary.coverage * 100).toFixed(1)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">任务完成率</div>
        <div class="metric-value">${(summary.taskCompletionRate * 100).toFixed(1)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">平均响应时间</div>
        <div class="metric-value">${summary.avgResponseTime.toFixed(2)} s</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">生存率</div>
        <div class="metric-value">${(summary.survivalRate * 100).toFixed(1)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">平均链路度</div>
        <div class="metric-value">${summary.avgLinkDegree.toFixed(2)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">实时状态</div>
        <div class="metric-label">Frame: ${frame.t}/${maxFrame}</div>
        <div class="metric-label">Alive Agents: ${aliveCount}</div>
        <div class="metric-label">Active Targets: ${activeTargets}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">解读提示</div>
        <div class="metric-label">覆盖率高 + 响应短：协同策略更稳。</div>
        <div class="metric-label">链路度低 + 生存率降：通信/失效压力更大。</div>
      </div>
    `;
  }

  renderCompare(primary, secondary, frameIndex, maxFrame) {
    const primaryAlive = primary.frame.agents.filter((agent) => agent.alive).length;
    const secondaryAlive = secondary.frame.agents.filter((agent) => agent.alive).length;
    const primaryActiveTargets = primary.frame.targets.filter((target) => target.active).length;
    const secondaryActiveTargets = secondary.frame.targets.filter((target) => target.active).length;

    this.container.innerHTML = `
      <h3 class="section-title">Metrics Board</h3>
      <div class="metric-card">
        <div class="metric-label">双视图同步帧</div>
        <div class="metric-value">${frameIndex} / ${maxFrame}</div>
      </div>
      ${this.compareCard("左视图", primary.runName, primary.summary, primaryAlive, primaryActiveTargets)}
      ${this.compareCard("右视图", secondary.runName, secondary.summary, secondaryAlive, secondaryActiveTargets)}
    `;
  }

  compareCard(title, runName, summary, aliveCount, activeTargets) {
    return `
      <div class="metric-card">
        <div class="metric-label">${title}</div>
        <div class="metric-value">${runName}</div>
        <div class="metric-label">完成率 ${(summary.taskCompletionRate * 100).toFixed(1)}%</div>
        <div class="metric-label">覆盖率 ${(summary.coverage * 100).toFixed(1)}%</div>
        <div class="metric-label">响应时间 ${summary.avgResponseTime.toFixed(2)} s</div>
        <div class="metric-label">存活 Agent ${aliveCount}</div>
        <div class="metric-label">剩余目标 ${activeTargets}</div>
      </div>
    `;
  }

  render(summary, frame, maxFrame, runName) {
    this.renderSingle(summary, frame, maxFrame, runName);
  }
}
