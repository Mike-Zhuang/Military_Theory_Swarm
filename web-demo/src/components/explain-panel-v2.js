import { iconSvg } from "./icon-set.js?v=20260409-03";

export function createExplainPanel(container) {
  container.innerHTML = `
    <h3 class="section-title section-title-with-icon">
      ${iconSvg("guide")}
      <span>Tactical Briefing</span>
    </h3>

    <div class="guide-strip">
      <div class="guide-title">Recommended Flow</div>
      <div class="guide-content">先切场景看战术差异，再看 ML 训练产物，最后应用置信度并回放对照。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("info")} Legend</div>
      <div class="legend-list">
        <div><span class="legend-dot agent"></span>Agent（友方无人节点）</div>
        <div><span class="legend-dot vehicle"></span>Vehicle（高价值机动目标）</div>
        <div><span class="legend-dot decoy"></span>Decoy（诱饵/干扰目标）</div>
        <div><span class="legend-dot civilian"></span>Civilian Object（非战目标）</div>
        <div><span class="legend-line"></span>Link（邻域通信链路）</div>
        <div><span class="legend-fail"></span>Fail-safe Agent（失效节点）</div>
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("metric")} Metric Notes</div>
      <div class="explain-row"><b>Coverage</b>：被蜂群访问过的网格占比，体现侦察面。</div>
      <div class="explain-row"><b>Completion Rate</b>：被成功清除的目标占比。</div>
      <div class="explain-row"><b>Response Time</b>：目标从被分配到被清除的平均时延。</div>
      <div class="explain-row"><b>Link Degree</b>：每个存活节点的平均可通信邻居数量。</div>
      <div class="explain-row"><b>Survival Rate</b>：仿真末尾仍可工作的节点比例。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("guide")} Scenario Focus</div>
      <div class="explain-row"><b>Recon-Coverage</b>：关注覆盖率与连通性稳定性。</div>
      <div class="explain-row"><b>Jam-Recovery</b>：关注干扰脉冲下恢复速度。</div>
      <div class="explain-row"><b>Multi-Target</b>：关注多目标竞争时的分配效率。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("warning")} Quick Diagnose</div>
      <div class="explain-row">后端连接失败：确认 <code>Backend: FastAPI</code> 已启动，地址为 <code>127.0.0.1:8001</code>。</div>
      <div class="explain-row">无训练产物：先执行准备数据，再等待任务状态到 <code>succeeded</code>。</div>
      <div class="explain-row">仿真未变化：确认已点击应用置信度，并观察画布 HUD 事件提示。</div>
    </div>
  `;
}
