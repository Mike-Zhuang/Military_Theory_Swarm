export function createExplainPanel(container) {
  container.innerHTML = `
    <h3 class="section-title">战术图例与术语</h3>

    <div class="metric-card">
      <div class="metric-label">图例</div>
      <div class="legend-list">
        <div><span class="legend-dot agent"></span>Agent（友方无人节点）</div>
        <div><span class="legend-dot vehicle"></span>vehicle（高价值机动目标）</div>
        <div><span class="legend-dot decoy"></span>decoy（诱饵/干扰目标）</div>
        <div><span class="legend-dot civilian"></span>civilian-object（非战目标）</div>
        <div><span class="legend-line"></span>通信链路（邻域可达）</div>
        <div><span class="legend-fail"></span>灰色节点（失效节点）</div>
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label">核心指标解释</div>
      <div class="explain-row"><b>覆盖率</b>：被蜂群访问过的网格占比，体现侦察面。</div>
      <div class="explain-row"><b>任务完成率</b>：被成功清除的目标占比。</div>
      <div class="explain-row"><b>平均响应时间</b>：目标从被分配到被清除的平均时延。</div>
      <div class="explain-row"><b>平均链路度</b>：每个存活节点的平均可通信邻居数量。</div>
      <div class="explain-row"><b>生存率</b>：仿真末尾仍可工作的节点比例。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label">场景观察点</div>
      <div class="explain-row"><b>recon-coverage</b>：看覆盖率与连通性稳定性。</div>
      <div class="explain-row"><b>jam-recovery</b>：看干扰脉冲下恢复速度。</div>
      <div class="explain-row"><b>multi-target</b>：看多目标竞争时分配效率。</div>
    </div>
  `;
}
