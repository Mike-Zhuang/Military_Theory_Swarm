import { iconSvg } from "./icon-set.js?v=20260409-03";

export function createExplainPanel(container) {
  container.innerHTML = `
    <h3 class="section-title section-title-with-icon">
      ${iconSvg("guide")}
      <span>战术图例与术语说明</span>
    </h3>

    <div class="guide-strip">
      <div class="guide-title">推荐演示顺序</div>
      <div class="guide-content">先切场景看战术差异，再看 ML 训练产物，最后应用置信度并回放对照。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("info")} 图例</div>
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
      <div class="metric-label card-title">${iconSvg("metric")} 核心指标解释</div>
      <div class="explain-row"><b>覆盖率</b>：被蜂群访问过的网格占比，体现侦察面。</div>
      <div class="explain-row"><b>任务完成率</b>：被成功清除的目标占比。</div>
      <div class="explain-row"><b>平均响应时间</b>：目标从被分配到被清除的平均时延。</div>
      <div class="explain-row"><b>平均链路度</b>：每个存活节点的平均可通信邻居数量。</div>
      <div class="explain-row"><b>生存率</b>：仿真末尾仍可工作的节点比例。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("guide")} 场景观察点</div>
      <div class="explain-row"><b>recon-coverage</b>：看覆盖率与连通性稳定性。</div>
      <div class="explain-row"><b>jam-recovery</b>：看干扰脉冲下恢复速度。</div>
      <div class="explain-row"><b>multi-target</b>：看多目标竞争时分配效率。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("warning")} 常见问题自诊断</div>
      <div class="explain-row">后端连接失败：确认 <code>Backend: FastAPI</code> 已启动，地址为 <code>127.0.0.1:8001</code>。</div>
      <div class="explain-row">无训练产物：先执行“准备数据”并等待任务状态变成 <code>succeeded</code>。</div>
      <div class="explain-row">仿真未变化：确认已点击“将置信度应用到仿真”，并观察右上角事件提示。</div>
    </div>
  `;
}
