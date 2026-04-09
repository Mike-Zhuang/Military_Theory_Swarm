import { iconSvg } from "./icon-set.js";

const DEFAULT_BACKEND = "http://127.0.0.1:8001";
const DEFAULT_TRAIN_IMAGES = "ml-module/data/visdrone/raw/VisDrone2019-DET-train/images";
const DEFAULT_TRAIN_ANNOTATIONS = "ml-module/data/visdrone/raw/VisDrone2019-DET-train/annotations";
const DEFAULT_VAL_IMAGES = "ml-module/data/visdrone/raw/VisDrone2019-DET-val/images";
const DEFAULT_VAL_ANNOTATIONS = "ml-module/data/visdrone/raw/VisDrone2019-DET-val/annotations";

function parseError(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function safeNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function createMlLab({
  container,
  onApplyConfidence,
}) {
  const state = {
    backendBaseUrl: DEFAULT_BACKEND,
    activeJobId: "",
    activeRunId: "",
    runs: [],
    pollingTimer: null,
    datasetSummary: null,
  };

  container.innerHTML = `
    <h3 class="section-title section-title-with-icon">
      ${iconSvg("guide")}
      <span>ML 在线实验台（完整教程模式）</span>
    </h3>

    <div class="workflow-grid">
      <article class="workflow-step">
        <div class="workflow-head">${iconSvg("dataset")} 第 1 步：准备数据</div>
        <div class="workflow-desc">推荐使用官方 train+val，保证演示结果可信且可复现实验。</div>
      </article>
      <article class="workflow-step">
        <div class="workflow-head">${iconSvg("train")} 第 2 步：提交训练</div>
        <div class="workflow-desc">训练完成后将自动生成曲线、模型摘要和类别置信度文件。</div>
      </article>
      <article class="workflow-step">
        <div class="workflow-head">${iconSvg("evaluate")} 第 3 步：查看评估</div>
        <div class="workflow-desc">重点看混淆矩阵与类别准确率，准备课堂解释材料。</div>
      </article>
      <article class="workflow-step">
        <div class="workflow-head">${iconSvg("apply")} 第 4 步：应用到仿真</div>
        <div class="workflow-desc">将当前 run 的置信度注入仿真，观察协同性能变化。</div>
      </article>
    </div>

    <div class="control-group">
      <label for="backend-url">后端地址</label>
      <input id="backend-url" type="text" value="${DEFAULT_BACKEND}" />
      <small>默认地址：127.0.0.1:8001，若更改端口需同步更新。</small>
    </div>

    <div class="control-group control-group-inline">
      <button id="health-btn">${iconSvg("info")} 检查后端状态</button>
      <button id="refresh-runs-btn">${iconSvg("guide")} 刷新 Run 列表</button>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("dataset")} 数据准备参数</div>
      <div class="control-group">
        <label for="split-mode">数据划分策略</label>
        <select id="split-mode">
          <option value="official-val">official-val（官方 train/val）</option>
          <option value="auto-split">auto-split（仅 train 自动切分）</option>
        </select>
      </div>
      <div class="control-group">
        <label for="train-images-dir">train images 目录</label>
        <input id="train-images-dir" type="text" value="${DEFAULT_TRAIN_IMAGES}" />
      </div>
      <div class="control-group">
        <label for="train-ann-dir">train annotations 目录</label>
        <input id="train-ann-dir" type="text" value="${DEFAULT_TRAIN_ANNOTATIONS}" />
      </div>
      <div class="control-group">
        <label for="val-images-dir">val images 目录</label>
        <input id="val-images-dir" type="text" value="${DEFAULT_VAL_IMAGES}" />
      </div>
      <div class="control-group">
        <label for="val-ann-dir">val annotations 目录</label>
        <input id="val-ann-dir" type="text" value="${DEFAULT_VAL_ANNOTATIONS}" />
      </div>
      <div class="control-group two-col">
        <div>
          <label for="subset-size">每类 train 样本上限</label>
          <input id="subset-size" type="number" min="120" step="20" value="900" />
        </div>
        <div>
          <label for="val-subset-size">每类 val 样本上限</label>
          <input id="val-subset-size" type="number" min="0" step="20" value="0" />
          <small>0 表示 val 全量样本。</small>
        </div>
      </div>
      <div class="control-group">
        <button id="prepare-btn">${iconSvg("dataset")} 准备/校验 VisDrone 子集</button>
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("train")} 训练参数</div>
      <div class="control-group two-col">
        <div>
          <label for="epochs">Epochs</label>
          <input id="epochs" type="number" min="1" max="200" value="18" />
        </div>
        <div>
          <label for="batch-size">Batch Size</label>
          <input id="batch-size" type="number" min="8" max="256" value="64" />
        </div>
      </div>
      <div class="control-group">
        <label for="learning-rate">Learning Rate</label>
        <input id="learning-rate" type="number" min="0.00001" max="0.01" step="0.0001" value="0.0006" />
      </div>
      <div class="control-group control-group-inline">
        <button id="train-btn">${iconSvg("train")} 提交训练任务</button>
        <button id="evaluate-btn">${iconSvg("evaluate")} 评估选中 Run</button>
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("apply")} Run 管理与仿真注入</div>
      <div class="control-group">
        <label for="run-select">历史 Run</label>
        <select id="run-select"></select>
      </div>
      <div id="run-meta" class="summary-block"></div>
      <div class="control-group">
        <button id="apply-confidence-btn">${iconSvg("apply")} 将置信度应用到仿真</button>
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("info")} 任务状态与日志</div>
      <div id="job-status" class="status-line">idle</div>
      <pre id="job-logs" class="job-log"></pre>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("dataset")} 数据准备摘要</div>
      <div id="dataset-summary" class="summary-block">尚未准备数据。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("evaluate")} 训练产物展示</div>
      <div id="artifact-links" class="artifact-links"></div>
      <img id="artifact-curve" class="artifact-image" alt="training curve" />
      <img id="artifact-confusion" class="artifact-image" alt="confusion matrix" />
      <img id="artifact-samples" class="artifact-image" alt="sample grid" />
    </div>
  `;

  const backendUrlInput = container.querySelector("#backend-url");
  const splitModeSelect = container.querySelector("#split-mode");
  const trainImagesDirInput = container.querySelector("#train-images-dir");
  const trainAnnDirInput = container.querySelector("#train-ann-dir");
  const valImagesDirInput = container.querySelector("#val-images-dir");
  const valAnnDirInput = container.querySelector("#val-ann-dir");
  const subsetSizeInput = container.querySelector("#subset-size");
  const valSubsetSizeInput = container.querySelector("#val-subset-size");
  const epochsInput = container.querySelector("#epochs");
  const batchSizeInput = container.querySelector("#batch-size");
  const lrInput = container.querySelector("#learning-rate");
  const runSelect = container.querySelector("#run-select");
  const runMeta = container.querySelector("#run-meta");
  const datasetSummaryPanel = container.querySelector("#dataset-summary");
  const jobStatus = container.querySelector("#job-status");
  const jobLogs = container.querySelector("#job-logs");
  const artifactLinks = container.querySelector("#artifact-links");
  const artifactCurve = container.querySelector("#artifact-curve");
  const artifactConfusion = container.querySelector("#artifact-confusion");
  const artifactSamples = container.querySelector("#artifact-samples");

  function normalizeBackendUrl() {
    const text = backendUrlInput.value.trim();
    state.backendBaseUrl = text || DEFAULT_BACKEND;
  }

  function artifactUrl(path) {
    if (!path) {
      return "";
    }
    if (path.startsWith("http://") || path.startsWith("https://")) {
      return path;
    }
    return `${state.backendBaseUrl}${path}`;
  }

  async function api(path, options = {}) {
    normalizeBackendUrl();
    const response = await fetch(`${state.backendBaseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status}: ${text}`);
    }
    return response.json();
  }

  function setJobStatus(text) {
    jobStatus.textContent = text;
  }

  function setLogs(lines) {
    if (!lines || lines.length === 0) {
      jobLogs.textContent = "(暂无日志)";
      return;
    }
    jobLogs.textContent = lines.slice(-100).join("\n");
    jobLogs.scrollTop = jobLogs.scrollHeight;
  }

  function setArtifactImage(element, src) {
    if (!src) {
      element.style.display = "none";
      element.removeAttribute("src");
      return;
    }
    element.style.display = "block";
    element.src = `${artifactUrl(src)}?t=${Date.now()}`;
  }

  function renderArtifacts(artifacts) {
    const entries = [
      ["history", artifacts.history],
      ["summary", artifacts.summary],
      ["classConfidence", artifacts.classConfidence],
      ["evaluationSummary", artifacts.evaluationSummary],
      ["confusionMatrixCsv", artifacts.confusionMatrixCsv],
      ["checkpoint", artifacts.checkpoint],
    ].filter(([, value]) => Boolean(value));

    artifactLinks.innerHTML = entries.length > 0
      ? entries.map(([name, path]) => `<a href="${artifactUrl(path)}" target="_blank" rel="noreferrer">${name}</a>`).join(" | ")
      : "当前 run 尚无可展示产物。";

    setArtifactImage(artifactCurve, artifacts.trainingCurve);
    setArtifactImage(artifactConfusion, artifacts.confusionMatrixPng);
    setArtifactImage(artifactSamples, artifacts.sampleGrid);
  }

  function renderDatasetSummary(summaryPayload) {
    if (!summaryPayload) {
      datasetSummaryPanel.textContent = "尚未准备数据。";
      return;
    }

    state.datasetSummary = summaryPayload;
    const outputCounts = summaryPayload.outputCounts || {};
    const trainCounts = outputCounts.train || {};
    const valCounts = outputCounts.val || {};
    const sourceTrainStats = summaryPayload.sources?.train?.stats || {};
    const sourceValStats = summaryPayload.sources?.val?.stats || {};

    datasetSummaryPanel.innerHTML = `
      <div><b>splitMode:</b> ${summaryPayload.splitMode || "-"}</div>
      <div><b>train 输出:</b> vehicle=${trainCounts.vehicle || 0}, civilian=${trainCounts["civilian-object"] || 0}, decoy=${trainCounts.decoy || 0}</div>
      <div><b>val 输出:</b> vehicle=${valCounts.vehicle || 0}, civilian=${valCounts["civilian-object"] || 0}, decoy=${valCounts.decoy || 0}</div>
      <div><b>train 过滤:</b> small=${sourceTrainStats.filteredSmallCount || 0}, invalid=${sourceTrainStats.filteredInvalidCount || 0}, unknown=${sourceTrainStats.filteredUnknownCategoryCount || 0}</div>
      <div><b>val 过滤:</b> small=${sourceValStats.filteredSmallCount || 0}, invalid=${sourceValStats.filteredInvalidCount || 0}, unknown=${sourceValStats.filteredUnknownCategoryCount || 0}</div>
    `;
  }

  function renderRunMeta(run) {
    const summary = run?.summary || {};
    if (!run) {
      runMeta.textContent = "暂无 run。";
      return;
    }

    const epochs = safeNumber(summary.epochs, 0);
    const totalTrainingSec = safeNumber(summary.totalTrainingSec, 0);
    const avgEpochSec = epochs > 0 ? totalTrainingSec / epochs : 0;

    runMeta.innerHTML = `
      <div><b>runId:</b> ${run.runId}</div>
      <div><b>device:</b> ${summary.device || "-"}</div>
      <div><b>样本规模:</b> train=${summary.trainSamples || 0}, val=${summary.valSamples || 0}</div>
      <div><b>训练时长:</b> 总计 ${totalTrainingSec.toFixed(2)}s / 每轮 ${avgEpochSec.toFixed(2)}s</div>
      <div><b>参数:</b> epochs=${summary.epochs || "-"}, batch=${summary.batchSize || "-"}, lr=${summary.learningRate || "-"}</div>
    `;
  }

  async function loadRunArtifacts(runId) {
    if (!runId) {
      renderArtifacts({});
      renderRunMeta(null);
      return;
    }
    const selectedRun = state.runs.find((run) => run.runId === runId) || null;
    renderRunMeta(selectedRun);
    const payload = await api(`/api/runs/${runId}/artifacts`);
    renderArtifacts(payload);
  }

  function stopPolling() {
    if (state.pollingTimer) {
      clearInterval(state.pollingTimer);
      state.pollingTimer = null;
    }
  }

  function startPolling(jobId) {
    stopPolling();
    state.activeJobId = jobId;
    state.pollingTimer = setInterval(async () => {
      try {
        const payload = await api(`/api/jobs/${jobId}`);
        setJobStatus(`${payload.status}${payload.runId ? ` | run=${payload.runId}` : ""}`);
        setLogs(payload.logs || []);

        if (payload.status === "succeeded" && payload.type === "dataset_prepare") {
          renderDatasetSummary(payload.artifacts?.manifest || null);
        }

        if (payload.status === "succeeded" || payload.status === "failed" || payload.status === "cancelled") {
          stopPolling();
          if (payload.runId) {
            await refreshRuns(payload.runId);
          }
        }
      } catch (error) {
        stopPolling();
        setJobStatus(`polling error: ${parseError(error)}`);
      }
    }, 1200);
  }

  async function refreshRuns(preferredRunId = "") {
    const payload = await api("/api/runs");
    state.runs = payload.runs || [];

    const optionsHtml = state.runs
      .map((run) => `<option value="${run.runId}">${run.runId}</option>`)
      .join("");
    runSelect.innerHTML = optionsHtml || `<option value="">(暂无 run)</option>`;

    const selectedRunId = preferredRunId || runSelect.value || (state.runs[0] ? state.runs[0].runId : "");
    runSelect.value = selectedRunId;
    state.activeRunId = selectedRunId;

    await loadRunArtifacts(selectedRunId);
  }

  container.querySelector("#health-btn").addEventListener("click", async () => {
    try {
      const payload = await api("/api/health");
      setJobStatus(`backend ok | queue=${payload.queueSize}`);
    } catch (error) {
      setJobStatus(`backend error: ${parseError(error)}`);
    }
  });

  container.querySelector("#refresh-runs-btn").addEventListener("click", async () => {
    try {
      await refreshRuns(state.activeRunId);
      setJobStatus("已刷新 run 列表");
    } catch (error) {
      setJobStatus(`refresh runs failed: ${parseError(error)}`);
    }
  });

  container.querySelector("#prepare-btn").addEventListener("click", async () => {
    try {
      const payload = await api("/api/dataset/prepare", {
        method: "POST",
        body: JSON.stringify({
          splitMode: splitModeSelect.value,
          trainImagesDir: trainImagesDirInput.value.trim(),
          trainAnnotationsDir: trainAnnDirInput.value.trim(),
          valImagesDir: valImagesDirInput.value.trim(),
          valAnnotationsDir: valAnnDirInput.value.trim(),
          subsetSizePerClass: safeNumber(subsetSizeInput.value, 900),
          valSubsetSizePerClass: safeNumber(valSubsetSizeInput.value, 0),
        }),
      });
      setJobStatus(`dataset job queued: ${payload.jobId}`);
      setLogs([`dataset prepare queued: ${payload.jobId}`]);
      startPolling(payload.jobId);
    } catch (error) {
      setJobStatus(`dataset prepare failed: ${parseError(error)}`);
    }
  });

  container.querySelector("#train-btn").addEventListener("click", async () => {
    try {
      const payload = await api("/api/jobs/train", {
        method: "POST",
        body: JSON.stringify({
          epochs: safeNumber(epochsInput.value, 18),
          batchSize: safeNumber(batchSizeInput.value, 64),
          learningRate: safeNumber(lrInput.value, 0.0006),
          runName: "visdrone-ui",
          evaluateAfterTrain: true,
        }),
      });
      setJobStatus(`train job queued: ${payload.jobId}`);
      setLogs([`train queued: ${payload.jobId}`]);
      startPolling(payload.jobId);
    } catch (error) {
      setJobStatus(`train failed: ${parseError(error)}`);
    }
  });

  container.querySelector("#evaluate-btn").addEventListener("click", async () => {
    const runId = runSelect.value;
    if (!runId) {
      setJobStatus("请先选择一个 run");
      return;
    }
    try {
      const payload = await api("/api/jobs/evaluate", {
        method: "POST",
        body: JSON.stringify({ runId }),
      });
      setJobStatus(`evaluate job queued: ${payload.jobId}`);
      setLogs([`evaluate queued: ${payload.jobId}`]);
      startPolling(payload.jobId);
    } catch (error) {
      setJobStatus(`evaluate failed: ${parseError(error)}`);
    }
  });

  runSelect.addEventListener("change", async () => {
    state.activeRunId = runSelect.value;
    try {
      await loadRunArtifacts(runSelect.value);
    } catch (error) {
      setJobStatus(`load artifacts failed: ${parseError(error)}`);
    }
  });

  container.querySelector("#apply-confidence-btn").addEventListener("click", async () => {
    const runId = runSelect.value;
    if (!runId) {
      setJobStatus("请先选择一个 run");
      return;
    }
    try {
      setJobStatus(`正在将 run=${runId} 的置信度注入仿真...`);
      await onApplyConfidence({
        runId,
        backendBaseUrl: state.backendBaseUrl,
      });
      setJobStatus(`已完成仿真注入：${runId}`);
    } catch (error) {
      setJobStatus(`apply confidence failed: ${parseError(error)}`);
    }
  });

  setLogs([]);
  refreshRuns().catch((error) => {
    setJobStatus(`init runs failed: ${parseError(error)}`);
  });
}
