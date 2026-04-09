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

function tail(items, size) {
  if (!Array.isArray(items)) {
    return [];
  }
  return items.slice(Math.max(0, items.length - size));
}

function generalizationAlert(liveMetrics) {
  const recent = tail(liveMetrics, 4);
  if (recent.length < 4) {
    return "";
  }
  const [a, b, c, d] = recent.map((item) => Number(item.valLoss));
  if (Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c) && Number.isFinite(d) && b > a && c > b && d > c) {
    return "连续 3 轮 val loss 上升，建议立即早停并检查增强/正则参数。";
  }
  return "";
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
  };

  container.innerHTML = `
    <h3 class="section-title section-title-with-icon">
      ${iconSvg("guide")}
      <span>ML 在线实验台（可调参 + 可早停）</span>
    </h3>

    <div class="workflow-grid">
      <article class="workflow-step">
        <div class="workflow-head">${iconSvg("dataset")} 第 1 步：准备数据</div>
        <div class="workflow-desc">默认使用 VisDrone 官方 train + val，保证结论可信。</div>
      </article>
      <article class="workflow-step">
        <div class="workflow-head">${iconSvg("train")} 第 2 步：提交训练</div>
        <div class="workflow-desc">支持预训练骨干、早停、学习率调度与增强等级。</div>
      </article>
      <article class="workflow-step">
        <div class="workflow-head">${iconSvg("evaluate")} 第 3 步：查看评估</div>
        <div class="workflow-desc">重点看混淆矩阵、macro-F1 与泛化差距。</div>
      </article>
      <article class="workflow-step">
        <div class="workflow-head">${iconSvg("apply")} 第 4 步：应用到仿真</div>
        <div class="workflow-desc">将 best run 置信度注入仿真并回放策略差异。</div>
      </article>
    </div>

    <div class="control-group">
      <label for="backend-url">后端地址</label>
      <input id="backend-url" type="text" value="${DEFAULT_BACKEND}" />
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
        </div>
      </div>
      <div class="control-group">
        <label class="toggle-row" for="use-ignored-decoy">
          <span>使用 ignored-region 作为 decoy</span>
          <input id="use-ignored-decoy" type="checkbox" />
        </label>
      </div>
      <div class="control-group">
        <button id="prepare-btn">${iconSvg("dataset")} 准备/校验数据集</button>
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("train")} 训练参数</div>
      <div class="control-group two-col">
        <div>
          <label for="model-name">模型</label>
          <select id="model-name">
            <option value="mobilenetv3-small">mobilenetv3-small（推荐）</option>
            <option value="tiny-cnn">tiny-cnn（轻量）</option>
          </select>
        </div>
        <div>
          <label for="augment-level">增强等级</label>
          <select id="augment-level">
            <option value="medium">medium</option>
            <option value="light">light</option>
            <option value="strong">strong</option>
          </select>
        </div>
      </div>
      <div class="control-group">
        <label class="toggle-row" for="pretrained-toggle">
          <span>使用预训练权重</span>
          <input id="pretrained-toggle" type="checkbox" checked />
        </label>
      </div>
      <div class="control-group two-col">
        <div>
          <label for="epochs">Epochs</label>
          <input id="epochs" type="number" min="1" max="200" value="80" />
        </div>
        <div>
          <label for="batch-size">Batch Size</label>
          <input id="batch-size" type="number" min="8" max="256" value="64" />
        </div>
      </div>
      <div class="control-group two-col">
        <div>
          <label for="learning-rate">Learning Rate</label>
          <input id="learning-rate" type="number" min="0.00001" max="0.01" step="0.00005" value="0.0003" />
        </div>
        <div>
          <label for="weight-decay">Weight Decay</label>
          <input id="weight-decay" type="number" min="0" max="0.1" step="0.0001" value="0.0001" />
        </div>
      </div>
      <div class="control-group two-col">
        <div>
          <label for="label-smoothing">Label Smoothing</label>
          <input id="label-smoothing" type="number" min="0" max="0.4" step="0.01" value="0.05" />
        </div>
        <div>
          <label for="freeze-epochs">Freeze Epochs</label>
          <input id="freeze-epochs" type="number" min="0" max="50" value="3" />
        </div>
      </div>
      <div class="control-group two-col">
        <div>
          <label for="scheduler">Scheduler</label>
          <select id="scheduler">
            <option value="cosine">cosine</option>
            <option value="plateau">plateau</option>
            <option value="none">none</option>
          </select>
        </div>
        <div>
          <label for="image-size">输入分辨率</label>
          <input id="image-size" type="number" min="64" max="384" step="32" value="128" />
        </div>
      </div>
      <div class="control-group two-col">
        <div>
          <label for="early-stop-patience">早停耐心值</label>
          <input id="early-stop-patience" type="number" min="1" max="50" value="8" />
        </div>
        <div>
          <label for="early-stop-delta">早停最小改进</label>
          <input id="early-stop-delta" type="number" min="0" max="1" step="0.0005" value="0.001" />
        </div>
      </div>
      <div class="control-group control-group-inline">
        <button id="train-btn">${iconSvg("train")} 提交训练任务</button>
        <button id="cancel-job-btn">${iconSvg("warning")} 停止当前任务</button>
        <button id="evaluate-btn">${iconSvg("evaluate")} 评估选中 Run</button>
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("apply")} Run 管理与仿真注入</div>
      <div class="control-group">
        <label for="run-select">历史 Run</label>
        <select id="run-select"></select>
      </div>
      <div id="run-meta" class="summary-block">暂无 run</div>
      <div class="control-group">
        <button id="apply-confidence-btn">${iconSvg("apply")} 将置信度应用到仿真</button>
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("info")} 任务状态与日志</div>
      <div id="job-status" class="status-line">idle</div>
      <div id="generalization-warning" class="status-warning"></div>
      <pre id="job-logs" class="job-log"></pre>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("dataset")} 数据准备摘要</div>
      <div id="dataset-summary" class="summary-block">尚未准备数据。</div>
    </div>

    <div class="metric-card">
      <div class="metric-label card-title">${iconSvg("evaluate")} 训练产物展示</div>
      <div id="artifact-links" class="artifact-links"></div>
      <img id="artifact-curve-live" class="artifact-image" alt="training curve live" />
      <img id="artifact-curve" class="artifact-image" alt="training curve final" />
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
  const useIgnoredDecoyInput = container.querySelector("#use-ignored-decoy");
  const modelNameSelect = container.querySelector("#model-name");
  const augmentLevelSelect = container.querySelector("#augment-level");
  const pretrainedToggle = container.querySelector("#pretrained-toggle");
  const epochsInput = container.querySelector("#epochs");
  const batchSizeInput = container.querySelector("#batch-size");
  const lrInput = container.querySelector("#learning-rate");
  const weightDecayInput = container.querySelector("#weight-decay");
  const labelSmoothingInput = container.querySelector("#label-smoothing");
  const freezeEpochsInput = container.querySelector("#freeze-epochs");
  const schedulerInput = container.querySelector("#scheduler");
  const imageSizeInput = container.querySelector("#image-size");
  const earlyStopPatienceInput = container.querySelector("#early-stop-patience");
  const earlyStopDeltaInput = container.querySelector("#early-stop-delta");
  const runSelect = container.querySelector("#run-select");
  const runMeta = container.querySelector("#run-meta");
  const datasetSummaryPanel = container.querySelector("#dataset-summary");
  const jobStatus = container.querySelector("#job-status");
  const generalizationWarning = container.querySelector("#generalization-warning");
  const jobLogs = container.querySelector("#job-logs");
  const artifactLinks = container.querySelector("#artifact-links");
  const artifactCurveLive = container.querySelector("#artifact-curve-live");
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
    jobLogs.textContent = lines.join("\n");
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
      ["bestCheckpoint", artifacts.bestCheckpoint],
      ["lastCheckpoint", artifacts.lastCheckpoint],
      ["history", artifacts.history],
      ["summary", artifacts.summary],
      ["classConfidence", artifacts.classConfidence],
      ["evaluationSummary", artifacts.evaluationSummary],
      ["confusionMatrixCsv", artifacts.confusionMatrixCsv],
    ].filter(([, value]) => Boolean(value));

    artifactLinks.innerHTML = entries.length > 0
      ? entries.map(([name, path]) => `<a href="${artifactUrl(path)}" target="_blank" rel="noreferrer">${name}</a>`).join(" | ")
      : "当前 run 尚无可展示产物。";

    setArtifactImage(artifactCurveLive, artifacts.trainingCurveLive || artifacts.trainingCurve);
    setArtifactImage(artifactCurve, artifacts.trainingCurve);
    setArtifactImage(artifactConfusion, artifacts.confusionMatrixPng);
    setArtifactImage(artifactSamples, artifacts.sampleGrid);
  }

  function renderDatasetSummary(summaryPayload) {
    if (!summaryPayload) {
      datasetSummaryPanel.textContent = "尚未准备数据。";
      return;
    }
    const outputCounts = summaryPayload.outputCounts || {};
    const trainCounts = outputCounts.train || {};
    const valCounts = outputCounts.val || {};
    const quality = summaryPayload.labelQuality || {};

    datasetSummaryPanel.innerHTML = `
      <div><b>splitMode:</b> ${summaryPayload.splitMode || "-"}</div>
      <div><b>train 输出:</b> vehicle=${trainCounts.vehicle || 0}, civilian=${trainCounts["civilian-object"] || 0}, decoy=${trainCounts.decoy || 0}</div>
      <div><b>val 输出:</b> vehicle=${valCounts.vehicle || 0}, civilian=${valCounts["civilian-object"] || 0}, decoy=${valCounts.decoy || 0}</div>
      <div><b>标签质量:</b> ignoredUsed=${quality.ignoredUsedCount || 0}, ignoredSkipped=${quality.ignoredSkippedCount || 0}, backgroundNegative=${quality.backgroundNegativeCount || 0}</div>
    `;
  }

  function renderRunMeta(run) {
    if (!run) {
      runMeta.textContent = "暂无 run。";
      return;
    }
    const summary = run.summary || {};
    const epochsExecuted = safeNumber(summary.epochsExecuted, safeNumber(summary.epochs, 0));
    const totalTrainingSec = safeNumber(summary.totalTrainingSec, 0);
    const avgEpochSec = epochsExecuted > 0 ? totalTrainingSec / epochsExecuted : 0;

    runMeta.innerHTML = `
      <div><b>runId:</b> ${run.runId}</div>
      <div><b>device:</b> ${summary.device || "-"}</div>
      <div><b>模型:</b> ${summary.modelName || "-"}</div>
      <div><b>样本规模:</b> train=${summary.trainSamples || 0}, val=${summary.valSamples || 0}</div>
      <div><b>最佳:</b> epoch=${summary.bestEpoch || "-"}, valLoss=${summary.bestValLoss ?? "-"}, valAcc=${summary.bestValAcc ?? "-"}</div>
      <div><b>最后:</b> valLoss=${summary.lastValLoss ?? "-"}, valAcc=${summary.lastValAcc ?? "-"}</div>
      <div><b>训练时长:</b> 总计 ${totalTrainingSec.toFixed(2)}s / 每轮 ${avgEpochSec.toFixed(2)}s</div>
    `;
  }

  async function loadRunArtifacts(runId) {
    if (!runId) {
      renderRunMeta(null);
      renderArtifacts({});
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

  async function refreshRuns(preferredRunId = "") {
    const payload = await api("/api/runs");
    state.runs = payload.runs || [];

    runSelect.innerHTML = state.runs
      .map((run) => `<option value="${run.runId}">${run.runId}</option>`)
      .join("") || `<option value="">(暂无 run)</option>`;

    const selectedRunId = preferredRunId || runSelect.value || (state.runs[0] ? state.runs[0].runId : "");
    runSelect.value = selectedRunId;
    state.activeRunId = selectedRunId;
    await loadRunArtifacts(selectedRunId);
  }

  function progressText(progress) {
    if (!progress || !progress.currentEpoch) {
      return "";
    }
    return ` | epoch ${progress.currentEpoch}/${progress.totalEpochs} | best=${progress.bestValLoss ?? "-"} | noImprove=${progress.noImproveEpochs ?? 0} | ETA=${progress.etaSec ?? "-"}s`;
  }

  function startPolling(jobId) {
    stopPolling();
    state.activeJobId = jobId;
    state.pollingTimer = setInterval(async () => {
      try {
        const payload = await api(`/api/jobs/${jobId}`);
        const suffix = payload.runId ? ` | run=${payload.runId}` : "";
        setJobStatus(`${payload.status}${suffix}${progressText(payload.progress)}`);
        setLogs(tail(payload.logs || [], 120));

        if (payload.liveMetrics && payload.liveMetrics.length > 0) {
          const warning = generalizationAlert(payload.liveMetrics);
          generalizationWarning.textContent = warning;
        } else {
          generalizationWarning.textContent = "";
        }

        if (payload.type === "dataset_prepare" && payload.status === "succeeded") {
          renderDatasetSummary(payload.artifacts?.manifest || null);
        }

        if (payload.runId) {
          state.activeRunId = payload.runId;
          try {
            const artifacts = await api(`/api/runs/${payload.runId}/artifacts`);
            renderArtifacts(artifacts);
          } catch (error) {
            setJobStatus(`加载实时产物失败: ${parseError(error)}`);
          }
        }

        if (["succeeded", "failed", "cancelled"].includes(payload.status)) {
          stopPolling();
          await refreshRuns(payload.runId || state.activeRunId);
        }
      } catch (error) {
        stopPolling();
        setJobStatus(`polling error: ${parseError(error)}`);
      }
    }, 1200);
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
          useIgnoredAsDecoy: Boolean(useIgnoredDecoyInput.checked),
        }),
      });
      generalizationWarning.textContent = "";
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
          modelName: modelNameSelect.value,
          pretrained: Boolean(pretrainedToggle.checked),
          augmentLevel: augmentLevelSelect.value,
          epochs: safeNumber(epochsInput.value, 80),
          batchSize: safeNumber(batchSizeInput.value, 64),
          learningRate: safeNumber(lrInput.value, 0.0003),
          weightDecay: safeNumber(weightDecayInput.value, 0.0001),
          labelSmoothing: safeNumber(labelSmoothingInput.value, 0.05),
          freezeEpochs: safeNumber(freezeEpochsInput.value, 3),
          scheduler: schedulerInput.value,
          imageSize: safeNumber(imageSizeInput.value, 128),
          earlyStopPatience: safeNumber(earlyStopPatienceInput.value, 8),
          earlyStopMinDelta: safeNumber(earlyStopDeltaInput.value, 0.001),
          runName: "visdrone-ui",
          evaluateAfterTrain: true,
        }),
      });
      generalizationWarning.textContent = "";
      setJobStatus(`train job queued: ${payload.jobId}`);
      setLogs([`train queued: ${payload.jobId}`]);
      startPolling(payload.jobId);
    } catch (error) {
      setJobStatus(`train failed: ${parseError(error)}`);
    }
  });

  container.querySelector("#cancel-job-btn").addEventListener("click", async () => {
    if (!state.activeJobId) {
      setJobStatus("当前没有可取消任务");
      return;
    }
    try {
      const payload = await api(`/api/jobs/${state.activeJobId}/cancel`, {
        method: "POST",
      });
      setJobStatus(`已请求取消任务: ${payload.jobId}`);
    } catch (error) {
      setJobStatus(`取消任务失败: ${parseError(error)}`);
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
      generalizationWarning.textContent = "";
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
