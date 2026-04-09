const DEFAULT_BACKEND = "http://127.0.0.1:8001";

function parseError(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
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
    <h3 class="section-title">ML 在线实验台</h3>

    <div class="control-group">
      <label for="backend-url">后端地址</label>
      <input id="backend-url" type="text" value="${DEFAULT_BACKEND}" />
      <small>默认本机服务端口：8001</small>
    </div>

    <div class="control-group">
      <button id="health-btn">检查后端状态</button>
      <button id="prepare-btn">准备/校验 VisDrone 子集</button>
    </div>

    <div class="control-group">
      <label for="subset-size">每类样本数</label>
      <input id="subset-size" type="number" min="120" step="20" value="900" />
    </div>

    <div class="control-group">
      <label for="epochs">Epochs</label>
      <input id="epochs" type="number" min="1" max="200" value="18" />
    </div>

    <div class="control-group">
      <label for="batch-size">Batch Size</label>
      <input id="batch-size" type="number" min="8" max="256" value="64" />
    </div>

    <div class="control-group">
      <label for="learning-rate">Learning Rate</label>
      <input id="learning-rate" type="number" min="0.00001" max="0.01" step="0.0001" value="0.0006" />
    </div>

    <div class="control-group">
      <button id="train-btn">提交训练任务</button>
      <button id="evaluate-btn">评估选中 Run</button>
    </div>

    <div class="control-group">
      <label for="run-select">历史 Run</label>
      <select id="run-select"></select>
      <button id="refresh-runs-btn">刷新 Run 列表</button>
      <button id="apply-confidence-btn">将置信度应用到仿真</button>
    </div>

    <div class="metric-card">
      <div class="metric-label">任务状态</div>
      <div id="job-status">idle</div>
      <pre id="job-logs" class="job-log"></pre>
    </div>

    <div class="metric-card">
      <div class="metric-label">训练产物</div>
      <div id="artifact-links" class="artifact-links"></div>
      <img id="artifact-curve" class="artifact-image" alt="training curve" />
      <img id="artifact-confusion" class="artifact-image" alt="confusion matrix" />
      <img id="artifact-samples" class="artifact-image" alt="sample grid" />
    </div>
  `;

  const backendUrlInput = container.querySelector("#backend-url");
  const subsetSizeInput = container.querySelector("#subset-size");
  const epochsInput = container.querySelector("#epochs");
  const batchSizeInput = container.querySelector("#batch-size");
  const lrInput = container.querySelector("#learning-rate");
  const runSelect = container.querySelector("#run-select");
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
    jobLogs.textContent = lines.slice(-80).join("\n");
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

    artifactLinks.innerHTML = entries
      .map(([name, path]) => `<a href="${artifactUrl(path)}" target="_blank" rel="noreferrer">${name}</a>`)
      .join(" | ");

    setArtifactImage(artifactCurve, artifacts.trainingCurve);
    setArtifactImage(artifactConfusion, artifacts.confusionMatrixPng);
    setArtifactImage(artifactSamples, artifacts.sampleGrid);
  }

  async function loadRunArtifacts(runId) {
    if (!runId) {
      renderArtifacts({});
      return;
    }
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
    }, 1500);
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

    if (selectedRunId) {
      await loadRunArtifacts(selectedRunId);
    } else {
      renderArtifacts({});
    }
  }

  container.querySelector("#health-btn").addEventListener("click", async () => {
    try {
      const payload = await api("/api/health");
      setJobStatus(`backend ok | queue=${payload.queueSize}`);
    } catch (error) {
      setJobStatus(`backend error: ${parseError(error)}`);
    }
  });

  container.querySelector("#prepare-btn").addEventListener("click", async () => {
    try {
      const payload = await api("/api/dataset/prepare", {
        method: "POST",
        body: JSON.stringify({
          subsetSizePerClass: Number(subsetSizeInput.value),
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
          epochs: Number(epochsInput.value),
          batchSize: Number(batchSizeInput.value),
          learningRate: Number(lrInput.value),
          runName: "visdrone-ui",
          dataDir: "ml-module/data/visdrone-ready",
          evaluateAfterTrain: true,
        }),
      });
      setJobStatus(`train job queued: ${payload.jobId}`);
      setLogs([`train queued: ${payload.jobId}`]);
      startPolling(payload.jobId);
    } catch (error) {
      setJobStatus(`train submit failed: ${parseError(error)}`);
    }
  });

  container.querySelector("#evaluate-btn").addEventListener("click", async () => {
    const runId = runSelect.value;
    if (!runId) {
      setJobStatus("请先选择一个 run 再评估。");
      return;
    }
    try {
      const payload = await api("/api/jobs/evaluate", {
        method: "POST",
        body: JSON.stringify({
          runId,
          dataDir: "ml-module/data/visdrone-ready",
          split: "val",
          batchSize: Number(batchSizeInput.value),
        }),
      });
      setJobStatus(`evaluate job queued: ${payload.jobId}`);
      setLogs([`evaluate queued: ${payload.jobId}`]);
      startPolling(payload.jobId);
    } catch (error) {
      setJobStatus(`evaluate submit failed: ${parseError(error)}`);
    }
  });

  container.querySelector("#refresh-runs-btn").addEventListener("click", async () => {
    try {
      await refreshRuns();
      setJobStatus("run 列表已刷新");
    } catch (error) {
      setJobStatus(`refresh run failed: ${parseError(error)}`);
    }
  });

  runSelect.addEventListener("change", async () => {
    state.activeRunId = runSelect.value;
    try {
      await loadRunArtifacts(state.activeRunId);
    } catch (error) {
      setJobStatus(`load artifacts failed: ${parseError(error)}`);
    }
  });

  container.querySelector("#apply-confidence-btn").addEventListener("click", async () => {
    const runId = runSelect.value;
    if (!runId) {
      setJobStatus("请先选择一个 run 再应用置信度。");
      return;
    }
    try {
      await onApplyConfidence({
        runId,
        backendBaseUrl: state.backendBaseUrl,
      });
      setJobStatus(`已应用 run: ${runId} 到仿真`);
    } catch (error) {
      setJobStatus(`apply confidence failed: ${parseError(error)}`);
    }
  });

  refreshRuns().catch((error) => {
    setJobStatus(`init failed: ${parseError(error)}`);
  });

  return {
    refreshRuns,
    stopPolling,
  };
}
