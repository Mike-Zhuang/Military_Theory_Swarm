export function createControlPanel({
  container,
  scenarioOptions,
  runs,
  onScenarioChange,
  onPrimaryRunChange,
  onSecondaryRunChange,
  onCompareToggle,
  onPlayToggle,
  onSpeedChange,
  onSeek,
  onReset,
}) {
  const panelHtml = `
    <h3 class="section-title">Control Panel</h3>

    <div class="control-group">
      <label for="scenario-select">演示场景</label>
      <select id="scenario-select">
        ${scenarioOptions.map((item, index) => `<option value="${index}">${item.label}</option>`).join("")}
      </select>
    </div>

    <div class="control-group">
      <label for="primary-run-select">主视图策略</label>
      <select id="primary-run-select">
        ${runs.map((run, index) => `<option value="${index}">${run.name}</option>`).join("")}
      </select>
    </div>

    <div class="control-group">
      <label class="toggle-row" for="compare-toggle">
        <span>双视图对照</span>
        <input id="compare-toggle" type="checkbox" checked />
      </label>
    </div>

    <div class="control-group" id="secondary-run-group">
      <label for="secondary-run-select">副视图策略</label>
      <select id="secondary-run-select">
        ${runs.map((run, index) => `<option value="${index}">${run.name}</option>`).join("")}
      </select>
    </div>

    <div class="control-group">
      <label for="speed-range">播放倍率</label>
      <input id="speed-range" type="range" min="0.5" max="3" step="0.5" value="1" />
      <small><span id="speed-value" class="timeline-value">1.0x</span></small>
    </div>

    <div class="control-group">
      <label for="timeline-range">时间轴</label>
      <input id="timeline-range" type="range" min="0" max="1" step="1" value="0" />
      <small><span id="timeline-value" class="timeline-value">0 / 0</span></small>
    </div>

    <div class="control-group">
      <button id="play-btn">暂停</button>
      <button id="reset-btn">回到起点</button>
    </div>
  `;
  container.innerHTML = panelHtml;

  const scenarioSelect = container.querySelector("#scenario-select");
  const primaryRunSelect = container.querySelector("#primary-run-select");
  const secondaryRunSelect = container.querySelector("#secondary-run-select");
  const secondaryRunGroup = container.querySelector("#secondary-run-group");
  const compareToggle = container.querySelector("#compare-toggle");
  const speedRange = container.querySelector("#speed-range");
  const speedValue = container.querySelector("#speed-value");
  const timelineRange = container.querySelector("#timeline-range");
  const timelineValue = container.querySelector("#timeline-value");
  const playBtn = container.querySelector("#play-btn");
  const resetBtn = container.querySelector("#reset-btn");

  let isPlaying = true;

  scenarioSelect.addEventListener("change", async (event) => {
    try {
      await onScenarioChange(Number(event.target.value));
    } catch (error) {
      console.error(error);
      alert(`场景加载失败: ${String(error)}`);
    }
  });

  primaryRunSelect.addEventListener("change", (event) => {
    onPrimaryRunChange(Number(event.target.value));
  });

  secondaryRunSelect.addEventListener("change", (event) => {
    onSecondaryRunChange(Number(event.target.value));
  });

  compareToggle.addEventListener("change", (event) => {
    const enabled = Boolean(event.target.checked);
    secondaryRunGroup.style.display = enabled ? "grid" : "none";
    onCompareToggle(enabled);
  });

  speedRange.addEventListener("input", (event) => {
    const value = Number(event.target.value);
    speedValue.textContent = `${value.toFixed(1)}x`;
    onSpeedChange(value);
  });

  timelineRange.addEventListener("input", (event) => {
    onSeek(Number(event.target.value));
  });

  playBtn.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? "暂停" : "播放";
    onPlayToggle(isPlaying);
  });

  resetBtn.addEventListener("click", () => {
    onReset();
  });

  return {
    setTimeline(frameIndex, maxFrame) {
      timelineRange.max = String(Math.max(1, maxFrame));
      timelineRange.value = String(frameIndex);
      timelineValue.textContent = `${frameIndex} / ${maxFrame}`;
    },
    setPlayState(nextPlaying) {
      isPlaying = nextPlaying;
      playBtn.textContent = isPlaying ? "暂停" : "播放";
    },
    setRunOptions(nextRuns, primaryIndex = 0, secondaryIndex = 1) {
      const optionsHtml = nextRuns
        .map((run, index) => `<option value="${index}">${run.name}</option>`)
        .join("");
      primaryRunSelect.innerHTML = optionsHtml;
      secondaryRunSelect.innerHTML = optionsHtml;
      primaryRunSelect.value = String(primaryIndex);
      secondaryRunSelect.value = String(Math.min(nextRuns.length - 1, secondaryIndex));
    },
    setScenarioIndex(index) {
      scenarioSelect.value = String(index);
    },
    setCompareMode(enabled) {
      compareToggle.checked = enabled;
      secondaryRunGroup.style.display = enabled ? "grid" : "none";
    },
  };
}
