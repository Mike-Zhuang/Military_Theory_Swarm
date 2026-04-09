import { createControlPanel } from "./components/control-panel.js";
import { createExplainPanel } from "./components/explain-panel.js";
import { createMlLab } from "./components/ml-lab.js";
import { MetricsBoard } from "./components/metrics-board.js";
import { SwarmCanvas } from "./components/swarm-canvas.js";
import { loadScenario } from "./lib/scenario-loader.js";

const BACKEND_BASE_URL = "http://127.0.0.1:8001";

function assetUrl(relativePath) {
  return new URL(relativePath, import.meta.url).toString();
}

function makeFallbackScenario(reason) {
  return {
    metadata: {
      world: {
        width: 1000,
        height: 700,
      },
    },
    runs: [
      {
        name: "fallback-demo",
        summary: {
          coverage: 0.12,
          taskCompletionRate: 0.08,
          avgResponseTime: 3.5,
          survivalRate: 1.0,
          avgLinkDegree: 2.0,
        },
        frames: [
          {
            t: 0,
            agents: [
              { id: 0, x: 170, y: 190, vx: 1.2, vy: 0.5, alive: true },
              { id: 1, x: 230, y: 250, vx: 1.0, vy: 0.3, alive: true },
              { id: 2, x: 310, y: 280, vx: 0.6, vy: 1.0, alive: true },
            ],
            targets: [
              { id: 0, x: 700, y: 210, className: "vehicle", active: true },
              { id: 1, x: 650, y: 460, className: "civilian-object", active: true },
              { id: 2, x: 770, y: 360, className: "decoy", active: true },
            ],
            links: [[0, 1], [1, 2]],
            events: [
              "启动阶段未加载到场景 JSON，已使用内置兜底场景",
              reason,
            ],
          },
        ],
      },
    ],
  };
}

function showStartupWarning(text) {
  const footer = document.querySelector(".hint");
  if (!footer) {
    return;
  }
  const warning = document.createElement("span");
  warning.className = "startup-warning";
  warning.textContent = `启动提示：${text}`;
  footer.appendChild(warning);
}

const SCENARIO_OPTIONS = [
  {
    key: "jam-recovery",
    label: "干扰重构 (jam-recovery)",
    paths: [
      assetUrl("../public/generated/jam-recovery-compare.json"),
      assetUrl("../public/scenarios/demo-compare.json"),
    ],
  },
  {
    key: "recon-coverage",
    label: "侦察覆盖 (recon-coverage)",
    paths: [
      assetUrl("../public/generated/recon-coverage-compare.json"),
      assetUrl("../public/scenarios/demo-compare.json"),
    ],
  },
  {
    key: "multi-target-allocation",
    label: "多目标分配 (multi-target)",
    paths: [
      assetUrl("../public/generated/multi-target-allocation-compare.json"),
      assetUrl("../public/scenarios/demo-compare.json"),
    ],
  },
  {
    key: "demo",
    label: "默认场景 (demo)",
    paths: [assetUrl("../public/scenarios/demo-compare.json")],
  },
];

const state = {
  scenarioIndex: 0,
  primaryRunIndex: 0,
  secondaryRunIndex: 1,
  compareMode: true,
  frameIndex: 0,
  isPlaying: true,
  speed: 1,
  lastTime: 0,
};

function bootstrap() {
  let scenario = makeFallbackScenario("页面初始化中");
  const canvas = new SwarmCanvas(document.getElementById("swarm-canvas"), scenario.metadata.world);
  const metricsBoard = new MetricsBoard(document.getElementById("metrics-board"));

  createExplainPanel(document.getElementById("explain-panel"));

  const controls = createControlPanel({
    container: document.getElementById("control-panel"),
    scenarioOptions: SCENARIO_OPTIONS,
    runs: scenario.runs,
    onScenarioChange: async (scenarioIndex) => {
      await switchScenario(scenarioIndex);
    },
    onPrimaryRunChange: (runIndex) => {
      state.primaryRunIndex = runIndex;
      state.frameIndex = 0;
      render();
    },
    onSecondaryRunChange: (runIndex) => {
      state.secondaryRunIndex = runIndex;
      state.frameIndex = 0;
      render();
    },
    onCompareToggle: (enabled) => {
      state.compareMode = enabled;
      render();
    },
    onPlayToggle: (isPlaying) => {
      state.isPlaying = isPlaying;
    },
    onSpeedChange: (speed) => {
      state.speed = speed;
    },
    onSeek: (frameIndex) => {
      state.frameIndex = frameIndex;
      render();
    },
    onReset: () => {
      state.frameIndex = 0;
      state.isPlaying = true;
      controls.setPlayState(true);
      render();
    },
  });

  createMlLab({
    container: document.getElementById("ml-lab-panel"),
    onApplyConfidence: async ({ runId, backendBaseUrl }) => {
      const scenarioKey = SCENARIO_OPTIONS[state.scenarioIndex].key;
      const mappedScenario = scenarioKey === "demo" ? "jam-recovery" : scenarioKey;
      const payload = await fetch(`${backendBaseUrl || BACKEND_BASE_URL}/api/simulate/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario: mappedScenario,
          runId,
          steps: 260,
          agents: 32,
          packetLoss: 0.25,
        }),
      }).then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text());
        }
        return response.json();
      });

      const scenarioUrl = `${backendBaseUrl || BACKEND_BASE_URL}${payload.scenarioUrl}`;
      const loaded = await loadScenario([scenarioUrl]);
      applyScenario(loaded, state.scenarioIndex);
    },
  });

  function applyScenario(nextScenario, scenarioIndex) {
    scenario = nextScenario;
    state.scenarioIndex = scenarioIndex;
    state.primaryRunIndex = 0;
    state.secondaryRunIndex = Math.min(1, scenario.runs.length - 1);
    state.frameIndex = 0;
    state.isPlaying = true;
    state.lastTime = 0;

    canvas.setWorld(scenario.metadata.world);
    controls.setRunOptions(scenario.runs, 0, Math.min(1, scenario.runs.length - 1));
    controls.setPlayState(true);
    controls.setScenarioIndex(scenarioIndex);
    controls.setCompareMode(state.compareMode && scenario.runs.length > 1);
    render();
  }

  async function switchScenario(scenarioIndex) {
    const selected = SCENARIO_OPTIONS[scenarioIndex];
    try {
      const loaded = await loadScenario(selected.paths);
      applyScenario(loaded, scenarioIndex);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showStartupWarning(`场景切换失败：${message}`);
      const fallback = makeFallbackScenario(message);
      applyScenario(fallback, scenarioIndex);
    }
  }

  function currentRun(index) {
    return scenario.runs[Math.max(0, Math.min(index, scenario.runs.length - 1))];
  }

  function currentFramesLength() {
    const primaryLength = currentRun(state.primaryRunIndex).frames.length;
    if (!state.compareMode) {
      return primaryLength;
    }
    const secondaryLength = currentRun(state.secondaryRunIndex).frames.length;
    return Math.max(primaryLength, secondaryLength);
  }

  function advance(deltaMs) {
    if (!state.isPlaying) {
      return;
    }

    const baseFps = 7;
    const framesPerMs = (baseFps * state.speed) / 1000;
    const nextFrame = state.frameIndex + deltaMs * framesPerMs;
    const maxFrames = currentFramesLength();

    if (nextFrame >= maxFrames - 1) {
      state.frameIndex = maxFrames - 1;
      state.isPlaying = false;
      controls.setPlayState(false);
      return;
    }

    state.frameIndex = nextFrame;
  }

  function render() {
    const frameIdx = Math.max(0, Math.floor(state.frameIndex));
    const primaryRun = currentRun(state.primaryRunIndex);
    const primaryFrame = primaryRun.frames[Math.min(frameIdx, primaryRun.frames.length - 1)];

    if (state.compareMode && scenario.runs.length > 1) {
      const secondaryRun = currentRun(state.secondaryRunIndex);
      const secondaryFrame = secondaryRun.frames[Math.min(frameIdx, secondaryRun.frames.length - 1)];
      canvas.renderCompare(
        primaryFrame,
        secondaryFrame,
        `左视图: ${primaryRun.name}`,
        `右视图: ${secondaryRun.name}`,
      );
      metricsBoard.renderCompare(
        {
          runName: primaryRun.name,
          summary: primaryRun.summary,
          frame: primaryFrame,
        },
        {
          runName: secondaryRun.name,
          summary: secondaryRun.summary,
          frame: secondaryFrame,
        },
        frameIdx,
        currentFramesLength() - 1,
      );
      controls.setTimeline(frameIdx, currentFramesLength() - 1);
      return;
    }

    canvas.render(primaryFrame);
    metricsBoard.render(primaryRun.summary, primaryFrame, primaryRun.frames.length - 1, primaryRun.name);
    controls.setTimeline(frameIdx, primaryRun.frames.length - 1);
  }

  function frameLoop(timestamp) {
    if (state.lastTime === 0) {
      state.lastTime = timestamp;
    }

    const delta = timestamp - state.lastTime;
    state.lastTime = timestamp;

    advance(delta);
    render();
    requestAnimationFrame(frameLoop);
  }

  async function loadInitialScenario() {
    try {
      const loaded = await loadScenario(SCENARIO_OPTIONS[state.scenarioIndex].paths);
      applyScenario(loaded, state.scenarioIndex);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showStartupWarning(`场景 JSON 未就绪，已使用内置兜底场景。${message}`);
    }
  }

  render();
  requestAnimationFrame(frameLoop);
  loadInitialScenario();
}

bootstrap();
