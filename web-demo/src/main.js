import { createControlPanel } from "./components/control-panel.js";
import { MetricsBoard } from "./components/metrics-board.js";
import { SwarmCanvas } from "./components/swarm-canvas.js";
import { loadScenario } from "./lib/scenario-loader.js";

const SCENARIO_OPTIONS = [
  {
    label: "干扰重构 (jam-recovery)",
    path: "./public/scenarios/jam-recovery-compare.json",
  },
  {
    label: "侦察覆盖 (recon-coverage)",
    path: "./public/scenarios/recon-coverage-compare.json",
  },
  {
    label: "多目标分配 (multi-target)",
    path: "./public/scenarios/multi-target-allocation-compare.json",
  },
  {
    label: "默认场景 (demo)",
    path: "./public/scenarios/demo-compare.json",
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

let scenario = await loadScenario(SCENARIO_OPTIONS[state.scenarioIndex].path);
const canvas = new SwarmCanvas(
  document.getElementById("swarm-canvas"),
  scenario.metadata.world,
);
const metricsBoard = new MetricsBoard(document.getElementById("metrics-board"));

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

async function switchScenario(scenarioIndex) {
  const selected = SCENARIO_OPTIONS[scenarioIndex];
  const loaded = await loadScenario(selected.path);
  scenario = loaded;
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

render();
requestAnimationFrame(frameLoop);
