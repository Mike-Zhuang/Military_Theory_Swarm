#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
TEMP_DIR="$(mktemp -d -t swarm-verify-XXXX)"

cleanup() {
  rm -rf "${TEMP_DIR}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[verify] 未找到虚拟环境解释器: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[verify] 1/6 compileall"
"${PYTHON_BIN}" -m compileall "${ROOT_DIR}/backend" "${ROOT_DIR}/sim-core" "${ROOT_DIR}/ml-module" "${ROOT_DIR}/docs"

echo "[verify] 2/6 后端健康检查（函数级）"
"${PYTHON_BIN}" -c "import asyncio; from backend.app import health; result = asyncio.run(health()); assert result['status'] == 'ok'; print(result)"

echo "[verify] 3/6 前端静态资源检查"
"${PYTHON_BIN}" -c "from pathlib import Path; root=Path('${ROOT_DIR}')/'web-demo'; index=(root/'index.html').read_text(encoding='utf-8'); assert './src/main.js' in index; assert (root/'src'/'main.js').exists(); assert (root/'src'/'styles.css').exists(); print('frontend static files ok')"

echo "[verify] 4/6 生成场景到 generated 目录"
(
  cd "${ROOT_DIR}/sim-core"
  "${PYTHON_BIN}" generate_scenario_pack.py --steps 60 --agents 12 --packet-loss 0.18
)

echo "[verify] 5/6 运行实验矩阵与图表"
(
  cd "${ROOT_DIR}/sim-core"
  "${PYTHON_BIN}" run_experiments.py --scenario jam-recovery --steps 60 --agents 12 --output-dir ../docs/outputs
)
(
  cd "${ROOT_DIR}/docs"
  "${PYTHON_BIN}" plot_metrics.py --csv outputs/experiment-matrix.csv --output-dir outputs/figures
)

echo "[verify] 6/6 ML 训练/评估 smoke test"
(
  cd "${ROOT_DIR}/ml-module"
  "${PYTHON_BIN}" data/synthetic-generator.py --output "${TEMP_DIR}/synthetic-data" --samples-per-class 28 --seed 21
  "${PYTHON_BIN}" train.py --data-dir "${TEMP_DIR}/synthetic-data" --model-name tiny-cnn --epochs 1 --batch-size 16 --output "${TEMP_DIR}/tiny-cnn.pt"
  "${PYTHON_BIN}" evaluate.py --checkpoint "${TEMP_DIR}/tiny-cnn.pt" --data-dir "${TEMP_DIR}/synthetic-data" --split val --batch-size 16 --output-dir "${TEMP_DIR}/eval"
)

echo "[verify] 检查关键文件"
[[ -f "${ROOT_DIR}/web-demo/public/generated/jam-recovery-compare.json" ]]
[[ -f "${ROOT_DIR}/docs/outputs/experiment-matrix.csv" ]]
[[ -f "${ROOT_DIR}/docs/outputs/figures/completion-vs-packet-loss.png" ]]
[[ -f "${TEMP_DIR}/eval/evaluation-summary.json" ]]

echo "[verify] 全部通过"
