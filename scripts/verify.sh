#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[verify] 未找到虚拟环境解释器: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[verify] 1/4 compileall"
"${PYTHON_BIN}" -m compileall "${ROOT_DIR}/backend" "${ROOT_DIR}/sim-core" "${ROOT_DIR}/ml-module" "${ROOT_DIR}/docs"

echo "[verify] 2/4 生成场景到 generated 目录"
(
  cd "${ROOT_DIR}/sim-core"
  "${PYTHON_BIN}" generate_scenario_pack.py --steps 60 --agents 12 --packet-loss 0.18
)

echo "[verify] 3/4 运行实验矩阵与图表"
(
  cd "${ROOT_DIR}/sim-core"
  "${PYTHON_BIN}" run_experiments.py --scenario jam-recovery --steps 60 --agents 12 --output-dir ../docs/outputs
)
(
  cd "${ROOT_DIR}/docs"
  "${PYTHON_BIN}" plot_metrics.py --csv outputs/experiment-matrix.csv --output-dir outputs/figures
)

echo "[verify] 4/4 检查关键文件"
[[ -f "${ROOT_DIR}/web-demo/public/generated/jam-recovery-compare.json" ]]
[[ -f "${ROOT_DIR}/docs/outputs/experiment-matrix.csv" ]]
[[ -f "${ROOT_DIR}/docs/outputs/figures/completion-vs-packet-loss.png" ]]

echo "[verify] 全部通过"
