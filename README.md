# 蜂群自主协同作战（教育仿真 V2）

本项目用于军事理论课程展示，目标是把“蜂群协同仿真 + 真实数据驱动 ML + 前端在线实验台”打通到同一套可演示流程。

> 重要边界：本仓库仅用于课堂教学与算法研究，不包含现实武器化接口、部署参数或实战实现细节。

## 当前能力

- `sim-core`：去中心化/集中式对照仿真，支持外部 ML 置信度注入。
- `ml-module`：VisDrone 子集准备、训练、推理、评估、样本网格导出。
- `backend`：FastAPI 本地异步任务队列，前端可提交训练/评估任务并查询状态。
- `web-demo`：战术沙盘 + 图例解释 + 在线训练台（同页交互）。
- `scripts/verify.sh`：固定验证门禁脚本。

## 目录结构

```text
Swarm/
├── backend/
│   ├── app.py
│   └── requirements.txt
├── ml-module/
│   ├── data/
│   │   ├── synthetic-generator.py
│   │   └── prepare_visdrone.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── model.py
│   ├── render_sample_grid.py
│   ├── requirements.txt
│   └── train.py
├── sim-core/
├── web-demo/
│   ├── public/
│   │   ├── generated/
│   │   └── scenarios/demo-compare.json
│   └── src/
├── scripts/
│   └── verify.sh
└── AGENT.md
```

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml-module/requirements.txt
pip install -r docs/requirements.txt
pip install -r backend/requirements.txt
```

## 启动方式

### 1) 启动后端（在线训练台依赖）

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
```

### 2) 启动前端

```bash
python -m http.server 5173 --directory web-demo
# 浏览器打开 http://127.0.0.1:5173
```

前端页面中 `ML 在线实验台` 默认后端地址就是 `http://127.0.0.1:8001`。

## VisDrone 真实数据流程

### A. 准备数据

可在前端点击“准备/校验 VisDrone 子集”，也可命令行执行：

```bash
cd ml-module
python data/prepare_visdrone.py \
  --raw-dir data/visdrone/raw \
  --output-dir data/visdrone-ready \
  --subset-size-per-class 900 \
  --val-ratio 0.2
```

如果要自动下载官方压缩包：

```bash
python data/prepare_visdrone.py --download
```

### B. 训练 + 评估

```bash
python train.py --data-dir data/visdrone-ready --epochs 18 --batch-size 64 --learning-rate 0.0006 --output runs/manual/checkpoints/tiny-cnn.pt
python infer.py --checkpoint runs/manual/checkpoints/tiny-cnn.pt --calibration-dir data/visdrone-ready/val --emit-class-confidence runs/manual/class-confidence.json
python evaluate.py --checkpoint runs/manual/checkpoints/tiny-cnn.pt --data-dir data/visdrone-ready --split val --output-dir runs/manual/eval
python render_sample_grid.py --data-dir data/visdrone-ready --split val --output runs/manual/sample-grid.png
```

## 仿真导出路径策略

- 保留样例：`web-demo/public/scenarios/demo-compare.json`（入库）
- 运行产物：`web-demo/public/generated/*.json`（忽略）

默认命令：

```bash
cd sim-core
python simulate.py --compare --scenario jam-recovery --output ../web-demo/public/generated/demo-compare.json
python generate_scenario_pack.py --output-dir ../web-demo/public/generated
```

## 验证门禁与推送

固定验证脚本：

```bash
./scripts/verify.sh
```

验证通过后执行：

```bash
git add .
git commit -m "feat: ..."
git push origin main
```

## VS Code 调试入口

已提供以下运行配置：

1. `Backend: FastAPI`
2. `Demo: Serve and Open`
3. `Sim: Compare Export`
4. `Sim: Generate Scenario Pack`
5. `ML: Evaluate Validation Set`
6. `Docs: Plot Metrics`

## 课堂讲解建议

1. 先讲图例和指标定义，降低“看不懂动图”的门槛。
2. 再做去中心化 vs 集中式对照。
3. 接着在前端提交训练任务，展示训练曲线与混淆矩阵。
4. 最后一键应用某个 run 的置信度到仿真回放，讲清“识别误差如何传导到协同决策”。
