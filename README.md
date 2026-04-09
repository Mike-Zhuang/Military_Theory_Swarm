# 蜂群自主协同作战（教育仿真）

本项目用于军事理论课程展示，目标是用可运行代码演示：
- 单体无人平台的运动学与控制逻辑
- 去中心化多智能体协同在干扰环境下的鲁棒性
- 轻量机器学习模型如何影响任务优先级与协同效率

> 重要边界：本仓库仅用于课堂教学与算法研究，不包含现实武器化接口、部署参数或实战实现细节。

## 当前交付内容

- `sim-core`：Python 多智能体仿真核心（含干扰注入、对照实验导出）
- `ml-module`：合成数据生成、轻量分类模型训练与推理
- `ml-module/evaluate.py`：验证集评估与混淆矩阵导出
- `web-demo`：战术沙盘风格前端，可播放对照场景
- `docs`：系统设计与实验指标模板
- `AGENT.md`：开发代理持续记录（每轮改动、下一步计划）

## 目录结构

```text
Swarm/
├── AGENT.md
├── README.md
├── docs/
│   ├── experiment-metrics.md
│   └── system-design.md
├── ml-module/
│   ├── data/
│   │   └── synthetic-generator.py
│   ├── infer.py
│   ├── model.py
│   ├── requirements.txt
│   └── train.py
├── sim-core/
│   ├── requirements.txt
│   ├── run_experiments.py
│   ├── simulate.py
│   └── src/
│       ├── coordination.py
│       ├── disturbance.py
│       ├── dynamics.py
│       ├── exporter.py
│       ├── models.py
│       └── simulator.py
└── web-demo/
    ├── index.html
    ├── public/
    │   └── scenarios/
    │       └── demo-compare.json
    └── src/
        ├── components/
        │   ├── control-panel.js
        │   ├── metrics-board.js
        │   └── swarm-canvas.js
        ├── lib/
        │   └── scenario-loader.js
        ├── main.js
        └── styles.css
```

## 快速开始

### 0) 初始化环境（推荐在项目根目录）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml-module/requirements.txt
pip install -r docs/requirements.txt
```

### 1) 运行仿真并导出对照数据

```bash
cd sim-core
python simulate.py --compare --scenario jam-recovery --steps 260 --agents 32 --packet-loss 0.25 --output ../web-demo/public/scenarios/demo-compare.json
python generate_scenario_pack.py --steps 220 --agents 28 --packet-loss 0.22
```

### 2) 生成并训练轻量 ML 模型

```bash
cd ../ml-module
python data/synthetic-generator.py --output data/generated --samples-per-class 1200
python train.py --data-dir data/generated --epochs 10 --batch-size 64 --output checkpoints/tiny-cnn.pt
python infer.py --checkpoint checkpoints/tiny-cnn.pt --calibration-dir data/generated/val --emit-class-confidence ../sim-core/class-confidence.json
python evaluate.py --checkpoint checkpoints/tiny-cnn.pt --data-dir data/generated --split val --output-dir reports/eval
```

关于“为什么没有外部数据集也能训练”，核心原因是这里用的是**我们自己生成的合成数据集**：脚本会自动画出俯视目标图像，并按类别写入 `train/val` 目录，所以训练依然是基于数据集完成的，只是数据集不是从网上下载，而是项目自己构造。详细说明见 `ml-module/DATASET.md`。

### 3) 导出实验矩阵与图表

```bash
cd ../sim-core
python run_experiments.py --scenario jam-recovery --steps 180 --agents 24 --output-dir ../docs/outputs
cd ../docs
python plot_metrics.py --csv outputs/experiment-matrix.csv --output-dir outputs/figures
```

### 4) 启动网页演示

```bash
cd ../
python -m http.server 5173 --directory web-demo
# 浏览器打开 http://localhost:5173
```

## VS Code 直接调试

已内置调试配置文件：

- `.vscode/launch.json`
- `.vscode/tasks.json`

在 VS Code 的 Run and Debug 里可以直接运行：

1. `Sim: Compare Export`
2. `Sim: Generate Scenario Pack`
3. `Sim: Experiment Matrix`
4. `ML: Generate Synthetic Dataset`
5. `ML: Train Tiny CNN`
6. `ML: Infer and Export Class Confidence`
7. `ML: Evaluate Validation Set`
8. `Docs: Plot Metrics`
9. `Demo: Serve and Open`

## 建议演示流程（录屏友好）

1. 打开双视图对照模式，同时播放 `去中心化 + ML` 与 `集中式 + 无 ML`。
2. 动态调整通信干扰强度，观察任务完成率与平均响应时间变化。
3. 切换场景（侦察覆盖 / 干扰重构 / 多目标分配）并强调鲁棒性差异。
4. 展示 `reports/eval/confusion-matrix.png` 与误检影响曲线，解释模型误差如何传导到协同决策。

## 后续扩展

- 引入图神经网络（GNN）做邻域信息融合
- 引入一致性滤波与鲁棒控制约束
- 增加“误报补偿策略”与“失联重构策略”的可视化对照
