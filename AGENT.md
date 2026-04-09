# AGENT 持续记录

## 项目背景

课程主题为“蜂群无人系统与自主协同作战”。本项目采用“教育仿真 + 可视化 + 轻量 ML”路线，目标是做出可录屏、可讲解、可量化的技术展示系统。

## 本轮改动（2026-04-09）

### 1) 文档与结构

1. 新建 `README.md`，提供项目定位、快速运行、录屏建议与边界声明。
2. 新建 `docs/system-design.md`，定义模块职责与数据协议。
3. 新建 `docs/experiment-metrics.md`，定义覆盖率、完成率、响应时间、鲁棒性保持率等指标。

### 2) 仿真核心（sim-core）

1. 建立完整数据模型：`Vec2`、`AgentState`、`TargetState`、`RunResult`。
2. 实现动力学更新：速度/加速度约束、转向约束、边界反弹。
3. 实现干扰模型：动态丢包脉冲、局部干扰区、随机个体失效。
4. 实现协同逻辑：
	- 去中心化：邻域投票一致性 + Boids（分离/对齐/聚合）
	- 集中式：全局偏置目标分配
5. 实现任务清除逻辑与事件流记录。
6. 实现 `simulate.py`：
	- 单策略导出
	- 去中心化 vs 集中式对照导出（`--compare`）
	- 支持外部 ML 置信度文件（`--ml-confidence`）
7. 实现 `run_experiments.py`：批量跑丢包率/策略/ML 开关组合，导出 CSV + JSON。

### 3) ML 模块（ml-module）

1. 新建合成数据生成器 `data/synthetic-generator.py`：
	- 三类目标：`vehicle`、`decoy`、`civilian-object`
	- 自动生成 train/val 数据与 manifest。
2. 新建轻量模型 `model.py`（Tiny CNN）。
3. 新建训练脚本 `train.py`：
	- 自动选择设备（MPS/CUDA/CPU）
	- 训练日志、checkpoint、history 导出。
4. 新建推理脚本 `infer.py`：
	- 单图/文件夹推理
	- 计算并导出类别置信度 JSON（供仿真读取）。

### 4) 网页战术沙盘（web-demo）

1. 新建 `index.html` + `styles.css`，构建“战术沙盘”风格界面。
2. 新建 `main.js`：统一状态管理、帧循环播放、时间轴控制。
3. 新建组件：
	- `control-panel.js`：策略切换、播放、倍速、时间轴
	- `swarm-canvas.js`：网格、通信链路、目标、agent 航迹绘制
	- `metrics-board.js`：实时指标展示
4. 新建 `public/scenarios/demo-compare.json` 示例场景文件。

### 5) 可运行性验证

已执行并通过：

1. Python 语法编译检查
	- `python -m compileall sim-core ml-module`
2. 仿真导出
	- `simulate.py --compare ... --output ../web-demo/public/scenarios/demo-compare.json`
3. 批量实验导出
	- 生成 `docs/outputs/experiment-matrix.csv`
	- 生成 `docs/outputs/experiment-matrix.json`
4. 网页可访问验证
	- `HEAD /` 返回 `200`
	- `HEAD /public/scenarios/demo-compare.json` 返回 `200`
5. IDE 错误检查：当前无错误。

## 本轮追加改动（2026-04-09，第二次迭代）

### 1) VS Code 调试体验补强

1. 补全 `.vscode/launch.json`
	- 新增 `Sim: Generate Scenario Pack`
	- 新增 `ML: Evaluate Validation Set`
	- 新增 `Docs: Plot Metrics`
2. 补全 `.vscode/tasks.json`
	- 新增场景包生成、验证集评估、图表导出任务。
3. 保持 `python.defaultInterpreterPath` 指向项目根目录 `.venv`，方便在 VS Code 中直接 F5 调试。

### 2) ML 证据链补强

1. 新增 `ml-module/evaluate.py`
	- 读取 checkpoint 与验证集
	- 导出 `evaluation-summary.json`
	- 导出 `confusion-matrix.csv`
	- 导出 `confusion-matrix.png`
2. 更新 `ml-module/DATASET.md`
	- 明确说明“不是无数据训练”，而是“先用代码生成合成数据集，再训练模型”。
3. 调整 `train.py`
	- 新增 `--num-workers`
	- 默认使用 `0`，避免受限环境与 VS Code 调试时触发共享内存权限报错。

### 3) 前端演示效果增强

1. `web-demo/src/main.js`
	- 新增双视图同步对照状态管理。
2. `web-demo/src/components/control-panel.js`
	- 新增主视图/副视图策略选择。
	- 新增“双视图对照”切换。
3. `web-demo/src/components/swarm-canvas.js`
	- 支持单视图与左右分屏双视图渲染。
4. `web-demo/src/components/metrics-board.js`
	- 支持单视图指标与双视图对照指标板。

### 4) 运行稳定性修正

1. 为 `docs/plot_metrics.py` 和 `ml-module/evaluate.py` 增加本地缓存目录与 `Agg` 后端设置。
2. 更新 `.gitignore`
	- 忽略评估报告、matplotlib 本地缓存与运行期生成物。

### 5) 本轮验证结果

1. `python -m compileall sim-core ml-module docs` 通过。
2. 小规模 synthetic dataset 生成、1 epoch 训练、类别置信度导出通过。
3. `evaluate.py` 成功生成：
	- `ml-module/reports/eval/evaluation-summary.json`
	- `ml-module/reports/eval/confusion-matrix.csv`
	- `ml-module/reports/eval/confusion-matrix.png`
4. `plot_metrics.py` 成功更新 `docs/outputs/figures/*.png`。
5. 受当前沙箱限制，本轮未能在命令行中重新绑定本地 `5173` 端口做 HTTP 服务验证；建议在 VS Code 中直接使用 `Demo: Serve and Open` 调试配置进行本机检查。

## 当前状态

1. 教学演示主链路已打通：仿真生成 -> JSON 回放 -> 指标展示。
2. ML 训练/推理脚本已具备，可直接产出置信度并接入仿真评分。
3. 实验矩阵导出能力已具备，可直接用于报告制图与结论支撑。
4. VS Code 调试入口已基本齐全，适合直接在课堂前做单步调试与录屏彩排。

## 下一步（优先顺序）

1. 接入真实训练结果到仿真：用 `infer.py` 导出的类别置信度驱动任务分配，并生成“有 ML / 无 ML”对照视频素材。
2. 增加场景模板：侦察覆盖、干扰重构、多目标分配三种可切换地图与目标布局。
3. 加入图表导出脚本：自动绘制完成率-丢包率曲线、响应时间箱线图、混淆矩阵图。
4. 增加演讲支持文件：5-8 分钟讲稿、录屏分镜、每页核心结论卡片。
5. 前端增强：并排双视图同屏对照（同时播放两策略）。
6. 接入 GitHub 仓库：初始化 git、提交首版，并在认证恢复后推送到远端仓库。
