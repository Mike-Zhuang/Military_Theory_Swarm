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

## 本轮追加改动（2026-04-09，第三次迭代）

### 1) 在线训练台后端（FastAPI）

1. 新增 `backend/app.py`，实现本地异步任务队列：`queued/running/succeeded/failed/cancelled`。
2. 新增 API：
	- `POST /api/dataset/prepare`
	- `POST /api/jobs/train`
	- `POST /api/jobs/evaluate`
	- `GET /api/jobs/{jobId}`
	- `GET /api/runs`
	- `GET /api/runs/{runId}/artifacts`
3. 新增 `POST /api/simulate/compare`，支持将指定 run 的 `class-confidence` 一键注入仿真并导出到 `web-demo/public/generated`。
4. 新增静态挂载：
	- `/artifacts/*`（训练产物）
	- `/generated/*`（仿真生成场景）

### 2) 真实数据接入（VisDrone）

1. 新增 `ml-module/data/prepare_visdrone.py`：
	- 支持下载/断点续传/解压（可选）
	- 支持本地 `images + annotations` 直连
	- 目标重映射到 `vehicle / civilian-object / decoy`
	- 输出 `train/val` 分类数据集与 `manifest.json`
2. 新增 `ml-module/render_sample_grid.py`，输出数据样本网格图供前端展示。
3. 更新 `ml-module/DATASET.md` 为 V2，明确真实数据主线 + 合成保底路线。

### 3) 训练产物增强

1. 重构 `ml-module/train.py`：
	- 输出 `tiny-cnn.history.json`
	- 输出 `tiny-cnn.summary.json`
	- 输出 `tiny-cnn.curve.png`
	- 记录每轮时长、总训练时长、设备与样本规模
2. 保持 `infer.py/evaluate.py` 与现有流程兼容。

### 4) 前端升级（同页战术沙盘 + 在线训练台）

1. `web-demo/index.html` 改为上下双层布局：
	- 上层：战术沙盘 + 控制 + 指标
	- 下层：图例解释面板 + ML 在线实验台
2. 新增 `web-demo/src/components/explain-panel.js`：固定图例、指标定义、场景观察点。
3. 新增 `web-demo/src/components/ml-lab.js`：
	- 提交数据准备/训练/评估任务
	- 轮询任务状态和日志
	- 展示训练曲线、混淆矩阵、样本网格、产物链接
	- 一键应用 run 置信度到仿真
4. `web-demo/src/main.js` 接入后端交互与动态场景加载。
5. `web-demo/src/components/swarm-canvas.js` 增加画布内固定图例。
6. `web-demo/src/components/metrics-board.js` 增加链路度与读图提示。

### 5) Git 产物策略与路径收敛

1. 仿真导出默认路径改为 `web-demo/public/generated`：
	- `sim-core/simulate.py`
	- `sim-core/generate_scenario_pack.py`
2. `.gitignore` 调整：
	- 忽略 `web-demo/public/generated/*.json`
	- `web-demo/public/scenarios/` 仅保留 `demo-compare.json`
	- 忽略 `ml-module/runs/` 和 VisDrone 数据目录
3. 已将以下运行产物从 Git 索引中移除（保留本地文件）：
	- `jam-recovery-compare.json`
	- `recon-coverage-compare.json`
	- `multi-target-allocation-compare.json`

### 6) 调试与验证流程

1. 新增 `backend/requirements.txt`。
2. `.vscode/launch.json` 新增 `Backend: FastAPI` 并加入 `Demo: Serve and Open` 复合启动。
3. `.vscode/tasks.json` 新增 `backend: serve api`。
4. 新增 `scripts/verify.sh` 作为固定门禁：
	- compileall
	- 生成场景包（generated）
	- 实验矩阵 + 图表
	- 关键产物存在性检查

### 7) 当前阻塞与下一步

1. 当前环境无法访问外网下载数据集时，`prepare_visdrone.py --download` 会受限，可通过本地提供 `images/annotations` 目录绕过。
2. 下一步优先在真实 VisDrone 子集上跑一轮完整前端链路，确认前端在线训练体验和时延表现。

## 本轮追加改动（2026-04-09，第四次迭代，V2.1 实施）

### 1) VisDrone 数据链路升级（train+val 双源）

1. 重构 `ml-module/data/prepare_visdrone.py`：
	- 新增 `--split-mode official-val|auto-split`。
	- 新增 `--train-images-dir/--train-annotations-dir`。
	- 新增 `--val-images-dir/--val-annotations-dir`。
	- 新增 `--val-subset-size-per-class`（`0` 表示 val 全量）。
	- 保留旧参数 `--source-images-dir/--source-annotations-dir` 兼容。
2. `manifest.json` 统计增强：
	- source 级别统计（image/annotation/缺失标注）。
	- 过滤统计（small/invalid/unknown）。
	- 类别映射与类别直方图（可用于课堂答辩）。
3. 已用本地真实数据验证：
	- `VisDrone2019-DET-train`：`6471/6471`；
	- `VisDrone2019-DET-val`：`548/548`；
	- `official-val` 模式可成功产出 `data/visdrone-ready/manifest.json`。

### 2) 后端接口扩展

1. `backend/app.py` 的 `DatasetPrepareRequest` 增加：
	- `splitMode`
	- `trainImagesDir/trainAnnotationsDir`
	- `valImagesDir/valAnnotationsDir`
	- `valSubsetSizePerClass`
2. `runDatasetPrepare` 支持以上参数透传到 `prepare_visdrone.py`，并保持旧字段兼容。

### 3) 前端教学化重构（去“AI 味”）

1. 视觉风格重构为“课堂简洁风 + 深色军工元素”：
	- 重写 `web-demo/src/styles.css`，降低背景噪声，强化可读性与信息层级。
2. 新增本地 SVG 图标系统：
	- `web-demo/src/components/icon-set.js`。
3. 重构 `ML 在线实验台`：
	- 4 步流程卡（准备数据 -> 训练 -> 评估 -> 应用仿真）。
	- 可配置 `official-val/auto-split` 与 train/val 目录。
	- 新增“数据准备摘要”与“run 元信息”展示（设备、样本量、总时长、每轮时长）。
4. 重构图例说明区：
	- 增加“推荐演示顺序”和“常见问题自诊断”。

### 4) Git 与验证流程强化

1. `.gitignore` 收敛：
	- 直接忽略 `web-demo/public/generated/`。
	- 新增 `.tmp/`/`tmp/` 忽略。
2. `scripts/verify.sh` 升级为 6 步门禁：
	- compileall
	- 后端健康检查（函数级）
	- 前端静态资源检查
	- 场景生成
	- 实验矩阵 + 图表
	- ML 训练/评估 smoke test（临时目录，不污染仓库）
3. VS Code 调试入口更新：
	- `ML: Prepare VisDrone (Official Val)` 配置新增；
	- 训练/评估默认路径切换到 `data/visdrone-ready` 与 `runs/dev`。

### 5) 本轮验证结果

已执行并通过：

1. `python3 -m compileall backend sim-core ml-module docs`
2. `node --check`（`main.js/ml-lab.js/control-panel.js/explain-panel.js/icon-set.js`）
3. 真实数据准备（`prepare_visdrone.py --split-mode official-val ...`）
4. `./scripts/verify.sh`（6/6 全部通过）

## 本轮追加改动（2026-04-09，第五次迭代，前端空白页修复）

### 1) 问题与根因

1. 现象：页面出现深浅色块，但多个面板无文字内容。
2. 根因：`main.js` 初始化阶段若 `loadScenario()` 抛错，会在创建各面板前中断，导致面板保持空容器状态。

### 2) 修复措施

1. 重构 `web-demo/src/main.js`：
	- 增加 `bootstrap()` 统一初始化与异常兜底。
	- 增加 `makeFallbackScenario()`，场景 JSON 加载失败时自动切换内置演示数据。
	- 场景路径改为 `new URL(..., import.meta.url)`，避免不同访问路径下相对路径失配。
	- 增加启动警告提示，明确“当前使用内置演示场景”。
2. 更新 `web-demo/index.html`：
	- 为四个面板增加静态占位文案（即使脚本异常也不会“纯空白”）。
3. 更新 `web-demo/src/styles.css`：
	- 新增 `.startup-warning` 样式，确保错误提示可见。

### 3) 本轮验证结果

已执行并通过：

1. `node --check web-demo/src/main.js web-demo/src/components/ml-lab.js`
2. `python3 -m compileall web-demo/src backend`
3. `./scripts/verify.sh`（6/6 全部通过）

## 本轮追加改动（2026-04-09，第六次迭代，前端“持续加载”修复）

### 1) 问题描述

1. 现象：页面长期停留“正在加载...”，用户体感像页面没有真正初始化。
2. 线索：HTTP 服务日志能看到模块请求，但 UI 仍停留初始占位状态。

### 2) 根因分析

1. 启动流程对场景加载成功依赖较强，场景文件读取异常时，面板初始化可能被延后或表现为“长期加载”。
2. 浏览器缓存旧版 `main.js` 可能导致用户仍命中旧逻辑（你日志中也出现了较多 304）。

### 3) 修复内容

1. 重构 `web-demo/src/main.js` 启动策略：
	- 启动时先用内置 `fallback` 场景立即完成 UI 初始化，不再等待远端/文件加载。
	- 场景异步加载成功后再覆盖显示，失败时仅提示告警，不阻塞页面可用。
	- 场景切换失败时自动回退到内置场景并提示原因。
2. `index.html` 增加版本化脚本地址：
	- `./src/main.js?v=20260409-02`，强制浏览器拉取新脚本，降低旧缓存命中概率。

### 4) 本轮验证结果

已执行并通过：

1. `node --check web-demo/src/main.js`
2. `python3 -m compileall web-demo/src`
3. `./scripts/verify.sh`（6/6 全部通过）

## 本轮追加改动（2026-04-09，第七次迭代，前端语法崩溃修复）

### 1) 问题与根因

1. 用户 F12 报错：`Uncaught SyntaxError: Unexpected identifier 'Backend'`。
2. 根因：`explain-panel.js` 的模板字符串中写了反引号文本（如 `` `Backend: FastAPI` ``），导致模板字符串被提前截断，整段模块语法报错，页面无法渲染。

### 2) 修复动作

1. `web-demo/src/components/explain-panel.js`
	- 将反引号文本改为 `<code>...</code>`，彻底消除语法歧义。
2. `web-demo/src/styles.css`
	- 新增 `.explain-row code` 样式，保证提示内容可读。
3. `web-demo/index.html`
	- 增加 `<link rel="icon" href="data:,">`，消除 `favicon.ico` 404 噪声。

### 3) 本轮验证

已执行并通过：

1. `node --check`（`main/control-panel/explain-panel/ml-lab/metrics/icon-set`）
2. `python3 -m compileall web-demo/src`
3. `./scripts/verify.sh`（6/6 全部通过）

## 本轮追加改动（2026-04-09，第八次迭代，强制模块缓存切换）

### 1) 现象补充

1. 用户端仍持续出现 `explain-panel.js:45 Unexpected identifier 'Backend'`。
2. 结合日志判断：页面反复命中缓存资源，用户可能仍在执行旧模块图（旧 `main.js` -> 旧 `explain-panel.js`）。

### 2) 处理方案

1. 新增 `web-demo/src/components/explain-panel-v2.js`，不再复用旧入口路径。
2. `web-demo/src/main.js` 的所有模块导入统一追加 `?v=20260409-03`，强制浏览器拉取新模块图。
3. `web-demo/index.html` 主脚本 URL 改为 `main.js?v=20260409-03`，进一步切断旧缓存链路。

### 3) 验证结果

已执行并通过：

1. `node --check web-demo/src/main.js web-demo/src/components/explain-panel-v2.js`
2. `python3 -m compileall web-demo/src`
3. `./scripts/verify.sh`（6/6 全部通过）
