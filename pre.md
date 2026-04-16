# 个人展示段 PPT 大纲（精简版，约 6-8 分钟）

## 0. 你的展示定位（先记住这句）
- 承接队友理论后，你的开场句：
  - Talk is cheap, show me the code.
  - 下面我用一个可运行系统，展示“感知-决策-协同”如何真的工作。
- 你的任务不是讲全理论，而是做三件事：
  - 让非 CS 同学看懂
  - 让老师看到专业性
  - 用视频和图表证明你不是在放动画

---

## 1. 时间分配（建议）
- 第 1 页（引入）：0:30
- 第 2 页（系统闭环）：0:45
- 第 3 页（数据集与任务）：1:00
- 第 4 页（模型与训练曲线）：1:15
- 第 5 页（视频 1：基线对照）：1:00
- 第 6 页（视频 2：抗干扰恢复）：1:00
- 第 7 页（视频 3：ML Lab + 回注）：1:15
- 第 8 页（结论）：0:30

合计约 7 分 15 秒。

---

## 第 1 页：承接引入（无视频）
### PPT 上屏文字（3 行）
- Talk is cheap, show me the code.
- 我们做了一个“感知-决策-协同”闭环系统。
- 重点看：它是否真的提升了协同作战表现。

### 图片
- 建议 1 张封面图（蜂群无人机/战场抽象图）。

### 图注（放图下方）
图 1  蜂群无人系统从“概念”走向“可运行验证”。

### 讲稿（脱稿友好，和上屏文字有重合）
这一段我只做一件事：show me the code。我们把“感知-决策-协同”做成了可运行闭环，不是概念图。接下来我会用三段视频和两张训练曲线，直接回答一个问题：这个系统是不是实打实改善了协同表现。

---

## 第 2 页：系统闭环（无视频）
### PPT 上屏文字（4 行）
- 感知层：三分类识别目标语义
- 决策层：按置信度调整目标优先级
- 协同层：在丢包/干扰下持续执行任务
- 输出：完成率、响应时间、链路稳定度

### 图片
- 建议你画一个简单流程图：
  - 数据集 -> 训练模型 -> 置信度 -> 仿真分配 -> 指标对比

### 图注
图 2  本项目“感知-决策-协同”闭环流程。

### 讲稿
这一页把系统结构说清楚。感知层负责看见目标，决策层负责先做什么，协同层负责在干扰情况下还能不能做完。最后我们不是看动画，而是看输出指标：完成率、响应时间和链路稳定度。

---

## 第 3 页：数据集与任务定义（无视频）
### PPT 上屏文字（4 行）
- 数据集：VisDrone（官方公开数据）
- 类别：vehicle / decoy / civilian-object
- 任务：三分类输出置信度
- 作用：把“识别结果”转成“资源分配依据”

### 图片
- 主图（推荐）：[docs/outputs/figures/visdrone-sample-grid-3x3.png](docs/outputs/figures/visdrone-sample-grid-3x3.png)

### 图注
图 3  VisDrone 数据样本示意（每类各 3 张，便于课堂快速理解）。

### 讲稿
这里我们用的是 VisDrone 数据集。为了让非 CS 同学也能直观看懂，我只保留了三类：vehicle、decoy、civilian-object。模型输出的不只是类别名称，更重要是置信度，这个置信度会直接影响后面的目标分配策略。

### 数据集国标引用（可直接贴到页脚）
[1] ZHU P F, WEN L Y, DU D W, et al. Detection and tracking meet drones challenge[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 44(11): 7380-7399.

### 引用在 PPT 里的放法
- 在本页右下角放 9-10 pt 灰色小字：
  - 数据来源：[1]
- 在最后一页再放一次完整参考文献条目（上面这行）。

---

## 第 4 页：模型与训练结果（无视频）
### PPT 上屏文字（4 行）
- 模型：MobileNetV3-Small（三分类）
- 指标：Loss 下降、Accuracy 上升
- 结论：模型很快收敛
- 含义：感知结果可用于后续协同决策

### 图片
- [docs/outputs/figures/tiny-cnn-loss-curve.png](docs/outputs/figures/tiny-cnn-loss-curve.png)
- [docs/outputs/figures/tiny-cnn-accuracy-curve.png](docs/outputs/figures/tiny-cnn-accuracy-curve.png)

### 图注
图 4  训练与验证损失曲线（Loss Curve）。
图 5  训练与验证准确率曲线（Accuracy Curve）。

### 讲稿
这一页是“专业性”核心。左图看 loss，右图看 accuracy。关键结论很简单：loss 下降、accuracy 上升，说明模型可以稳定收敛。注意我不是只追求好看的分数，而是要把这个结果用于后续协同任务分配，这才是军事场景里真正有意义的地方。

---

## 第 5 页：视频 1（基线对照）
### PPT 上屏文字（4 行）
- 场景：Recon Coverage
- 左右同帧对照，保证公平
- 看覆盖效率与队形稳定
- 比“感觉”更重要的是“同条件对比”

### 视频
- V1：基线对照视频（建议 35-45 秒）

### 图注
视频 1  Recon Coverage 同帧双视图对照（左/右策略并行）。

### 讲稿
这段视频主要看基线表现。我要强调“同帧对照”，因为这样最公平。观察点只有两个：覆盖效率和队形稳定。我们不下绝对结论，而是给后续干扰场景做参照。

---

## 第 6 页：视频 2（抗干扰恢复）
### PPT 上屏文字（4 行）
- 场景：Jam Recovery
- 注入丢包和干扰
- 看恢复速度与任务连续性
- 鲁棒性比峰值性能更重要

### 视频
- V2：抗干扰恢复视频（建议 35-45 秒）

### 图注
视频 2  Jam Recovery 场景下的抗干扰恢复过程。

### 讲稿
这一段最有“军事味”。战场里一定有干扰，所以我们看的不是某一帧多漂亮，而是干扰后能不能恢复、任务是否连续。这里体现的是系统鲁棒性，而不是理想条件下的极限成绩。

---

## 第 7 页：视频 3（ML Lab + 回注）
### PPT 上屏文字（4 行）
- ML Lab 实时训练可见
- 彩色进度条 + 产物落盘
- 置信度回注仿真决策
- 从“识别”走向“协同收益”

### 视频
- V3：训练与回注视频（建议 45-60 秒）

### 图注
视频 3  ML Lab 训练过程与置信度回注后协同变化。

### 讲稿
这段视频证明它不是静态 demo。我们能看到实时训练、进度条、产物生成，然后把置信度回注到仿真。也就是说，模型输出会改变任务分配行为，这才是“AI 真正进入作战链路”。

---

## 第 8 页：结论（无 Q&A 版）
### PPT 上屏文字（3 行）
- 这不是单纯 UI，而是可运行闭环验证
- 模型输出确实影响协同决策
- 非理想条件下仍能体现鲁棒性价值

### 图片
- 建议放系统主界面总览截图 1 张。

### 图注
图 6  项目演示总览：感知、决策、协同的一体化展示。

### 讲稿
最后总结三句话。第一，我们做的是可运行闭环，不是纯展示。第二，模型结果确实影响了协同决策。第三，在干扰条件下系统依然体现鲁棒性价值。这就是我这部分展示想交付的核心结论。

---

## 视频录制详细操作（你照着做就行）

## 录制前统一准备（一次即可）
1. 启动后端：
   - 终端执行："/Users/mike/Documents/University/Grade_1B/classes/Military Theory/Pre/Swarm/.venv/bin/python" -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
2. 启动前端静态服务：
   - 终端执行："/Users/mike/Documents/University/Grade_1B/classes/Military Theory/Pre/Swarm/.venv/bin/python" -m http.server 5173 --directory web-demo
3. 打开浏览器：
   - http://127.0.0.1:5173/
4. 录屏工具：
   - macOS 用 Shift + Command + 5，选择“录制所选部分”，只框住网页主区域。
5. 统一参数：
   - 播放速度 speed = 1.0
   - 对比模式 compare = on
   - 每段视频时长控制在建议范围内

## V1 录制步骤（Recon Coverage）
1. 场景切换到 Recon Coverage。
2. 时间轴回到 0，点击播放。
3. 保持双视图都在画面内，录制 35-45 秒。
4. 中间不要切页，只让系统自然播放。
5. 导出命名：v1-recon-compare.mp4。

## V2 录制步骤（Jam Recovery）
1. 场景切换到 Jam Recovery。
2. 时间轴回到 0，点击播放。
3. 录制 35-45 秒，重点覆盖“干扰发生后到恢复”的时间段。
4. 左下角黄色事件提示（cleared by agent）现在已延长，可等待出现后再结束录制。
5. 导出命名：v2-jam-recovery.mp4。

## V3 录制步骤（ML Lab + 回注）
1. 切到 ML Lab 页面。
2. 点击训练任务启动（使用默认参数即可）。
3. 录制实时进度条、状态变化、产物出现（20-30 秒）。
4. 点击应用置信度到仿真。
5. 切回仿真播放 15-20 秒，展示前后差异。
6. 导出命名：v3-ml-train-apply.mp4。

---

## 你自己截图的方法（不用我已有截图）

## 方法 A：手动截图（推荐）
1. 浏览器打开目标页面。
2. 暂停在你想要的帧。
3. Shift + Command + 4，框选网页核心区域。
4. 保存到 docs/outputs/screenshots-self/。
5. 建议命名：
   - s1-system-overview.png
   - s2-recon-compare.png
   - s3-jam-recovery.png
   - s4-ml-lab-progress.png

## 方法 B：命令行自动截图（可选）
- 先确保 playwright 可用，再执行：
  - npx --yes playwright screenshot --device="Desktop Chrome" --full-page "http://127.0.0.1:5173/#scene-hero" "docs/outputs/screenshots-self/s1-system-overview.png"
  - npx --yes playwright screenshot --device="Desktop Chrome" --full-page "http://127.0.0.1:5173/#scene-sandbox" "docs/outputs/screenshots-self/s2-recon-compare.png"
  - npx --yes playwright screenshot --device="Desktop Chrome" --full-page "http://127.0.0.1:5173/#scene-lab" "docs/outputs/screenshots-self/s4-ml-lab-progress.png"

---

## 图像适配结论（已做图片检查）
- 适合放 PPT：
  - visdrone-sample-grid-3x3.png（数据分布直观）
  - tiny-cnn-loss-curve.png（loss 变化明显）
  - tiny-cnn-accuracy-curve.png（accuracy 变化明显）
- 不建议作为主证据：
  - completion-vs-packet-loss.png
  - response-time-vs-packet-loss.png
  - ml-gain-vs-packet-loss.png
- 原因：三张 packet-loss 图基本是平线，课堂观感弱，解释成本高。

---

## 你要讲清楚的“模型军事价值”一句话模板
- 模型的价值不在“识别本身”，而在“把目标语义转化为协同优先级”，减少对诱饵的资源浪费，提高任务执行效率与鲁棒性。

---

## 最后一页可放的参考文献（国标风格）
[1] ZHU P F, WEN L Y, DU D W, et al. Detection and tracking meet drones challenge[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 44(11): 7380-7399.
