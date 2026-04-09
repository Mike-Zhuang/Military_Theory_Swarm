# 为什么可以直接训练？

很多课程项目会误以为“没有公开数据集就不能训练”。其实训练的核心不是“必须有公开数据集”，而是“必须有带标签的数据样本”。这个项目采用的是**可控合成数据**路线：

1. 用 [data/synthetic-generator.py](data/synthetic-generator.py) 自动生成俯视图样本。
2. 样本类别固定为 `vehicle`、`decoy`、`civilian-object`。
3. 用这些样本训练轻量模型，验证“识别误差如何影响协同决策”。

也就是说，这里并不是“没有数据就直接训练”，而是**先用代码造数据，再用造出的数据集训练**。

## 合成数据的价值

- 版权安全：不依赖受限战场图像。
- 成本低：无需人工标注。
- 可控性高：可主动调节噪声、模糊、类间重叠程度。
- 教学友好：便于解释模型误差来源和实验可复现性。

## 局限

- 域差异明显：合成图像分布与真实遥感/无人机图像存在偏差。
- 结论外推受限：只能支撑“方法有效性演示”，不能直接替代真实部署评估。

## 如何替换为真实数据

1. 准备目录结构：

```text
your-data/
├── train/
│   ├── vehicle/
│   ├── decoy/
│   └── civilian-object/
└── val/
    ├── vehicle/
    ├── decoy/
    └── civilian-object/
```

2. 训练命令改为：

```bash
python train.py --data-dir your-data --epochs 10 --batch-size 64 --output checkpoints/tiny-cnn.pt
```

3. 导出类别置信度并接入仿真：

```bash
python infer.py \
  --checkpoint checkpoints/tiny-cnn.pt \
  --calibration-dir your-data/val \
  --emit-class-confidence ../sim-core/class-confidence.json
```
