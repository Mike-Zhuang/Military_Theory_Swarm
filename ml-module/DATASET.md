# 数据集说明（V2）

当前项目采用“双轨数据策略”：

1. `VisDrone` 真实数据子集（主线）
2. 合成数据（保底与快速调试）

## 1) 真实数据主线：VisDrone 子集

### 类别重映射

为了对齐课程演示目标，`prepare_visdrone.py` 将检测标注映射为 3 类分类任务：

- `vehicle`：`car/van/bus/truck/motor`
- `civilian-object`：`pedestrian/people/bicycle/tricycle/awning-tricycle`
- `decoy`：`ignored-region` + 随机背景硬负样本

### 处理流程

```bash
cd ml-module
python data/prepare_visdrone.py \
  --raw-dir data/visdrone/raw \
  --output-dir data/visdrone-ready \
  --subset-size-per-class 900 \
  --val-ratio 0.2
```

可选：自动下载并解压官方训练集压缩包

```bash
python data/prepare_visdrone.py --download
```

输出目录结构：

```text
data/visdrone-ready/
├── train/
│   ├── vehicle/
│   ├── civilian-object/
│   └── decoy/
├── val/
│   ├── vehicle/
│   ├── civilian-object/
│   └── decoy/
└── manifest.json
```

## 2) 合成数据保底路线

在你没有下载真实数据或课堂前需要快速回归时，可以继续用合成数据：

```bash
python data/synthetic-generator.py --output data/generated --samples-per-class 1200
python train.py --data-dir data/generated --epochs 10 --batch-size 64 --output checkpoints/tiny-cnn.pt
```

## 3) 训练和评估（通用）

```bash
python train.py --data-dir data/visdrone-ready --epochs 18 --batch-size 64 --learning-rate 0.0006 --output runs/manual/checkpoints/tiny-cnn.pt
python infer.py --checkpoint runs/manual/checkpoints/tiny-cnn.pt --calibration-dir data/visdrone-ready/val --emit-class-confidence runs/manual/class-confidence.json
python evaluate.py --checkpoint runs/manual/checkpoints/tiny-cnn.pt --data-dir data/visdrone-ready --split val --output-dir runs/manual/eval
python render_sample_grid.py --data-dir data/visdrone-ready --split val --output runs/manual/sample-grid.png
```

## 4) 关于“训练看起来很快”

如果你觉得训练太快、像玩具实验，可以从以下维度提高真实度：

1. 增加 `--subset-size-per-class`。
2. 增加 `--epochs`。
3. 在前端展示 `summary.json` 中的样本量、设备和总训练时长，解释实验成本。

## 5) 局限与边界

- 该三类映射是课程演示映射，不等同于真实作战分类标准。
- VisDrone 采集分布与目标课题场景仍有域差异，结论重点是“方法可行性”而不是“部署可行性”。
