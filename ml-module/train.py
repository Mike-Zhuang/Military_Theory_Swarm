from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

MATPLOTLIB_CACHE_DIR = Path(__file__).resolve().parent / ".matplotlib-cache"
XDG_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MODEL_NAMES, buildModel, setBackboneFrozen

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classification model with anti-overfitting defaults")
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--model-name", type=str, default="mobilenetv3-small", choices=MODEL_NAMES)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "plateau"])
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--augment-level", type=str, default="medium", choices=["light", "medium", "strong"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output", type=str, default="checkpoints/tiny-cnn.pt")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


def chooseDevice() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def buildTransforms(imageSize: int, augmentLevel: str) -> Tuple[transforms.Compose, transforms.Compose]:
    evalTransform = transforms.Compose(
        [
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )

    light = [
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    medium = [
        transforms.RandomResizedCrop(size=imageSize, scale=(0.78, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03),
    ]
    strong = [
        transforms.RandomResizedCrop(size=imageSize, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.28, contrast=0.28, saturation=0.28, hue=0.06),
        transforms.RandomAffine(degrees=12, translate=(0.08, 0.08)),
    ]

    augmentByLevel = {
        "light": light,
        "medium": medium,
        "strong": strong,
    }

    trainTransform = transforms.Compose(
        [
            *augmentByLevel[augmentLevel],
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
            transforms.RandomErasing(p=0.2 if augmentLevel == "strong" else 0.12, scale=(0.02, 0.16)),
        ]
    )
    return trainTransform, evalTransform


def classWeightsFromDataset(dataset: datasets.ImageFolder) -> torch.Tensor:
    labelCounts: Dict[int, int] = {}
    for _, classIndex in dataset.samples:
        labelCounts[classIndex] = labelCounts.get(classIndex, 0) + 1
    total = sum(labelCounts.values())
    classCount = len(dataset.class_to_idx)
    weights: List[float] = []
    for classIndex in range(classCount):
        classSamples = max(1, labelCounts.get(classIndex, 0))
        weights.append(total / (classCount * classSamples))
    return torch.tensor(weights, dtype=torch.float32)


def buildLoaders(
    dataDir: str,
    imageSize: int,
    augmentLevel: str,
    batchSize: int,
    numWorkers: int,
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder]:
    trainTransform, evalTransform = buildTransforms(imageSize=imageSize, augmentLevel=augmentLevel)
    trainDataset = datasets.ImageFolder(root=str(Path(dataDir) / "train"), transform=trainTransform)
    valDataset = datasets.ImageFolder(root=str(Path(dataDir) / "val"), transform=evalTransform)

    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=False,
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=False,
    )
    return trainLoader, valLoader, trainDataset, valDataset


def runEpoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    isTrain = optimizer is not None
    model.train(isTrain)

    totalLoss = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if isTrain:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if isTrain:
            loss.backward()
            optimizer.step()

        totalLoss += float(loss.item()) * labels.size(0)
        predicted = logits.argmax(dim=1)
        correct += int((predicted == labels).sum().item())
        count += labels.size(0)

    avgLoss = totalLoss / max(1, count)
    accuracy = correct / max(1, count)
    return avgLoss, accuracy


def saveCurve(history: List[Dict[str, float]], outputPath: Path) -> None:
    if not history:
        return

    epochs = [int(item["epoch"]) for item in history]
    trainLoss = [float(item["trainLoss"]) for item in history]
    valLoss = [float(item["valLoss"]) for item in history]
    trainAcc = [float(item["trainAcc"]) for item in history]
    valAcc = [float(item["valAcc"]) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].plot(epochs, trainLoss, marker="o", label="train")
    axes[0].plot(epochs, valLoss, marker="s", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, trainAcc, marker="o", label="train")
    axes[1].plot(epochs, valAcc, marker="s", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputPath, dpi=180)
    plt.close(fig)


def saveJson(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def saveCheckpoint(
    model: nn.Module,
    path: Path,
    classToIdx: Dict[str, int],
    modelName: str,
    imageSize: int,
    history: List[Dict[str, float]],
    bestEpoch: int,
    bestValLoss: float,
    bestValAcc: float,
) -> None:
    payload = {
        "stateDict": model.state_dict(),
        "classToIdx": classToIdx,
        "modelName": modelName,
        "imageSize": imageSize,
        "normalization": {
            "mean": IMAGE_NET_MEAN,
            "std": IMAGE_NET_STD,
        },
        "history": history,
        "bestEpoch": bestEpoch,
        "bestValLoss": round(bestValLoss, 6),
        "bestValAcc": round(bestValAcc, 6),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def pickOutputPaths(args: argparse.Namespace) -> Dict[str, Path]:
    outputPath = Path(args.output)
    checkpointDir = Path(args.output_dir) if args.output_dir else outputPath.parent
    checkpointDir.mkdir(parents=True, exist_ok=True)
    return {
        "legacyOutput": outputPath,
        "checkpointDir": checkpointDir,
        "best": checkpointDir / "best.pt",
        "last": checkpointDir / "last.pt",
        "history": checkpointDir / "history.json",
        "summary": checkpointDir / "summary.json",
        "curve": checkpointDir / "curve.png",
        "curveLive": checkpointDir / "curve-live.png",
        "liveMetrics": checkpointDir / "live-metrics.jsonl",
        "progress": checkpointDir / "progress.json",
        "legacyHistory": outputPath.with_suffix(".history.json"),
        "legacySummary": outputPath.with_suffix(".summary.json"),
        "legacyCurve": outputPath.with_suffix(".curve.png"),
    }


def main() -> None:
    args = parseArgs()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = chooseDevice()
    paths = pickOutputPaths(args)
    if paths["liveMetrics"].exists():
        paths["liveMetrics"].unlink()

    trainLoader, valLoader, trainDataset, valDataset = buildLoaders(
        dataDir=args.data_dir,
        imageSize=args.image_size,
        augmentLevel=args.augment_level,
        batchSize=args.batch_size,
        numWorkers=args.num_workers,
    )
    if len(trainDataset) == 0 or len(valDataset) == 0:
        raise SystemExit(f"Dataset is empty: train={len(trainDataset)} val={len(valDataset)}")

    classToIdx = trainDataset.class_to_idx
    model = buildModel(
        modelName=args.model_name,
        classCount=len(classToIdx),
        pretrained=bool(args.pretrained),
    ).to(device)

    # 复杂逻辑说明：使用两阶段训练，先冻结骨干再解冻，可降低小样本训练初期过拟合。
    if args.freeze_epochs > 0:
        setBackboneFrozen(model=model, modelName=args.model_name, frozen=True)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(2, args.epochs))
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, args.early_stop_patience // 2),
        )

    classWeights = classWeightsFromDataset(trainDataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=classWeights, label_smoothing=args.label_smoothing)

    history: List[Dict[str, float]] = []
    bestValLoss = float("inf")
    bestValAcc = 0.0
    bestEpoch = 0
    noImproveEpochs = 0
    startTime = time.perf_counter()
    baseLr = args.learning_rate

    for epoch in range(1, args.epochs + 1):
        epochStart = time.perf_counter()

        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            setBackboneFrozen(model=model, modelName=args.model_name, frozen=False)
            newParameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and not any(parameter is p for group in optimizer.param_groups for p in group["params"])
            ]
            if newParameters:
                optimizer.add_param_group(
                    {
                        "params": newParameters,
                        "lr": baseLr * 0.5,
                        "weight_decay": args.weight_decay,
                    }
                )

        trainLoss, trainAcc = runEpoch(model, trainLoader, criterion, optimizer, device)
        valLoss, valAcc = runEpoch(model, valLoader, criterion, None, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valLoss)
            else:
                scheduler.step()

        epochDuration = time.perf_counter() - epochStart
        lrNow = min(group["lr"] for group in optimizer.param_groups)
        row = {
            "epoch": float(epoch),
            "trainLoss": round(trainLoss, 6),
            "trainAcc": round(trainAcc, 6),
            "valLoss": round(valLoss, 6),
            "valAcc": round(valAcc, 6),
            "learningRate": round(lrNow, 8),
            "epochTimeSec": round(epochDuration, 4),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        with paths["liveMetrics"].open("a", encoding="utf-8") as fileObj:
            fileObj.write(json.dumps(row, ensure_ascii=False) + "\n")

        improved = (bestValLoss - valLoss) > args.early_stop_min_delta
        if improved:
            bestValLoss = valLoss
            bestValAcc = valAcc
            bestEpoch = epoch
            noImproveEpochs = 0
            saveCheckpoint(
                model=model,
                path=paths["best"],
                classToIdx=classToIdx,
                modelName=args.model_name,
                imageSize=args.image_size,
                history=history,
                bestEpoch=bestEpoch,
                bestValLoss=bestValLoss,
                bestValAcc=bestValAcc,
            )
        else:
            noImproveEpochs += 1

        saveCheckpoint(
            model=model,
            path=paths["last"],
            classToIdx=classToIdx,
            modelName=args.model_name,
            imageSize=args.image_size,
            history=history,
            bestEpoch=bestEpoch,
            bestValLoss=bestValLoss if bestEpoch > 0 else valLoss,
            bestValAcc=bestValAcc if bestEpoch > 0 else valAcc,
        )
        saveCurve(history=history, outputPath=paths["curveLive"])

        elapsed = time.perf_counter() - startTime
        avgEpochSec = elapsed / epoch
        etaSec = max(0.0, avgEpochSec * (args.epochs - epoch))
        progress = {
            "currentEpoch": epoch,
            "totalEpochs": args.epochs,
            "bestEpoch": bestEpoch,
            "bestValLoss": round(bestValLoss, 6) if bestEpoch > 0 else None,
            "bestValAcc": round(bestValAcc, 6) if bestEpoch > 0 else None,
            "noImproveEpochs": noImproveEpochs,
            "etaSec": round(etaSec, 2),
            "status": "running",
        }
        saveJson(paths["progress"], progress)

        if noImproveEpochs >= args.early_stop_patience:
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "bestEpoch": bestEpoch,
                        "bestValLoss": round(bestValLoss, 6),
                    },
                    ensure_ascii=False,
                )
            )
            break

    totalTrainingSec = time.perf_counter() - startTime
    if not paths["best"].exists():
        saveCheckpoint(
            model=model,
            path=paths["best"],
            classToIdx=classToIdx,
            modelName=args.model_name,
            imageSize=args.image_size,
            history=history,
            bestEpoch=max(1, len(history)),
            bestValLoss=history[-1]["valLoss"] if history else 0.0,
            bestValAcc=history[-1]["valAcc"] if history else 0.0,
        )

    saveCurve(history=history, outputPath=paths["curve"])

    summary = {
        "device": str(device),
        "epochsRequested": args.epochs,
        "epochsExecuted": len(history),
        "batchSize": args.batch_size,
        "numWorkers": args.num_workers,
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "labelSmoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "augmentLevel": args.augment_level,
        "modelName": args.model_name,
        "pretrained": bool(args.pretrained),
        "freezeEpochs": args.freeze_epochs,
        "trainSamples": len(trainDataset),
        "valSamples": len(valDataset),
        "classToIdx": classToIdx,
        "totalTrainingSec": round(totalTrainingSec, 4),
        "bestEpoch": bestEpoch,
        "bestValLoss": round(bestValLoss, 6) if bestEpoch > 0 else None,
        "bestValAcc": round(bestValAcc, 6) if bestEpoch > 0 else None,
        "lastValLoss": history[-1]["valLoss"] if history else None,
        "lastValAcc": history[-1]["valAcc"] if history else None,
        "generalizationGapAtBest": (
            round(history[max(0, bestEpoch - 1)]["valLoss"] - history[max(0, bestEpoch - 1)]["trainLoss"], 6)
            if bestEpoch > 0
            else None
        ),
    }

    saveJson(paths["summary"], summary)
    paths["history"].write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保持旧接口兼容，避免已有脚本失效。
    torch.save(torch.load(paths["last"], map_location="cpu"), paths["legacyOutput"])
    paths["legacyHistory"].write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    saveJson(paths["legacySummary"], summary)
    saveCurve(history=history, outputPath=paths["legacyCurve"])

    progressFinal = {
        "currentEpoch": len(history),
        "totalEpochs": args.epochs,
        "bestEpoch": bestEpoch,
        "bestValLoss": round(bestValLoss, 6) if bestEpoch > 0 else None,
        "bestValAcc": round(bestValAcc, 6) if bestEpoch > 0 else None,
        "noImproveEpochs": noImproveEpochs,
        "etaSec": 0.0,
        "status": "succeeded",
    }
    saveJson(paths["progress"], progressFinal)

    print(f"Checkpoint saved: {paths['legacyOutput']}")
    print(f"Best checkpoint: {paths['best']}")
    print(f"Last checkpoint: {paths['last']}")
    print(f"Summary saved: {paths['summary']}")
    print(f"Curve saved: {paths['curve']}")


if __name__ == "__main__":
    main()
