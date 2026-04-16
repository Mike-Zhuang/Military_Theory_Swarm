from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from model import MODEL_NAMES, buildModel, setBackboneFrozen

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, labelSmoothing: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.labelSmoothing = labelSmoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ceLoss = functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.labelSmoothing,
        )
        pt = torch.exp(-ceLoss)
        focalLoss = ((1.0 - pt) ** self.gamma) * ceLoss
        return focalLoss.mean()


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classification model with dual-validation monitoring")
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
    parser.add_argument("--loss-type", type=str, default="focal", choices=["cross-entropy", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--balanced-sampler", dest="balancedSampler", action="store_true")
    parser.add_argument("--no-balanced-sampler", dest="balancedSampler", action="store_false")
    parser.add_argument("--monitor-split", type=str, default="dev-val")
    parser.add_argument("--official-split", type=str, default="official-val")
    parser.add_argument("--output", type=str, default="checkpoints/tiny-cnn.pt")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(pretrained=True, balancedSampler=True)
    return parser.parse_args()


def chooseDevice() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolveSplitDir(dataDir: Path, preferredNames: List[str]) -> Tuple[Path, str]:
    for splitName in preferredNames:
        candidate = dataDir / splitName
        if candidate.exists():
            return candidate, splitName
    raise RuntimeError(f"Split directory not found, candidates={preferredNames}, dataDir={dataDir}")


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
        transforms.RandomResizedCrop(size=imageSize, scale=(0.84, 1.0), ratio=(0.88, 1.12)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02),
    ]
    strong = [
        transforms.RandomResizedCrop(size=imageSize, scale=(0.74, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05)),
    ]

    trainTransform = transforms.Compose(
        [
            *(light if augmentLevel == "light" else medium if augmentLevel == "medium" else strong),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
            transforms.RandomErasing(p=0.08 if augmentLevel == "light" else 0.12, scale=(0.02, 0.12)),
        ]
    )
    return trainTransform, evalTransform


def countSamplesByClass(dataset: datasets.ImageFolder) -> Dict[str, int]:
    countsByIndex: Dict[int, int] = {}
    for _, classIndex in dataset.samples:
        countsByIndex[classIndex] = countsByIndex.get(classIndex, 0) + 1
    return {
        className: countsByIndex.get(classIndex, 0)
        for className, classIndex in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])
    }


def buildSampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    countsByIndex: Dict[int, int] = {}
    for _, classIndex in dataset.samples:
        countsByIndex[classIndex] = countsByIndex.get(classIndex, 0) + 1
    sampleWeights = [1.0 / max(1, countsByIndex[classIndex]) for _, classIndex in dataset.samples]
    return WeightedRandomSampler(
        weights=torch.tensor(sampleWeights, dtype=torch.double),
        num_samples=len(sampleWeights),
        replacement=True,
    )


def buildLoaders(
    dataDir: str,
    imageSize: int,
    augmentLevel: str,
    batchSize: int,
    numWorkers: int,
    monitorSplit: str,
    officialSplit: str,
    balancedSampler: bool,
) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder], Dict[str, str]]:
    dataRoot = Path(dataDir)
    trainTransform, evalTransform = buildTransforms(imageSize=imageSize, augmentLevel=augmentLevel)

    trainDir, trainSplit = resolveSplitDir(dataRoot, ["train"])
    monitorDir, resolvedMonitorSplit = resolveSplitDir(dataRoot, [monitorSplit, "dev-val", "val"])
    officialDir, resolvedOfficialSplit = resolveSplitDir(
        dataRoot,
        [officialSplit, "official-val", resolvedMonitorSplit, "val"],
    )

    datasetsBySplit = {
        "train": datasets.ImageFolder(root=str(trainDir), transform=trainTransform),
        "monitor": datasets.ImageFolder(root=str(monitorDir), transform=evalTransform),
        "official": datasets.ImageFolder(root=str(officialDir), transform=evalTransform),
    }

    trainSampler = buildSampler(datasetsBySplit["train"]) if balancedSampler else None
    trainLoader = DataLoader(
        datasetsBySplit["train"],
        batch_size=batchSize,
        shuffle=trainSampler is None,
        sampler=trainSampler,
        num_workers=numWorkers,
        pin_memory=False,
    )

    monitorLoader = DataLoader(
        datasetsBySplit["monitor"],
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=False,
    )
    officialLoader = DataLoader(
        datasetsBySplit["official"],
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=False,
    )

    return {
        "train": trainLoader,
        "monitor": monitorLoader,
        "official": officialLoader,
    }, datasetsBySplit, {
        "train": trainSplit,
        "monitor": resolvedMonitorSplit,
        "official": resolvedOfficialSplit,
    }


def createCriterion(lossType: str, labelSmoothing: float, focalGamma: float) -> nn.Module:
    if lossType == "focal":
        return FocalLoss(gamma=focalGamma, labelSmoothing=labelSmoothing)
    return nn.CrossEntropyLoss(label_smoothing=labelSmoothing)


def runEpoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    highConfidenceThreshold: float = 0.9,
) -> Dict[str, float]:
    isTrain = optimizer is not None
    model.train(isTrain)

    totalLoss = 0.0
    correct = 0
    count = 0
    totalConfidence = 0.0
    wrongHighConfidenceCount = 0

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

        probabilities = torch.softmax(logits, dim=1)
        maxProbabilities, predicted = probabilities.max(dim=1)

        totalLoss += float(loss.item()) * labels.size(0)
        correct += int((predicted == labels).sum().item())
        count += labels.size(0)
        totalConfidence += float(maxProbabilities.sum().item())
        wrongHighConfidenceCount += int(((predicted != labels) & (maxProbabilities >= highConfidenceThreshold)).sum().item())

    return {
        "loss": totalLoss / max(1, count),
        "accuracy": correct / max(1, count),
        "maxProbMean": totalConfidence / max(1, count),
        "wrongHighConfidenceCount": wrongHighConfidenceCount,
    }


def evaluateDetailed(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    idxToClass: Dict[int, str],
    device: torch.device,
    eceBins: int = 10,
    highConfidenceThreshold: float = 0.9,
) -> Dict[str, object]:
    classNames = [idxToClass[idx] for idx in sorted(idxToClass)]
    confusion = {
        actualClass: {predictedClass: 0 for predictedClass in classNames}
        for actualClass in classNames
    }
    totals = {className: 0 for className in classNames}
    predictedCount = {className: 0 for className in classNames}
    correctByClass = {className: 0 for className in classNames}

    totalLoss = 0.0
    totalCount = 0
    totalCorrect = 0
    totalConfidence = 0.0
    wrongHighConfidenceCount = 0
    confidences: List[float] = []
    correctness: List[int] = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            maxProbabilities, predicted = probabilities.max(dim=1)

            totalLoss += float(loss.item()) * labels.size(0)
            totalCount += labels.size(0)
            totalConfidence += float(maxProbabilities.sum().item())
            wrongHighConfidenceCount += int(((predicted != labels) & (maxProbabilities >= highConfidenceThreshold)).sum().item())

            for actualIndex, predictedIndex, confidence in zip(
                labels.cpu().tolist(),
                predicted.cpu().tolist(),
                maxProbabilities.cpu().tolist(),
            ):
                actualClass = idxToClass[int(actualIndex)]
                predictedClass = idxToClass[int(predictedIndex)]
                confusion[actualClass][predictedClass] += 1
                totals[actualClass] += 1
                predictedCount[predictedClass] += 1
                isCorrect = 1 if actualIndex == predictedIndex else 0
                totalCorrect += isCorrect
                correctByClass[actualClass] += isCorrect
                confidences.append(float(confidence))
                correctness.append(isCorrect)

    perClass = {}
    macroF1 = 0.0
    for className in classNames:
        precision = correctByClass[className] / max(1, predictedCount[className])
        recall = correctByClass[className] / max(1, totals[className])
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        macroF1 += f1
        perClass[className] = {
            "support": totals[className],
            "accuracy": round(correctByClass[className] / max(1, totals[className]), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    macroF1 = macroF1 / max(1, len(classNames))

    ece = 0.0
    if confidences:
        for binIndex in range(eceBins):
            lower = binIndex / eceBins
            upper = (binIndex + 1) / eceBins
            members = [
                index
                for index, confidence in enumerate(confidences)
                if lower <= confidence < upper or (binIndex == eceBins - 1 and confidence == 1.0)
            ]
            if not members:
                continue
            binAccuracy = sum(correctness[index] for index in members) / len(members)
            binConfidence = sum(confidences[index] for index in members) / len(members)
            ece += (len(members) / len(confidences)) * abs(binAccuracy - binConfidence)

    return {
        "loss": totalLoss / max(1, totalCount),
        "accuracy": totalCorrect / max(1, totalCount),
        "macroF1": macroF1,
        "ece": ece,
        "maxProbMean": totalConfidence / max(1, totalCount),
        "wrongHighConfidenceCount": wrongHighConfidenceCount,
        "perClass": perClass,
        "confusionMatrix": confusion,
        "samples": totalCount,
    }


def supportsAnsiColor() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "dumb").lower() == "dumb":
        return False
    return sys.stdout.isatty()


def renderEpochProgressLine(
    epoch: int,
    totalEpochs: int,
    trainLoss: float,
    devValLoss: float,
    trainAcc: float,
    devValAcc: float,
    learningRate: float,
    noImproveEpochs: int,
) -> str:
    total = max(1, totalEpochs)
    ratio = min(1.0, max(0.0, epoch / total))
    barWidth = 26
    filled = int(round(ratio * barWidth))
    empty = barWidth - filled
    bar = "#" * filled + "-" * empty

    if supportsAnsiColor():
        green = "\033[92m"
        cyan = "\033[96m"
        yellow = "\033[93m"
        magenta = "\033[95m"
        reset = "\033[0m"
        return (
            f"{green}[{bar}]{reset} "
            f"{cyan}{epoch}/{total}{reset} "
            f"train-loss={trainLoss:.4f} dev-loss={devValLoss:.4f} "
            f"train-acc={trainAcc:.3f} dev-acc={devValAcc:.3f} "
            f"{yellow}lr={learningRate:.6f}{reset} "
            f"{magenta}no-improve={noImproveEpochs}{reset}"
        )

    return (
        f"[{bar}] {epoch}/{total} "
        f"train-loss={trainLoss:.4f} dev-loss={devValLoss:.4f} "
        f"train-acc={trainAcc:.3f} dev-acc={devValAcc:.3f} "
        f"lr={learningRate:.6f} no-improve={noImproveEpochs}"
    )


def saveCurve(history: List[Dict[str, float]], outputPath: Path, showDevVal: bool = True) -> None:
    if not history:
        return

    epochs = [int(item["epoch"]) for item in history]
    trainLoss = [float(item["trainLoss"]) for item in history]
    devValLoss = [float(item["devValLoss"]) for item in history]
    trainAcc = [float(item["trainAcc"]) for item in history]
    devValAcc = [float(item["devValAcc"]) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].plot(epochs, trainLoss, marker="o", label="train")
    if showDevVal:
        axes[0].plot(epochs, devValLoss, marker="s", label="dev-val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, trainAcc, marker="o", label="train")
    if showDevVal:
        axes[1].plot(epochs, devValAcc, marker="s", label="dev-val")
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
    bestDevValLoss: float,
    bestDevValAcc: float,
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
        "bestDevValLoss": round(bestDevValLoss, 6),
        "bestDevValAcc": round(bestDevValAcc, 6),
        "bestValLoss": round(bestDevValLoss, 6),
        "bestValAcc": round(bestDevValAcc, 6),
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
        "curveTrain": checkpointDir / "curve-train.png",
        "curveLive": checkpointDir / "curve-live.png",
        "curveLiveTrain": checkpointDir / "curve-live-train.png",
        "liveMetrics": checkpointDir / "live-metrics.jsonl",
        "progress": checkpointDir / "progress.json",
        "legacyHistory": outputPath.with_suffix(".history.json"),
        "legacySummary": outputPath.with_suffix(".summary.json"),
        "legacyCurve": outputPath.with_suffix(".curve.png"),
    }


def loadCheckpointModel(checkpointPath: Path, device: torch.device) -> Tuple[nn.Module, Dict[int, str]]:
    payload = torch.load(checkpointPath, map_location=device)
    classToIdx = payload["classToIdx"]
    idxToClass = {index: className for className, index in classToIdx.items()}
    model = buildModel(
        modelName=payload.get("modelName", "tiny-cnn"),
        classCount=len(classToIdx),
        pretrained=False,
    ).to(device)
    model.load_state_dict(payload["stateDict"])
    model.eval()
    return model, idxToClass


def describeLossGapReason(trainSamples: int, monitorSamples: int, officialSamples: int) -> str:
    if officialSamples > max(trainSamples * 4, monitorSamples * 4):
        return "official-val 更大且类别更不均衡，loss 通常会显著高于训练期 dev-val。"
    if officialSamples > monitorSamples:
        return "official-val 分布更接近真实场景，因此 loss 会高于训练监控集。"
    return "训练期与最终评估集规模接近，loss 差距主要来自样本难度与过置信。"


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

    loaders, datasetsBySplit, resolvedSplits = buildLoaders(
        dataDir=args.data_dir,
        imageSize=args.image_size,
        augmentLevel=args.augment_level,
        batchSize=args.batch_size,
        numWorkers=args.num_workers,
        monitorSplit=args.monitor_split,
        officialSplit=args.official_split,
        balancedSampler=args.balancedSampler,
    )

    trainDataset = datasetsBySplit["train"]
    monitorDataset = datasetsBySplit["monitor"]
    officialDataset = datasetsBySplit["official"]

    if len(trainDataset) == 0 or len(monitorDataset) == 0 or len(officialDataset) == 0:
        raise SystemExit(
            f"Dataset is empty: train={len(trainDataset)} monitor={len(monitorDataset)} official={len(officialDataset)}"
        )

    classToIdx = trainDataset.class_to_idx
    model = buildModel(
        modelName=args.model_name,
        classCount=len(classToIdx),
        pretrained=bool(args.pretrained),
    ).to(device)

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

    criterion = createCriterion(
        lossType=args.loss_type,
        labelSmoothing=args.label_smoothing,
        focalGamma=args.focal_gamma,
    )

    history: List[Dict[str, float]] = []
    bestDevValLoss = float("inf")
    bestDevValAcc = 0.0
    bestEpoch = 0
    noImproveEpochs = 0
    startTime = time.perf_counter()
    baseLearningRate = args.learning_rate

    for epoch in range(1, args.epochs + 1):
        epochStart = time.perf_counter()

        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            setBackboneFrozen(model=model, modelName=args.model_name, frozen=False)
            newParameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and not any(parameter is candidate for group in optimizer.param_groups for candidate in group["params"])
            ]
            if newParameters:
                optimizer.add_param_group(
                    {
                        "params": newParameters,
                        "lr": baseLearningRate * 0.5,
                        "weight_decay": args.weight_decay,
                    }
                )

        trainMetrics = runEpoch(model, loaders["train"], criterion, optimizer, device)
        devValMetrics = runEpoch(model, loaders["monitor"], criterion, None, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(devValMetrics["loss"])
            else:
                scheduler.step()

        epochDuration = time.perf_counter() - epochStart
        learningRateNow = min(group["lr"] for group in optimizer.param_groups)
        row = {
            "epoch": float(epoch),
            "trainLoss": round(trainMetrics["loss"], 6),
            "trainAcc": round(trainMetrics["accuracy"], 6),
            "trainMaxProbMean": round(trainMetrics["maxProbMean"], 6),
            "devValLoss": round(devValMetrics["loss"], 6),
            "devValAcc": round(devValMetrics["accuracy"], 6),
            "devValMaxProbMean": round(devValMetrics["maxProbMean"], 6),
            "devValWrongHighConfidenceCount": int(devValMetrics["wrongHighConfidenceCount"]),
            "valLoss": round(devValMetrics["loss"], 6),
            "valAcc": round(devValMetrics["accuracy"], 6),
            "learningRate": round(learningRateNow, 8),
            "epochTimeSec": round(epochDuration, 4),
        }
        history.append(row)
        print(
            renderEpochProgressLine(
                epoch=epoch,
                totalEpochs=args.epochs,
                trainLoss=row["trainLoss"],
                devValLoss=row["devValLoss"],
                trainAcc=row["trainAcc"],
                devValAcc=row["devValAcc"],
                learningRate=row["learningRate"],
                noImproveEpochs=noImproveEpochs,
            )
        )
        print(json.dumps(row, ensure_ascii=False))

        with paths["liveMetrics"].open("a", encoding="utf-8") as fileObj:
            fileObj.write(json.dumps(row, ensure_ascii=False) + "\n")

        improved = (bestDevValLoss - devValMetrics["loss"]) > args.early_stop_min_delta
        if improved:
            bestDevValLoss = devValMetrics["loss"]
            bestDevValAcc = devValMetrics["accuracy"]
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
                bestDevValLoss=bestDevValLoss,
                bestDevValAcc=bestDevValAcc,
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
            bestDevValLoss=bestDevValLoss if bestEpoch > 0 else devValMetrics["loss"],
            bestDevValAcc=bestDevValAcc if bestEpoch > 0 else devValMetrics["accuracy"],
        )
        saveCurve(history=history, outputPath=paths["curveLive"], showDevVal=True)
        saveCurve(history=history, outputPath=paths["curveLiveTrain"], showDevVal=False)

        elapsed = time.perf_counter() - startTime
        averageEpochSec = elapsed / epoch
        progress = {
            "currentEpoch": epoch,
            "totalEpochs": args.epochs,
            "bestEpoch": bestEpoch,
            "bestDevValLoss": round(bestDevValLoss, 6) if bestEpoch > 0 else None,
            "bestDevValAcc": round(bestDevValAcc, 6) if bestEpoch > 0 else None,
            "bestValLoss": round(bestDevValLoss, 6) if bestEpoch > 0 else None,
            "bestValAcc": round(bestDevValAcc, 6) if bestEpoch > 0 else None,
            "noImproveEpochs": noImproveEpochs,
            "etaSec": round(max(0.0, averageEpochSec * (args.epochs - epoch)), 2),
            "status": "running",
            "monitorSplit": resolvedSplits["monitor"],
        }
        saveJson(paths["progress"], progress)

        if noImproveEpochs >= args.early_stop_patience:
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "bestEpoch": bestEpoch,
                        "bestDevValLoss": round(bestDevValLoss, 6),
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
            bestDevValLoss=history[-1]["devValLoss"] if history else 0.0,
            bestDevValAcc=history[-1]["devValAcc"] if history else 0.0,
        )

    bestModel, idxToClass = loadCheckpointModel(paths["best"], device)
    officialMetrics = evaluateDetailed(
        model=bestModel,
        loader=loaders["official"],
        criterion=criterion,
        idxToClass=idxToClass,
        device=device,
    )

    saveCurve(history=history, outputPath=paths["curve"], showDevVal=True)
    saveCurve(history=history, outputPath=paths["curveTrain"], showDevVal=False)

    classDistribution = {
        "train": countSamplesByClass(trainDataset),
        "dev-val": countSamplesByClass(monitorDataset),
        "official-val": countSamplesByClass(officialDataset),
    }

    summary = {
        "device": str(device),
        "epochsRequested": args.epochs,
        "epochsExecuted": len(history),
        "batchSize": args.batch_size,
        "numWorkers": args.num_workers,
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "labelSmoothing": args.label_smoothing,
        "lossType": args.loss_type,
        "focalGamma": args.focal_gamma,
        "balancedSampler": bool(args.balancedSampler),
        "scheduler": args.scheduler,
        "augmentLevel": args.augment_level,
        "modelName": args.model_name,
        "pretrained": bool(args.pretrained),
        "freezeEpochs": args.freeze_epochs,
        "trainSamples": len(trainDataset),
        "devValSamples": len(monitorDataset),
        "officialValSamples": len(officialDataset),
        "valSamples": len(monitorDataset),
        "classToIdx": classToIdx,
        "classDistribution": classDistribution,
        "monitorSplit": resolvedSplits["monitor"],
        "officialSplit": resolvedSplits["official"],
        "totalTrainingSec": round(totalTrainingSec, 4),
        "bestEpoch": bestEpoch,
        "bestDevValLoss": round(bestDevValLoss, 6) if bestEpoch > 0 else None,
        "bestDevValAcc": round(bestDevValAcc, 6) if bestEpoch > 0 else None,
        "bestValLoss": round(bestDevValLoss, 6) if bestEpoch > 0 else None,
        "bestValAcc": round(bestDevValAcc, 6) if bestEpoch > 0 else None,
        "lastDevValLoss": history[-1]["devValLoss"] if history else None,
        "lastDevValAcc": history[-1]["devValAcc"] if history else None,
        "lastValLoss": history[-1]["devValLoss"] if history else None,
        "lastValAcc": history[-1]["devValAcc"] if history else None,
        "officialValLoss": round(float(officialMetrics["loss"]), 6),
        "officialValAcc": round(float(officialMetrics["accuracy"]), 6),
        "officialMacroF1": round(float(officialMetrics["macroF1"]), 6),
        "officialEce": round(float(officialMetrics["ece"]), 6),
        "maxProbMean": round(float(officialMetrics["maxProbMean"]), 6),
        "wrongHighConfidenceCount": int(officialMetrics["wrongHighConfidenceCount"]),
        "generalizationGapAtBest": (
            round(history[max(0, bestEpoch - 1)]["devValLoss"] - history[max(0, bestEpoch - 1)]["trainLoss"], 6)
            if bestEpoch > 0
            else None
        ),
        "lossGapReason": describeLossGapReason(
            trainSamples=len(trainDataset),
            monitorSamples=len(monitorDataset),
            officialSamples=len(officialDataset),
        ),
    }

    saveJson(paths["summary"], summary)
    paths["history"].write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    torch.save(torch.load(paths["last"], map_location="cpu"), paths["legacyOutput"])
    paths["legacyHistory"].write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    saveJson(paths["legacySummary"], summary)
    saveCurve(history=history, outputPath=paths["legacyCurve"], showDevVal=True)

    progressFinal = {
        "currentEpoch": len(history),
        "totalEpochs": args.epochs,
        "bestEpoch": bestEpoch,
        "bestDevValLoss": round(bestDevValLoss, 6) if bestEpoch > 0 else None,
        "bestDevValAcc": round(bestDevValAcc, 6) if bestEpoch > 0 else None,
        "bestValLoss": round(bestDevValLoss, 6) if bestEpoch > 0 else None,
        "bestValAcc": round(bestDevValAcc, 6) if bestEpoch > 0 else None,
        "officialValLoss": round(float(officialMetrics["loss"]), 6),
        "officialValAcc": round(float(officialMetrics["accuracy"]), 6),
        "noImproveEpochs": noImproveEpochs,
        "etaSec": 0.0,
        "status": "succeeded",
        "monitorSplit": resolvedSplits["monitor"],
        "officialSplit": resolvedSplits["official"],
    }
    saveJson(paths["progress"], progressFinal)

    print(f"Checkpoint saved: {paths['legacyOutput']}")
    print(f"Best checkpoint: {paths['best']}")
    print(f"Last checkpoint: {paths['last']}")
    print(f"Summary saved: {paths['summary']}")
    print(f"Curve saved: {paths['curve']}")


if __name__ == "__main__":
    main()
