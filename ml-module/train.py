from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

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

from model import TinyConvNet


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny-CNN on target classification dataset")
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="checkpoints/tiny-cnn.pt")
    return parser.parse_args()


def chooseDevice() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def buildLoaders(
    dataDir: str,
    batchSize: int,
    numWorkers: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], int, int]:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    trainDataset = datasets.ImageFolder(root=str(Path(dataDir) / "train"), transform=transform)
    valDataset = datasets.ImageFolder(root=str(Path(dataDir) / "val"), transform=transform)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
    return trainLoader, valLoader, trainDataset.class_to_idx, len(trainDataset), len(valDataset)


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

        totalLoss += loss.item() * labels.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        count += labels.size(0)

    avgLoss = totalLoss / max(1, count)
    accuracy = correct / max(1, count)
    return avgLoss, accuracy


def saveCurve(history: list[Dict[str, float]], outputPath: Path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    trainLoss = [float(item["trainLoss"]) for item in history]
    valLoss = [float(item["valLoss"]) for item in history]
    trainAcc = [float(item["trainAcc"]) for item in history]
    valAcc = [float(item["valAcc"]) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
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
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputPath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parseArgs()
    device = chooseDevice()

    trainLoader, valLoader, classToIdx, trainCount, valCount = buildLoaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )

    model = TinyConvNet(classCount=len(classToIdx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[Dict[str, float]] = []
    startTime = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epochStart = time.perf_counter()
        trainLoss, trainAcc = runEpoch(model, trainLoader, criterion, optimizer, device)
        valLoss, valAcc = runEpoch(model, valLoader, criterion, None, device)
        epochDuration = time.perf_counter() - epochStart

        row = {
            "epoch": float(epoch),
            "trainLoss": round(trainLoss, 6),
            "trainAcc": round(trainAcc, 6),
            "valLoss": round(valLoss, 6),
            "valAcc": round(valAcc, 6),
            "epochTimeSec": round(epochDuration, 4),
        }
        history.append(row)
        print(row)

    outputPath = Path(args.output)
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "stateDict": model.state_dict(),
        "classToIdx": classToIdx,
        "history": history,
        "device": str(device),
    }
    torch.save(checkpoint, outputPath)

    historyPath = outputPath.with_suffix(".history.json")
    historyPath.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "batchSize": args.batch_size,
        "numWorkers": args.num_workers,
        "learningRate": args.learning_rate,
        "trainSamples": trainCount,
        "valSamples": valCount,
        "classToIdx": classToIdx,
        "totalTrainingSec": round(time.perf_counter() - startTime, 4),
    }
    summaryPath = outputPath.with_suffix(".summary.json")
    summaryPath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    curvePath = outputPath.with_suffix(".curve.png")
    saveCurve(history, curvePath)

    print(f"Checkpoint saved: {outputPath}")
    print(f"History saved: {historyPath}")
    print(f"Summary saved: {summaryPath}")
    print(f"Curve saved: {curvePath}")


if __name__ == "__main__":
    main()
