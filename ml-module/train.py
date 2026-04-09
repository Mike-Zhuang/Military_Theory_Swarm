from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import TinyConvNet


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny-CNN on synthetic target dataset")
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
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    trainDataset = datasets.ImageFolder(root=str(Path(dataDir) / "train"), transform=transform)
    valDataset = datasets.ImageFolder(root=str(Path(dataDir) / "val"), transform=transform)

    # 默认使用单进程加载，避免在受限环境或 VS Code 调试场景下触发共享内存权限问题。
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
    return trainLoader, valLoader, trainDataset.class_to_idx


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


def main() -> None:
    args = parseArgs()
    device = chooseDevice()
    trainLoader, valLoader, classToIdx = buildLoaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )

    model = TinyConvNet(classCount=len(classToIdx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        trainLoss, trainAcc = runEpoch(model, trainLoader, criterion, optimizer, device)
        valLoss, valAcc = runEpoch(model, valLoader, criterion, None, device)

        row = {
            "epoch": epoch,
            "trainLoss": round(trainLoss, 6),
            "trainAcc": round(trainAcc, 6),
            "valLoss": round(valLoss, 6),
            "valAcc": round(valAcc, 6),
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
    print(f"Checkpoint saved: {outputPath}")
    print(f"History saved: {historyPath}")


if __name__ == "__main__":
    main()
