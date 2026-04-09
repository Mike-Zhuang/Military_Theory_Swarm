from __future__ import annotations

import argparse
import csv
import json
import os
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from infer import chooseDevice, loadModel


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tiny-CNN and export confusion matrix artifacts")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="reports/eval")
    return parser.parse_args()


def buildLoader(
    dataDir: Path,
    split: str,
    batchSize: int,
    numWorkers: int,
) -> Tuple[DataLoader, Dict[int, str]]:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.ImageFolder(root=str(dataDir / split), transform=transform)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
    idxToClass = {idx: className for className, idx in dataset.class_to_idx.items()}
    return loader, idxToClass


def emptyMatrix(classNames: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        actualClass: {predictedClass: 0 for predictedClass in classNames}
        for actualClass in classNames
    }


def evaluate(
    checkpointPath: str,
    dataDir: Path,
    split: str,
    batchSize: int,
    numWorkers: int,
    outputDir: Path,
) -> Dict[str, object]:
    device = chooseDevice()
    model, idxToClass = loadModel(checkpointPath, device)
    loader, datasetIdxToClass = buildLoader(dataDir, split, batchSize, numWorkers)

    if idxToClass != datasetIdxToClass:
        raise SystemExit("Checkpoint class mapping does not match dataset class mapping.")

    classNames = [idxToClass[idx] for idx in sorted(idxToClass)]
    confusion = emptyMatrix(classNames)
    totals = {className: 0 for className in classNames}
    correctByClass = {className: 0 for className in classNames}

    totalCount = 0
    totalCorrect = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predicted = logits.argmax(dim=1)

            for actualIdx, predictedIdx in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                actualClass = idxToClass[int(actualIdx)]
                predictedClass = idxToClass[int(predictedIdx)]
                confusion[actualClass][predictedClass] += 1
                totals[actualClass] += 1
                totalCount += 1
                if actualIdx == predictedIdx:
                    correctByClass[actualClass] += 1
                    totalCorrect += 1

    perClass = {}
    for className in classNames:
        support = totals[className]
        correct = correctByClass[className]
        accuracy = correct / max(1, support)
        perClass[className] = {
            "support": support,
            "accuracy": round(accuracy, 4),
        }

    summary = {
        "split": split,
        "samples": totalCount,
        "overallAccuracy": round(totalCorrect / max(1, totalCount), 4),
        "perClass": perClass,
        "confusionMatrix": confusion,
    }

    outputDir.mkdir(parents=True, exist_ok=True)
    (outputDir / "evaluation-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (outputDir / "confusion-matrix.csv").open("w", newline="", encoding="utf-8") as fileObj:
        writer = csv.writer(fileObj)
        writer.writerow(["actual/predicted", *classNames])
        for actualClass in classNames:
            writer.writerow([actualClass, *[confusion[actualClass][predictedClass] for predictedClass in classNames]])

    plotConfusionMatrix(confusion, classNames, outputDir / "confusion-matrix.png")
    return summary


def plotConfusionMatrix(
    confusion: Dict[str, Dict[str, int]],
    classNames: List[str],
    outputPath: Path,
) -> None:
    matrix = [
        [confusion[actualClass][predictedClass] for predictedClass in classNames]
        for actualClass in classNames
    ]

    plt.figure(figsize=(6.4, 5.2))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(classNames)), classNames, rotation=20)
    plt.yticks(range(len(classNames)), classNames)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for rowIndex, row in enumerate(matrix):
        for colIndex, value in enumerate(row):
            color = "white" if value > max(1, max(map(max, matrix))) * 0.45 else "#173042"
            plt.text(colIndex, rowIndex, str(value), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(outputPath, dpi=180)
    plt.close()


def main() -> None:
    args = parseArgs()
    summary = evaluate(
        checkpointPath=args.checkpoint,
        dataDir=Path(args.data_dir),
        split=args.split,
        batchSize=args.batch_size,
        numWorkers=args.num_workers,
        outputDir=Path(args.output_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
