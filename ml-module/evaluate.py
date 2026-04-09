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
    parser = argparse.ArgumentParser(description="Evaluate model and export confusion matrix artifacts")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--split", type=str, default="official-val", choices=["train", "dev-val", "official-val", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="reports/eval")
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args()


def buildLoader(
    dataDir: Path,
    split: str,
    batchSize: int,
    numWorkers: int,
    imageSize: int,
    mean: List[float],
    std: List[float],
) -> Tuple[DataLoader, Dict[int, str]]:
    splitDir = dataDir / split
    if not splitDir.exists() and split == "val":
        if (dataDir / "official-val").exists():
            splitDir = dataDir / "official-val"
            split = "official-val"
        elif (dataDir / "dev-val").exists():
            splitDir = dataDir / "dev-val"
            split = "dev-val"
    transform = transforms.Compose(
        [
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = datasets.ImageFolder(root=str(splitDir), transform=transform)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
    idxToClass = {idx: className for className, idx in dataset.class_to_idx.items()}
    return loader, idxToClass


def emptyMatrix(classNames: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        actualClass: {predictedClass: 0 for predictedClass in classNames}
        for actualClass in classNames
    }


def safeDivide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def computeExpectedCalibrationError(
    confidences: List[float],
    correctness: List[int],
    bins: int,
) -> float:
    if not confidences:
        return 0.0

    total = len(confidences)
    ece = 0.0
    for binIndex in range(bins):
        lower = binIndex / bins
        upper = (binIndex + 1) / bins
        members = [index for index, confidence in enumerate(confidences) if lower <= confidence < upper or (binIndex == bins - 1 and confidence == 1.0)]
        if not members:
            continue
        binAcc = sum(correctness[index] for index in members) / len(members)
        binConf = sum(confidences[index] for index in members) / len(members)
        ece += (len(members) / total) * abs(binAcc - binConf)
    return ece


def evaluate(
    checkpointPath: str,
    dataDir: Path,
    split: str,
    batchSize: int,
    numWorkers: int,
    outputDir: Path,
    eceBins: int,
) -> Dict[str, object]:
    device = chooseDevice()
    model, idxToClass, checkpoint = loadModel(checkpointPath, device)
    imageSize = int(checkpoint.get("imageSize", 64))
    normalization = checkpoint.get("normalization", {})
    mean = normalization.get("mean", [0.5, 0.5, 0.5])
    std = normalization.get("std", [0.5, 0.5, 0.5])

    loader, datasetIdxToClass = buildLoader(
        dataDir=dataDir,
        split=split,
        batchSize=batchSize,
        numWorkers=numWorkers,
        imageSize=imageSize,
        mean=mean,
        std=std,
    )

    if idxToClass != datasetIdxToClass:
        raise SystemExit("Checkpoint class mapping does not match dataset class mapping.")

    classNames = [idxToClass[idx] for idx in sorted(idxToClass)]
    confusion = emptyMatrix(classNames)
    totals = {className: 0 for className in classNames}
    correctByClass = {className: 0 for className in classNames}
    predictedCount = {className: 0 for className in classNames}

    confidences: List[float] = []
    correctness: List[int] = []
    totalCount = 0
    totalCorrect = 0
    totalConfidence = 0.0
    wrongHighConfidenceCount = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confidenceTensor, predictedTensor = probs.max(dim=1)
            totalConfidence += float(confidenceTensor.sum().item())
            wrongHighConfidenceCount += int(((predictedTensor != labels) & (confidenceTensor >= 0.9)).sum().item())

            for actualIdx, predictedIdx, confidence in zip(
                labels.cpu().tolist(),
                predictedTensor.cpu().tolist(),
                confidenceTensor.cpu().tolist(),
            ):
                actualClass = idxToClass[int(actualIdx)]
                predictedClass = idxToClass[int(predictedIdx)]
                confusion[actualClass][predictedClass] += 1
                totals[actualClass] += 1
                predictedCount[predictedClass] += 1
                totalCount += 1
                isCorrect = 1 if actualIdx == predictedIdx else 0
                if isCorrect:
                    correctByClass[actualClass] += 1
                    totalCorrect += 1
                confidences.append(float(confidence))
                correctness.append(isCorrect)

    perClass = {}
    macroF1 = 0.0
    for className in classNames:
        support = totals[className]
        correct = correctByClass[className]
        precision = safeDivide(correct, predictedCount[className])
        recall = safeDivide(correct, support)
        f1 = safeDivide(2 * precision * recall, precision + recall)
        macroF1 += f1
        perClass[className] = {
            "support": support,
            "accuracy": round(safeDivide(correct, support), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    macroF1 = safeDivide(macroF1, len(classNames))
    ece = computeExpectedCalibrationError(confidences=confidences, correctness=correctness, bins=eceBins)

    summaryPath = Path(checkpointPath).with_name("summary.json")
    legacySummaryPath = Path(checkpointPath).with_name("tiny-cnn.summary.json")
    trainSummary = {}
    if summaryPath.exists():
        trainSummary = json.loads(summaryPath.read_text(encoding="utf-8"))
    elif legacySummaryPath.exists():
        trainSummary = json.loads(legacySummaryPath.read_text(encoding="utf-8"))

    summary = {
        "split": split,
        "samples": totalCount,
        "overallAccuracy": round(totalCorrect / max(1, totalCount), 4),
        "macroF1": round(macroF1, 4),
        "ece": round(ece, 4),
        "maxProbMean": round(totalConfidence / max(1, totalCount), 4),
        "wrongHighConfidenceCount": wrongHighConfidenceCount,
        "perClass": perClass,
        "confusionMatrix": confusion,
        "modelName": checkpoint.get("modelName", "tiny-cnn"),
        "bestEpoch": checkpoint.get("bestEpoch", trainSummary.get("bestEpoch")),
        "bestValLoss": checkpoint.get("bestValLoss", trainSummary.get("bestValLoss")),
        "bestValAcc": checkpoint.get("bestValAcc", trainSummary.get("bestValAcc")),
        "lastValLoss": trainSummary.get("lastValLoss"),
        "lastValAcc": trainSummary.get("lastValAcc"),
        "generalizationGapAtBest": trainSummary.get("generalizationGapAtBest"),
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
    maxValue = max(1, max(max(row) for row in matrix))

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
            color = "white" if value > maxValue * 0.45 else "#173042"
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
        eceBins=args.ece_bins,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
