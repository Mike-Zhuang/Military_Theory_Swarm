from __future__ import annotations

import argparse
import json
import random
import zipfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from PIL import Image

VEHICLE_CATEGORIES = {4, 5, 6, 9, 10}
CIVILIAN_CATEGORIES = {1, 2, 3, 7, 8}
DECOY_CATEGORIES = {0}


@dataclass
class CropCandidate:
    imagePath: Path
    bbox: Tuple[int, int, int, int]
    className: str


@dataclass
class ParsedBox:
    left: int
    top: int
    width: int
    height: int
    category: int


@dataclass
class SourceStats:
    imageCount: int = 0
    annotationCount: int = 0
    missingAnnotationCount: int = 0
    totalBoxCount: int = 0
    filteredSmallCount: int = 0
    filteredInvalidCount: int = 0
    filteredUnknownCategoryCount: int = 0
    backgroundCropCount: int = 0
    ignoredUsedCount: int = 0
    ignoredSkippedCount: int = 0
    backgroundNegativeCount: int = 0
    classBoxCount: Dict[str, int] | None = None
    categoryHistogram: Dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.classBoxCount is None:
            self.classBoxCount = {
                "vehicle": 0,
                "civilian-object": 0,
                "decoy": 0,
            }
        if self.categoryHistogram is None:
            self.categoryHistogram = {}

    def toDict(self) -> Dict[str, Any]:
        return {
            "imageCount": self.imageCount,
            "annotationCount": self.annotationCount,
            "missingAnnotationCount": self.missingAnnotationCount,
            "totalBoxCount": self.totalBoxCount,
            "filteredSmallCount": self.filteredSmallCount,
            "filteredInvalidCount": self.filteredInvalidCount,
            "filteredUnknownCategoryCount": self.filteredUnknownCategoryCount,
            "backgroundCropCount": self.backgroundCropCount,
            "ignoredUsedCount": self.ignoredUsedCount,
            "ignoredSkippedCount": self.ignoredSkippedCount,
            "backgroundNegativeCount": self.backgroundNegativeCount,
            "classBoxCount": self.classBoxCount,
            "categoryHistogram": self.categoryHistogram,
        }


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VisDrone subset for classification training")
    parser.add_argument("--raw-dir", type=str, default="data/visdrone/raw")
    parser.add_argument("--output-dir", type=str, default="data/visdrone-ready")
    parser.add_argument("--split-mode", type=str, default="official-val", choices=["official-val", "auto-split"])
    parser.add_argument("--train-images-dir", type=str, default="")
    parser.add_argument("--train-annotations-dir", type=str, default="")
    parser.add_argument("--val-images-dir", type=str, default="")
    parser.add_argument("--val-annotations-dir", type=str, default="")
    parser.add_argument("--source-images-dir", type=str, default="")
    parser.add_argument("--source-annotations-dir", type=str, default="")
    parser.add_argument("--archive-path", type=str, default="")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-url",
        type=str,
        default="https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip",
    )
    parser.add_argument("--subset-size-per-class", type=int, default=900)
    parser.add_argument("--dev-val-size-per-class", type=int, default=360)
    parser.add_argument("--val-subset-size-per-class", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-box-size", type=int, default=16)
    parser.add_argument("--background-crops-per-image", type=int, default=2)
    parser.add_argument("--use-ignored-as-decoy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensureDir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def downloadWithResume(url: str, archivePath: Path) -> None:
    ensureDir(archivePath.parent)
    start = archivePath.stat().st_size if archivePath.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start > 0 else {}
    mode = "ab" if start > 0 else "wb"

    with requests.get(url, stream=True, timeout=60, headers=headers) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"Download failed: {response.status_code} {response.text[:200]}")
        with archivePath.open(mode) as fileObj:
            for chunk in response.iter_content(chunk_size=1024 * 512):
                if chunk:
                    fileObj.write(chunk)


def extractArchive(archivePath: Path, rawDir: Path) -> None:
    ensureDir(rawDir)
    with zipfile.ZipFile(archivePath, "r") as zipFile:
        zipFile.extractall(rawDir)


def guessTrainSourceDirs(rawDir: Path) -> Tuple[Path, Path]:
    candidates = [
        (
            rawDir / "VisDrone2019-DET-train" / "images",
            rawDir / "VisDrone2019-DET-train" / "annotations",
        ),
        (rawDir / "images", rawDir / "annotations"),
    ]
    for imagesDir, annDir in candidates:
        if imagesDir.exists() and annDir.exists():
            return imagesDir, annDir
    raise RuntimeError(
        "Unable to locate VisDrone train images/annotations directory. "
        "Use --train-images-dir and --train-annotations-dir explicitly."
    )


def guessValSourceDirs(rawDir: Path) -> Tuple[Path, Path]:
    candidates = [
        (
            rawDir / "VisDrone2019-DET-val" / "images",
            rawDir / "VisDrone2019-DET-val" / "annotations",
        ),
        (
            rawDir / "val" / "images",
            rawDir / "val" / "annotations",
        ),
    ]
    for imagesDir, annDir in candidates:
        if imagesDir.exists() and annDir.exists():
            return imagesDir, annDir
    raise RuntimeError(
        "Unable to locate VisDrone val images/annotations directory. "
        "Use --val-images-dir and --val-annotations-dir explicitly."
    )


def parseAnnotationFile(path: Path) -> List[ParsedBox]:
    boxes: List[ParsedBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        left, top, width, height = [int(float(parts[idx])) for idx in range(4)]
        category = int(float(parts[5]))
        boxes.append(ParsedBox(left=left, top=top, width=width, height=height, category=category))
    return boxes


def intersects(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> bool:
    leftA, topA, rightA, bottomA = boxA
    leftB, topB, rightB, bottomB = boxB
    return not (rightA <= leftB or rightB <= leftA or bottomA <= topB or bottomB <= topA)


def randomBackgroundCandidates(
    imagePath: Path,
    width: int,
    height: int,
    occupied: List[Tuple[int, int, int, int]],
    samples: int,
    rng: random.Random,
) -> List[CropCandidate]:
    candidates: List[CropCandidate] = []
    tries = 0
    while len(candidates) < samples and tries < samples * 16:
        tries += 1
        cropW = rng.randint(26, max(28, min(96, width // 3)))
        cropH = rng.randint(26, max(28, min(96, height // 3)))
        if width <= cropW + 1 or height <= cropH + 1:
            break
        left = rng.randint(0, width - cropW - 1)
        top = rng.randint(0, height - cropH - 1)
        bbox = (left, top, left + cropW, top + cropH)
        if any(intersects(bbox, occ) for occ in occupied):
            continue
        candidates.append(CropCandidate(imagePath=imagePath, bbox=bbox, className="decoy"))
    return candidates


def collectCandidates(
    imagesDir: Path,
    annDir: Path,
    minBoxSize: int,
    backgroundPerImage: int,
    useIgnoredAsDecoy: bool,
    rng: random.Random,
) -> Tuple[Dict[str, List[CropCandidate]], SourceStats]:
    grouped: Dict[str, List[CropCandidate]] = {
        "vehicle": [],
        "civilian-object": [],
        "decoy": [],
    }
    stats = SourceStats()

    for imagePath in sorted(imagesDir.glob("*.jpg")):
        stats.imageCount += 1
        annPath = annDir / f"{imagePath.stem}.txt"
        if not annPath.exists():
            stats.missingAnnotationCount += 1
            continue
        stats.annotationCount += 1

        with Image.open(imagePath) as imageObj:
            imageWidth, imageHeight = imageObj.size

        boxes = parseAnnotationFile(annPath)
        stats.totalBoxCount += len(boxes)
        occupied: List[Tuple[int, int, int, int]] = []

        for box in boxes:
            categoryKey = str(box.category)
            stats.categoryHistogram[categoryKey] = stats.categoryHistogram.get(categoryKey, 0) + 1
            if box.width < minBoxSize or box.height < minBoxSize:
                stats.filteredSmallCount += 1
                continue
            left = max(0, box.left)
            top = max(0, box.top)
            right = min(imageWidth, left + box.width)
            bottom = min(imageHeight, top + box.height)
            if right - left < minBoxSize or bottom - top < minBoxSize:
                stats.filteredInvalidCount += 1
                continue

            bbox = (left, top, right, bottom)
            occupied.append(bbox)

            if box.category in VEHICLE_CATEGORIES:
                grouped["vehicle"].append(CropCandidate(imagePath=imagePath, bbox=bbox, className="vehicle"))
                stats.classBoxCount["vehicle"] += 1
            elif box.category in CIVILIAN_CATEGORIES:
                grouped["civilian-object"].append(
                    CropCandidate(imagePath=imagePath, bbox=bbox, className="civilian-object")
                )
                stats.classBoxCount["civilian-object"] += 1
            elif box.category in DECOY_CATEGORIES:
                if useIgnoredAsDecoy:
                    grouped["decoy"].append(CropCandidate(imagePath=imagePath, bbox=bbox, className="decoy"))
                    stats.classBoxCount["decoy"] += 1
                    stats.ignoredUsedCount += 1
                else:
                    stats.ignoredSkippedCount += 1
            else:
                stats.filteredUnknownCategoryCount += 1

        backgroundCandidates = randomBackgroundCandidates(
            imagePath=imagePath,
            width=imageWidth,
            height=imageHeight,
            occupied=occupied,
            samples=backgroundPerImage,
            rng=rng,
        )
        grouped["decoy"].extend(backgroundCandidates)
        stats.backgroundCropCount += len(backgroundCandidates)
        stats.backgroundNegativeCount += len(backgroundCandidates)
        stats.classBoxCount["decoy"] += len(backgroundCandidates)

    return grouped, stats


def chooseSubset(
    grouped: Dict[str, List[CropCandidate]],
    subsetSizePerClass: int,
    rng: random.Random,
) -> Dict[str, List[CropCandidate]]:
    selected: Dict[str, List[CropCandidate]] = {}
    for className, items in grouped.items():
        copied = list(items)
        rng.shuffle(copied)
        selected[className] = copied[: min(len(copied), subsetSizePerClass)]
    return selected


def splitTrainVal(
    selected: Dict[str, List[CropCandidate]],
    valRatio: float,
) -> Dict[str, Dict[str, List[CropCandidate]]]:
    output: Dict[str, Dict[str, List[CropCandidate]]] = {
        "train": {"vehicle": [], "civilian-object": [], "decoy": []},
        "dev-val": {"vehicle": [], "civilian-object": [], "decoy": []},
        "official-val": {"vehicle": [], "civilian-object": [], "decoy": []},
    }
    for className, items in selected.items():
        splitIdx = int(len(items) * (1.0 - valRatio))
        output["train"][className] = items[:splitIdx]
        output["dev-val"][className] = items[splitIdx:]
        output["official-val"][className] = list(output["dev-val"][className])
    return output


def selectedCountByClass(items: Dict[str, List[CropCandidate]]) -> Dict[str, int]:
    return {className: len(classItems) for className, classItems in items.items()}


def sourcePayload(name: str, imagesDir: Path, annDir: Path, stats: SourceStats) -> Dict[str, Any]:
    return {
        "name": name,
        "imagesDir": str(imagesDir),
        "annotationsDir": str(annDir),
        "stats": stats.toDict(),
    }


def saveCrops(outputDir: Path, splitItems: Dict[str, Dict[str, List[CropCandidate]]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}

    for splitName, byClass in splitItems.items():
        counts[splitName] = {"vehicle": 0, "civilian-object": 0, "decoy": 0}
        for className, items in byClass.items():
            saveDir = ensureDir(outputDir / splitName / className)
            for idx, candidate in enumerate(items):
                with Image.open(candidate.imagePath) as imageObj:
                    cropped = imageObj.crop(candidate.bbox)
                    savePath = saveDir / f"{candidate.imagePath.stem}_{idx:05d}.png"
                    cropped.save(savePath)
                counts[splitName][className] += 1
    return counts


def clearOutput(outputDir: Path) -> None:
    if not outputDir.exists():
        return
    for path in sorted(outputDir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
    for path in sorted(outputDir.rglob("*"), reverse=True):
        if path.is_dir() and path != outputDir:
            path.rmdir()


def main() -> None:
    args = parseArgs()
    rng = random.Random(args.seed)

    rawDir = Path(args.raw_dir)
    outputDir = Path(args.output_dir)

    if args.download:
        archivePath = Path(args.archive_path) if args.archive_path else rawDir / "VisDrone2019-DET-train.zip"
        print(f"Downloading VisDrone archive -> {archivePath}")
        downloadWithResume(args.download_url, archivePath)
        print("Extracting archive...")
        extractArchive(archivePath, rawDir)

    splitMode = args.split_mode
    sources: Dict[str, Dict[str, Any]] = {}
    splitItems: Dict[str, Dict[str, List[CropCandidate]]]
    selectedCounts: Dict[str, Dict[str, int]]
    candidateCounts: Dict[str, Dict[str, int]]
    valGrouped: Dict[str, List[CropCandidate]] | None = None

    if args.source_images_dir and args.source_annotations_dir:
        # 兼容旧参数：自动切分模式下旧参数等价于 train 输入目录。
        trainImagesDir = Path(args.source_images_dir)
        trainAnnDir = Path(args.source_annotations_dir)
    elif args.train_images_dir and args.train_annotations_dir:
        trainImagesDir = Path(args.train_images_dir)
        trainAnnDir = Path(args.train_annotations_dir)
    else:
        trainImagesDir, trainAnnDir = guessTrainSourceDirs(rawDir)

    if not trainImagesDir.exists() or not trainAnnDir.exists():
        raise RuntimeError(f"Train source dirs not found: images={trainImagesDir} annotations={trainAnnDir}")

    print(f"Train source images: {trainImagesDir}")
    print(f"Train source annotations: {trainAnnDir}")

    trainGrouped, trainStats = collectCandidates(
        imagesDir=trainImagesDir,
        annDir=trainAnnDir,
        minBoxSize=args.min_box_size,
        backgroundPerImage=args.background_crops_per_image,
        useIgnoredAsDecoy=args.use_ignored_as_decoy,
        rng=rng,
    )
    sources["train"] = sourcePayload("train", trainImagesDir, trainAnnDir, trainStats)

    if splitMode == "official-val":
        if args.val_images_dir and args.val_annotations_dir:
            valImagesDir = Path(args.val_images_dir)
            valAnnDir = Path(args.val_annotations_dir)
        else:
            valImagesDir, valAnnDir = guessValSourceDirs(rawDir)

        if not valImagesDir.exists() or not valAnnDir.exists():
            raise RuntimeError(f"Val source dirs not found: images={valImagesDir} annotations={valAnnDir}")

        print(f"Val source images: {valImagesDir}")
        print(f"Val source annotations: {valAnnDir}")

        valGrouped, valStats = collectCandidates(
            imagesDir=valImagesDir,
            annDir=valAnnDir,
            minBoxSize=args.min_box_size,
            backgroundPerImage=args.background_crops_per_image,
            useIgnoredAsDecoy=args.use_ignored_as_decoy,
            rng=random.Random(args.seed + 97),
        )
        sources["val"] = sourcePayload("val", valImagesDir, valAnnDir, valStats)

        trainSelected = chooseSubset(
            grouped=trainGrouped,
            subsetSizePerClass=args.subset_size_per_class,
            rng=random.Random(args.seed + 17),
        )
        devValSizePerClass = args.dev_val_size_per_class
        if args.val_subset_size_per_class > 0:
            devValSizePerClass = args.val_subset_size_per_class

        if devValSizePerClass > 0:
            devValSelected = chooseSubset(
                grouped=valGrouped,
                subsetSizePerClass=devValSizePerClass,
                rng=random.Random(args.seed + 41),
            )
        else:
            devValSelected = {className: list(items) for className, items in valGrouped.items()}

        splitItems = {
            "train": trainSelected,
            "dev-val": devValSelected,
            "official-val": {className: list(items) for className, items in valGrouped.items()},
        }
        candidateCounts = {
            "train": selectedCountByClass(trainGrouped),
            "dev-val": selectedCountByClass(valGrouped),
            "official-val": selectedCountByClass(valGrouped),
        }
        selectedCounts = {
            "train": selectedCountByClass(trainSelected),
            "dev-val": selectedCountByClass(devValSelected),
            "official-val": selectedCountByClass(valGrouped),
        }
    else:
        trainSelected = chooseSubset(
            grouped=trainGrouped,
            subsetSizePerClass=args.subset_size_per_class,
            rng=random.Random(args.seed + 17),
        )
        splitItems = splitTrainVal(trainSelected, valRatio=args.val_ratio)
        candidateCounts = {
            "train": selectedCountByClass(trainGrouped),
            "dev-val": selectedCountByClass(splitItems["dev-val"]),
            "official-val": selectedCountByClass(splitItems["official-val"]),
        }
        selectedCounts = {
            "train": selectedCountByClass(splitItems["train"]),
            "dev-val": selectedCountByClass(splitItems["dev-val"]),
            "official-val": selectedCountByClass(splitItems["official-val"]),
        }

    ensureDir(outputDir)
    clearOutput(outputDir)
    counts = saveCrops(outputDir, splitItems)

    payload = {
        "dataset": "VisDrone-Subset",
        "splitMode": splitMode,
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "sources": sources,
        "subsetSizePerClass": args.subset_size_per_class,
        "devValSizePerClass": args.dev_val_size_per_class,
        "valSubsetSizePerClass": args.val_subset_size_per_class,
        "valRatio": args.val_ratio,
        "seed": args.seed,
        "candidateCounts": candidateCounts,
        "selectedCounts": selectedCounts,
        "outputCounts": counts,
        "monitorSplit": "dev-val",
        "devValCounts": counts.get("dev-val", {}),
        "officialValCounts": counts.get("official-val", {}),
        "classes": ["vehicle", "civilian-object", "decoy"],
        "mappingRules": {
            "vehicle": sorted(VEHICLE_CATEGORIES),
            "civilian-object": sorted(CIVILIAN_CATEGORIES),
            "decoy": sorted(DECOY_CATEGORIES),
            "decoyExtra": "background hard negatives",
        },
        "filters": {
            "minBoxSize": args.min_box_size,
            "backgroundCropsPerImage": args.background_crops_per_image,
            "ignoredUnknownCategory": True,
            "useIgnoredAsDecoy": bool(args.use_ignored_as_decoy),
        },
        "labelQuality": {
            "ignoredUsedCount": sum(source["stats"]["ignoredUsedCount"] for source in sources.values()),
            "ignoredSkippedCount": sum(source["stats"]["ignoredSkippedCount"] for source in sources.values()),
            "backgroundNegativeCount": sum(source["stats"]["backgroundNegativeCount"] for source in sources.values()),
        },
    }
    (outputDir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Prepared dataset saved in: {outputDir}")


if __name__ == "__main__":
    main()
