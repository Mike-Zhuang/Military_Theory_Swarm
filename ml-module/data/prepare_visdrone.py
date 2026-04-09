from __future__ import annotations

import argparse
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from PIL import Image

VEHICLE_CATEGORIES = {4, 5, 6, 9, 10}
CIVILIAN_CATEGORIES = {1, 2, 3, 7, 8}


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


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VisDrone subset for classification training")
    parser.add_argument("--raw-dir", type=str, default="data/visdrone/raw")
    parser.add_argument("--output-dir", type=str, default="data/visdrone-ready")
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
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-box-size", type=int, default=16)
    parser.add_argument("--background-crops-per-image", type=int, default=2)
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


def guessSourceDirs(rawDir: Path) -> Tuple[Path, Path]:
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
        "Unable to locate source images/annotations directory. "
        "Use --source-images-dir and --source-annotations-dir explicitly."
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
    rng: random.Random,
) -> Dict[str, List[CropCandidate]]:
    grouped: Dict[str, List[CropCandidate]] = {
        "vehicle": [],
        "civilian-object": [],
        "decoy": [],
    }

    for imagePath in sorted(imagesDir.glob("*.jpg")):
        annPath = annDir / f"{imagePath.stem}.txt"
        if not annPath.exists():
            continue

        with Image.open(imagePath) as imageObj:
            imageWidth, imageHeight = imageObj.size

        boxes = parseAnnotationFile(annPath)
        occupied: List[Tuple[int, int, int, int]] = []

        for box in boxes:
            if box.width < minBoxSize or box.height < minBoxSize:
                continue
            left = max(0, box.left)
            top = max(0, box.top)
            right = min(imageWidth, left + box.width)
            bottom = min(imageHeight, top + box.height)
            if right - left < minBoxSize or bottom - top < minBoxSize:
                continue

            bbox = (left, top, right, bottom)
            occupied.append(bbox)

            if box.category in VEHICLE_CATEGORIES:
                grouped["vehicle"].append(CropCandidate(imagePath=imagePath, bbox=bbox, className="vehicle"))
            elif box.category in CIVILIAN_CATEGORIES:
                grouped["civilian-object"].append(
                    CropCandidate(imagePath=imagePath, bbox=bbox, className="civilian-object")
                )
            elif box.category == 0:
                grouped["decoy"].append(CropCandidate(imagePath=imagePath, bbox=bbox, className="decoy"))

        grouped["decoy"].extend(
            randomBackgroundCandidates(
                imagePath=imagePath,
                width=imageWidth,
                height=imageHeight,
                occupied=occupied,
                samples=backgroundPerImage,
                rng=rng,
            )
        )

    return grouped


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
        "val": {"vehicle": [], "civilian-object": [], "decoy": []},
    }
    for className, items in selected.items():
        splitIdx = int(len(items) * (1.0 - valRatio))
        output["train"][className] = items[:splitIdx]
        output["val"][className] = items[splitIdx:]
    return output


def saveCrops(outputDir: Path, splitItems: Dict[str, Dict[str, List[CropCandidate]]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        "train": {"vehicle": 0, "civilian-object": 0, "decoy": 0},
        "val": {"vehicle": 0, "civilian-object": 0, "decoy": 0},
    }

    for splitName, byClass in splitItems.items():
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

    if args.source_images_dir and args.source_annotations_dir:
        imagesDir = Path(args.source_images_dir)
        annDir = Path(args.source_annotations_dir)
    else:
        imagesDir, annDir = guessSourceDirs(rawDir)

    if not imagesDir.exists() or not annDir.exists():
        raise RuntimeError(f"Source data dirs not found: images={imagesDir} annotations={annDir}")

    print(f"Source images: {imagesDir}")
    print(f"Source annotations: {annDir}")

    grouped = collectCandidates(
        imagesDir=imagesDir,
        annDir=annDir,
        minBoxSize=args.min_box_size,
        backgroundPerImage=args.background_crops_per_image,
        rng=rng,
    )

    selected = chooseSubset(grouped, subsetSizePerClass=args.subset_size_per_class, rng=rng)
    splitItems = splitTrainVal(selected, valRatio=args.val_ratio)

    ensureDir(outputDir)
    clearOutput(outputDir)
    counts = saveCrops(outputDir, splitItems)

    payload = {
        "dataset": "VisDrone-Subset",
        "sourceImagesDir": str(imagesDir),
        "sourceAnnotationsDir": str(annDir),
        "subsetSizePerClass": args.subset_size_per_class,
        "valRatio": args.val_ratio,
        "seed": args.seed,
        "candidateCounts": {key: len(value) for key, value in grouped.items()},
        "selectedCounts": {key: len(value) for key, value in selected.items()},
        "outputCounts": counts,
        "classes": ["vehicle", "civilian-object", "decoy"],
    }
    (outputDir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Prepared dataset saved in: {outputDir}")


if __name__ == "__main__":
    main()
