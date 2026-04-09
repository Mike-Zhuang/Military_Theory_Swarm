from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dataset sample grid for UI preview")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="official-val")
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def collectSamples(dataDir: Path, split: str, samplesPerClass: int, rng: random.Random) -> Dict[str, List[Path]]:
    splitDir = dataDir / split
    if not splitDir.exists() and split == "official-val":
        fallbackDir = dataDir / "val"
        splitDir = fallbackDir if fallbackDir.exists() else dataDir / "dev-val"
    if not splitDir.exists():
        raise RuntimeError(f"Split directory not found: {splitDir}")

    grouped: Dict[str, List[Path]] = {}
    for classDir in sorted(splitDir.iterdir()):
        if not classDir.is_dir():
            continue
        files = sorted(classDir.glob("*.png"))
        rng.shuffle(files)
        grouped[classDir.name] = files[: min(len(files), samplesPerClass)]
    if not grouped:
        raise RuntimeError(f"No samples found in: {splitDir}")
    return grouped


def renderGrid(samples: Dict[str, List[Path]], outputPath: Path) -> None:
    classes = sorted(samples.keys())
    cols = max(len(items) for items in samples.values())
    rows = len(classes)

    tileSize = 92
    gap = 10
    titleHeight = 28
    leftLabelWidth = 170

    width = leftLabelWidth + cols * (tileSize + gap) + gap
    height = titleHeight + rows * (tileSize + gap) + gap

    canvas = Image.new("RGB", (width, height), color=(12, 25, 35))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle([(0, 0), (width, titleHeight)], fill=(18, 40, 55))
    draw.text((10, 8), "VisDrone Samples", fill=(220, 239, 248), font=font)

    for rowIdx, className in enumerate(classes):
        y = titleHeight + gap + rowIdx * (tileSize + gap)
        draw.text((10, y + tileSize // 2 - 6), className, fill=(170, 214, 234), font=font)

        items = samples[className]
        for colIdx in range(cols):
            x = leftLabelWidth + gap + colIdx * (tileSize + gap)
            draw.rectangle([(x - 1, y - 1), (x + tileSize + 1, y + tileSize + 1)], outline=(62, 107, 128), width=1)
            if colIdx >= len(items):
                continue
            with Image.open(items[colIdx]) as imageObj:
                tile = imageObj.convert("RGB").resize((tileSize, tileSize))
            canvas.paste(tile, (x, y))

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(outputPath)


def main() -> None:
    args = parseArgs()
    rng = random.Random(args.seed)
    samples = collectSamples(Path(args.data_dir), args.split, args.samples_per_class, rng)
    renderGrid(samples, Path(args.output))
    print(f"Sample grid saved: {args.output}")


if __name__ == "__main__":
    main()
