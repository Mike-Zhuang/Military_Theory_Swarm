from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFilter


CLASS_NAMES = ["vehicle", "decoy", "civilian-object"]


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic top-view target dataset")
    parser.add_argument("--output", type=str, default="data/generated")
    parser.add_argument("--samples-per-class", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def noisyBackground(draw: ImageDraw.ImageDraw, size: int, rng: random.Random) -> None:
    for _ in range(80):
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        shade = rng.randint(15, 55)
        draw.point((x, y), fill=(shade, shade, shade))


def drawVehicle(draw: ImageDraw.ImageDraw, size: int, rng: random.Random) -> None:
    cx, cy = size // 2 + rng.randint(-6, 6), size // 2 + rng.randint(-6, 6)
    bodyW, bodyH = rng.randint(20, 30), rng.randint(12, 18)
    draw.rectangle(
        [cx - bodyW // 2, cy - bodyH // 2, cx + bodyW // 2, cy + bodyH // 2],
        outline=(210, 210, 210),
        fill=(120, 120, 120),
        width=2,
    )
    barrelLen = rng.randint(10, 16)
    draw.line(
        [cx, cy, cx + barrelLen, cy + rng.randint(-2, 2)],
        fill=(225, 225, 225),
        width=2,
    )


def drawDecoy(draw: ImageDraw.ImageDraw, size: int, rng: random.Random) -> None:
    cx, cy = size // 2 + rng.randint(-8, 8), size // 2 + rng.randint(-8, 8)
    scale = rng.randint(12, 17)
    points = [
        (cx, cy - scale),
        (cx + scale, cy + scale),
        (cx - scale, cy + scale),
    ]
    draw.polygon(points, outline=(220, 220, 220), fill=(95, 95, 95))
    draw.line([points[0], points[1]], fill=(230, 230, 230), width=1)


def drawCivilianObject(draw: ImageDraw.ImageDraw, size: int, rng: random.Random) -> None:
    cx, cy = size // 2 + rng.randint(-7, 7), size // 2 + rng.randint(-7, 7)
    radius = rng.randint(8, 14)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=(220, 220, 220),
        fill=(135, 135, 135),
        width=2,
    )
    boxW, boxH = rng.randint(8, 14), rng.randint(8, 13)
    draw.rectangle(
        [cx - boxW // 2, cy - boxH // 2, cx + boxW // 2, cy + boxH // 2],
        fill=(165, 165, 165),
    )


def renderImage(className: str, imageSize: int, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (imageSize, imageSize), color=(25, 25, 28))
    draw = ImageDraw.Draw(image)
    noisyBackground(draw, imageSize, rng)

    if className == "vehicle":
        drawVehicle(draw, imageSize, rng)
    elif className == "decoy":
        drawDecoy(draw, imageSize, rng)
    else:
        drawCivilianObject(draw, imageSize, rng)

    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))

    return image


def ensurePath(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parseArgs()
    rng = random.Random(args.seed)

    outputDir = Path(args.output)
    trainDir = outputDir / "train"
    valDir = outputDir / "val"
    for splitDir in [trainDir, valDir]:
        for className in CLASS_NAMES:
            ensurePath(splitDir / className)

    manifest: Dict[str, Dict[str, int]] = {
        "train": {className: 0 for className in CLASS_NAMES},
        "val": {className: 0 for className in CLASS_NAMES},
    }

    for className in CLASS_NAMES:
        for idx in range(args.samples_per_class):
            split = "train" if rng.random() < args.train_ratio else "val"
            image = renderImage(className=className, imageSize=args.image_size, rng=rng)
            fileName = f"{className}_{idx:04d}.png"
            savePath = outputDir / split / className / fileName
            image.save(savePath)
            manifest[split][className] += 1

    info = {
        "classes": CLASS_NAMES,
        "samplesPerClass": args.samples_per_class,
        "trainRatio": args.train_ratio,
        "seed": args.seed,
        "counts": manifest,
    }
    (outputDir / "manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Synthetic dataset generated in: {outputDir}")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
