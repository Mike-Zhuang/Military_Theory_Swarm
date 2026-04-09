from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from model import TinyConvNet


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference utility for Tiny-CNN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--calibration-dir", type=str, default="")
    parser.add_argument("--emit-class-confidence", type=str, default="")
    return parser.parse_args()


def chooseDevice() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def loadModel(checkpointPath: str, device: torch.device) -> Tuple[TinyConvNet, Dict[int, str]]:
    checkpoint = torch.load(checkpointPath, map_location=device)
    classToIdx = checkpoint["classToIdx"]
    idxToClass = {idx: className for className, idx in classToIdx.items()}

    model = TinyConvNet(classCount=len(classToIdx)).to(device)
    model.load_state_dict(checkpoint["stateDict"])
    model.eval()
    return model, idxToClass


def imageTensor(imagePath: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(imagePath).convert("RGB")
    return transform(image).unsqueeze(0)


def predictOne(
    model: TinyConvNet,
    tensor: torch.Tensor,
    idxToClass: Dict[int, str],
    device: torch.device,
) -> Dict[str, object]:
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    scoreByClass = {
        idxToClass[idx]: float(prob)
        for idx, prob in enumerate(probs)
    }
    bestClass = max(scoreByClass.items(), key=lambda item: item[1])[0]
    return {
        "bestClass": bestClass,
        "scoreByClass": scoreByClass,
    }


def collectImages(args: argparse.Namespace) -> List[Path]:
    images: List[Path] = []
    if args.image:
        images.append(Path(args.image))
    if args.folder:
        images.extend(sorted(Path(args.folder).glob("*.png")))
    return images


def exportClassConfidence(
    model: TinyConvNet,
    idxToClass: Dict[int, str],
    device: torch.device,
    calibrationDir: Path,
    outputPath: Path,
) -> Dict[str, object]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for classDir in sorted(calibrationDir.iterdir()):
        if not classDir.is_dir():
            continue
        className = classDir.name
        for imagePath in classDir.glob("*.png"):
            result = predictOne(
                model=model,
                tensor=imageTensor(imagePath),
                idxToClass=idxToClass,
                device=device,
            )
            confidence = result["scoreByClass"].get(className, 0.0)
            sums[className] = sums.get(className, 0.0) + float(confidence)
            counts[className] = counts.get(className, 0) + 1

    classConfidence = {
        className: (sums.get(className, 0.0) / max(1, counts.get(className, 0)))
        for className in sorted(counts)
    }
    payload = {
        "classConfidence": classConfidence,
        "counts": counts,
    }
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parseArgs()
    device = chooseDevice()
    model, idxToClass = loadModel(args.checkpoint, device)

    results = []
    for imagePath in collectImages(args):
        result = predictOne(
            model=model,
            tensor=imageTensor(imagePath),
            idxToClass=idxToClass,
            device=device,
        )
        results.append({"image": str(imagePath), **result})

    if results:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))

    if args.emit_class_confidence and args.calibration_dir:
        payload = exportClassConfidence(
            model=model,
            idxToClass=idxToClass,
            device=device,
            calibrationDir=Path(args.calibration_dir),
            outputPath=Path(args.emit_class_confidence),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
