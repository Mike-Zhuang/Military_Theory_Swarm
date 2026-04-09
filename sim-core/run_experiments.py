from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src.exporter import writeJson
from src.models import SimulationConfig
from src.simulator import runSimulation


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiment matrix for report figures.")
    parser.add_argument("--output-dir", type=str, default="../docs/outputs")
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scenario",
        choices=["recon-coverage", "jam-recovery", "multi-target-allocation"],
        default="jam-recovery",
    )
    return parser.parse_args()


def ensureDir(path: str) -> Path:
    outputDir = Path(path)
    outputDir.mkdir(parents=True, exist_ok=True)
    return outputDir


def main() -> None:
    args = parseArgs()
    outputDir = ensureDir(args.output_dir)

    packetLossLevels = [0.0, 0.15, 0.3, 0.45]
    strategies = ["decentralized", "centralized"]
    mlProfiles = {
        "off": {
            "vehicle": 1.0,
            "decoy": 1.0,
            "civilian-object": 1.0,
        },
        "on": {
            "vehicle": 0.93,
            "decoy": 0.36,
            "civilian-object": 0.14,
        },
    }

    rows: List[Dict[str, object]] = []
    for packetLoss in packetLossLevels:
        for strategy in strategies:
            for mlMode, classConfidence in mlProfiles.items():
                config = SimulationConfig(
                    stepCount=args.steps,
                    agentCount=args.agents,
                    packetLoss=packetLoss,
                    seed=args.seed,
                    scenarioName=args.scenario,
                )
                run = runSimulation(
                    config=config,
                    strategy=strategy,
                    classConfidence=classConfidence,
                    seedOffset=0 if strategy == "decentralized" else 101,
                )
                rows.append(
                    {
                        "scenario": args.scenario,
                        "strategy": strategy,
                        "ml": mlMode,
                        "packetLoss": packetLoss,
                        **run.summary,
                    }
                )

    csvPath = outputDir / "experiment-matrix.csv"
    jsonPath = outputDir / "experiment-matrix.json"

    with csvPath.open("w", newline="", encoding="utf-8") as fileObj:
        writer = csv.DictWriter(fileObj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    writeJson({"rows": rows}, str(jsonPath))
    print(f"Experiment matrix exported: {csvPath}")
    print(f"Experiment matrix exported: {jsonPath}")


if __name__ == "__main__":
    main()
