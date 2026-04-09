from __future__ import annotations

import argparse
from pathlib import Path

from src.exporter import writeScenario
from src.models import SimulationConfig, runToDict
from src.simulator import runSimulation


SCENARIOS = [
    "recon-coverage",
    "jam-recovery",
    "multi-target-allocation",
]


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compare files for all built-in scenarios")
    parser.add_argument("--output-dir", type=str, default="../web-demo/public/scenarios")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--agents", type=int, default=28)
    parser.add_argument("--packet-loss", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    outputDir = Path(args.output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)

    for idx, scenarioName in enumerate(SCENARIOS):
        config = SimulationConfig(
            stepCount=args.steps,
            agentCount=args.agents,
            packetLoss=args.packet_loss,
            seed=args.seed + idx,
            scenarioName=scenarioName,
        )

        decentralized = runSimulation(
            config=config,
            strategy="decentralized",
            seedOffset=0,
        )
        centralized = runSimulation(
            config=config,
            strategy="centralized",
            seedOffset=77,
        )

        payload = {
            "metadata": {
                "title": f"Decentralized vs Centralized ({scenarioName})",
                "scenarioType": "compare",
                "scenario": scenarioName,
                "world": {
                    "width": config.worldWidth,
                    "height": config.worldHeight,
                },
                "dt": config.dt,
            },
            "runs": [runToDict(decentralized), runToDict(centralized)],
        }

        outputPath = outputDir / f"{scenarioName}-compare.json"
        writeScenario(payload=payload, outputPath=str(outputPath))
        print(f"Generated: {outputPath}")


if __name__ == "__main__":
    main()
