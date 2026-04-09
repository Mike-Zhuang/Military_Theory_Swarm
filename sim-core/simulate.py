from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from src.exporter import writeScenario
from src.models import SimulationConfig, runToDict
from src.simulator import runSimulation


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate educational swarm simulation scenarios.",
    )
    parser.add_argument("--strategy", choices=["decentralized", "centralized"], default="decentralized")
    parser.add_argument("--compare", action="store_true", help="Export decentralized and centralized runs in one file")
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--packet-loss", type=float, default=0.2)
    parser.add_argument("--fail-rate", type=float, default=0.0015)
    parser.add_argument("--comm-radius", type=float, default=185.0)
    parser.add_argument(
        "--scenario",
        choices=["recon-coverage", "jam-recovery", "multi-target-allocation"],
        default="jam-recovery",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="../web-demo/public/generated/demo-compare.json")
    parser.add_argument("--ml-confidence", type=str, default="")
    parser.add_argument("--ml-off", action="store_true")
    return parser.parse_args()


def loadClassConfidence(path: str) -> Dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "classConfidence" in payload and isinstance(payload["classConfidence"], dict):
        return {str(key): float(value) for key, value in payload["classConfidence"].items()}
    return {str(key): float(value) for key, value in payload.items()}


def main() -> None:
    args = parseArgs()

    config = SimulationConfig(
        stepCount=args.steps,
        agentCount=args.agents,
        packetLoss=args.packet_loss,
        failureRate=args.fail_rate,
        communicationRadius=args.comm_radius,
        seed=args.seed,
        scenarioName=args.scenario,
    )

    classConfidence: Dict[str, float] = {
        "vehicle": 1.0,
        "decoy": 1.0,
        "civilian-object": 1.0,
    }
    if args.ml_confidence:
        classConfidence.update(loadClassConfidence(args.ml_confidence))
    if args.ml_off:
        classConfidence = {
            "vehicle": 1.0,
            "decoy": 1.0,
            "civilian-object": 1.0,
        }

    if args.compare:
        decentralized = runSimulation(
            config=config,
            strategy="decentralized",
            classConfidence=classConfidence,
            seedOffset=0,
        )
        centralized = runSimulation(
            config=config,
            strategy="centralized",
            classConfidence=classConfidence,
            seedOffset=77,
        )
        payload = {
            "metadata": {
                "title": f"Decentralized vs Centralized ({args.scenario})",
                "scenarioType": "compare",
                "scenario": args.scenario,
                "world": {
                    "width": config.worldWidth,
                    "height": config.worldHeight,
                },
                "dt": config.dt,
            },
            "runs": [runToDict(decentralized), runToDict(centralized)],
        }
    else:
        run = runSimulation(
            config=config,
            strategy=args.strategy,
            classConfidence=classConfidence,
            seedOffset=0,
        )
        payload = {
            "metadata": {
                "title": f"Single strategy: {args.strategy} ({args.scenario})",
                "scenarioType": "single",
                "scenario": args.scenario,
                "world": {
                    "width": config.worldWidth,
                    "height": config.worldHeight,
                },
                "dt": config.dt,
            },
            "runs": [runToDict(run)],
        }

    writeScenario(payload=payload, outputPath=args.output)
    print(f"Scenario exported: {args.output}")


if __name__ == "__main__":
    main()
