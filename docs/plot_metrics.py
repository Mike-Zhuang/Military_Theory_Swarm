from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
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


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment metrics from CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/experiment-matrix.csv",
        help="Path to experiment matrix CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory for generated plots",
    )
    return parser.parse_args()


def loadRows(csvPath: Path) -> List[Dict[str, str]]:
    with csvPath.open("r", encoding="utf-8") as fileObj:
        reader = csv.DictReader(fileObj)
        return [dict(row) for row in reader]


def toFloat(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def saveCompletionPlot(rows: List[Dict[str, str]], outputDir: Path) -> None:
    grouped: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        key = (row["strategy"], row["ml"])
        grouped[key].append((toFloat(row["packetLoss"]), toFloat(row["taskCompletionRate"])))

    plt.figure(figsize=(8.2, 5.1))
    for (strategy, mlMode), series in sorted(grouped.items()):
        series.sort(key=lambda item: item[0])
        xs = [item[0] for item in series]
        ys = [item[1] * 100.0 for item in series]
        plt.plot(xs, ys, marker="o", label=f"{strategy} + ml:{mlMode}")

    plt.title("Completion Rate vs Packet Loss")
    plt.xlabel("Packet Loss")
    plt.ylabel("Task Completion Rate (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputDir / "completion-vs-packet-loss.png", dpi=180)
    plt.close()


def saveResponsePlot(rows: List[Dict[str, str]], outputDir: Path) -> None:
    grouped: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        key = (row["strategy"], row["ml"])
        grouped[key].append((toFloat(row["packetLoss"]), toFloat(row["avgResponseTime"])))

    plt.figure(figsize=(8.2, 5.1))
    for (strategy, mlMode), series in sorted(grouped.items()):
        series.sort(key=lambda item: item[0])
        xs = [item[0] for item in series]
        ys = [item[1] for item in series]
        plt.plot(xs, ys, marker="s", label=f"{strategy} + ml:{mlMode}")

    plt.title("Average Response Time vs Packet Loss")
    plt.xlabel("Packet Loss")
    plt.ylabel("Avg Response Time (s)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputDir / "response-time-vs-packet-loss.png", dpi=180)
    plt.close()


def saveMlGainPlot(rows: List[Dict[str, str]], outputDir: Path) -> None:
    byPair: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["strategy"], toFloat(row["packetLoss"]))
        byPair[key][row["ml"]] = toFloat(row["taskCompletionRate"])

    valuesByStrategy: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (strategy, packetLoss), record in byPair.items():
        if "on" in record and "off" in record:
            gain = (record["on"] - record["off"]) * 100.0
            valuesByStrategy[strategy].append((packetLoss, gain))

    plt.figure(figsize=(8.2, 5.1))
    for strategy, values in sorted(valuesByStrategy.items()):
        values.sort(key=lambda item: item[0])
        xs = [item[0] for item in values]
        ys = [item[1] for item in values]
        plt.plot(xs, ys, marker="^", label=strategy)

    plt.axhline(0.0, color="#999", linewidth=1, linestyle="--")
    plt.title("ML Gain on Completion Rate")
    plt.xlabel("Packet Loss")
    plt.ylabel("Gain (%) = completion(ml:on) - completion(ml:off)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputDir / "ml-gain-vs-packet-loss.png", dpi=180)
    plt.close()


def main() -> None:
    args = parseArgs()
    csvPath = Path(args.csv)
    outputDir = Path(args.output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)

    rows = loadRows(csvPath)
    if not rows:
        raise SystemExit("No rows found in CSV. Run sim-core/run_experiments.py first.")

    saveCompletionPlot(rows, outputDir)
    saveResponsePlot(rows, outputDir)
    saveMlGainPlot(rows, outputDir)

    print(f"Figures generated in: {outputDir}")


if __name__ == "__main__":
    main()
