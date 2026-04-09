from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

from .models import AgentState, SimulationConfig


JamZone = Tuple[float, float, float, float]


def buildDefaultJamZones(
    config: SimulationConfig,
    scenarioName: str = "jam-recovery",
) -> List[JamZone]:
    if scenarioName == "recon-coverage":
        return [
            (
                config.worldWidth * 0.52,
                config.worldHeight * 0.35,
                95.0,
                0.22,
            )
        ]

    if scenarioName == "multi-target-allocation":
        return [
            (
                config.worldWidth * 0.66,
                config.worldHeight * 0.30,
                115.0,
                0.30,
            ),
            (
                config.worldWidth * 0.76,
                config.worldHeight * 0.66,
                130.0,
                0.36,
            ),
        ]

    # Default is a stronger central interference zone for recovery demonstrations.
    return [
        (
            config.worldWidth * 0.58,
            config.worldHeight * 0.45,
            140.0,
            0.45,
        )
    ]


def packetLossAtStep(baseLoss: float, stepIdx: int) -> float:
    pulse = 0.18 if (stepIdx % 70) in range(20, 32) else 0.0
    return max(0.0, min(0.95, baseLoss + pulse))


def localJammingPenalty(x: float, y: float, jamZones: Sequence[JamZone]) -> float:
    penalty = 0.0
    for centerX, centerY, radius, intensity in jamZones:
        distance = math.hypot(x - centerX, y - centerY)
        if distance < radius:
            penalty = max(penalty, intensity * (1.0 - distance / radius))
    return max(0.0, min(0.95, penalty))


def buildLinks(
    agents: Sequence[AgentState],
    config: SimulationConfig,
    stepIdx: int,
    rng: random.Random,
    jamZones: Sequence[JamZone],
) -> Tuple[List[List[int]], float]:
    links: List[List[int]] = []
    packetLoss = packetLossAtStep(config.packetLoss, stepIdx)

    for idx in range(len(agents)):
        left = agents[idx]
        if not left.alive:
            continue
        for jdx in range(idx + 1, len(agents)):
            right = agents[jdx]
            if not right.alive:
                continue

            distance = math.hypot(
                left.position.x - right.position.x,
                left.position.y - right.position.y,
            )
            if distance > config.communicationRadius:
                continue

            leftPenalty = localJammingPenalty(
                left.position.x,
                left.position.y,
                jamZones,
            )
            rightPenalty = localJammingPenalty(
                right.position.x,
                right.position.y,
                jamZones,
            )
            dropChance = min(0.98, packetLoss + max(leftPenalty, rightPenalty))
            if rng.random() < dropChance:
                continue

            links.append([left.id, right.id])

    return links, packetLoss


def applyRandomFailures(
    agents: Sequence[AgentState],
    config: SimulationConfig,
    stepIdx: int,
    rng: random.Random,
) -> List[str]:
    events: List[str] = []
    for agent in agents:
        if not agent.alive:
            continue
        if rng.random() < config.failureRate:
            agent.alive = False
            events.append(
                f"step={stepIdx}: agent-{agent.id} entered fail-safe state"
            )
    return events
