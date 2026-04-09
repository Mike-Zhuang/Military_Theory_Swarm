from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from .coordination import computeDesiredVelocities
from .disturbance import applyRandomFailures, buildDefaultJamZones, buildLinks
from .dynamics import stepAgent
from .models import (
    DEFAULT_CLASS_CONFIDENCE,
    AgentState,
    FrameState,
    RunResult,
    SimulationConfig,
    TargetState,
    Vec2,
    toAgentDict,
    toTargetDict,
)


def spawnAgents(config: SimulationConfig, rng: random.Random) -> List[AgentState]:
    agents: List[AgentState] = []
    scenario = config.scenarioName
    if scenario == "recon-coverage":
        originX = config.worldWidth * 0.14
        originY = config.worldHeight * 0.50
        spreadX, spreadY = 95.0, 125.0
    elif scenario == "multi-target-allocation":
        originX = config.worldWidth * 0.20
        originY = config.worldHeight * 0.48
        spreadX, spreadY = 80.0, 95.0
    else:
        originX = config.worldWidth * 0.18
        originY = config.worldHeight * 0.52
        spreadX, spreadY = 65.0, 85.0

    for idx in range(config.agentCount):
        offsetX = rng.uniform(-spreadX, spreadX)
        offsetY = rng.uniform(-spreadY, spreadY)
        velocity = Vec2(rng.uniform(8.0, 14.0), rng.uniform(-2.0, 2.0))
        agents.append(
            AgentState(
                id=idx + 1,
                position=Vec2(originX + offsetX, originY + offsetY),
                velocity=velocity,
            )
        )
    return agents


def spawnTargets(config: SimulationConfig) -> List[TargetState]:
    scenario = config.scenarioName
    if scenario == "recon-coverage":
        targetBlueprints = [
            (1, "vehicle", 0.72, 0.20, 1.18),
            (2, "decoy", 0.82, 0.30, 0.92),
            (3, "vehicle", 0.65, 0.52, 1.24),
            (4, "civilian-object", 0.87, 0.48, 0.35),
            (5, "decoy", 0.60, 0.76, 0.90),
            (6, "vehicle", 0.84, 0.78, 1.30),
        ]
    elif scenario == "multi-target-allocation":
        targetBlueprints = [
            (1, "vehicle", 0.73, 0.22, 1.52),
            (2, "vehicle", 0.80, 0.30, 1.41),
            (3, "decoy", 0.68, 0.45, 1.03),
            (4, "vehicle", 0.76, 0.62, 1.35),
            (5, "decoy", 0.86, 0.70, 0.94),
            (6, "civilian-object", 0.90, 0.52, 0.25),
            (7, "vehicle", 0.64, 0.78, 1.33),
            (8, "civilian-object", 0.58, 0.30, 0.30),
        ]
    else:
        targetBlueprints = [
            (1, "vehicle", 0.78, 0.25, 1.45),
            (2, "vehicle", 0.82, 0.66, 1.38),
            (3, "decoy", 0.71, 0.42, 1.00),
            (4, "decoy", 0.62, 0.17, 0.95),
            (5, "civilian-object", 0.56, 0.78, 0.35),
            (6, "civilian-object", 0.90, 0.50, 0.30),
        ]

    targets: List[TargetState] = []
    for targetId, className, xRatio, yRatio, priority in targetBlueprints:
        targets.append(
            TargetState(
                id=targetId,
                className=className,
                position=Vec2(config.worldWidth * xRatio, config.worldHeight * yRatio),
                basePriority=priority,
            )
        )
    return targets


def toGridCell(position: Vec2, config: SimulationConfig) -> Tuple[int, int]:
    cellX = int(position.x // config.gridCell)
    cellY = int(position.y // config.gridCell)
    return cellX, cellY


def markTargetCompletions(
    agents: Sequence[AgentState],
    targets: Sequence[TargetState],
    config: SimulationConfig,
    stepIdx: int,
) -> List[str]:
    events: List[str] = []
    for target in targets:
        if not target.active:
            continue

        for agent in agents:
            if not agent.alive:
                continue
            if agent.assignedTargetId != target.id:
                continue

            distance = math.hypot(
                target.position.x - agent.position.x,
                target.position.y - agent.position.y,
            )
            if distance <= config.completionRadius:
                target.active = False
                target.completionStep = stepIdx
                events.append(
                    f"step={stepIdx}: target-{target.id} cleared by agent-{agent.id}"
                )
                break
    return events


def simulationSummary(
    agents: Sequence[AgentState],
    targets: Sequence[TargetState],
    coverageCells: int,
    totalCells: int,
    assignmentStepByTarget: Dict[int, int],
    packetLossSeries: Sequence[float],
    avgLinkDegreeSeries: Sequence[float],
    config: SimulationConfig,
) -> Dict[str, float]:
    completedTargets = [target for target in targets if not target.active]

    responseTimes: List[float] = []
    for target in completedTargets:
        assignStep = assignmentStepByTarget.get(target.id, 0)
        completeStep = target.completionStep or config.stepCount
        responseTimes.append(max(0.0, (completeStep - assignStep) * config.dt))

    completionRate = len(completedTargets) / max(1, len(targets))
    coverage = coverageCells / max(1, totalCells)
    survivalRate = sum(1 for agent in agents if agent.alive) / max(1, len(agents))
    avgResponse = (
        sum(responseTimes) / len(responseTimes)
        if responseTimes
        else config.stepCount * config.dt
    )
    avgLoss = (
        sum(packetLossSeries) / len(packetLossSeries)
        if packetLossSeries
        else config.packetLoss
    )
    avgDegree = (
        sum(avgLinkDegreeSeries) / len(avgLinkDegreeSeries)
        if avgLinkDegreeSeries
        else 0.0
    )
    avgTravel = sum(agent.distanceTravelled for agent in agents) / max(1, len(agents))

    return {
        "coverage": round(coverage, 4),
        "taskCompletionRate": round(completionRate, 4),
        "avgResponseTime": round(avgResponse, 3),
        "survivalRate": round(survivalRate, 4),
        "avgPacketLoss": round(avgLoss, 4),
        "avgLinkDegree": round(avgDegree, 4),
        "avgTravelDistance": round(avgTravel, 3),
    }


def runSimulation(
    config: SimulationConfig,
    strategy: str,
    classConfidence: Dict[str, float] | None = None,
    seedOffset: int = 0,
) -> RunResult:
    confidence = DEFAULT_CLASS_CONFIDENCE.copy()
    if classConfidence:
        confidence.update(classConfidence)

    rng = random.Random(config.seed + seedOffset)
    agents = spawnAgents(config, rng)
    targets = spawnTargets(config)
    jamZones = buildDefaultJamZones(config, config.scenarioName)

    widthCells = math.ceil(config.worldWidth / config.gridCell)
    heightCells = math.ceil(config.worldHeight / config.gridCell)
    totalCells = widthCells * heightCells
    coverageSet = set()

    frames: List[FrameState] = []
    assignmentStepByTarget: Dict[int, int] = {}
    packetLossSeries: List[float] = []
    avgLinkDegreeSeries: List[float] = []

    for stepIdx in range(config.stepCount):
        links, packetLoss = buildLinks(
            agents=agents,
            config=config,
            stepIdx=stepIdx,
            rng=rng,
            jamZones=jamZones,
        )
        packetLossSeries.append(packetLoss)

        aliveCount = max(1, sum(1 for agent in agents if agent.alive))
        avgLinkDegreeSeries.append(2.0 * len(links) / aliveCount)

        events = applyRandomFailures(
            agents=agents,
            config=config,
            stepIdx=stepIdx,
            rng=rng,
        )

        desiredById = computeDesiredVelocities(
            agents=agents,
            targets=targets,
            links=links,
            strategy=strategy,
            classConfidence=confidence,
            config=config,
            stepIdx=stepIdx,
            rng=rng,
        )

        for agent in agents:
            if not agent.alive:
                continue
            if agent.assignedTargetId is not None and agent.assignedTargetId not in assignmentStepByTarget:
                assignmentStepByTarget[agent.assignedTargetId] = stepIdx

            desiredVelocity = desiredById.get(agent.id, Vec2(0.0, 0.0))
            stepAgent(agent=agent, desiredVelocity=desiredVelocity, config=config)
            coverageSet.add(toGridCell(agent.position, config))

        events.extend(markTargetCompletions(
            agents=agents,
            targets=targets,
            config=config,
            stepIdx=stepIdx,
        ))

        frames.append(
            FrameState(
                t=stepIdx,
                agents=[toAgentDict(agent) for agent in agents],
                targets=[toTargetDict(target) for target in targets],
                links=links,
                events=events,
            )
        )

        if all(not target.active for target in targets):
            break

    summary = simulationSummary(
        agents=agents,
        targets=targets,
        coverageCells=len(coverageSet),
        totalCells=totalCells,
        assignmentStepByTarget=assignmentStepByTarget,
        packetLossSeries=packetLossSeries,
        avgLinkDegreeSeries=avgLinkDegreeSeries,
        config=config,
    )

    return RunResult(
        name=strategy,
        config={
            "scenario": config.scenarioName,
            "agentCount": config.agentCount,
            "stepCount": config.stepCount,
            "packetLoss": config.packetLoss,
            "failureRate": config.failureRate,
            "communicationRadius": config.communicationRadius,
            "dt": config.dt,
        },
        summary=summary,
        frames=frames,
    )
