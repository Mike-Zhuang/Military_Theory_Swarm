from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Sequence

from .dynamics import normalizeVector, vectorNorm
from .models import AgentState, SimulationConfig, TargetState, Vec2


def buildNeighborMap(
    agents: Sequence[AgentState],
    links: Sequence[List[int]],
) -> Dict[int, List[AgentState]]:
    byId = {agent.id: agent for agent in agents}
    neighborMap: Dict[int, List[AgentState]] = {agent.id: [] for agent in agents}
    for leftId, rightId in links:
        left = byId.get(leftId)
        right = byId.get(rightId)
        if left is None or right is None:
            continue
        neighborMap[leftId].append(right)
        neighborMap[rightId].append(left)
    return neighborMap


def chooseTargetByScore(
    agent: AgentState,
    targets: Sequence[TargetState],
    classConfidence: Dict[str, float],
) -> int | None:
    bestTargetId = None
    bestScore = -1e9

    for target in targets:
        if not target.active:
            continue

        confidence = classConfidence.get(target.className, 0.55)
        distance = math.hypot(
            target.position.x - agent.position.x,
            target.position.y - agent.position.y,
        )
        score = target.basePriority * confidence * 100.0 - 0.35 * distance
        if score > bestScore:
            bestScore = score
            bestTargetId = target.id

    return bestTargetId


def consensusTarget(
    ownChoice: int | None,
    neighbors: Sequence[AgentState],
) -> int | None:
    voteCounter: Counter[int] = Counter()
    for neighbor in neighbors:
        if neighbor.alive and neighbor.assignedTargetId is not None:
            voteCounter[neighbor.assignedTargetId] += 1

    if ownChoice is not None:
        voteCounter[ownChoice] += 1

    if not voteCounter:
        return ownChoice

    topTarget, topCount = voteCounter.most_common(1)[0]
    if topCount >= 3:
        return topTarget
    return ownChoice


def boidsVector(
    agent: AgentState,
    neighbors: Sequence[AgentState],
) -> Vec2:
    if not neighbors:
        return Vec2(0.0, 0.0)

    sep = Vec2(0.0, 0.0)
    ali = Vec2(0.0, 0.0)
    coh = Vec2(0.0, 0.0)

    for neighbor in neighbors:
        delta = agent.position.sub(neighbor.position)
        distance = max(1e-6, vectorNorm(delta))
        if distance < 38.0:
            sep = sep.add(normalizeVector(delta).scale(1.0 / distance))
        ali = ali.add(neighbor.velocity)
        coh = coh.add(neighbor.position)

    inv = 1.0 / len(neighbors)
    ali = ali.scale(inv).sub(agent.velocity)
    coh = coh.scale(inv).sub(agent.position)

    # A simple weighted blend for easy classroom explanation.
    return sep.scale(1.2).add(ali.scale(0.45)).add(coh.scale(0.38))


def targetVector(
    agent: AgentState,
    targetsById: Dict[int, TargetState],
) -> Vec2:
    if agent.assignedTargetId is None:
        return Vec2(0.0, 0.0)

    target = targetsById.get(agent.assignedTargetId)
    if target is None or not target.active:
        return Vec2(0.0, 0.0)

    return target.position.sub(agent.position)


def computeDesiredVelocities(
    agents: Sequence[AgentState],
    targets: Sequence[TargetState],
    links: Sequence[List[int]],
    strategy: str,
    classConfidence: Dict[str, float],
    config: SimulationConfig,
    stepIdx: int,
    rng: random.Random,
) -> Dict[int, Vec2]:
    neighborMap = buildNeighborMap(agents, links)
    targetsById = {target.id: target for target in targets}

    for agent in agents:
        if not agent.alive:
            agent.assignedTargetId = None
            continue

        ownChoice = chooseTargetByScore(agent, targets, classConfidence)
        if strategy == "decentralized":
            finalChoice = consensusTarget(ownChoice, neighborMap.get(agent.id, []))
        else:
            finalChoice = ownChoice
        agent.assignedTargetId = finalChoice
        if agent.firstAssignedStep is None and finalChoice is not None:
            agent.firstAssignedStep = stepIdx

    desiredById: Dict[int, Vec2] = {}
    for agent in agents:
        if not agent.alive:
            desiredById[agent.id] = Vec2(0.0, 0.0)
            continue

        boids = boidsVector(agent, neighborMap.get(agent.id, []))
        towardTarget = targetVector(agent, targetsById)

        if strategy == "decentralized":
            merged = towardTarget.scale(1.05).add(boids)
        else:
            merged = towardTarget.scale(1.25).add(boids.scale(0.45))

        if vectorNorm(merged) < 1e-6:
            randomHeading = Vec2(
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            )
            merged = randomHeading

        desiredById[agent.id] = normalizeVector(merged).scale(config.maxSpeed)

    return desiredById
