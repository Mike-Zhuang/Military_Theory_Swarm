from __future__ import annotations

import math
from typing import Tuple

from .models import AgentState, SimulationConfig, Vec2


def vectorNorm(vec: Vec2) -> float:
    return math.hypot(vec.x, vec.y)


def normalizeVector(vec: Vec2) -> Vec2:
    norm = vectorNorm(vec)
    if norm < 1e-9:
        return Vec2(0.0, 0.0)
    return Vec2(vec.x / norm, vec.y / norm)


def clampVector(vec: Vec2, maxNorm: float) -> Vec2:
    norm = vectorNorm(vec)
    if norm <= maxNorm or norm < 1e-9:
        return vec
    ratio = maxNorm / norm
    return Vec2(vec.x * ratio, vec.y * ratio)


def angleDifference(target: float, source: float) -> float:
    diff = (target - source + math.pi) % (2 * math.pi) - math.pi
    return diff


def rotateTowards(current: Vec2, desired: Vec2, maxTurn: float) -> Vec2:
    if vectorNorm(desired) < 1e-9:
        return current
    if vectorNorm(current) < 1e-9:
        return desired

    currentAngle = math.atan2(current.y, current.x)
    desiredAngle = math.atan2(desired.y, desired.x)
    diff = angleDifference(desiredAngle, currentAngle)
    turn = max(-maxTurn, min(maxTurn, diff))
    resultAngle = currentAngle + turn
    speed = vectorNorm(desired)
    return Vec2(math.cos(resultAngle) * speed, math.sin(resultAngle) * speed)


def applyWorldBounds(agent: AgentState, config: SimulationConfig) -> None:
    if agent.position.x < 0:
        agent.position.x = 0.0
        agent.velocity.x = abs(agent.velocity.x)
    elif agent.position.x > config.worldWidth:
        agent.position.x = float(config.worldWidth)
        agent.velocity.x = -abs(agent.velocity.x)

    if agent.position.y < 0:
        agent.position.y = 0.0
        agent.velocity.y = abs(agent.velocity.y)
    elif agent.position.y > config.worldHeight:
        agent.position.y = float(config.worldHeight)
        agent.velocity.y = -abs(agent.velocity.y)


def stepAgent(agent: AgentState, desiredVelocity: Vec2, config: SimulationConfig) -> float:
    if not agent.alive:
        return 0.0

    desiredVelocity = clampVector(desiredVelocity, config.maxSpeed)
    desiredVelocity = rotateTowards(
        current=agent.velocity,
        desired=desiredVelocity,
        maxTurn=config.maxTurnRate * config.dt,
    )

    accel = desiredVelocity.sub(agent.velocity)
    accel = clampVector(accel, config.maxAccel * config.dt)
    agent.velocity = clampVector(agent.velocity.add(accel), config.maxSpeed)

    prevX, prevY = agent.position.x, agent.position.y
    agent.position = agent.position.add(agent.velocity.scale(config.dt))
    applyWorldBounds(agent, config)

    travelled = math.hypot(agent.position.x - prevX, agent.position.y - prevY)
    agent.distanceTravelled += travelled
    return travelled
