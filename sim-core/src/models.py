from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Vec2:
    x: float
    y: float

    def add(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def sub(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def scale(self, value: float) -> "Vec2":
        return Vec2(self.x * value, self.y * value)


@dataclass
class AgentState:
    id: int
    position: Vec2
    velocity: Vec2
    alive: bool = True
    assignedTargetId: Optional[int] = None
    firstAssignedStep: Optional[int] = None
    distanceTravelled: float = 0.0


@dataclass
class TargetState:
    id: int
    className: str
    position: Vec2
    basePriority: float
    active: bool = True
    completionStep: Optional[int] = None


@dataclass
class SimulationConfig:
    worldWidth: int = 1000
    worldHeight: int = 700
    stepCount: int = 260
    dt: float = 0.2
    agentCount: int = 32
    maxSpeed: float = 32.0
    maxAccel: float = 38.0
    maxTurnRate: float = 2.2
    communicationRadius: float = 185.0
    packetLoss: float = 0.15
    failureRate: float = 0.0015
    completionRadius: float = 28.0
    gridCell: int = 30
    seed: int = 7
    scenarioName: str = "jam-recovery"


@dataclass
class FrameState:
    t: int
    agents: List[Dict[str, object]]
    targets: List[Dict[str, object]]
    links: List[List[int]]
    events: List[str]


@dataclass
class RunResult:
    name: str
    config: Dict[str, object]
    summary: Dict[str, float]
    frames: List[FrameState] = field(default_factory=list)


DEFAULT_CLASS_CONFIDENCE: Dict[str, float] = {
    "vehicle": 1.0,
    "decoy": 1.0,
    "civilian-object": 1.0,
}


def toAgentDict(agent: AgentState) -> Dict[str, object]:
    return {
        "id": agent.id,
        "x": round(agent.position.x, 3),
        "y": round(agent.position.y, 3),
        "vx": round(agent.velocity.x, 3),
        "vy": round(agent.velocity.y, 3),
        "alive": agent.alive,
        "assignedTargetId": agent.assignedTargetId,
    }


def toTargetDict(target: TargetState) -> Dict[str, object]:
    return {
        "id": target.id,
        "className": target.className,
        "x": round(target.position.x, 3),
        "y": round(target.position.y, 3),
        "basePriority": round(target.basePriority, 3),
        "active": target.active,
    }


def frameToDict(frame: FrameState) -> Dict[str, object]:
    return {
        "t": frame.t,
        "agents": frame.agents,
        "targets": frame.targets,
        "links": frame.links,
        "events": frame.events,
    }


def runToDict(run: RunResult) -> Dict[str, object]:
    return {
        "name": run.name,
        "config": run.config,
        "summary": run.summary,
        "frames": [frameToDict(frame) for frame in run.frames],
    }
