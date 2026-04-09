from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml-module"
SIM_DIR = PROJECT_ROOT / "sim-core"
WEB_PUBLIC_DIR = PROJECT_ROOT / "web-demo" / "public"
RUNS_DIR = ML_DIR / "runs"
GENERATED_SCENARIO_DIR = WEB_PUBLIC_DIR / "generated"
DEFAULT_DATASET_DIR = ML_DIR / "data" / "visdrone-ready"
DEFAULT_RAW_DIR = ML_DIR / "data" / "visdrone" / "raw"
DEFAULT_TRAIN_IMAGES_DIR = DEFAULT_RAW_DIR / "VisDrone2019-DET-train" / "images"
DEFAULT_TRAIN_ANNOTATIONS_DIR = DEFAULT_RAW_DIR / "VisDrone2019-DET-train" / "annotations"
DEFAULT_VAL_IMAGES_DIR = DEFAULT_RAW_DIR / "VisDrone2019-DET-val" / "images"
DEFAULT_VAL_ANNOTATIONS_DIR = DEFAULT_RAW_DIR / "VisDrone2019-DET-val" / "annotations"

STATUS_VALUES: List[Literal["queued", "running", "succeeded", "failed", "cancelled"]] = [
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]


def resolvePath(rawPath: str) -> Path:
    pathObj = Path(rawPath)
    if pathObj.is_absolute():
        return pathObj
    return (PROJECT_ROOT / pathObj).resolve()


class DatasetPrepareRequest(BaseModel):
    download: bool = False
    downloadUrl: str = ""
    archivePath: str = ""
    rawDir: str = str(DEFAULT_RAW_DIR)
    outputDir: str = str(DEFAULT_DATASET_DIR)
    splitMode: Literal["official-val", "auto-split"] = "official-val"
    trainImagesDir: str = str(DEFAULT_TRAIN_IMAGES_DIR)
    trainAnnotationsDir: str = str(DEFAULT_TRAIN_ANNOTATIONS_DIR)
    valImagesDir: str = str(DEFAULT_VAL_IMAGES_DIR)
    valAnnotationsDir: str = str(DEFAULT_VAL_ANNOTATIONS_DIR)
    sourceImagesDir: str = ""
    sourceAnnotationsDir: str = ""
    subsetSizePerClass: int = Field(default=900, ge=120)
    valSubsetSizePerClass: int = Field(default=0, ge=0, le=20000)
    valRatio: float = Field(default=0.2, gt=0.05, lt=0.5)
    seed: int = 42


class TrainJobRequest(BaseModel):
    dataDir: str = str(DEFAULT_DATASET_DIR)
    epochs: int = Field(default=18, ge=1, le=200)
    batchSize: int = Field(default=64, ge=8, le=256)
    learningRate: float = Field(default=0.0006, gt=0.0, le=0.01)
    numWorkers: int = Field(default=0, ge=0, le=8)
    runName: str = ""
    evaluateAfterTrain: bool = True


class EvaluateJobRequest(BaseModel):
    runId: str
    dataDir: str = str(DEFAULT_DATASET_DIR)
    split: Literal["train", "val"] = "val"
    batchSize: int = Field(default=64, ge=8, le=256)
    numWorkers: int = Field(default=0, ge=0, le=8)


class SimulateRequest(BaseModel):
    scenario: Literal["recon-coverage", "jam-recovery", "multi-target-allocation"] = "jam-recovery"
    steps: int = Field(default=260, ge=30, le=1200)
    agents: int = Field(default=32, ge=4, le=240)
    packetLoss: float = Field(default=0.25, ge=0.0, le=0.95)
    runId: str = ""


@dataclass
class JobRecord:
    id: str
    type: str
    status: str
    params: Dict[str, Any]
    createdAt: float
    startedAt: float | None = None
    finishedAt: float | None = None
    error: str | None = None
    logs: List[str] = field(default_factory=list)
    runId: str | None = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def appendLog(self, line: str) -> None:
        self.logs.append(line.rstrip())
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, JobRecord] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.lock = asyncio.Lock()

    async def submit(self, jobType: str, params: Dict[str, Any]) -> JobRecord:
        jobId = f"job-{uuid.uuid4().hex[:10]}"
        record = JobRecord(
            id=jobId,
            type=jobType,
            status="queued",
            params=params,
            createdAt=time.time(),
        )
        async with self.lock:
            self.jobs[jobId] = record
        await self.queue.put(jobId)
        return record

    async def get(self, jobId: str) -> JobRecord:
        async with self.lock:
            job = self.jobs.get(jobId)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {jobId}")
        return job

    async def runWorker(self) -> None:
        while True:
            jobId = await self.queue.get()
            job = await self.get(jobId)
            if job.status == "cancelled":
                self.queue.task_done()
                continue

            job.status = "running"
            job.startedAt = time.time()
            try:
                artifacts = await asyncio.to_thread(runJob, job)
                job.status = "succeeded"
                job.artifacts = artifacts
            except Exception as error:  # noqa: BLE001
                job.status = "failed"
                job.error = str(error)
                job.appendLog(f"[error] {error}")
            finally:
                job.finishedAt = time.time()
                self.queue.task_done()


def pythonBin() -> str:
    venvPython = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venvPython.exists():
        return str(venvPython)
    return sys.executable


def safeName(text: str) -> str:
    lowered = text.strip().lower()
    tokens = [token for token in lowered.replace("_", "-").split("-") if token]
    if not tokens:
        return "run"
    return "-".join(tokens)


def ensureDir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def runCommand(command: List[str], cwd: Path, job: JobRecord) -> None:
    job.appendLog("$ " + " ".join(shlex.quote(item) for item in command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    )
    assert process.stdout is not None
    for line in process.stdout:
        job.appendLog(line.rstrip())
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")


def detectPathsByRun(runId: str) -> Dict[str, Path]:
    runDir = RUNS_DIR / runId
    checkpoint = runDir / "checkpoints" / "tiny-cnn.pt"
    history = runDir / "checkpoints" / "tiny-cnn.history.json"
    summary = runDir / "checkpoints" / "tiny-cnn.summary.json"
    curve = runDir / "checkpoints" / "tiny-cnn.curve.png"
    classConfidence = runDir / "class-confidence.json"
    evalDir = runDir / "eval"
    sampleGrid = runDir / "sample-grid.png"
    return {
        "runDir": runDir,
        "checkpoint": checkpoint,
        "history": history,
        "summary": summary,
        "curve": curve,
        "classConfidence": classConfidence,
        "evalDir": evalDir,
        "sampleGrid": sampleGrid,
        "manifest": runDir / "artifacts.json",
    }


def renderArtifactManifest(runId: str) -> Dict[str, Any]:
    paths = detectPathsByRun(runId)

    def existsUrl(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        relative = path.relative_to(RUNS_DIR)
        return f"/artifacts/{relative.as_posix()}"

    payload = {
        "runId": runId,
        "checkpoint": existsUrl(paths["checkpoint"]),
        "history": existsUrl(paths["history"]),
        "summary": existsUrl(paths["summary"]),
        "trainingCurve": existsUrl(paths["curve"]),
        "classConfidence": existsUrl(paths["classConfidence"]),
        "sampleGrid": existsUrl(paths["sampleGrid"]),
        "evaluationSummary": existsUrl(paths["evalDir"] / "evaluation-summary.json"),
        "confusionMatrixCsv": existsUrl(paths["evalDir"] / "confusion-matrix.csv"),
        "confusionMatrixPng": existsUrl(paths["evalDir"] / "confusion-matrix.png"),
    }
    ensureDir(paths["runDir"])
    paths["manifest"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def runDatasetPrepare(job: JobRecord) -> Dict[str, Any]:
    params = DatasetPrepareRequest.model_validate(job.params)
    outputDir = resolvePath(params.outputDir)
    rawDir = resolvePath(params.rawDir)
    ensureDir(outputDir)
    ensureDir(rawDir)

    command = [
        pythonBin(),
        "data/prepare_visdrone.py",
        "--raw-dir",
        str(rawDir),
        "--output-dir",
        str(outputDir),
        "--split-mode",
        params.splitMode,
        "--subset-size-per-class",
        str(params.subsetSizePerClass),
        "--val-subset-size-per-class",
        str(params.valSubsetSizePerClass),
        "--val-ratio",
        str(params.valRatio),
        "--seed",
        str(params.seed),
    ]
    if params.trainImagesDir and params.trainAnnotationsDir:
        command.extend(
            [
                "--train-images-dir",
                str(resolvePath(params.trainImagesDir)),
                "--train-annotations-dir",
                str(resolvePath(params.trainAnnotationsDir)),
            ]
        )
    if params.valImagesDir and params.valAnnotationsDir:
        command.extend(
            [
                "--val-images-dir",
                str(resolvePath(params.valImagesDir)),
                "--val-annotations-dir",
                str(resolvePath(params.valAnnotationsDir)),
            ]
        )
    if params.sourceImagesDir:
        command.extend(["--source-images-dir", str(resolvePath(params.sourceImagesDir))])
    if params.sourceAnnotationsDir:
        command.extend(["--source-annotations-dir", str(resolvePath(params.sourceAnnotationsDir))])
    if params.download:
        command.append("--download")
        if params.downloadUrl:
            command.extend(["--download-url", params.downloadUrl])
    if params.archivePath:
        command.extend(["--archive-path", params.archivePath])

    runCommand(command, ML_DIR, job)

    manifestPath = outputDir / "manifest.json"
    if not manifestPath.exists():
        raise RuntimeError("Dataset preparation succeeded but manifest.json was not found.")

    payload = json.loads(manifestPath.read_text(encoding="utf-8"))
    return {
        "datasetDir": str(outputDir),
        "manifest": payload,
    }


def runTrain(job: JobRecord) -> Dict[str, Any]:
    params = TrainJobRequest.model_validate(job.params)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    runName = safeName(params.runName) if params.runName else "visdrone"
    runId = f"{timestamp}-{runName}"
    job.runId = runId

    paths = detectPathsByRun(runId)
    ensureDir(paths["checkpoint"].parent)

    dataDir = resolvePath(params.dataDir)
    if not (dataDir / "train").exists() or not (dataDir / "val").exists():
        raise RuntimeError(f"Dataset directory is incomplete: {dataDir}")

    trainCommand = [
        pythonBin(),
        "train.py",
        "--data-dir",
        str(dataDir),
        "--epochs",
        str(params.epochs),
        "--batch-size",
        str(params.batchSize),
        "--learning-rate",
        str(params.learningRate),
        "--num-workers",
        str(params.numWorkers),
        "--output",
        str(paths["checkpoint"]),
    ]
    runCommand(trainCommand, ML_DIR, job)

    inferCommand = [
        pythonBin(),
        "infer.py",
        "--checkpoint",
        str(paths["checkpoint"]),
        "--calibration-dir",
        str(dataDir / "val"),
        "--emit-class-confidence",
        str(paths["classConfidence"]),
    ]
    runCommand(inferCommand, ML_DIR, job)

    sampleGridCommand = [
        pythonBin(),
        "render_sample_grid.py",
        "--data-dir",
        str(dataDir),
        "--split",
        "val",
        "--output",
        str(paths["sampleGrid"]),
    ]
    runCommand(sampleGridCommand, ML_DIR, job)

    if params.evaluateAfterTrain:
        evalCommand = [
            pythonBin(),
            "evaluate.py",
            "--checkpoint",
            str(paths["checkpoint"]),
            "--data-dir",
            str(dataDir),
            "--split",
            "val",
            "--batch-size",
            str(params.batchSize),
            "--num-workers",
            str(params.numWorkers),
            "--output-dir",
            str(paths["evalDir"]),
        ]
        runCommand(evalCommand, ML_DIR, job)

    artifactManifest = renderArtifactManifest(runId)
    return {
        "runId": runId,
        "artifacts": artifactManifest,
    }


def runEvaluate(job: JobRecord) -> Dict[str, Any]:
    params = EvaluateJobRequest.model_validate(job.params)
    paths = detectPathsByRun(params.runId)

    if not paths["checkpoint"].exists():
        raise RuntimeError(f"Checkpoint not found for run: {params.runId}")

    command = [
        pythonBin(),
        "evaluate.py",
        "--checkpoint",
        str(paths["checkpoint"]),
        "--data-dir",
        str(resolvePath(params.dataDir)),
        "--split",
        params.split,
        "--batch-size",
        str(params.batchSize),
        "--num-workers",
        str(params.numWorkers),
        "--output-dir",
        str(paths["evalDir"]),
    ]
    runCommand(command, ML_DIR, job)

    artifactManifest = renderArtifactManifest(params.runId)
    job.runId = params.runId
    return {
        "runId": params.runId,
        "artifacts": artifactManifest,
    }


def runJob(job: JobRecord) -> Dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    if job.type == "dataset_prepare":
        return runDatasetPrepare(job)
    if job.type == "train":
        return runTrain(job)
    if job.type == "evaluate":
        return runEvaluate(job)
    raise RuntimeError(f"Unsupported job type: {job.type}")


def listRuns() -> List[Dict[str, Any]]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for runDir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not runDir.is_dir():
            continue
        runId = runDir.name
        manifestPath = runDir / "artifacts.json"
        if manifestPath.exists():
            manifest = json.loads(manifestPath.read_text(encoding="utf-8"))
        else:
            manifest = renderArtifactManifest(runId)

        summaryPath = runDir / "checkpoints" / "tiny-cnn.summary.json"
        summary = {}
        if summaryPath.exists():
            summary = json.loads(summaryPath.read_text(encoding="utf-8"))

        runs.append(
            {
                "runId": runId,
                "summary": summary,
                "artifacts": manifest,
            }
        )
    return runs


jobManager = JobManager()
app = FastAPI(title="Swarm ML Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensureDir(RUNS_DIR)
ensureDir(GENERATED_SCENARIO_DIR)

app.mount("/artifacts", StaticFiles(directory=str(RUNS_DIR)), name="artifacts")
app.mount("/generated", StaticFiles(directory=str(GENERATED_SCENARIO_DIR)), name="generated")


@app.on_event("startup")
async def startupEvent() -> None:
    asyncio.create_task(jobManager.runWorker())


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "queueSize": jobManager.queue.qsize(),
        "runsDir": str(RUNS_DIR),
    }


@app.post("/api/dataset/prepare")
async def prepareDataset(request: DatasetPrepareRequest) -> Dict[str, Any]:
    job = await jobManager.submit("dataset_prepare", request.model_dump())
    return {
        "jobId": job.id,
        "status": job.status,
    }


@app.post("/api/jobs/train")
async def submitTrainJob(request: TrainJobRequest) -> Dict[str, Any]:
    job = await jobManager.submit("train", request.model_dump())
    return {
        "jobId": job.id,
        "status": job.status,
    }


@app.post("/api/jobs/evaluate")
async def submitEvaluateJob(request: EvaluateJobRequest) -> Dict[str, Any]:
    job = await jobManager.submit("evaluate", request.model_dump())
    return {
        "jobId": job.id,
        "status": job.status,
    }


@app.get("/api/jobs/{jobId}")
async def queryJob(jobId: str) -> Dict[str, Any]:
    job = await jobManager.get(jobId)
    return {
        "jobId": job.id,
        "type": job.type,
        "status": job.status,
        "params": job.params,
        "createdAt": job.createdAt,
        "startedAt": job.startedAt,
        "finishedAt": job.finishedAt,
        "error": job.error,
        "runId": job.runId,
        "artifacts": job.artifacts,
        "logs": job.logs[-120:],
    }


@app.get("/api/runs")
async def runs() -> Dict[str, Any]:
    return {
        "runs": listRuns(),
    }


@app.get("/api/runs/{runId}/artifacts")
async def runArtifacts(runId: str) -> Dict[str, Any]:
    runPath = RUNS_DIR / runId
    if not runPath.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {runId}")

    manifestPath = runPath / "artifacts.json"
    if manifestPath.exists():
        payload = json.loads(manifestPath.read_text(encoding="utf-8"))
    else:
        payload = renderArtifactManifest(runId)

    return payload


@app.post("/api/simulate/compare")
async def simulateCompare(request: SimulateRequest) -> Dict[str, Any]:
    runId = request.runId.strip()
    classConfidencePath = ""
    if runId:
        candidate = detectPathsByRun(runId)["classConfidence"]
        if not candidate.exists():
            raise HTTPException(status_code=404, detail=f"class-confidence.json not found for run: {runId}")
        classConfidencePath = str(candidate)

    ensureDir(GENERATED_SCENARIO_DIR)
    scenarioFileName = f"{request.scenario}-compare-{int(time.time())}.json"
    outputPath = GENERATED_SCENARIO_DIR / scenarioFileName

    command = [
        pythonBin(),
        "simulate.py",
        "--compare",
        "--scenario",
        request.scenario,
        "--steps",
        str(request.steps),
        "--agents",
        str(request.agents),
        "--packet-loss",
        str(request.packetLoss),
        "--output",
        str(outputPath),
    ]
    if classConfidencePath:
        command.extend(["--ml-confidence", classConfidencePath])

    try:
        process = subprocess.run(
            command,
            cwd=str(SIM_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stdout or error.stderr or str(error)
        raise HTTPException(status_code=500, detail=detail) from error

    return {
        "scenario": request.scenario,
        "scenarioUrl": f"/generated/{scenarioFileName}",
        "logs": (process.stdout or "").splitlines()[-20:],
    }
