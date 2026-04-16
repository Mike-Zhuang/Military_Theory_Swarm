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


class JobCancelledError(RuntimeError):
    pass


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
    devValSizePerClass: int = Field(default=360, ge=60, le=5000)
    valSubsetSizePerClass: int = Field(default=0, ge=0, le=20000)
    valRatio: float = Field(default=0.2, gt=0.05, lt=0.5)
    useIgnoredAsDecoy: bool = False
    seed: int = 42


class TrainJobRequest(BaseModel):
    dataDir: str = str(DEFAULT_DATASET_DIR)
    epochs: int = Field(default=80, ge=1, le=200)
    batchSize: int = Field(default=64, ge=8, le=256)
    learningRate: float = Field(default=0.0003, gt=0.0, le=0.01)
    numWorkers: int = Field(default=0, ge=0, le=8)
    modelName: Literal["mobilenetv3-small", "tiny-cnn"] = "mobilenetv3-small"
    pretrained: bool = True
    freezeEpochs: int = Field(default=3, ge=0, le=50)
    weightDecay: float = Field(default=1e-4, ge=0.0, le=0.1)
    labelSmoothing: float = Field(default=0.05, ge=0.0, le=0.4)
    lossType: Literal["cross-entropy", "focal"] = "focal"
    focalGamma: float = Field(default=1.5, ge=0.0, le=5.0)
    scheduler: Literal["none", "cosine", "plateau"] = "cosine"
    earlyStopPatience: int = Field(default=8, ge=1, le=50)
    earlyStopMinDelta: float = Field(default=1e-3, ge=0.0, le=1.0)
    augmentLevel: Literal["light", "medium", "strong"] = "medium"
    imageSize: int = Field(default=128, ge=64, le=384)
    runName: str = ""
    evaluateAfterTrain: bool = True


class EvaluateJobRequest(BaseModel):
    runId: str
    dataDir: str = str(DEFAULT_DATASET_DIR)
    split: Literal["train", "dev-val", "official-val", "val"] = "official-val"
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
    cancelRequested: bool = False
    activeProcess: subprocess.Popen[str] | None = None

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

    async def cancel(self, jobId: str) -> JobRecord:
        async with self.lock:
            job = self.jobs.get(jobId)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job not found: {jobId}")
            if job.status in {"succeeded", "failed", "cancelled"}:
                return job

            job.cancelRequested = True
            job.appendLog("[cancel] 用户请求取消任务")
            if job.status == "queued":
                job.status = "cancelled"
                job.finishedAt = time.time()
                return job

            activeProcess = job.activeProcess

        if activeProcess and activeProcess.poll() is None:
            try:
                activeProcess.terminate()
            except Exception:  # noqa: BLE001
                pass
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
                if job.cancelRequested:
                    job.status = "cancelled"
                    job.appendLog("[cancel] 任务已取消")
                else:
                    job.status = "succeeded"
                    job.artifacts = artifacts
            except JobCancelledError as error:
                job.status = "cancelled"
                job.error = str(error)
                job.appendLog(f"[cancel] {error}")
            except Exception as error:  # noqa: BLE001
                job.status = "failed"
                job.error = str(error)
                job.appendLog(f"[error] {error}")
            finally:
                job.activeProcess = None
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
    job.activeProcess = process
    assert process.stdout is not None
    for line in process.stdout:
        if job.cancelRequested:
            try:
                process.terminate()
            except Exception:  # noqa: BLE001
                pass
            break
        job.appendLog(line.rstrip())

    if job.cancelRequested:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise JobCancelledError("Job cancelled by user")

    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")
    job.activeProcess = None


def detectPathsByRun(runId: str) -> Dict[str, Path]:
    runDir = RUNS_DIR / runId
    checkpointDir = runDir / "checkpoints"
    bestCheckpoint = checkpointDir / "best.pt"
    lastCheckpoint = checkpointDir / "last.pt"
    legacyCheckpoint = checkpointDir / "tiny-cnn.pt"
    checkpoint = bestCheckpoint if bestCheckpoint.exists() else (lastCheckpoint if lastCheckpoint.exists() else legacyCheckpoint)
    history = checkpointDir / "history.json"
    summary = checkpointDir / "summary.json"
    curve = checkpointDir / "curve.png"
    curveTrain = checkpointDir / "curve-train.png"
    curveLive = checkpointDir / "curve-live.png"
    curveLiveTrain = checkpointDir / "curve-live-train.png"
    liveMetrics = checkpointDir / "live-metrics.jsonl"
    progress = checkpointDir / "progress.json"
    classConfidence = runDir / "class-confidence.json"
    evalDir = runDir / "eval"
    evalOfficialDir = evalDir / "official-val"
    evalDevDir = evalDir / "dev-val"
    sampleGrid = runDir / "sample-grid.png"
    if not history.exists():
        history = checkpointDir / "tiny-cnn.history.json"
    if not summary.exists():
        summary = checkpointDir / "tiny-cnn.summary.json"
    if not curve.exists():
        curve = checkpointDir / "tiny-cnn.curve.png"
    return {
        "runDir": runDir,
        "checkpointDir": checkpointDir,
        "checkpoint": checkpoint,
        "bestCheckpoint": bestCheckpoint,
        "lastCheckpoint": lastCheckpoint,
        "history": history,
        "summary": summary,
        "curve": curve,
        "curveTrain": curveTrain,
        "curveLive": curveLive,
        "curveLiveTrain": curveLiveTrain,
        "liveMetrics": liveMetrics,
        "progress": progress,
        "classConfidence": classConfidence,
        "evalDir": evalDir,
        "evalOfficialDir": evalOfficialDir,
        "evalDevDir": evalDevDir,
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
        "bestCheckpoint": existsUrl(paths["bestCheckpoint"]),
        "lastCheckpoint": existsUrl(paths["lastCheckpoint"]),
        "history": existsUrl(paths["history"]),
        "summary": existsUrl(paths["summary"]),
        "trainingCurve": existsUrl(paths["curve"]),
        "trainingCurveTrainOnly": existsUrl(paths["curveTrain"]),
        "trainingCurveLive": existsUrl(paths["curveLive"]),
        "trainingCurveLiveTrainOnly": existsUrl(paths["curveLiveTrain"]),
        "liveMetrics": existsUrl(paths["liveMetrics"]),
        "progress": existsUrl(paths["progress"]),
        "classConfidence": existsUrl(paths["classConfidence"]),
        "sampleGrid": existsUrl(paths["sampleGrid"]),
        "evaluationSummary": existsUrl(paths["evalOfficialDir"] / "evaluation-summary.json") or existsUrl(paths["evalDir"] / "evaluation-summary.json"),
        "confusionMatrixCsv": existsUrl(paths["evalOfficialDir"] / "confusion-matrix.csv") or existsUrl(paths["evalDir"] / "confusion-matrix.csv"),
        "confusionMatrixPng": existsUrl(paths["evalOfficialDir"] / "confusion-matrix.png") or existsUrl(paths["evalDir"] / "confusion-matrix.png"),
        "devEvaluationSummary": existsUrl(paths["evalDevDir"] / "evaluation-summary.json"),
        "officialEvaluationSummary": existsUrl(paths["evalOfficialDir"] / "evaluation-summary.json"),
    }
    ensureDir(paths["runDir"])
    paths["manifest"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def readProgressPayload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def readLiveMetrics(path: Path, maxRows: int = 12) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for rawLine in path.read_text(encoding="utf-8").splitlines():
            if rawLine.strip():
                rows.append(json.loads(rawLine))
    except Exception:  # noqa: BLE001
        return []
    return rows[-maxRows:]


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
        "--dev-val-size-per-class",
        str(params.devValSizePerClass),
        "--val-subset-size-per-class",
        str(params.valSubsetSizePerClass),
        "--val-ratio",
        str(params.valRatio),
        "--seed",
        str(params.seed),
    ]
    if params.useIgnoredAsDecoy:
        command.append("--use-ignored-as-decoy")
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
    ensureDir(paths["checkpointDir"])

    dataDir = resolvePath(params.dataDir)
    monitorDir = dataDir / "dev-val" if (dataDir / "dev-val").exists() else dataDir / "val"
    officialDir = dataDir / "official-val" if (dataDir / "official-val").exists() else monitorDir
    if not (dataDir / "train").exists() or not monitorDir.exists():
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
        "--model-name",
        params.modelName,
        "--freeze-epochs",
        str(params.freezeEpochs),
        "--weight-decay",
        str(params.weightDecay),
        "--label-smoothing",
        str(params.labelSmoothing),
        "--loss-type",
        params.lossType,
        "--focal-gamma",
        str(params.focalGamma),
        "--scheduler",
        params.scheduler,
        "--early-stop-patience",
        str(params.earlyStopPatience),
        "--early-stop-min-delta",
        str(params.earlyStopMinDelta),
        "--augment-level",
        params.augmentLevel,
        "--image-size",
        str(params.imageSize),
        "--monitor-split",
        monitorDir.name,
        "--official-split",
        officialDir.name,
        "--output",
        str(paths["checkpointDir"] / "tiny-cnn.pt"),
        "--output-dir",
        str(paths["checkpointDir"]),
    ]
    if not params.pretrained:
        trainCommand.append("--no-pretrained")
    runCommand(trainCommand, ML_DIR, job)

    bestCheckpoint = paths["bestCheckpoint"] if paths["bestCheckpoint"].exists() else paths["checkpoint"]

    inferCommand = [
        pythonBin(),
        "infer.py",
        "--checkpoint",
        str(bestCheckpoint),
        "--calibration-dir",
        str(officialDir),
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
        officialDir.name,
        "--output",
        str(paths["sampleGrid"]),
    ]
    runCommand(sampleGridCommand, ML_DIR, job)

    if params.evaluateAfterTrain:
        evalCommand = [
            pythonBin(),
            "evaluate.py",
            "--checkpoint",
            str(bestCheckpoint),
            "--data-dir",
            str(dataDir),
            "--split",
            "dev-val" if monitorDir.name == "dev-val" else "val",
            "--batch-size",
            str(params.batchSize),
            "--num-workers",
            str(params.numWorkers),
            "--output-dir",
            str(paths["evalDevDir"]),
        ]
        runCommand(evalCommand, ML_DIR, job)

        officialEvalCommand = [
            pythonBin(),
            "evaluate.py",
            "--checkpoint",
            str(bestCheckpoint),
            "--data-dir",
            str(dataDir),
            "--split",
            officialDir.name,
            "--batch-size",
            str(params.batchSize),
            "--num-workers",
            str(params.numWorkers),
            "--output-dir",
            str(paths["evalOfficialDir"]),
        ]
        runCommand(officialEvalCommand, ML_DIR, job)

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

    outputDir = paths["evalDir"]
    if params.split in {"dev-val", "val"}:
        outputDir = paths["evalDevDir"]
    elif params.split == "official-val":
        outputDir = paths["evalOfficialDir"]

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
        str(outputDir),
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
    if job.cancelRequested:
        raise JobCancelledError("Job cancelled before execution")

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

        summaryPath = detectPathsByRun(runId)["summary"]
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


@app.get("/api/dataset/manifest")
async def datasetManifest() -> Dict[str, Any]:
    manifestPath = DEFAULT_DATASET_DIR / "manifest.json"
    if not manifestPath.exists():
        raise HTTPException(status_code=404, detail=f"Dataset manifest not found: {manifestPath}")
    payload = json.loads(manifestPath.read_text(encoding="utf-8"))
    return {
        "datasetDir": str(DEFAULT_DATASET_DIR),
        "manifest": payload,
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


@app.post("/api/jobs/{jobId}/cancel")
async def cancelJob(jobId: str) -> Dict[str, Any]:
    job = await jobManager.cancel(jobId)
    return {
        "jobId": job.id,
        "status": job.status,
        "cancelRequested": job.cancelRequested,
    }


@app.get("/api/jobs/{jobId}")
async def queryJob(jobId: str) -> Dict[str, Any]:
    job = await jobManager.get(jobId)
    progress: Dict[str, Any] = {}
    liveMetrics: List[Dict[str, Any]] = []
    bestSnapshot: Dict[str, Any] = {}
    if job.type == "train" and job.runId:
        paths = detectPathsByRun(job.runId)
        progress = readProgressPayload(paths["progress"])
        liveMetrics = readLiveMetrics(paths["liveMetrics"])
        summaryPath = paths["summary"]
        if summaryPath.exists():
            try:
                summaryPayload = json.loads(summaryPath.read_text(encoding="utf-8"))
                bestSnapshot = {
                    "bestEpoch": summaryPayload.get("bestEpoch"),
                    "bestDevValLoss": summaryPayload.get("bestDevValLoss"),
                    "bestDevValAcc": summaryPayload.get("bestDevValAcc"),
                    "bestValLoss": summaryPayload.get("bestValLoss"),
                    "bestValAcc": summaryPayload.get("bestValAcc"),
                    "officialValLoss": summaryPayload.get("officialValLoss"),
                    "officialValAcc": summaryPayload.get("officialValAcc"),
                    "lastValLoss": summaryPayload.get("lastValLoss"),
                    "lastValAcc": summaryPayload.get("lastValAcc"),
                }
            except Exception:  # noqa: BLE001
                bestSnapshot = {}
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
        "progress": progress,
        "liveMetrics": liveMetrics,
        "bestSnapshot": bestSnapshot,
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
