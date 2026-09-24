"""Adapter between real PipeSonar BP rings and the rigid-ring ML checkpoint.

The Qt application normally runs in a small PyQt-only environment, while the
trained model lives in ``3dML/sonar-ml`` and has its own PyTorch virtual
environment.  The public function in this module launches an isolated worker
and returns only the compact per-ring pose corrections to the GUI process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from bp_reader import BpRecording, RingPoseCorrection


PROJECT_DIR = Path(__file__).resolve().parent
SONAR_ML_ROOT = PROJECT_DIR.parent / "3dML" / "sonar-ml"
DEFAULT_CHECKPOINT = (
    SONAR_ML_ROOT / "runs" / "oneobj_energy_only_pose_v1" / "best.pt"
)
DEFAULT_MODEL_PYTHON = SONAR_ML_ROOT / ".venv" / "Scripts" / "python.exe"


class RigidModelError(RuntimeError):
    """Raised when the optional rigid-ring optimizer cannot run."""


@dataclass(frozen=True)
class RingSegment:
    indices: Tuple[int, ...]
    timestamp: int
    angular_coverage_deg: float


@dataclass(frozen=True)
class RigidOptimizationResult:
    corrections: Tuple[RingPoseCorrection, ...]
    checkpoint: str
    model_version: str
    device: str


def _signed_angle_delta(current: float, previous: float) -> float:
    return (current - previous + 180.0) % 360.0 - 180.0


def _scan_direction(angles: Sequence[float]) -> float:
    deltas = [
        _signed_angle_delta(current, previous)
        for previous, current in zip(angles, angles[1:])
    ]
    plausible = [delta for delta in deltas if 0.05 <= abs(delta) <= 2.0]
    if not plausible:
        return 1.0
    return 1.0 if statistics.median(plausible[:256]) >= 0.0 else -1.0


def segment_complete_rings(
    recording: BpRecording,
    minimum_coverage_deg: float = 330.0,
    maximum_step_deg: float = 2.0,
) -> List[RingSegment]:
    """Split acquisition-ordered beams into robust near-complete revolutions."""

    scans = recording.echo_scans
    if len(scans) < 2:
        return []
    direction = _scan_direction([scan.angle_deg for scan in scans[:1024]])
    segments: List[RingSegment] = []
    indices: List[int] = [0]
    coverage = 0.0
    previous_angle = scans[0].angle_deg

    def finish() -> None:
        nonlocal indices, coverage
        if indices and coverage >= minimum_coverage_deg:
            first_time = scans[indices[0]].timestamp
            last_time = scans[indices[-1]].timestamp
            segments.append(
                RingSegment(tuple(indices), (first_time + last_time) // 2, coverage)
            )
        indices = []
        coverage = 0.0

    for index in range(1, len(scans)):
        current_angle = scans[index].angle_deg
        forward = direction * _signed_angle_delta(current_angle, previous_angle)
        previous_angle = current_angle
        if -0.05 <= forward <= maximum_step_deg:
            indices.append(index)
            coverage += max(0.0, forward)
            if coverage >= 359.0:
                finish()
            continue

        # A discontinuity can be a truncated recording or corrupt/interleaved
        # packet sequence. Preserve a substantially complete ring and restart.
        finish()
        indices = [index]

    finish()
    return segments


def _resolve_worker_python() -> Path:
    configured = os.environ.get("SONAR_RIGID_MODEL_PYTHON")
    candidate = Path(configured) if configured else DEFAULT_MODEL_PYTHON
    if candidate.is_file():
        return candidate
    raise RigidModelError(
        "未找到环级刚性模型的 PyTorch 运行环境：\n"
        f"{candidate}\n\n"
        "请保留 3dML/sonar-ml/.venv，或设置 SONAR_RIGID_MODEL_PYTHON。"
    )


def optimize_recording(
    bp_path: str,
    checkpoint: os.PathLike[str] | str = DEFAULT_CHECKPOINT,
    timeout_seconds: int = 300,
    cancelled: Optional[Callable[[], bool]] = None,
) -> RigidOptimizationResult:
    """Run model inference out of process and decode per-ring corrections."""

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise RigidModelError(f"未找到环级刚性模型权重：\n{checkpoint_path}")
    worker_python = _resolve_worker_python()
    command = [
        str(worker_python),
        str(Path(__file__).resolve()),
        "--worker",
        "--bp",
        str(Path(bp_path).resolve()),
        "--checkpoint",
        str(checkpoint_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            creationflags=creation_flags,
        )
        deadline = time.monotonic() + max(10, int(timeout_seconds))
        try:
            while True:
                if cancelled and cancelled():
                    raise RigidModelError("已取消环级刚性模型推理。")
                if time.monotonic() >= deadline:
                    raise RigidModelError("环级刚性模型推理超时。")
                try:
                    stdout, stderr = process.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    except OSError as error:
        raise RigidModelError(f"无法启动环级刚性模型：{error}") from error
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip()
        raise RigidModelError(f"环级刚性模型推理失败：\n{detail}")
    try:
        payload = json.loads(stdout)
        corrections = tuple(
            RingPoseCorrection(
                timestamp=int(item["timestamp"]),
                local_x_m=float(item["local_x_m"]),
                local_y_m=float(item["local_y_m"]),
                roll_deg=float(item["roll_deg"]),
                pitch_deg=float(item["pitch_deg"]),
                depth_correction_m=float(item["depth_correction_m"]),
                yaw_correction_deg=float(item["yaw_correction_deg"]),
            )
            for item in payload["rings"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RigidModelError("环级刚性模型返回了无法解析的结果。") from error
    if not corrections:
        raise RigidModelError("没有找到覆盖角度足够的完整扫描环，无法运行环级模型。")
    return RigidOptimizationResult(
        corrections=corrections,
        checkpoint=str(payload.get("checkpoint", checkpoint_path)),
        model_version=str(payload.get("model_version", "unknown")),
        device=str(payload.get("device", "unknown")),
    )


def _nearest_scan_index(
    target_angle: float, indices: Iterable[int], recording: BpRecording
) -> int:
    return min(
        indices,
        key=lambda index: abs(
            _signed_angle_delta(recording.echo_scans[index].angle_deg, target_angle)
        ),
    )


def _build_model_inputs(recording: BpRecording, segments: Sequence[RingSegment]):
    # Imports stay inside the worker so launching the Qt application does not
    # require NumPy or PyTorch in its own interpreter.
    import numpy as np

    range_rows = 350
    angle_columns = 100
    training_min_range_m = 0.05
    training_max_range_m = 6.0
    target_ranges = np.linspace(
        training_min_range_m, training_max_range_m, range_rows, dtype=np.float32
    )
    target_angles = np.arange(angle_columns, dtype=np.float32) * (360.0 / angle_columns)
    energy = np.zeros((len(segments), range_rows, angle_columns), dtype=np.float32)
    nav = np.zeros((len(segments), 6), dtype=np.float32)

    for ring_index, segment in enumerate(segments):
        for column, target_angle in enumerate(target_angles):
            scan_index = _nearest_scan_index(target_angle, segment.indices, recording)
            scan = recording.echo_scans[scan_index]
            raw = np.frombuffer(recording.echo_data(scan), dtype=np.uint8).astype(np.float32)
            if raw.size == 0:
                continue
            source_ranges = (np.arange(raw.size, dtype=np.float32) + 0.5) / raw.size * scan.range_m
            near_to_far = np.interp(
                target_ranges, source_ranges, raw, left=0.0, right=0.0
            )
            # Stonefish training PGM rows are stored far-to-near.
            energy[ring_index, :, column] = near_to_far[::-1] / 255.0

        sensor = recording.sensor_at(segment.timestamp)
        measured_depth = sensor.water_height if sensor else 0.0
        measured_yaw = sensor.compass_deg if sensor else 0.0
        yaw_radians = math.radians(measured_yaw)
        progress = ring_index / max(1, len(segments) - 1)
        nav[ring_index] = (
            measured_depth / 6.0,
            math.sin(yaw_radians),
            math.cos(yaw_radians),
            0.5,  # Real BP files do not record gain/power; training default is 10/20.
            0.5,
            progress,
        )

    network_images = np.log1p(15.0 * energy) / math.log(16.0)
    return network_images[:, None, :, :], nav


def _worker(bp_path: str, checkpoint_path: str) -> dict:
    import torch

    sys.path.insert(0, str(SONAR_ML_ROOT))
    from sonar_recon.model import (  # pylint: disable=import-outside-toplevel
        PoseCorrectionNet,
        checkpoint_encoder_mode,
        load_checkpoint_model_state,
    )

    with BpRecording(bp_path) as recording:
        if not recording.sensor_samples:
            raise RigidModelError("同名 TXT 中没有可用的 waterHeight/comPassAngle 数据。")
        segments = segment_complete_rings(recording)
        if not segments:
            raise RigidModelError("没有找到覆盖角度达到 330° 的完整扫描环。")
        images, nav = _build_model_inputs(recording, segments)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_mode = checkpoint_encoder_mode(checkpoint)
    model = PoseCorrectionNet(encoder_mode=encoder_mode).to(device)
    load_checkpoint_model_state(model, checkpoint)
    model.eval()
    with torch.inference_mode():
        output = model(
            torch.from_numpy(images).unsqueeze(0).to(device),
            torch.from_numpy(nav).unsqueeze(0).to(device),
        )[0].cpu().numpy()

    rings = []
    for segment, values in zip(segments, output):
        rings.append(
            {
                "timestamp": segment.timestamp,
                "scan_count": len(segment.indices),
                "angular_coverage_deg": segment.angular_coverage_deg,
                "local_x_m": float(values[0]),
                "local_y_m": float(values[1]),
                "roll_deg": float(values[2]),
                "pitch_deg": float(values[3]),
                "depth_correction_m": float(values[4]),
                "yaw_correction_deg": float(values[5]),
            }
        )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "model_version": str(checkpoint.get("model_version", "unknown")),
        "device": str(device),
        "rings": rings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PipeSonar rigid-ring model adapter")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--bp", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    arguments = parser.parse_args()
    if not arguments.worker:
        parser.error("this command is intended to be launched with --worker")
    try:
        payload = _worker(arguments.bp, arguments.checkpoint)
    except Exception as error:  # Keep the subprocess protocol concise for Qt.
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
