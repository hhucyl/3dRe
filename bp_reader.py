"""PipeSonar .bp recording indexer and decoder.

The layout is based on the structures shipped with PipeSonarSDK and verified
against the recordings in ``data``.  A recording is an ISX packet stream.  The
indexer keeps only packet offsets in memory; echo samples are decoded lazily.
"""

from __future__ import annotations

import bisect
import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


ISX = b"ISX"
FOOTER = b"\x56\x48\x08\x00\x00\x5a\x00\xfb"


class BpFormatError(ValueError):
    """Raised when a file contains no usable PipeSonar frames."""


@dataclass(frozen=True)
class PacketRef:
    offset: int
    data_offset: int
    data_size: int
    current: int
    total: int


@dataclass(frozen=True)
class EchoScan:
    timestamp: int
    angle_deg: float
    range_m: float
    total_sample_units: int
    packets: Tuple[PacketRef, ...]


@dataclass(frozen=True)
class ProfileScan:
    timestamp: int
    range_m: float
    total_sample_units: int
    packets: Tuple[PacketRef, ...]


@dataclass(frozen=True)
class SensorSample:
    timestamp: int
    compass_deg: float
    date_text: str = ""
    water_height: float = 0.0
    raw_fields: tuple = ()


@dataclass(frozen=True)
class PointCloud:
    """Down-sampled world-space sonar points and profile contours."""

    points: Tuple[Tuple[float, float, float, int], ...]
    contours: Tuple[Tuple[Tuple[float, float, float], ...], ...]
    min_height: float
    max_height: float
    interpolated: tuple = ()
    interpolation_note: str = ""


@dataclass(frozen=True)
class RingPoseCorrection:
    """Rigid pose predicted for one complete mechanical scan ring.

    ``timestamp`` is the midpoint acquisition time used to associate raw BP
    beams and contour packets with the nearest inferred ring.
    """

    timestamp: int
    local_x_m: float
    local_y_m: float
    roll_deg: float
    pitch_deg: float
    depth_correction_m: float
    yaw_correction_deg: float
    # Slice rotations use a constant orientation within each mechanical sweep.
    # None retains the legacy continuously interpolated pose convention.
    slice_start: Optional[int] = None


def _transform_polar_point(
    angle_deg: float,
    distance_m: float,
    water_height_m: float,
    heading_deg: float,
    correction: RingPoseCorrection,
) -> Tuple[float, float, float]:
    """Apply the predicted rigid ring pose in the viewer's ENU-like frame."""

    # The BP head angle and compass heading increase clockwise in the viewer
    # (X points right/east and Y points up/north). Convert the latter to a
    # conventional counter-clockwise Euler yaw for the rotation matrix.
    angle = math.radians(angle_deg)
    x = math.sin(angle) * distance_m
    y = math.cos(angle) * distance_m
    z = 0.0

    roll = math.radians(correction.roll_deg)
    pitch = math.radians(correction.pitch_deg)
    yaw = math.radians(-(heading_deg + correction.yaw_correction_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Rz(yaw) @ Ry(pitch) @ Rx(roll), matching the training reconstructor.
    world_x = (cy * cp * x + (cy * sp * sr - sy * cr) * y
               + (cy * sp * cr + sy * sr) * z)
    world_y = (sy * cp * x + (sy * sp * sr + cy * cr) * y
               + (sy * sp * cr - cy * sr) * z)
    world_z = -sp * x + cp * sr * y + cp * cr * z
    return (
        world_x + correction.local_x_m,
        world_y + correction.local_y_m,
        world_z + water_height_m + correction.depth_correction_m,
    )


def _u16(data: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u64(data: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _decode_angle(data: mmap.mmap, offset: int) -> float:
    # The MSB of the low position byte is the direction flag.  The remaining
    # 11 bits form a 0..799 head position, at 0.45 degrees per position.
    low = data[offset + 5]
    high = data[offset + 6]
    position = (high << 7) | (low & 0x7F)
    return (position * 0.45) % 360.0


def load_sensor_log(bp_path: str) -> List[SensorSample]:
    """Load the optional JSON-lines sidecar with the same base name."""

    txt_path = os.path.splitext(bp_path)[0] + ".txt"
    samples: List[SensorSample] = []
    if not os.path.isfile(txt_path):
        return samples

    with open(txt_path, "r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                samples.append(
                    SensorSample(
                        timestamp=int(item.get("currTime", 0)),
                        compass_deg=float(item.get("comPassAngle", 0.0)) % 360.0,
                        date_text=str(item.get("currDateStr", "")),
                        water_height=float(item.get("waterHeight", 0.0)),
                        raw_fields=tuple(item.items()),
                    )
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    samples.sort(key=lambda sample: sample.timestamp)
    return samples


class BpRecording:
    """Memory-mapped recording with a light-weight frame index."""

    def __init__(self, path: str, cancelled: Optional[Callable[[], bool]] = None) -> None:
        self._cancelled = cancelled
        self.path = os.path.abspath(path)
        self._stream = open(self.path, "rb")
        if os.fstat(self._stream.fileno()).st_size == 0:
            self._stream.close()
            raise BpFormatError("文件为空")
        self._map = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.echo_scans: List[EchoScan] = []
        self.echo_times: List[int] = []
        self.profile_scans: List[ProfileScan] = []
        self.sensor_samples = load_sensor_log(self.path)
        self._sensor_times = [sample.timestamp for sample in self.sensor_samples]
        self._profile_times: List[int] = []
        try:
            self._build_index()
            from acquisition_conditions import RangeChecks
            self._range_checks = RangeChecks(self)
            self.range_switches = self._range_checks.events
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None  # type: ignore[assignment]
        if getattr(self, "_stream", None) is not None:
            self._stream.close()
            self._stream = None  # type: ignore[assignment]

    def __enter__(self) -> "BpRecording":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        return len(self.echo_scans)

    @property
    def duration_ms(self) -> int:
        if len(self.echo_scans) < 2:
            return 0
        return max(0, self.echo_scans[-1].timestamp - self.echo_scans[0].timestamp)

    def _packet_end(self, start: int, next_start: int, is_final_packet: bool) -> int:
        """Return the exclusive raw-packet end, after stripping record footer."""

        end = next_start
        if is_final_packet and end - start >= 9:
            footer = self._map[end - 8 : end]
            # Firmware variants change bytes 5-6 of the record footer (0x5A
            # on recent captures and 0x00 on older ones).
            if footer[:5] == FOOTER[:5] and footer[-1:] == FOOTER[-1:]:
                end -= 8
        return end

    def _build_index(self) -> None:
        positions: List[int] = []
        cursor = 0
        size = len(self._map)
        while True:
            if len(positions) % 1024 == 0 and self._cancelled and self._cancelled():
                raise ValueError("已取消读取")
            cursor = self._map.find(ISX, cursor)
            if cursor < 0:
                break
            if cursor + 20 <= size and self._map[cursor + 3] in (0x10, 0x20):
                positions.append(cursor)
            cursor += 3

        if not positions:
            raise BpFormatError("文件中未找到有效的 ISX 声呐帧")

        pending_echo: List[PacketRef] = []
        pending_profile: List[PacketRef] = []
        pending_echo_key: Optional[Tuple[int, int, int]] = None
        pending_profile_key: Optional[Tuple[int, int]] = None

        for index, start in enumerate(positions):
            if index % 1024 == 0 and self._cancelled and self._cancelled():
                raise ValueError("已取消读取")
            next_start = positions[index + 1] if index + 1 < len(positions) else size
            if next_start - start < 21:
                continue

            head_id = self._map[start + 3]
            frame_code = self._map[start + 4]
            total = frame_code >> 4
            current = frame_code & 0x0F
            if total < 1 or current < 1 or current > total:
                continue

            timestamp = _u64(self._map, start + 12)
            range_m = self._map[start + 7] / 10.0
            current_units = _u16(self._map, start + 8)
            total_units = _u16(self._map, start + 10)
            packet_end = self._packet_end(start, next_start, current == total)
            if packet_end <= start + 20 or self._map[packet_end - 1] != 0xFC:
                continue

            if head_id == 0x20:
                # Echo samples are stored as one 8-bit intensity for each two
                # sample units.  Metadata precedes them, so locate from 0xFC.
                data_size = (current_units + 1) // 2
                data_offset = packet_end - 1 - data_size
                if data_offset < start + 20:
                    continue
                angle = _decode_angle(self._map, start)
                key = (int(round(angle * 100)), int(range_m * 10), total_units)
                if current == 1 or pending_echo_key != key:
                    pending_echo = []
                    pending_echo_key = key
                pending_echo.append(PacketRef(start, data_offset, data_size, current, total))
                if current == total and len(pending_echo) == total:
                    self.echo_scans.append(
                        EchoScan(timestamp, angle, range_m, total_units, tuple(pending_echo))
                    )
                    pending_echo = []
                    pending_echo_key = None

            elif head_id == 0x10:
                # Profile packets contain little-endian 16-bit range points.
                # current_units excludes the proprietary per-packet metadata.
                data_size = min(current_units, packet_end - 1 - (start + 20))
                data_size -= data_size % 2
                data_offset = packet_end - 1 - data_size
                key = (int(range_m * 10), total_units)
                if current == 1 or pending_profile_key != key:
                    pending_profile = []
                    pending_profile_key = key
                pending_profile.append(PacketRef(start, data_offset, data_size, current, total))
                if current == total and len(pending_profile) == total:
                    self.profile_scans.append(
                        ProfileScan(timestamp, range_m, total_units, tuple(pending_profile))
                    )
                    pending_profile = []
                    pending_profile_key = None

        if not self.echo_scans:
            raise BpFormatError("ISX 帧存在，但没有可解码的强度数据")

        self.echo_scans.sort(key=lambda scan: scan.timestamp)
        self.echo_times = [scan.timestamp for scan in self.echo_scans]
        self.profile_scans.sort(key=lambda scan: scan.timestamp)
        self._profile_times = [scan.timestamp for scan in self.profile_scans]

    def echo_data(self, scan: EchoScan) -> bytes:
        return b"".join(
            self._map[packet.data_offset : packet.data_offset + packet.data_size]
            for packet in scan.packets
        )

    def profile_distances(self, profile: ProfileScan) -> List[float]:
        raw = b"".join(
            self._map[packet.data_offset : packet.data_offset + packet.data_size]
            for packet in profile.packets
        )
        if len(raw) < 2 or profile.total_sample_units <= 0:
            return []
        values = struct.unpack("<{}H".format(len(raw) // 2), raw)
        scale = profile.range_m / profile.total_sample_units
        return [value * scale for value in values]

    def sensor_at(self, timestamp: int) -> Optional[SensorSample]:
        if not self.sensor_samples:
            return None
        index = bisect.bisect_left(self._sensor_times, timestamp)
        if index <= 0:
            return self.sensor_samples[0]
        if index >= len(self.sensor_samples):
            return self.sensor_samples[-1]
        before = self.sensor_samples[index - 1]
        after = self.sensor_samples[index]
        return before if timestamp - before.timestamp <= after.timestamp - timestamp else after

    def profile_at(self, timestamp: int) -> Optional[ProfileScan]:
        if not self.profile_scans:
            return None
        index = bisect.bisect_right(self._profile_times, timestamp) - 1
        if index < 0:
            return None
        profile = self.profile_scans[index]
        checks = getattr(self, '_range_checks', None)
        if checks is not None:
            event = bisect.bisect_right(checks.times, timestamp)-1
            if event >= 0 and profile.timestamp < checks.times[event]:
                return None
            if checks.profile_issue(profile):
                return None
        return profile

    def scan_index_at(self, timestamp: int) -> int:
        return max(
            0,
            min(len(self.echo_times) - 1, bisect.bisect_right(self.echo_times, timestamp) - 1),
        )

    def build_point_cloud(
        self,
        max_points: int = 60000,
        intensity_threshold: int = 32,
        ring_corrections: Optional[Sequence[RingPoseCorrection]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> PointCloud:
        """Create a reduced 3-D cloud with ``waterHeight`` as the base Z coordinate.

        Strong local echo maxima are retained instead of every range bin.  This
        preserves pipe-wall returns while keeping software projection smooth on
        machines without a dedicated OpenGL Python package. Slice angles are held
        constant within each sweep; legacy poses use continuous interpolation.
        Echoes and surfaces share the same sensor and rotation transform.
        """

        max_points = max(1000, int(max_points))
        intensity_threshold = max(0, min(255, int(intensity_threshold)))
        # Use exactly the same sensor timestamps and rotations as the mesh.
        from surface_reconstruction import _pose_interpolator, interpolate_sensor
        pose_at = _pose_interpolator(ring_corrections or ())

        def beam_transform(timestamp):
            height, heading, _ = interpolate_sensor(self, timestamp)
            rotation, translation = pose_at(timestamp)
            yaw = math.radians(-heading)
            cy, sy = math.cos(yaw), math.sin(yaw)

            def transform(angle_deg, distance):
                angle = math.radians(angle_deg)
                x, y, z = rotation @ (math.sin(angle) * distance, math.cos(angle) * distance, 0.0)
                return (cy * x - sy * y + translation[0],
                        sy * x + cy * y + translation[1], z + height + translation[2])
            return transform, height

        ray_budget = max(1, max_points // 10)
        scan_stride = max(1, math.ceil(len(self.echo_scans) / ray_budget))
        points: List[Tuple[float, float, float, int]] = []

        for scan_index in range(0, len(self.echo_scans), scan_stride):
            if cancelled and cancelled():
                raise ValueError("已取消三维叠加")
            scan = self.echo_scans[scan_index]
            echo = self.echo_data(scan)
            if len(echo) < 3:
                continue
            transform, water_height = beam_transform(scan.timestamp)

            # The adaptive floor follows changing gain while the user floor
            # rejects low-level water-column noise.
            probe_step = max(1, len(echo) // 128)
            probe = sorted(echo[::probe_step])
            adaptive = probe[min(len(probe) - 1, int(len(probe) * 0.86))]
            cutoff = max(intensity_threshold, adaptive)
            candidates: List[Tuple[int, int]] = []
            blind_zone = max(1, int(len(echo) * 0.04))
            for sample_index in range(blind_zone, len(echo) - 1):
                value = echo[sample_index]
                if value >= cutoff and value >= echo[sample_index - 1] and value >= echo[sample_index + 1]:
                    candidates.append((value, sample_index))
            # No more than ten returns per ray; favor the strongest echoes.
            candidates = sorted(candidates, reverse=True)[:10]
            for value, sample_index in candidates:
                distance = (sample_index + 0.5) / len(echo) * scan.range_m
                point = transform(scan.angle_deg, distance)
                points.append((*point, value))

        if len(points) > max_points:
            keep_stride = len(points) / max_points
            points = [points[int(index * keep_stride)] for index in range(max_points)]

        contours: List[Tuple[Tuple[float, float, float], ...]] = []
        from acquisition_conditions import RangeChecks
        checks = RangeChecks(self)
        profile_stride = max(1, math.ceil(len(self.profile_scans) / 120))
        for profile in self.profile_scans[::profile_stride]:
            if cancelled and cancelled():
                raise ValueError("已取消三维叠加")
            if checks.profile_issue(profile):
                continue
            transform, water_height = beam_transform(profile.timestamp)
            layer: List[Tuple[float, float, float]] = []
            for point_index, distance in enumerate(self.profile_distances(profile)):
                if distance <= 0.0 or distance > profile.range_m * 1.02:
                    layer.append((math.nan, math.nan, water_height))
                    continue
                angle_deg = point_index * 0.45
                layer.append(transform(angle_deg, distance))
            contours.append(tuple(layer))

        # Report only the height interval actually overlapping sonar frames;
        # sidecar logs may begin before or continue after the .bp recording.
        heights = [point[2] for point in points]
        if not heights:
            heights = [sample.water_height for sample in self.sensor_samples] or [0.0]
        return PointCloud(
            tuple(points),
            tuple(contours),
            min(heights),
            max(heights),
        )
