"""Confidence-weighted, oriented local SDF reconstruction of sonar walls.

Numerical dependencies are loaded only when reconstruction is requested. The
field is a truncated moving-least-squares approximation, not an exact Euclidean
SDF. Unknown space is masked and never used to close the observed surface.
Coordinates follow the existing viewer: Z = waterHeight, in metres.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from bp_reader import BpRecording, RingPoseCorrection


class ReconstructionCancelled(Exception):
    pass


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: Any
    faces: Any
    confidence: Any
    voxel_size: float
    observation_count: int
    median_residual: float
    note: str = ""
    interpolated_points: int = 0
    interpolated_cells: int = 0
    boundaries: tuple = ()
    spatial_interpolated_points: int = 0


def _check(cancelled: Optional[Callable[[], bool]]) -> None:
    if cancelled and cancelled():
        raise ReconstructionCancelled()


def interpolate_sensor(recording: BpRecording, timestamp: int):
    """Interpolate within the log; endpoint extrapolation has reduced weight."""
    samples = recording.sensor_samples
    if not samples:
        return 0.0, 0.0, 0.15
    index = bisect.bisect_left(recording._sensor_times, timestamp)
    if index == 0 or index == len(samples):
        sample = samples[0 if index == 0 else -1]
        age = abs(timestamp - sample.timestamp) / 1000.0
        return sample.water_height, sample.compass_deg, 1.0 / (1.0 + age * age)
    a, b = samples[index - 1], samples[index]
    gap = max(1, b.timestamp - a.timestamp)
    fraction = (timestamp - a.timestamp) / gap
    yaw_delta = (b.compass_deg - a.compass_deg + 180.0) % 360.0 - 180.0
    return (
        a.water_height + fraction * (b.water_height - a.water_height),
        (a.compass_deg + fraction * yaw_delta) % 360.0,
        min(1.0, 2000.0 / gap),
    )


def _pose_interpolator(corrections: Sequence[RingPoseCorrection]):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    # Duplicate ring timestamps must not be passed to Slerp.
    ordered = sorted({c.timestamp: c for c in corrections}.values(), key=lambda c: c.timestamp)
    if not ordered:
        return lambda timestamp: (np.eye(3), np.zeros(3))
    times = np.array([c.timestamp for c in ordered], dtype=float)
    translations = np.array([[c.local_x_m, c.local_y_m, c.depth_correction_m] for c in ordered])
    # Rz(-yaw_correction) @ Ry(pitch) @ Rx(roll), as in bp_reader.
    rotations = Rotation.from_euler(
        "xyz", [[c.roll_deg, c.pitch_deg, -c.yaw_correction_deg] for c in ordered], degrees=True
    )
    if all(c.slice_start is not None for c in ordered):
        starts = [c.slice_start for c in ordered]
        matrices = rotations.as_matrix()

        def slice_at(timestamp):
            index = max(0, bisect.bisect_right(starts, timestamp) - 1)
            return matrices[index], translations[index]

        return slice_at
    slerp = Slerp(times - times[0], rotations) if len(times) > 1 else None

    def at(timestamp):
        time = float(np.clip(timestamp, times[0], times[-1]))
        rotation = slerp([time - times[0]]).as_matrix()[0] if slerp else rotations.as_matrix()[0]
        translation = np.array([np.interp(time, times, translations[:, i]) for i in range(3)])
        return rotation, translation

    return at


def _select_candidates(candidates, cancelled=None, range_limits=None):
    """Sequence optimisation: reward clear peaks, allow genuine range steps."""
    import numpy as np

    selected = []
    start = 0
    while start < len(candidates):
        _check(cancelled)
        end = start + 1
        while end < len(candidates) and end - start < 1600:
            previous, current = candidates[end - 1], candidates[end]
            delta = abs((current[1] - previous[1] + 180) % 360 - 180)
            if current[0] - previous[0] > 1000 or delta > 3.0:
                break
            if range_limits is not None and not math.isclose(range_limits[end], range_limits[end-1], abs_tol=1e-6):
                break  # acquisition discontinuity must not force peak continuity
            end += 1
        costs = -np.log(np.maximum(candidates[start][3], 1e-6))
        back = []
        for i in range(start + 1, end):
            prev_ranges = candidates[i - 1][2]
            ranges, quality = candidates[i][2:4]
            scale = np.maximum(0.06, 0.035 * ranges)
            transition = np.minimum(np.abs(prev_ranges[:, None] - ranges[None, :]) / scale, 3.0)
            total = costs[:, None] + 0.65 * transition
            parents = total.argmin(axis=0)
            costs = total[parents, np.arange(len(ranges))] - np.log(np.maximum(quality, 1e-6))
            back.append(parents)
        choice = int(costs.argmin())
        block = []
        for i in range(end - 1, start - 1, -1):
            block.append((i, choice))
            if i > start:
                choice = int(back[i - start - 1][choice])
        selected.extend(reversed(block))
        start = end
    return selected


def _locally_supported_candidates(candidates, denoise=True, cancelled=None, range_limits=None):
    """Retain multiple coherent returns without a radius/shape prior.

    Range continuity is only evidence, not a hard boundary condition: strong
    peaks survive grazing-angle range changes along an open pipe sidewall.
    """
    import numpy as np

    primary = dict(_select_candidates(candidates, cancelled, range_limits))
    result = []
    for index, (timestamp, angle, ranges, scores) in enumerate(candidates):
        if index % 256 == 0:
            _check(cancelled)
        support = np.zeros(len(ranges), dtype=int)
        for offset in (-4, -2, -1, 1, 2, 4):
            neighbor = index + offset
            if not 0 <= neighbor < len(candidates):
                continue
            nt, na, nr, ns = candidates[neighbor]
            if abs(nt - timestamp) > 800 or abs((na - angle + 180) % 360 - 180) > 3:
                continue
            tolerance = np.maximum(0.04, 0.07 * ranges)
            good = ns >= max(0.08, ns.max() * 0.25)
            if good.any():
                matching = abs(ranges[:, None] - nr[good]) < tolerance[:, None]
                if range_limits is not None and not math.isclose(range_limits[index], range_limits[neighbor], abs_tol=1e-6):
                    common = min(range_limits[index], range_limits[neighbor])*.98
                    matching &= (ranges[:, None] < common) & (nr[good] < common)
                support += np.any(matching, axis=1)
        quality = scores * (0.4 + 0.6 * np.minimum(support / 3, 1))
        if denoise:
            keep = ((support >= 2) & (scores >= max(0.10, scores.max() * 0.30))) | (scores >= 0.80)
            first = primary[index]
            if scores[first] >= 0.25:
                keep[first] = True
        else:
            keep = scores >= max(0.06, scores.max() * 0.20)
        choices = np.flatnonzero(keep)
        choices = choices[np.argsort(-quality[choices])[:4]]
        for choice in choices:
            result.append((index, int(choice), float(quality[choice])))
    return result


def read_echo_candidates(recording, threshold=32, progress=None, cancelled=None, near_range=0.0,
                         range_fraction=1.0):
    """Keep competing peaks until scene-level evidence can resolve them."""
    import numpy as np
    from scipy.signal import find_peaks

    candidates = []
    for index, scan in enumerate(recording.echo_scans):
        if index % 256 == 0:
            _check(cancelled)
            if progress:
                progress(5 + int(25 * index / max(1, recording.frame_count)), "提取原始回波壁面")
        echo = np.frombuffer(recording.echo_data(scan), dtype=np.uint8).astype(float)
        if len(echo) < 5 or scan.range_m <= 0:
            continue
        smooth = np.convolve(echo, [0.25, 0.5, 0.25], mode="same")
        floor = float(np.percentile(smooth, 40))
        noise = max(2.0, float(np.median(np.abs(np.diff(smooth)))))
        peaks, properties = find_peaks(
            smooth, height=max(threshold, floor + 2 * noise),
            prominence=max(3.0, 2 * noise), distance=max(1, int(0.02 / scan.range_m * len(echo))),
        )
        good = (peaks >= max(2, int(near_range / scan.range_m * len(echo))))
        good &= peaks < min(len(echo) - 2, int(range_fraction * len(echo)))
        peaks = peaks[good]
        prominence = properties["prominences"][good]
        if not len(peaks):
            continue
        score = np.clip(prominence / (prominence + 4 * noise), 0, 1)
        score *= np.clip((smooth[peaks] - floor) / max(10.0, smooth.max() - floor), 0.1, 1)
        keep = np.argsort(score)[-8:]
        ranges = (peaks[keep] + 0.5) / len(echo) * scan.range_m
        candidates.append((scan.timestamp, scan.angle_deg, ranges, score[keep]))
    if len(candidates) < 30:
        raise ValueError("有效壁面回波过少，请降低回波阈值后重新生成。")
    return candidates


def extract_wall_observations(recording, threshold=32, corrections=(), progress=None, cancelled=None,
                              near_range=0.0, denoise=True, return_free_mask=False,
                              return_range_limits=False):
    """Retain locally supported surfaces, including open pipe sidewalls."""
    import numpy as np

    candidates = read_echo_candidates(recording, threshold, progress, cancelled, near_range=near_range)
    scan_limits = {(scan.timestamp, scan.angle_deg): scan.range_m for scan in recording.echo_scans}
    from acquisition_conditions import RangeChecks
    checks = RangeChecks(recording)
    acquisition_weights = {(s.timestamp, s.angle_deg): checks.echo_weight(s, len(recording.echo_data(s)))
                           for s in recording.echo_scans}
    selected = _locally_supported_candidates(candidates, denoise, cancelled,
                                            [scan_limits[t, a] for t, a, _, _ in candidates])
    nearest = {}
    for index, choice, _ in selected:
        nearest[index] = min(nearest.get(index, math.inf), candidates[index][2][choice])
    pose_at = _pose_interpolator(corrections)
    points, origins, weights, free_mask, range_limits = [], [], [], [], []
    last_index, beam_transform = None, None
    for position, (index, choice, quality) in enumerate(selected):
        if position % 256 == 0:
            _check(cancelled)
        timestamp, angle, ranges, scores = candidates[index]
        height, heading, time_weight = interpolate_sensor(recording, timestamp)
        if not all(math.isfinite(v) for v in (height, heading, time_weight)) or time_weight < 0.05:
            continue
        if index != last_index:
            rotation, translation = pose_at(timestamp)
            yaw = math.radians(-heading)
            heading_matrix = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                       [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            beam_transform = heading_matrix @ rotation
            last_index = index
        theta = math.radians(angle)
        local = np.array([math.sin(theta), math.cos(theta), 0]) * ranges[choice]
        origin = translation + np.array([0, 0, height])
        point = origin + beam_transform @ local
        if np.isfinite(point).all():
            points.append(point)
            origins.append(origin)
            weights.append(float(quality * time_weight * acquisition_weights[timestamp, angle]))
            # A far return must never carve away a nearer retained surface.
            free_mask.append(bool(ranges[choice] <= nearest[index] + 1e-8))
            range_limits.append(scan_limits[timestamp, angle])
    observations = (np.asarray(points).reshape(-1, 3), np.asarray(origins).reshape(-1, 3), np.asarray(weights))
    if return_free_mask:
        observations = (*observations, np.asarray(free_mask, dtype=bool))
    if return_range_limits:
        observations = (*observations, np.asarray(range_limits))
    return observations


def _oriented_samples(points, origins, confidence, voxel, cancelled=None, connection_radius=0.2,
                      denoise=True, return_indices=False, vertical_continuity=False):
    import numpy as np
    from scipy.spatial import cKDTree

    # Density balancing after all beams have been considered, independent of
    # the display point limit. Select a real observation, never average walls.
    cells = np.floor(points / (voxel * 0.65)).astype(np.int64)
    order = np.argsort(-confidence, kind="stable")
    _, first = np.unique(cells[order], axis=0, return_index=True)
    keep = order[first]
    points, origins, confidence = points[keep], origins[keep], confidence[keep]
    if len(points) < 20:
        raise ValueError("壁面有效点过少，无法估计局部曲面。")
    tree = cKDTree(points)
    k = min(48, len(points))
    normals = np.empty_like(points)
    radii = np.empty(len(points))
    quality = np.empty(len(points))
    for start in range(0, len(points), 2048):
        _check(cancelled)
        stop = min(start + 2048, len(points))
        distances, neighbors = tree.query(points[start:stop], k=k)
        # Cap support to keep distant walls/openings out of local fits.
        maximum_radius = max(voxel * 3, connection_radius)
        radius = np.clip(distances[:, -1] * 1.25, max(voxel * 3, maximum_radius * 0.65), maximum_radius)
        w = confidence[neighbors] * np.exp(-0.5 * (distances / (radius[:, None] * 0.5)) ** 2)
        w[distances > radius[:, None]] = 0
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        xyz = points[neighbors]
        center = np.sum(xyz * w[..., None], axis=1)
        delta = xyz - center[:, None]
        covariance = np.einsum("nki,nkj,nk->nij", delta, delta, w)
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, :, 0]
        # Refit a plane through the target's own neighborhood. A robust
        # residual weight keeps nearby corners/opposite walls from being
        # averaged into a bevel or into the middle of the pipe.
        residual = np.abs(np.einsum("nki,ni->nk", xyz - points[start:stop, None], normal))
        robust_w = w / (1 + (residual / max(voxel * 0.8, 0.015)) ** 4)
        robust_w /= np.maximum(robust_w.sum(axis=1, keepdims=True), 1e-12)
        center = np.sum(xyz * robust_w[..., None], axis=1)
        delta = xyz - center[:, None]
        covariance = np.einsum("nki,nkj,nk->nij", delta, delta, robust_w)
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, :, 0]
        ray = origins[start:stop] - points[start:stop]
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
        flip = np.einsum("ij,ij->i", normal, ray) < 0
        normal[flip] *= -1
        planarity = np.clip((values[:, 1] - values[:, 0]) / np.maximum(values[:, 2], 1e-12), 0, 1)
        # A scan of a long pipe often supplies lines, not a complete circular
        # section. Project the viewing direction perpendicular to the measured
        # line tangent; radial normals would falsely bend these walls.
        line_like = values[:, 1] < np.maximum((voxel * 0.15) ** 2, values[:, 2] * 0.025)
        tangent = vectors[:, :, 2]
        line_normal = ray - np.sum(ray * tangent, axis=1, keepdims=True) * tangent
        line_length = np.linalg.norm(line_normal, axis=1)
        usable_line = line_like & (line_length > 0.03)
        line_normal /= np.maximum(line_length[:, None], 1e-12)
        normal[usable_line] = line_normal[usable_line]
        planarity[usable_line] = 0.25
        unsupported = line_like & ~usable_line
        normal[unsupported] = ray[unsupported]
        planarity[unsupported] = 0.01
        contour_supported = np.zeros(stop - start, dtype=bool)
        if vertical_continuity:
            # Thin horizontal scans with range noise look planar to 3-D PCA:
            # it returns a vertical normal and creates horizontal ribbons.
            # Fit the within-slice contour tangent before assuming a floor.
            slice_w = w * (abs(xyz[:, :, 2] - points[start:stop, None, 2]) <= voxel * 0.75)
            slice_w /= np.maximum(slice_w.sum(axis=1, keepdims=True), 1e-12)
            xy = xyz[:, :, :2]
            xy_center = np.sum(xy * slice_w[..., None], axis=1)
            xy_delta = xy - xy_center[:, None]
            xy_cov = np.einsum("nki,nkj,nk->nij", xy_delta, xy_delta, slice_w)
            xy_values, xy_vectors = np.linalg.eigh(xy_cov)
            contour_normal = np.column_stack((xy_vectors[:, :, 0], np.zeros(stop - start)))
            contour_normal[np.sum(contour_normal * ray, axis=1) < 0] *= -1
            linearity = 1 - xy_values[:, 0] / np.maximum(xy_values[:, 1], 1e-12)
            contour_supported = (abs(normal[:, 2]) >= 0.8) & (linearity > 0.70)
            contour_supported &= (np.sum(slice_w > 0.01, axis=1) >= 4) & (xy_values[:, 1] > (voxel * 0.3) ** 2)
            normal[contour_supported] = contour_normal[contour_supported]
            planarity[contour_supported] = linearity[contour_supported] * 0.5
        normals[start:stop] = normal
        radii[start:stop] = radius
        quality[start:stop] = 0.15 + 0.85 * np.sqrt(planarity)
        if denoise:
            scattered = values[:, 0] / np.maximum(values.sum(axis=1), 1e-12) > 0.12
            sparse = np.sum(distances < radius[:, None], axis=1) < 4
            quality[start:stop][(scattered & ~contour_supported) | sparse] *= 0.08
    result = (points, origins, normals, confidence * quality, radii)
    return (*result, keep) if return_indices else result


def interpolate_vertical_samples(points, origins, normals, confidence, radii, voxel,
                                 max_gap=0.5, xy_radius=0.2, weight=0.3,
                                 progress=None, cancelled=None, max_new_points=200000, range_limits=None):
    """Bounded interpolation of compatible wall patches between observed Zs.

    No circular correspondence, no extrapolation, and no generated free-space
    rays. A conflicting observed layer stops a match instead of being skipped.
    Linear segments avoid overshoot; their normals constrain the later local
    SDF fit, yielding a continuous surface without a global shape model.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    _check(cancelled)
    if not (math.isfinite(max_gap) and max_gap > 0 and math.isfinite(xy_radius) and xy_radius > 0
            and math.isfinite(weight) and 0 < weight <= 0.5):
        raise ValueError("Z 插值的最大层距、横向匹配范围或插值权重无效。")
    original_count = len(points)
    def finish(result):
        return (*result, range_limits) if range_limits is not None else result

    if original_count == 0:
        return finish((points, origins, normals, confidence, radii, 0))
    original_tree = cKDTree(points)
    band_width = max(voxel, 0.015)
    band_id = np.floor((points[:, 2] - points[:, 2].min()) / band_width).astype(np.int64)
    bands = np.unique(band_id)
    members = [np.flatnonzero(band_id == band) for band in bands]
    trees = [cKDTree(points[ids, :2]) for ids in members]
    generated = []
    remaining_budget = min(max_new_points, max(1000, original_count * 6))
    for lower_index, lower_ids in enumerate(members[:-1]):
        _check(cancelled)
        if progress:
            progress(37 + int(5 * lower_index / max(1, len(members) - 1)), "沿 Z 匹配相邻壁面并插值")
        active = (confidence[lower_ids] >= 0.10) & (abs(normals[lower_ids, 2]) < 0.8)
        for upper_index in range(lower_index + 1, len(members)):
            _check(cancelled)
            if (bands[upper_index] - bands[lower_index] - 1) * band_width > max_gap:
                break
            indices = lower_ids[active]
            if not len(indices) or remaining_budget <= 0:
                break
            upper_ids = members[upper_index]
            distance, neighbor = trees[upper_index].query(points[indices, :2], k=[1, 2, 3, 4])
            exists = np.isfinite(distance) & (distance <= xy_radius)
            safe = np.minimum(neighbor, len(upper_ids) - 1)
            candidate_ids = upper_ids[safe]
            delta = points[candidate_ids] - points[indices, None]
            dz = delta[:, :, 2]
            angle_agreement = np.einsum("nki,ni->nk", normals[candidate_ids], normals[indices])
            compatible = exists & (dz >= voxel * 1.5) & (dz <= max_gap + 1e-9)
            compatible &= (confidence[candidate_ids] >= 0.10) & (angle_agreement >= 0.8)
            compatible &= abs(normals[candidate_ids, 2]) < 0.8
            # Prefer short lateral moves and agreeing surface directions.
            cost = distance + xy_radius * (1 - angle_agreement)
            cost[~compatible] = np.inf
            best = cost.argmin(axis=1)
            found = np.isfinite(cost[np.arange(len(indices)), best])
            chosen_lower = indices[found]
            chosen_upper = candidate_ids[np.arange(len(indices)), best][found]
            chosen_cost = cost[np.arange(len(indices)), best][found]
            # One-to-one correspondence within this pair of bands prevents
            # many lower points collapsing onto a single upper observation.
            ordering = np.argsort(chosen_cost, kind="stable")
            _, first = np.unique(chosen_upper[ordering], return_index=True)
            selection = ordering[first]
            a, b = chosen_lower[selection], chosen_upper[selection]
            if len(a):
                direction = points[b] - points[a]
                length = np.linalg.norm(direction, axis=1)
                direction /= np.maximum(length[:, None], 1e-12)
                normal = normals[a] + normals[b]
                normal -= np.sum(normal * direction, axis=1, keepdims=True) * direction
                normal_length = np.linalg.norm(normal, axis=1)
                usable = normal_length > 0.1
                a, b, normal = a[usable], b[usable], normal[usable]
                normal /= normal_length[usable, None]
                steps = np.ceil((points[b, 2] - points[a, 2]) / (voxel * 0.8)).astype(int)
                for step in range(1, int(steps.max(initial=1))):
                    _check(cancelled)
                    keep = step < steps
                    aa, bb = a[keep], b[keep]
                    fraction = (step / steps[keep])[:, None]
                    xyz = points[aa] * (1 - fraction) + points[bb] * fraction
                    source_distance, _ = original_tree.query(xyz)
                    missing = source_distance > voxel * 0.65
                    if not missing.any():
                        continue
                    source = origins[aa] * (1 - fraction) + origins[bb] * fraction
                    if range_limits is not None:
                        # A range limit is visibility metadata, never a wall
                        # return. A changing limit is bounded by both scans.
                        limit = np.minimum(range_limits[aa], range_limits[bb])
                        missing &= np.linalg.norm(xyz - source, axis=1) <= limit
                    q = np.minimum(confidence[aa], confidence[bb]) * weight
                    support = np.minimum(radii[aa], radii[bb])
                    ids = np.flatnonzero(missing)[:remaining_budget]
                    block = (xyz[ids], source[ids], normal[keep][ids], q[ids], support[ids])
                    if range_limits is not None:
                        block = (*block, limit[ids])
                    generated.append(block)
                    remaining_budget -= len(ids)
                    if remaining_budget <= 0:
                        break
            # A measured incompatible layer in this XY neighborhood is a
            # boundary, not permission to jump to a more distant matching one.
            separated_layer = np.any(exists & (dz >= voxel * 1.5), axis=1)
            stopped = indices[separated_layer]
            active[np.isin(lower_ids, stopped)] = False
        if remaining_budget <= 0:
            break
    if not generated:
        return finish((points, origins, normals, confidence, radii, 0))
    arrays = [np.concatenate([block[i] for block in generated]) for i in range(len(generated[0]))]
    # Bound density independently of the number of matching pairs. Generated
    # samples remain lower-confidence support, not additional measurements.
    order = np.argsort(-arrays[3], kind="stable")
    cells = np.floor(arrays[0][order] / (voxel * 0.65)).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    keep = order[first]
    result = tuple(np.concatenate((original, extra[keep])) for original, extra in
                   zip((points, origins, normals, confidence, radii), arrays))
    _check(cancelled)
    if range_limits is not None:
        return (*result, len(keep), np.r_[range_limits, arrays[5][keep]])
    return (*result, len(keep))


def interpolate_vertical_field(field, known, confidence, normals, voxel, max_gap,
                               xy_radius, weight, range_margin=None, progress=None, cancelled=None,
                               source_height=None):
    """Shape-preserving cubic interpolation of bounded SDF column gaps.

    Fill surface support itself, not just isolated correspondence samples.
    Only existing anchors are used (never recursively generated anchors).
    Opposite-facing walls and large transverse steps split the fit. A range
    boundary only masks cells; assigning a sign to it would manufacture a cap.
    """
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    columns = field.reshape(-1, field.shape[-1])
    masks = known.reshape(columns.shape)
    quality = confidence.reshape(columns.shape)
    directions = normals.reshape(*columns.shape, 3)
    margins = range_margin.reshape(columns.shape) if range_margin is not None else None
    heights = source_height.reshape(columns.shape) if source_height is not None else None
    filled = 0
    for column in range(len(columns)):
        if column % 256 == 0:
            _check(cancelled)
            if progress:
                progress(76 + int(5 * column / max(1, len(columns))), "沿 Z 连续拟合距离场（保留量程开边界）")
        ids = np.flatnonzero(masks[column])
        if len(ids) < 2 or not np.any(np.diff(ids) > 1):
            continue
        values = columns[column, ids]
        ns = directions[column, ids]
        gaps = np.diff(ids)
        agreement = np.sum(ns[:-1] * ns[1:], axis=1)
        compatible = (gaps * voxel <= max_gap + 1e-9) & (agreement >= 0.8)
        if heights is not None:
            compatible &= abs(np.diff(heights[column, ids])) <= max_gap + 1e-9
        compatible &= (abs(ns[:-1, 2]) < 0.8) & (abs(ns[1:, 2]) < 0.8)
        compatible &= abs(np.diff(values)) <= xy_radius
        # Split at conflicting observations before estimating cubic slopes.
        boundaries = np.r_[0, np.flatnonzero(~compatible) + 1, len(ids)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            anchors = ids[left:right]
            if len(anchors) < 2 or not np.any(np.diff(anchors) > 1):
                continue
            targets = np.arange(anchors[0] + 1, anchors[-1])
            targets = targets[~masks[column, targets]]
            if not len(targets):
                continue
            upper = np.searchsorted(ids, targets)
            lower = upper - 1
            t = (targets - ids[lower]) / (ids[upper] - ids[lower])
            if margins is not None:
                # No extrapolation into space outside either endpoint's
                # acquisition range. Interpolate the visibility margin only.
                margin = margins[column, ids[lower]] * (1 - t) + margins[column, ids[upper]] * t
                visible = (margin >= 0) & ~(np.isfinite(margins[column, targets]) & (margins[column, targets] < 0))
                targets, upper, lower, t = (a[visible] for a in (targets, upper, lower, t))
                if not len(targets):
                    continue
            columns[column, targets] = PchipInterpolator(anchors, values[left:right])(targets)
            quality[column, targets] = np.minimum(quality[column, ids[lower]], quality[column, ids[upper]]) * weight
            masks[column, targets] = True
            filled += len(targets)
    _check(cancelled)
    return filled


def reconstruct_surface(points, origins, confidence, voxel_size=0.05, progress=None, cancelled=None,
                        max_grid_cells=2_000_000, connection_radius=0.2, denoise=True, free_mask=None,
                        z_interpolation=False, z_max_gap=0.5, z_weight=0.3, range_limits=None, synthetic_mask=None):
    """Fuse local signed planes with measured free-ray vetoes, then mesh.

No padding is classified as material; an all-corners-known cell mask prevents
Marching Cubes from inventing surfaces at unknown/known transitions.
"""
    import numpy as np
    from scipy.ndimage import gaussian_filter, map_coordinates
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from skimage.measure import marching_cubes

    _check(cancelled)
    points, origins, confidence = (np.asarray(a, dtype=float) for a in (points, origins, confidence))
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("网格尺寸必须为正数。")
    if points.ndim != 2 or points.shape[1:] != (3,) or origins.shape != points.shape or confidence.shape != (len(points),):
        raise ValueError("壁面观测格式无效。")
    valid = np.isfinite(points).all(axis=1) & np.isfinite(origins).all(axis=1) & np.isfinite(confidence) & (confidence > 0)
    if free_mask is None:
        free_mask = np.ones(len(points), dtype=bool)
    free_mask = np.asarray(free_mask, dtype=bool)
    if free_mask.shape != (len(points),):
        raise ValueError("自由空间观测标记格式无效。")
    free_mask = free_mask[valid]
    if synthetic_mask is None:
        synthetic_mask = np.zeros(len(points), dtype=bool)
    synthetic_mask = np.asarray(synthetic_mask, dtype=bool)
    if synthetic_mask.shape != (len(points),):
        raise ValueError("空间插值观测标记格式无效。")
    synthetic_mask = synthetic_mask[valid]
    free_mask &= ~synthetic_mask
    spatial_count = int(synthetic_mask.sum())
    if range_limits is not None:
        range_limits = np.asarray(range_limits, dtype=float)
        if range_limits.shape != (len(points),) or not np.all(np.isfinite(range_limits) & (range_limits > 0)):
            raise ValueError("声束量程格式无效。")
        range_limits = range_limits[valid]
    points, origins, confidence = points[valid], origins[valid], confidence[valid]
    observation_count = int((~synthetic_mask).sum())
    if observation_count < 30:
        raise ValueError("可用壁面点过少，请调整回波阈值。")
    confidence = np.clip(confidence, 0, 1)
    voxel = float(voxel_size)
    if not math.isfinite(connection_radius) or connection_radius <= 0 or max_grid_cells < 512:
        raise ValueError("局部连接半径或网格预算无效。")
    extent = np.ptp(points, axis=0)
    # Budget includes padding. Recompute until the allocation really fits.
    while np.prod(np.ceil(extent / voxel).astype(np.int64) + 7, dtype=np.int64) > max_grid_cells:
        voxel *= 1.15
    note = ""
    if voxel > voxel_size * 1.001:
        note = f"已按内存预算调整网格至 {voxel * 100:.1f} cm"
    if progress:
        progress(35, "估计跨圈壁面法向")
    points, origins, normals, confidence, radii, source_indices = _oriented_samples(
        points, origins, confidence, voxel, cancelled, connection_radius, denoise, return_indices=True,
        vertical_continuity=z_interpolation)
    free_mask = free_mask[source_indices]
    synthetic_mask = synthetic_mask[source_indices]
    if range_limits is not None:
        range_limits = range_limits[source_indices]
    observed_points, observed_normals = points[~synthetic_mask], normals[~synthetic_mask]
    observed_tree = cKDTree(observed_points)
    interpolated_count = 0
    if z_interpolation:
        interpolated = interpolate_vertical_samples(
            points, origins, normals, confidence, radii, voxel, max_gap=z_max_gap,
            xy_radius=connection_radius, weight=z_weight, progress=progress, cancelled=cancelled,
            range_limits=range_limits)
        points, origins, normals, confidence, radii, interpolated_count = interpolated[:6]
        if range_limits is not None:
            range_limits = interpolated[6]
        free_mask = np.r_[free_mask, np.zeros(interpolated_count, dtype=bool)]
    synthetic = np.r_[synthetic_mask, np.ones(interpolated_count, dtype=bool)]
    tree = cKDTree(points)
    lower = points.min(axis=0) - voxel * 3
    shape = np.ceil((points.max(axis=0) - lower) / voxel).astype(int) + 4
    total = int(np.prod(shape))
    field = np.zeros(total, dtype=np.float32)
    weight_grid = np.zeros(total, dtype=np.float32)
    known = np.zeros(total, dtype=bool)
    normal_grid = np.zeros((total, 3), dtype=np.float32) if z_interpolation else None
    source_height = np.zeros(total, dtype=np.float32) if z_interpolation else None
    range_margin = np.full(total, np.nan, dtype=np.float32) if range_limits is not None else None
    k = min(16, len(points))
    for start in range(0, total, 8192):
        _check(cancelled)
        stop = min(start + 8192, total)
        if progress and start % (8192 * 8) == 0:
            progress(43 + int(32 * start / total), "融合法向约束距离场")
        grid = np.column_stack(np.unravel_index(np.arange(start, stop), tuple(shape)))
        xyz = lower + grid * voxel
        distance, index = tree.query(xyz, k=k)
        delta = xyz[:, None] - points[index]
        signed = np.einsum("nki,nki->nk", delta, normals[index])
        nearest_normal = normals[index[:, 0]]
        agreement = np.maximum(0, np.einsum("nki,ni->nk", normals[index], nearest_normal)) ** 4
        support = radii[index]
        w = confidence[index] * agreement * np.exp(-2 * (distance / support) ** 2)
        w[(distance > support) | (np.abs(signed) > 4 * voxel)] = 0
        if range_limits is not None:
            margin = range_limits[index] - np.linalg.norm(xyz[:, None] - origins[index], axis=2)
            significant = w > np.maximum(w.max(axis=1, keepdims=True) * 0.1, 1e-9)
            local_margin = np.max(np.where(significant, margin, -np.inf), axis=1)
            local_margin[~significant.any(axis=1)] = np.nan
            range_margin[start:stop] = local_margin
            w[margin < 0] = 0
        sum_w = w.sum(axis=1)
        value = (w * np.clip(signed, -4 * voxel, 4 * voxel)).sum(axis=1) / np.maximum(sum_w, 1e-12)
        supported = (sum_w > 0.025) & (distance[:, 0] < radii[index[:, 0]] * 0.8)
        # Interpolate between local observations, but do not extrude a wall
        # far beyond the last observed slice/range. Only mutually compatible
        # neighbors contribute to these local support bounds.
        contributing = w > np.maximum(w.max(axis=1, keepdims=True) * 0.1, 1e-9)
        local_low = np.where(contributing[..., None], points[index], np.inf).min(axis=1)
        local_high = np.where(contributing[..., None], points[index], -np.inf).max(axis=1)
        supported &= ((xyz >= local_low - voxel * 1.25) & (xyz <= local_high + voxel * 1.25)).all(axis=1)
        field[start:stop] = value
        weight_grid[start:stop] = np.clip(sum_w / 3, 0, 1)
        if z_interpolation:
            direction = np.sum(w[..., None] * normals[index], axis=1)
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
            normal_grid[start:stop] = direction
            source_height[start:stop] = (w * points[index, 2]).sum(axis=1) / np.maximum(sum_w, 1e-12)
        if interpolated_count or spatial_count:
            synthetic_share = (w * synthetic[index]).sum(axis=1) / np.maximum(sum_w, 1e-12)
            weight_grid[start:stop] *= 1 - 0.7 * synthetic_share
        known[start:stop] = supported

    field = field.reshape(tuple(shape))
    weight_grid = weight_grid.reshape(tuple(shape))
    known = known.reshape(tuple(shape))
    interpolated_cells = 0
    if z_interpolation:
        interpolated_cells = interpolate_vertical_field(
            field, known, weight_grid, normal_grid.reshape(*shape, 3), voxel,
            z_max_gap, connection_radius, z_weight, range_margin, progress, cancelled, source_height)
    _check(cancelled)
    # Normalized smoothing only inside already observed support.
    weights = weight_grid * known
    smoothed_weight = gaussian_filter(weights, 0.35)
    field = gaussian_filter(field * weights, 0.35) / np.maximum(smoothed_weight, 1e-9)

    # Trusted direct returns contribute free-space samples. Use a narrow tube
    # around each center ray; this is explicitly a finite-beam approximation.
    free = np.zeros(tuple(shape), dtype=np.float32)
    ray_length = np.linalg.norm(points - origins, axis=1)
    for start in range(0, len(points), 256):
        _check(cancelled)
        for i in range(start, min(start + 256, len(points))):
            if not free_mask[i] or confidence[i] < 0.35 or ray_length[i] < voxel * 5:
                continue
            steps = min(2048, max(2, int(ray_length[i] / voxel)))
            distance = np.linspace(voxel, ray_length[i] - voxel * 3, steps)
            xyz = origins[i] + distance[:, None] * (points[i] - origins[i]) / ray_length[i]
            cells = np.rint((xyz - lower) / voxel).astype(int)
            inside = ((cells >= 0) & (cells < shape)).all(axis=1)
            cells = cells[inside]
            np.maximum.at(free, tuple(cells.T), confidence[i])
    # A contradictory region stays unknown rather than creating a new surface
    # along the edge of a carved ray tube.
    contradiction = (free > 0.15) & (field < voxel)
    known[contradiction] = False
    _check(cancelled)
    if progress:
        progress(83, "提取连续三角网格")
    if not known.any() or not (np.any(field[known] < 0) and np.any(field[known] > 0)):
        raise ValueError("没有形成有效壁面距离场，请调整回波阈值或网格尺寸。")
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(voxel,) * 3, mask=known, allow_degenerate=False)
    # Conservatively accept triangles only in cells with all 8 known corners.
    centers = vertices[faces].mean(axis=1) / voxel
    cells = np.clip(np.floor(centers).astype(int), 0, shape - 2)
    supported_faces = np.ones(len(faces), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                supported_faces &= known[cells[:, 0] + dx, cells[:, 1] + dy, cells[:, 2] + dz]
    faces = faces[supported_faces]
    if not len(faces):
        raise ValueError("有效曲面为空，请增大网格尺寸或降低回波阈值。")
    # Remove only tiny disconnected fragments, retaining separate real walls.
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)))
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels[faces[:, 0]])
    faces = faces[counts[labels[faces[:, 0]]] >= 3]
    used, inverse = np.unique(faces, return_inverse=True)
    vertices = vertices[used]
    faces = inverse.reshape(-1, 3).astype(np.int32)
    vertex_confidence = map_coordinates(weight_grid, (vertices / voxel).T, order=1, mode="nearest")
    vertices += lower
    # Face winding follows normals toward the measured water/free-space side.
    _, nearest = tree.query(vertices[faces].mean(axis=1))
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.sum(face_normals * normals[nearest], axis=1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    # In-sample point-to-plane residual, not a claim of real-world accuracy.
    _, nearest = observed_tree.query(vertices)
    residual = np.abs(np.sum((vertices - observed_points[nearest]) * observed_normals[nearest], axis=1))
    _check(cancelled)
    if progress:
        progress(100, "曲面重建完成")
    note = "局部开放曲面；保留非圆形、变径和管道侧壁" + (f"；{note}" if note else "")
    if z_interpolation:
        note += f"；Z 向插值 {interpolated_count:,} 个辅助点、{interpolated_cells:,} 个连续场网格（非实测）"
    if range_limits is not None:
        note += "；量程边界保持开放"
    if spatial_count:
        note += f"；使用空间插值 {spatial_count:,} 个辅助点（非实测）"
    return SurfaceMesh(vertices.astype(np.float32), faces, vertex_confidence.astype(np.float32), voxel,
                       observation_count, float(np.median(residual)), note, interpolated_count, interpolated_cells,
                       spatial_interpolated_points=spatial_count)


def build_surface(recording, threshold=32, corrections=(), voxel_size=0.05, progress=None, cancelled=None,
                  near_range=0.0, connection_radius=0.2, denoise=True,
                  z_interpolation=False, z_max_gap=0.5, z_weight=0.3, boundaries=(), observations=None):
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import skimage  # noqa: F401
    except ImportError as error:
        raise ValueError("曲面重建需要 numpy、scipy 和 scikit-image，请使用运行程序的 Python 安装 requirements.txt。") from error
    if observations is None:
        points, origins, weights, free_mask, range_limits = extract_wall_observations(
            recording, threshold, corrections, progress, cancelled, near_range, denoise,
            return_free_mask=True, return_range_limits=True)
    else:
        points, origins, weights, free_mask, range_limits = (observations.points, observations.origins,
            observations.confidence, observations.free_mask, observations.range_limits)
    from surface_segments import reconstruct_sections
    return reconstruct_sections(points, origins, weights, boundaries, voxel_size=voxel_size, progress=progress,
                               cancelled=cancelled, connection_radius=connection_radius,
                               denoise=denoise, free_mask=free_mask, z_interpolation=z_interpolation,
                               z_max_gap=z_max_gap, z_weight=z_weight, range_limits=range_limits,
                               synthetic_mask=observations.synthetic if observations is not None else None)


def export_mesh(mesh: SurfaceMesh, path: str) -> None:
    """Write the complete mesh, not the reduced interactive preview."""
    import os
    import tempfile

    suffix = os.path.splitext(path)[1].lower()
    if suffix not in (".obj", ".ply"):
        raise ValueError("模型格式必须为 OBJ 或 PLY。")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", delete=False,
                                         dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp") as stream:
            temporary = stream.name
            if suffix == ".ply":
                stream.write("ply\nformat ascii 1.0\ncomment units metres; z = waterHeight\n")
                stream.write(f"comment z_interpolated_support_points {mesh.interpolated_points}\n")
                stream.write(f"comment z_interpolated_field_cells {mesh.interpolated_cells}\n")
                stream.write(f"comment spatial_interpolated_points {mesh.spatial_interpolated_points}\n")
                stream.write("comment structure_boundaries_z_m " + " ".join(f"{v:g}" for v in mesh.boundaries) + "\n")
                stream.write(f"element vertex {len(mesh.vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty float confidence\n")
                stream.write(f"element face {len(mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n")
                for vertex, confidence in zip(mesh.vertices, mesh.confidence):
                    stream.write("{:.7g} {:.7g} {:.7g} {:.5g}\n".format(*vertex, confidence))
                for face in mesh.faces:
                    stream.write("3 {} {} {}\n".format(*face))
            else:
                stream.write("# PipeSonar reconstructed inner surface; metres; z = waterHeight\n")
                stream.write(f"# z_interpolated_support_points {mesh.interpolated_points}\n")
                stream.write(f"# z_interpolated_field_cells {mesh.interpolated_cells}\n")
                stream.write(f"# spatial_interpolated_points {mesh.spatial_interpolated_points}\n")
                stream.write("# structure_boundaries_z_m " + " ".join(f"{v:g}" for v in mesh.boundaries) + "\n")
                for vertex in mesh.vertices:
                    stream.write("v {:.7g} {:.7g} {:.7g}\n".format(*vertex))
                for face in mesh.faces:
                    stream.write("f {} {} {}\n".format(*(face + 1)))
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
