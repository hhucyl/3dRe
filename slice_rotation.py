"""Fixed-origin sweep rotations and robust, local surface alignment.

Angles in degrees: a=pitch about Y, b=roll about X, c=clockwise heading
correction. World ray = Rz(-(compass+c)) Ry(a) Rx(b) local ray.
Only orientation is fitted; measured origins and ranges never move.
"""

from dataclasses import dataclass, replace
import math

from bp_reader import RingPoseCorrection
from surface_segments import validate_boundaries, section_ids


@dataclass(frozen=True)
class SliceRotationResult:
    corrections: tuple
    note: str = "手动角度"
    before_m: float = 0.0
    after_m: float = 0.0
    boundaries: tuple = ()
    segment_metrics: tuple = ()


def make_slice_corrections(recording):
    """Cover every beam, including incomplete first/last sweeps and reversals."""
    scans = recording.echo_scans
    if not scans:
        return ()
    groups, start, travel, direction = [], 0, 0.0, 0
    for i in range(1, len(scans)):
        delta = (scans[i].angle_deg - scans[i-1].angle_deg + 180) % 360 - 180
        sign = 1 if delta > 0 else -1 if delta < 0 else 0
        discontinuity = (abs(delta) > 3 or scans[i].timestamp - scans[i-1].timestamp > 5000
                         or (direction and sign and sign != direction))
        if travel >= 359.0 or discontinuity:
            groups.append((start, i-1))
            start, travel, direction = i, 0.0, 0
        else:
            travel += abs(delta)
            direction = sign or direction
    groups.append((start, len(scans)-1))
    return tuple(RingPoseCorrection(
        (scans[first].timestamp + scans[last].timestamp)//2, 0, 0, 0, 0, 0, 0,
        slice_start=scans[first].timestamp,
    ) for first, last in groups)


def validate_corrections(corrections):
    if not corrections:
        raise ValueError("没有可调整的声呐切片。")
    previous = -math.inf
    for c in corrections:
        if c.slice_start is None or c.slice_start <= previous:
            raise ValueError("切片起始时间必须严格递增。")
        previous = c.slice_start
        if any(v != 0 for v in (c.local_x_m, c.local_y_m, c.depth_correction_m)):
            raise ValueError("切片校正仅允许旋转，不能改变探头位置或水深。")
        if not all(math.isfinite(v) for v in (c.pitch_deg, c.roll_deg, c.yaw_correction_deg)):
            raise ValueError("角度必须为有限数值。")
        if abs(c.pitch_deg) > 85 or abs(c.roll_deg) > 85 or abs(c.yaw_correction_deg) > 180:
            raise ValueError("a、b 范围为 ±85°，c 修正范围为 ±180°。")


def fit_rotations(local, origins, headings, slice_ids, initial, max_gap=0.5,
                  progress=None, cancelled=None, iterations=7, boundaries=(), range_limits=None,
                  observation_weights=None):
    """Alternating local plane fitting and bounded rotation-only least squares.

    Reference planes require support from multiple OTHER sweeps. No cylinder,
    circle, endpoint or translation parameter is introduced. The first c is
    anchored because a shared heading offset cannot be inferred from smoothness.
    """
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from surface_reconstruction import _check

    validate_corrections(initial)
    boundaries = validate_boundaries(boundaries)
    local, origins = np.asarray(local), np.asarray(origins)
    ids, headings = np.asarray(slice_ids, dtype=int), np.asarray(headings)
    from acquisition_conditions import common_visibility
    limits = np.full(len(local), np.inf) if range_limits is None else np.asarray(range_limits, dtype=float)
    weights = np.ones(len(local)) if observation_weights is None else np.asarray(observation_weights, dtype=float)
    if limits.shape != (len(local),) or np.any(np.isnan(limits) | (limits <= 0)):
        raise ValueError('逐点量程必须为正数且与观测点对应。')
    if weights.shape != (len(local),) or not np.all(np.isfinite(weights) & (weights > 0) & (weights <= 1)):
        raise ValueError('回波权重必须在 (0, 1] 内且与观测点对应。')
    n = len(initial)
    if n < 3 or len(local) < 90:
        raise ValueError("自动拟合至少需要三圈且有足够的重叠壁面；仍可手动调整 a/b/c。")
    angles0 = np.array([[c.pitch_deg, c.roll_deg, c.yaw_correction_deg] for c in initial])
    angles = angles0.copy()
    cy, sy = np.cos(np.deg2rad(-headings)), np.sin(np.deg2rad(-headings))

    def world(values):
        matrices = Rotation.from_euler('xyz', values[:, [1, 0, 2]] * [1, 1, -1], degrees=True).as_matrix()
        rays = np.einsum('nij,nj->ni', matrices[ids], local)
        x, y = rays[:, 0].copy(), rays[:, 1].copy()
        rays[:, 0], rays[:, 1] = cy*x - sy*y, sy*x + cy*y
        return rays + origins

    centers = np.array([np.median(origins[ids == i, 2]) if np.any(ids == i) else np.nan for i in range(n)])
    initial_world = world(angles0)
    first_metric, final_metric, used = None, None, 0
    evaluation = None
    best_angles = angles.copy()
    best_score = math.inf
    segment_metrics = ()
    for iteration in range(iterations):
        _check(cancelled)
        points = world(angles)
        labels = section_ids(points, boundaries)
        source, anchors, normals, match_weights = [], [], [], []
        for i in range(n):
            _check(cancelled)
            own = np.flatnonzero(ids == i)
            neighbors = np.flatnonzero((np.abs(centers - centers[i]) <= max_gap) & (np.arange(n) != i))
            other = np.flatnonzero(np.isin(ids, neighbors))
            if len(own) < 10 or len(other) < 24 or len(neighbors) < 2:
                continue
            for section in np.unique(labels[own]):
                own_part = own[labels[own] == section]
                other_part = other[labels[other] == section]
                if len(other_part) < 24:
                    continue
                distances, near = cKDTree(points[other_part]).query(points[own_part], k=min(24, len(other_part)))
                for row, index in enumerate(own_part):
                    valid = distances[row] < max(0.25, max_gap * 1.8)
                    target = other_part[near[row][valid]]
                    if range_limits is not None:
                        target = target[common_visibility(points, origins, limits, index, target)]
                    if len(target) < 8 or len(np.unique(ids[target])) < 2:
                        continue
                    support = points[target]
                    center = np.average(support, axis=0, weights=weights[target])
                    delta = support - center
                    eigen, vectors = np.linalg.eigh((delta*weights[target, None]).T @ delta / weights[target].sum())
                    # Local patches may be planar or curved; a crease/step is
                    # not a shared smooth plane. No fit crosses a Z section.
                    if eigen[1] < 1e-5 or eigen[0] > 0.3 * eigen[1] or eigen[1] < 0.015 * eigen[2]:
                        continue
                    normal = vectors[:, 0]
                    if abs(np.dot(points[index] - center, normal)) > 0.25:
                        continue
                    source.append(index)
                    anchors.append(center)
                    normals.append(normal)
                    match_weights.append(math.sqrt(weights[index]*float(np.mean(weights[target]))))
        if len(source) < 60:
            if first_metric is None:
                raise ValueError("相邻切片缺少可匹配的局部壁面，无法可靠自动拟合。请手动调整或增大最大层距。")
            break
        source, anchors, normals = np.array(source), np.array(anchors), np.array(normals)
        if evaluation is None:
            evaluation = (source.copy(), anchors.copy(), normals.copy(), labels[source].copy())
            first_metric = float(np.median(np.abs(np.einsum('ij,ij->i', initial_world[source]-anchors, normals))))
            final_metric = first_metric
            used = len(source)
            ev_errors = np.abs(np.einsum('ij,ij->i', initial_world[source]-anchors, normals))
            active_sections = np.unique(labels[source])
            best_score = float(np.mean([np.median(ev_errors[labels[source] == label]) for label in active_sections]))
        # Each residual depends on one sweep, plus mild temporal/initial priors.
        m = len(source)
        active, counts = np.unique(labels[source], return_counts=True)
        balance = np.ones(m)
        for label, count in zip(active, counts):
            balance[labels[source] == label] = math.sqrt(m/(len(active)*count))
        balance *= np.sqrt(match_weights)
        sparsity = lil_matrix((m + n*3 + (n-1)*3 + 1, n*3), dtype=int)
        for row, index in enumerate(ids[source]):
            sparsity[row, index*3:index*3+3] = 1
        for j in range(n*3):
            sparsity[m+j, j] = 1
        for j in range((n-1)*3):
            sparsity[m+n*3+j, j] = sparsity[m+n*3+j, j+3] = 1
        sparsity[-1, 2] = 1

        def residual(flat):
            _check(cancelled)
            values = flat.reshape(n, 3)
            geometry = np.einsum('ij,ij->i', world(values)[source] - anchors, normals)
            # Robustify observations only: the quadratic angular priors must
            # not saturate and allow spurious abrupt inter-sweep rotations.
            robust = geometry * np.sqrt(2 / (np.sqrt(1+(geometry/0.025)**2)+1))
            return np.concatenate((robust * balance, ((values-angles0)*0.003).ravel(),
                                   # Structure boundaries do not imply jumps
                                   # in the physical head's orientation.
                                   (np.diff(values, axis=0)*0.015).ravel(),
                                   [(values[0, 2]-angles0[0, 2])*1000]))

        start_residual = residual(angles.ravel())
        solved = least_squares(residual, angles.ravel(), bounds=(
            np.maximum(angles0-20, [-85, -85, -180]).ravel(),
            np.minimum(angles0+20, [85, 85, 180]).ravel()),
            jac_sparsity=sparsity.tocsr(), max_nfev=35)
        _check(cancelled)
        candidate = solved.x.reshape(n, 3)
        candidate[0, 2] = angles0[0, 2]
        # Keep an independent fixed set of initial planes for all iterations.
        ev_source, ev_anchors, ev_normals, ev_labels = evaluation
        errors = np.abs(np.einsum('ij,ij->i', world(candidate)[ev_source]-ev_anchors, ev_normals))
        after = float(np.median(errors))
        score = float(np.mean([np.median(errors[ev_labels == label]) for label in np.unique(ev_labels)]))
        if np.linalg.norm(residual(solved.x)) > np.linalg.norm(start_residual) * 1.05:
            break
        if score < best_score:
            best_score = score
            final_metric = after
            best_angles = candidate.copy()
        change = float(np.max(np.abs(candidate-angles)))
        angles = candidate
        if progress:
            progress(15 + int((iteration+1)/iterations*65), f"切片角度拟合 {iteration+1}/{iterations}，局部残差 {after*100:.2f} cm")
        if change < 0.02:
            break
    if first_metric is None:
        raise ValueError("未获得稳定的角度拟合结果，请手动调整。")
    corrections = tuple(replace(c, pitch_deg=float(v[0]), roll_deg=float(v[1]), yaw_correction_deg=float(v[2]))
                        for c, v in zip(initial, best_angles))
    if boundaries:
        ev_source, ev_anchors, ev_normals, ev_labels = evaluation
        before_errors = np.abs(np.einsum('ij,ij->i', initial_world[ev_source]-ev_anchors, ev_normals))
        after_errors = np.abs(np.einsum('ij,ij->i', world(best_angles)[ev_source]-ev_anchors, ev_normals))
        segment_metrics = tuple((int(label)+1, int((ev_labels == label).sum()),
                                 float(np.median(before_errors[ev_labels == label])),
                                 float(np.median(after_errors[ev_labels == label]))) for label in np.unique(ev_labels))
    note = (f"{n} 圈，仅旋转；{used} 条初始局部平面约束，残差 {first_metric*100:.2f} → {final_metric*100:.2f} cm；"
            "首圈 c 锚定，弱观测角度保留初值附近")
    if np.any(np.abs(best_angles-angles0) > 19.8):
        note += "；部分角度达到本次 ±20° 搜索边界，请检查回波和初值"
    if boundaries:
        detail = '；'.join(f"段 {label} ({count} 点) {before*100:.2f}→{after*100:.2f} cm"
                          for label, count, before, after in segment_metrics)
        note += f"；结构分段 {len(boundaries)+1} 段，段间不匹配壁面；" + detail
        missing = sorted(set(range(1, len(boundaries)+2)) - {row[0] for row in segment_metrics})
        if missing:
            note += "；缺少拟合支持的段：" + ', '.join(map(str, missing))
    return SliceRotationResult(corrections, note, first_metric, final_metric, boundaries, segment_metrics)


def optimize_recording(recording, initial, threshold=32, near_range=0.0, max_gap=0.5,
                       progress=None, cancelled=None, boundaries=()):
    import numpy as np
    from surface_reconstruction import read_echo_candidates, _locally_supported_candidates, interpolate_sensor, _check
    validate_corrections(initial)
    candidates = read_echo_candidates(recording, threshold, progress, cancelled, near_range=near_range)
    from acquisition_conditions import RangeChecks
    checks = RangeChecks(recording)
    scan_metadata = {(s.timestamp, s.angle_deg): (s.range_m, checks.echo_weight(s, len(recording.echo_data(s))))
                     for s in recording.echo_scans}
    selected = _locally_supported_candidates(candidates, True, cancelled,
                                            [scan_metadata[t, a][0] for t, a, _, _ in candidates])
    # One best locally supported return per beam for fitting. Full multi-echo
    # observations remain intact for point clouds and surface reconstruction.
    best = {}
    for index, choice, quality in selected:
        if index not in best or quality > best[index][1]:
            best[index] = (choice, quality)
    starts = np.array([c.slice_start for c in initial])
    buckets = [[] for _ in initial]
    for index in sorted(best):
        timestamp, angle, ranges, _ = candidates[index]
        choice, quality = best[index]
        height, heading, time_weight = interpolate_sensor(recording, timestamp)
        if time_weight < 0.05 or quality < 0.08 or not np.isfinite([height, heading]).all():
            continue
        sid = max(0, int(np.searchsorted(starts, timestamp, side='right')-1))
        theta = math.radians(angle)
        limit, acquisition_weight = scan_metadata[timestamp, angle]
        buckets[sid].append(([math.sin(theta)*ranges[choice], math.cos(theta)*ranges[choice], 0],
                             [0, 0, height], heading, sid, limit, quality*time_weight*acquisition_weight))
    rows = []
    for bucket in buckets:
        _check(cancelled)
        rows.extend(bucket[::max(1, math.ceil(len(bucket)/180))])
    if not rows:
        raise ValueError("没有足够壁面回波供角度拟合。")
    local, origins, headings, ids, limits, weights = zip(*rows)
    result = fit_rotations(local, origins, headings, ids, initial, max_gap, progress, cancelled, boundaries=boundaries,
                           range_limits=limits, observation_weights=weights)
    if checks.events:
        result = replace(result, note=result.note + f'；{len(checks.events)} 次量程切换：跨量程匹配已检查共同可见范围，切换初始回波降权')
    return result
