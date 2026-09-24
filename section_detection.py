"""Multi-section proposals using chamber shape AND observed pipe sidewalls.

Labels describe evidence, not certified pipe presence/absence. Range saturation
or missing returns alone never establish a pipe. Proposals use measured head Z;
actual reconstruction still assigns rotated points by world Z.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SectionEvidence:
    low: float
    high: float
    state: str
    levels: int


@dataclass(frozen=True)
class SectionProposal:
    boundaries: tuple
    reasons: tuple
    sections: tuple
    acquisition_notes: tuple = ()


def _runs(mask):
    """Circular contiguous indices, merging the angular wrap correctly."""
    import numpy as np
    mask = np.asarray(mask, dtype=bool)
    if mask.all():
        return [np.arange(len(mask))]
    if not mask.any():
        return []
    start = (int(np.flatnonzero(~mask)[0])+1) % len(mask)
    indices = (np.arange(len(mask))+start) % len(mask)
    groups, group = [], []
    for index in indices:
        if mask[index]:
            group.append(index)
        elif group:
            groups.append(np.array(group))
            group = []
    if group:
        groups.append(np.array(group))
    return groups


def profile_features(ranges, range_m):
    """Find paired straight flanks in extended sectors, preserving 0.45° bins."""
    import numpy as np
    values = np.full(800, np.nan)
    raw = np.asarray(ranges, dtype=float)
    values[:min(800, len(raw))] = raw[:800]
    valid = np.isfinite(values) & (values > .05) & (values < range_m*.98)
    if valid.sum() < 80:
        return np.full(5, np.nan), -1, 0.0
    quantiles = np.quantile(values[valid], [.1, .25, .5, .75, .9])
    baseline = max(float(quantiles[2]), .15)
    missing = ~valid
    extended = missing | (values > max(baseline*1.35, baseline+.25))
    theta = np.arange(800)*np.deg2rad(.45)
    xy = np.column_stack((np.sin(theta), np.cos(theta)))*values[:, None]
    pipes = 0
    for run in _runs(extended):
        if not 12 <= len(run) <= 245:
            continue
        midpoint = len(run)//2
        halves = (run[:midpoint], run[midpoint:])
        lines = []
        for half in halves:
            indices = half[valid[half]]
            if len(indices) < 6:
                break
            points = xy[indices]
            center = np.median(points, axis=0)
            eigen, vectors = np.linalg.eigh((points-center).T @ (points-center)/len(points))
            tangent = vectors[:, 1]
            residual = abs((points-center) @ vectors[:, 0])
            span = np.ptp(points @ tangent)
            if (span < max(.25, baseline*.25) or eigen[0] > eigen[1]*.025 or
                    np.quantile(residual, .8) > max(.025, span*.035)):
                break
            lines.append((center, tangent))
        if len(lines) != 2:
            continue
        (left, a), (right, b) = lines
        axis_angle = theta[run[midpoint]]
        axis = np.array([math.sin(axis_angle), math.cos(axis_angle)])
        width = abs(np.dot(right-left, [-a[1], a[0]]))
        if abs(a @ b) < .93 or min(abs(a @ axis), abs(b @ axis)) < .82:
            continue
        if not .12 <= width <= baseline*2.5:
            continue
        pipes += 1
    # Unknown includes clipped/open sectors without corroborating sidewalls.
    ambiguous_extension = quantiles[4] > baseline*2 and quantiles[4]-baseline > .4
    state = pipes if pipes else (-1 if missing.mean() > .12 or ambiguous_extension else 0)
    return quantiles, state, float(valid.mean())


def _levels(recording, ceiling=None, checks=None):
    import numpy as np
    from surface_reconstruction import interpolate_sensor
    from acquisition_conditions import RangeChecks
    checks = checks or RangeChecks(recording)
    bins = {}
    for profile in recording.profile_scans:
        if checks.profile_issue(profile):
            continue
        height, _, quality = interpolate_sensor(recording, profile.timestamp)
        if not math.isfinite(height) or quality < .05:
            continue
        limit = min(profile.range_m, ceiling) if ceiling is not None else profile.range_m
        features, state, coverage = profile_features(recording.profile_distances(profile), limit)
        bins.setdefault(round(height/.1), []).append((height, features, state, coverage, profile.range_m))
    levels = []
    for key in sorted(bins):
        rows = bins[key]
        finite = [r[1] for r in rows if np.isfinite(r[1]).all()]
        descriptor = np.median(finite, axis=0) if finite else np.full(5, np.nan)
        states = [r[2] for r in rows]
        unique, counts = np.unique(states, return_counts=True)
        # Contradictory revisits are not confidently labeled no-pipe.
        state = int(unique[np.argmax(counts)]) if counts.max()/len(states) >= .6 else -1
        levels.append((float(np.median([r[0] for r in rows])), descriptor, state,
                       min(r[4] for r in rows), max(r[4] for r in rows)))
    return levels


def _shape_cuts(levels):
    """Robust piecewise-constant change points; two observed levels per segment."""
    import numpy as np
    n = len(levels)
    if n < 4:
        return []
    data = np.array([r[1] for r in levels])
    valid = np.isfinite(data).all(axis=1)
    if valid.sum() < 4:
        return []
    scale = max(.12, float(np.median(data[valid]))*.15)
    costs = np.full((n+1, n+1), np.inf)
    for begin in range(n):
        for end in range(begin+2, n+1):
            values = data[begin:end][valid[begin:end]]
            if len(values) >= 2:
                costs[begin, end] = np.minimum(abs(values-np.median(values, axis=0))/scale, 6).mean(axis=1).sum()
    best, previous = np.full(n+1, np.inf), np.full(n+1, -1, dtype=int)
    best[0] = -2.5
    for end in range(2, n+1):
        for begin in range(0, end-1):
            value = best[begin]+costs[begin, end]+2.5
            if value < best[end]:
                best[end], previous[end] = value, begin
    cuts, end = [], n
    while previous[end] > 0:
        end = previous[end]
        cuts.append(int(end))
    return sorted(cuts)


def _state_runs(levels):
    groups = []
    for i, row in enumerate(levels):
        if not groups or groups[-1][2] != row[2]:
            groups.append([i, i+1, row[2]])
        else:
            groups[-1][1] = i+1
    return groups


def describe_sections(levels, boundaries):
    import numpy as np
    if not levels:
        return ()
    bounds = [levels[0][0], *boundaries, levels[-1][0]]
    result = []
    for i, (low, high) in enumerate(zip(bounds, bounds[1:])):
        rows = [r for r in levels if r[0] >= low and (r[0] < high or i == len(bounds)-2 and r[0] <= high)]
        states = np.array([r[2] for r in rows])
        if len(states) >= 2 and np.mean(states > 0) >= .6:
            state = f"疑似有管道（最多 {int(states.max())} 处侧壁组合）"
        elif len(states) >= 2 and np.mean(states == 0) >= .8:
            state = "未见管道特征（不等于确认无管）"
        else:
            state = "不确定 / 观测不足或跨层不一致"
        result.append(SectionEvidence(float(low), float(high), state, len(rows)))
    return tuple(result)


def _candidate_cuts(levels, include_pipes):
    cuts = {i: '井室轮廓变化' for i in _shape_cuts(levels)}
    if include_pipes:
        runs = _state_runs(levels)
        for left, right in zip(runs, runs[1:]):
            # A one-level blip is insufficient evidence of appearing/disappearing.
            if left[1]-left[0] < 2 or right[1]-right[0] < 2:
                continue
            a, b = left[2], right[2]
            if a == -1 or b == -1:
                reason = '观测可靠性变化（不能判定管道有无）'
            elif a == 0 and b > 0:
                reason = '管道侧壁特征出现'
            elif a > 0 and b == 0:
                reason = '管道侧壁特征消失'
            else:
                reason = '管道侧壁组合数量变化'
            i = right[0]
            cuts[i] = cuts.get(i, '') + ('；' if i in cuts else '') + reason
    return cuts


def propose_sections(recording, include_pipes=True):
    """Compare acquisition regimes in their common observable distance domain."""
    from acquisition_conditions import RangeChecks
    checks = RangeChecks(recording)
    levels = _levels(recording, checks=checks)
    notes = list(checks.report())
    suspect = sum(bool(checks.profile_issue(p)) for p in recording.profile_scans)
    if suspect:
        notes.append(f'{suspect} 幅切换附近或量程不一致轮廓已排除自动分段；原始回波仍按逐帧量程处理')
    if len(levels) < 4:
        return SectionProposal((), (), describe_sections(levels, ()), tuple(notes))
    cuts = _candidate_cuts(levels, include_pipes)
    if any(r[3] != levels[0][3] or r[4] != levels[0][3] for r in levels):
        # A common-domain pass prevents raw descriptor changes at a range
        # switch from hiding a nearby real change. Native-range candidates
        # remain useful for subsequent changes in the newly observed region.
        common = _levels(recording, min(r[3] for r in levels), checks)
        cuts.update(_candidate_cuts(common, include_pipes))
        normalized = {}
        for i in list(cuts):
            window = levels[max(0, i-2):min(len(levels), i+2)]
            if min(r[3] for r in window) == max(r[4] for r in window):
                continue
            ceiling = min(r[3] for r in window)
            if ceiling not in normalized:
                normalized[ceiling] = _levels(recording, ceiling, checks)
            begin, end = max(0, i-2), min(len(levels), i+2)
            verified = _candidate_cuts(normalized[ceiling][begin:end], include_pipes)
            reason = verified.get(i-begin)
            # Missing/visible evidence differences alone cannot establish a
            # physical boundary between acquisition regimes.
            if reason and ('轮廓' in reason or '侧壁' in reason):
                cuts[i] = reason + f'（共同可见范围 0–{ceiling*.98:g} m 核验）'
            else:
                del cuts[i]
                notes.append(f'Z≈{(levels[i][0]+levels[i-1][0])/2:.3f} m 的变化缺少共同可见范围内结构证据，未设分界')
    retained = []
    for i, reason in sorted(cuts.items()):
        if levels[i][0]-levels[i-1][0] > .8:
            continue
        if retained and i-retained[-1][0] < 2:
            previous, previous_reason = retained.pop()
            def priority(text):
                return 2 if '侧壁' in text else 1 if '轮廓' in text else 0
            chosen = i if priority(reason) > priority(previous_reason) else previous
            retained.append((chosen, previous_reason+'；'+reason))
        else:
            retained.append((i, reason))
    # Do not turn a dense recording into dozens of tiny fragments automatically.
    # Users can revise the selected boundaries manually.
    retained = retained[:16]
    boundaries = tuple(round((levels[i][0]+levels[i-1][0])/2, 3) for i, _ in retained)
    reasons = tuple(reason for _, reason in retained)
    return SectionProposal(boundaries, reasons, describe_sections(levels, boundaries), tuple(notes))
