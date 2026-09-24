"""Acquisition range changes, independent of geometric section boundaries."""
from bisect import bisect_right
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RangeSwitch:
    timestamp: int
    before_m: float
    after_m: float
    source: str
    stable_after: float

    @property
    def description(self):
        extra = (f"；{self.before_m:g}–{self.after_m:g} m 新增可见区域：此前未知"
                 if self.after_m > self.before_m else "；缩小后范围外区域：当前未知")
        return f"量程 {self.before_m:g}→{self.after_m:g} m（采集条件变化，不是结构分界）" + extra


def range_switches(recording):
    """Prefer beam headers. With only profiles, the changed profile is suspect.

    A rolling vendor contour may contain old bins until every angular bin has
    been revisited. Require a complete continuous turn in the new range, not a
    fixed timeout. An interrupted/reversed scan restarts this coverage check.
    """
    echoes = getattr(recording, 'echo_scans', ())
    scans = echoes or getattr(recording, 'profile_scans', ())
    events = []
    active, travel, direction = None, 0.0, 0
    for i in range(1, len(scans)):
        previous, scan = scans[i-1], scans[i]
        if not math.isclose(scan.range_m, previous.range_m, abs_tol=1e-6):
            events.append(RangeSwitch(scan.timestamp, previous.range_m, scan.range_m,
                                      'echo' if echoes else 'profile', math.inf))
            active, travel, direction = len(events)-1, 0.0, 0
        elif active is not None:
            if not echoes:
                e = events[active]
                events[active] = RangeSwitch(e.timestamp, e.before_m, e.after_m, e.source, scan.timestamp)
                active = None
                continue
            delta = (scan.angle_deg-previous.angle_deg+180) % 360-180
            sign = 1 if delta > 0 else -1 if delta < 0 else 0
            if abs(delta) > 3 or scan.timestamp-previous.timestamp > 5000 or direction and sign and sign != direction:
                travel, direction = 0.0, 0
            else:
                travel += abs(delta)
                direction = sign or direction
            if travel >= 359.5:
                e = events[active]
                events[active] = RangeSwitch(e.timestamp, e.before_m, e.after_m, e.source, scan.timestamp)
                active = None
    return tuple(events)


class RangeChecks:
    def __init__(self, recording):
        self.events = range_switches(recording)
        self.times = [e.timestamp for e in self.events]
        self.scans = getattr(recording, 'echo_scans', ())
        self.scan_times = [s.timestamp for s in self.scans]

    def profile_issue(self, profile):
        i = bisect_right(self.times, profile.timestamp)-1
        if i >= 0 and profile.timestamp < self.events[i].stable_after:
            return '切换后未完成一圈，轮廓可能混合新旧量程' if self.scans else '切换首幅轮廓待核验'
        j = bisect_right(self.scan_times, profile.timestamp)-1
        if j >= 0 and not math.isclose(profile.range_m, self.scans[j].range_m, abs_tol=1e-6):
            return '轮廓量程与同期原始回波不一致'
        return ''

    def echo_weight(self, scan, sample_count):
        # Metric sample spacing is a conservative uncertainty proxy, not a
        # vendor amplitude/gain calibration. Never rescale measured distances.
        resolution = scan.range_m/max(1, sample_count)
        weight = 1/math.sqrt(1+(resolution/.01)**2)
        j = bisect_right(self.scan_times, scan.timestamp)-1
        i = bisect_right(self.times, scan.timestamp)-1
        if i >= 0:
            start = bisect_right(self.scan_times, self.times[i])-1
            if 0 <= j-start < 3:
                weight *= .35  # first three beams: conservative transient guard
        return weight

    def report(self):
        if not self.events:
            return ()
        base = self.scan_times[0] if self.scan_times else self.times[0]
        return tuple(f"t={(e.timestamp-base)/1000:.2f} s：{e.description}" for e in self.events)


def common_visibility(points, origins, limits, index, targets):
    """Mutual range-sphere visibility for cross-range local plane matches.

    This necessary condition does not certify visibility through occlusions.
    Same-range matches retain the original behavior.
    """
    import numpy as np
    cross = ~np.isclose(limits[targets], limits[index], atol=1e-6)
    source_ray = points[index]-origins[index]
    target_rays = points[targets]-origins[targets]
    mutual = (np.linalg.norm(points[targets]-origins[index], axis=1) < limits[index]*.98)
    mutual &= np.linalg.norm(points[index]-origins[targets], axis=1) < limits[targets]*.98
    mutual &= np.einsum('ij,j->i', target_rays, source_ray) > .2*np.linalg.norm(target_rays, axis=1)*np.linalg.norm(source_ray)
    return ~cross | mutual
