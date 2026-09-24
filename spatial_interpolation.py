"""Bounded 3-D interpolation of compatible wall observations before meshing."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpatialObservations:
    points: object
    origins: object
    confidence: object
    free_mask: object
    range_limits: object
    synthetic: object
    note: str

    @property
    def added_count(self):
        return int(self.synthetic.sum())


def interpolate_spatial(points, origins, confidence, free_mask, range_limits, spacing=.05,
                        max_gap=.3, boundaries=(), support=.2, denoise=True,
                        progress=None, cancelled=None, max_new_points=200000):
    import numpy as np
    from scipy.spatial import cKDTree
    from surface_reconstruction import _oriented_samples, _check
    from surface_segments import section_ids, validate_boundaries
    _check(cancelled)
    boundaries = validate_boundaries(boundaries)
    if not (math.isfinite(spacing) and spacing > 0 and math.isfinite(max_gap) and max_gap >= 2*spacing):
        raise ValueError("空间插值间距必须为正，最大连接距离至少为间距的 2 倍。")
    if not math.isfinite(support) or support <= 0 or max_new_points < 0:
        raise ValueError("空间插值支持范围或点数预算无效。")
    p, o, q = (np.array(a, dtype=float, copy=True) for a in (points, origins, confidence))
    free, limits = np.array(free_mask, dtype=bool, copy=True), np.array(range_limits, dtype=float, copy=True)
    if (p.ndim != 2 or p.shape[1:] != (3,) or o.shape != p.shape or
            any(a.shape != (len(p),) for a in (q, free, limits))):
        raise ValueError("空间插值观测数组格式无效。")
    if not (np.isfinite(p).all() and np.isfinite(o).all() and np.isfinite(q).all()
            and np.isfinite(limits).all() and (limits > 0).all()):
        raise ValueError("空间插值观测包含非法数值。")
    labels = section_ids(p, boundaries)
    generated, occupied, saturated = [], set(), False
    budget = min(int(max_new_points), max(1000, len(p)*4))
    for label in np.unique(labels):
        _check(cancelled)
        ids = np.flatnonzero(labels == label)
        if len(ids) < 30:
            continue
        try:
            anchors, sources, normals, quality, _, keep = _oriented_samples(
                p[ids], o[ids], q[ids], spacing, cancelled, max(support, max_gap),
                denoise, return_indices=True, vertical_continuity=False)
        except ValueError as error:
            if '壁面有效点过少' not in str(error):
                raise
            continue
        anchor_limits = limits[ids[keep]]
        rays = anchors-sources
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        tree, observed_tree = cKDTree(anchors), cKDTree(p[ids])
        k = min(64, len(anchors))
        distance, neighbors = tree.query(anchors, k=k, distance_upper_bound=max_gap)
        for i in range(len(anchors)):
            if i % 128 == 0:
                _check(cancelled)
                if progress:
                    progress(30+int(30*i/max(1, len(anchors))), f"结构段 {label+1}：三维邻域空间插值")
            if quality[i] < .08:
                continue
            js = neighbors[i][np.isfinite(distance[i]) & (distance[i] >= spacing*1.6)]
            js = js[(js != i) & (js < len(anchors))]
            if not len(js):
                continue
            delta = anchors[js]-anchors[i]
            length = np.linalg.norm(delta, axis=1)
            direction = delta/length[:, None]
            compatible = (normals[js] @ normals[i] >= .9) & (quality[js] >= .08)
            # Ambiguous PCA normals in a narrow pipe must not connect two
            # opposing returns through the water/head's free-space region.
            compatible &= rays[js] @ rays[i] > .2
            compatible &= (abs(direction @ normals[i]) <= .26)
            compatible &= abs(np.einsum('ij,ij->i', direction, normals[js])) <= .26
            js, direction = js[compatible], direction[compatible]
            if not len(js):
                continue
            # Choose the nearest compatible anchor in each tangent direction,
            # rather than densely filling all pairs or assuming horizontal scans.
            axis = np.eye(3)[np.argmin(abs(normals[i]))]
            u = np.cross(normals[i], axis); u /= np.linalg.norm(u)
            v = np.cross(normals[i], u)
            sectors = np.floor((np.arctan2(direction @ v, direction @ u)+np.pi)/(np.pi/4)).astype(int) % 8
            _, first = np.unique(sectors, return_index=True)
            for j in js[first]:
                if j < i:
                    continue
                steps = int(np.ceil(np.linalg.norm(anchors[j]-anchors[i])/spacing))
                fraction = np.arange(1, steps)[:, None]/steps
                xyz = anchors[i]*(1-fraction)+anchors[j]*fraction
                source = sources[i]*(1-fraction)+sources[j]*fraction
                limit = min(anchor_limits[i], anchor_limits[j])
                nearest, _ = observed_tree.query(xyz)
                valid = (nearest > spacing*.55) & (np.linalg.norm(xyz-source, axis=1) <= limit)
                valid &= section_ids(xyz, boundaries) == label
                for at in np.flatnonzero(valid):
                    cell = tuple(np.floor(xyz[at]/(spacing*.65)).astype(int))
                    if cell in occupied:
                        continue
                    if len(generated) >= budget:
                        saturated = True
                        break
                    occupied.add(cell)
                    generated.append((xyz[at], source[at], min(quality[i], quality[j])*.3, limit))
                if saturated:
                    break
            if saturated:
                break
        if saturated:
            break
    n = len(generated)
    synthetic = np.r_[np.zeros(len(p), dtype=bool), np.ones(n, dtype=bool)]
    if generated:
        xyz, source, quality, limit = zip(*generated)
        p, o, q = np.vstack((p, xyz)), np.vstack((o, source)), np.r_[q, quality]
        free, limits = np.r_[free, np.zeros(n, dtype=bool)], np.r_[limits, limit]
    note = f"空间插值 {n:,} 个辅助点（非实测），间距 {spacing*100:g} cm，最大连接 {max_gap*100:g} cm"
    if saturated:
        note += "；已达到插值点数预算"
    if not n:
        note += "；无可补充的相容邻域，可调整间距或连接距离"
    for a in (p, o, q, free, limits, synthetic):
        a.setflags(write=False)
    _check(cancelled)
    return SpatialObservations(p, o, q, free, limits, synthetic, note)


def prepare_spatial(recording, corrections=(), threshold=32, near_range=0, **kwargs):
    from surface_reconstruction import extract_wall_observations
    observations = extract_wall_observations(recording, threshold, corrections,
        progress=kwargs.get('progress'), cancelled=kwargs.get('cancelled'), near_range=near_range,
        denoise=kwargs.get('denoise', True), return_free_mask=True, return_range_limits=True)
    return interpolate_spatial(*observations, **kwargs)


def preview_cloud(cloud, observations, max_points):
    from dataclasses import replace
    import numpy as np
    extra = observations.points[observations.synthetic]
    count = min(len(extra), max(1000, int(max_points)//2))
    if count:
        extra = extra[np.linspace(0, len(extra)-1, count, dtype=int)]
    return replace(cloud, interpolated=tuple(map(tuple, extra)), interpolation_note=observations.note)
