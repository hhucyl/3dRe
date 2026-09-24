"""Robust enclosing-wall selection and periodic radius-field reconstruction.

A manhole shaft has one enclosing wall radius per azimuth and height relative
to a fitted reference centre. Its radius is free to vary: the reference circle
is used to reject clutter, not to replace the measurements with a cylinder.
The resulting implicit field is F(x,y,z) = R(atan2(y-cy,x-cx),z)-rho.
We mesh its zero set directly on the periodic grid, leaving both ends open.
"""

import math

from surface_reconstruction import SurfaceMesh, _check, _pose_interpolator, read_echo_candidates, interpolate_sensor


def _circle_from_three(points):
    import numpy as np

    a, b, c = points
    matrix = 2 * np.array([b - a, c - a])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    center = np.linalg.solve(matrix, [b @ b - a @ a, c @ c - a @ a])
    radius = float(np.linalg.norm(a - center))
    return center, radius


def fit_reference_wall(points, origins, confidence, diameter=0.0, cancelled=None):
    """RANSAC enclosing circle, scored by angular coverage instead of density."""
    import numpy as np
    from scipy.optimize import least_squares

    xy = points[:, :2]
    center0 = np.median(origins[:, :2], axis=0)
    radial = np.linalg.norm(xy - center0, axis=1)
    # Spatially balanced candidates stop stationary scans dominating the fit.
    cell = max(0.025, float(np.quantile(radial, 0.5)) * 0.025)
    order = np.argsort(-confidence, kind="stable")
    _, first = np.unique(np.floor(points[order] / cell).astype(np.int64), axis=0, return_index=True)
    selected = order[first]
    if len(selected) > 12000:
        selected = selected[np.linspace(0, len(selected) - 1, 12000).astype(int)]
    samples, weights = xy[selected], confidence[selected]
    zlow, zhigh = np.quantile(points[selected, 2], [0.01, 0.99])
    height_sectors = 12 if zhigh - zlow > cell * 3 else 1
    height_index = np.clip(((points[selected, 2] - zlow) / max(zhigh - zlow, 1e-6) * height_sectors).astype(int),
                           0, height_sectors - 1)
    radius_min = max(0.28, diameter * 0.35) if diameter else 0.28
    radius_max = diameter * 0.65 if diameter else max(radius_min * 2, float(np.quantile(radial, 0.96)))
    hypotheses = []
    if diameter:
        hypotheses.append((center0, diameter / 2))
    hist, edges = np.histogram(radial, bins=100, range=(radius_min, radius_max), weights=confidence)
    for index in np.argsort(hist)[-8:]:
        hypotheses.append((center0, float((edges[index] + edges[index + 1]) / 2)))
    rng = np.random.default_rng(8128)
    for _ in range(500):
        if len(samples) < 3:
            break
        circle = _circle_from_three(samples[rng.choice(len(samples), 3, replace=False)])
        if circle:
            hypotheses.append(circle)
    best, best_score = None, -1.0
    origin_sample = origins[::max(1, len(origins) // 100), :2]
    for iteration, (center, radius) in enumerate(hypotheses):
        if iteration % 32 == 0:
            _check(cancelled)
        if not radius_min <= radius <= radius_max:
            continue
        # The enclosing wall must contain the scanner trajectory.
        if np.mean(np.linalg.norm(origin_sample - center, axis=1) < radius * 0.95) < 0.8:
            continue
        distance = np.linalg.norm(samples - center, axis=1)
        tolerance = max(0.055, radius * 0.10)
        residual = abs(distance - radius)
        inlier = residual < tolerance
        if inlier.sum() < 20:
            continue
        angle = np.mod(np.arctan2(samples[:, 1] - center[1], samples[:, 0] - center[0]), 2 * np.pi)
        sectors = np.minimum(71, (angle / (2 * np.pi) * 72).astype(int))
        support = np.zeros(72 * height_sectors)
        bins = height_index * 72 + sectors
        np.maximum.at(support, bins[inlier], weights[inlier] * (1 - residual[inlier] / tolerance))
        coverage = np.count_nonzero(support > 0.1) / len(support)
        score = support.sum() * coverage ** 1.5
        if diameter:
            score *= math.exp(-0.5 * ((2 * radius - diameter) / (diameter * 0.15)) ** 2)
        # A weak range preference breaks otherwise equal enclosing hypotheses.
        score /= 1 + 0.03 * radius
        if score > best_score:
            best_score, best = score, (center, radius)
    if best is None:
        raise ValueError("未识别到稳定的检查井外围井壁，请填写参考内径或调整近场排除距离。")
    center, radius = best
    for _ in range(3):
        _check(cancelled)
        residual = np.abs(np.linalg.norm(samples - center, axis=1) - radius)
        keep = residual < max(0.09, 0.18 * radius)
        if keep.sum() < 20:
            break
        selected_points, selected_weights = samples[keep], np.sqrt(weights[keep])

        def error(parameters):
            return (np.linalg.norm(selected_points - parameters[:2], axis=1) - parameters[2]) * selected_weights

        result = least_squares(error, [*center, radius], loss="soft_l1", f_scale=0.03,
                               bounds=([-np.inf, -np.inf, radius_min], [np.inf, np.inf, radius_max]), max_nfev=50)
        center, radius = result.x[:2], float(result.x[2])
    return center, radius


def _candidate_cloud(recording, threshold, corrections, near_range, progress, cancelled):
    import numpy as np

    candidates = read_echo_candidates(recording, threshold, progress, cancelled, near_range, 0.97)
    pose_at = _pose_interpolator(corrections)
    points, origins, weights, beams = [], [], [], []
    for index, (timestamp, angle, ranges, scores) in enumerate(candidates):
        if index % 256 == 0:
            _check(cancelled)
        height, heading, time_weight = interpolate_sensor(recording, timestamp)
        if not all(math.isfinite(v) for v in (height, heading, time_weight)) or time_weight < 0.05:
            continue
        rotation, translation = pose_at(timestamp)
        theta, yaw = math.radians(angle), math.radians(-heading)
        heading_matrix = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                   [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        direction = heading_matrix @ rotation @ np.array([math.sin(theta), math.cos(theta), 0])
        origin = translation + np.array([0, 0, height])
        points.extend(origin + ranges[:, None] * direction)
        origins.extend(np.tile(origin, (len(ranges), 1)))
        weights.extend(scores * time_weight)
        beams.extend([index] * len(ranges))
    return np.asarray(points), np.asarray(origins), np.asarray(weights), np.asarray(beams)


def select_enclosing_wall(points, origins, confidence, beams, center, radius, cancelled=None, return_external=False):
    """Select at most one shell return per beam; far-only rays mark openings."""
    import numpy as np

    radial = np.linalg.norm(points[:, :2] - center, axis=1)
    band = max(0.12, radius * 0.32)
    residual = abs(radial - radius)
    score = confidence * np.exp(-0.5 * (residual / max(0.07, radius * 0.16)) ** 2)
    acceptable = (residual <= band) & (confidence >= 0.07)
    selected, openings, external = [], [], []
    starts = np.r_[0, np.flatnonzero(np.diff(beams)) + 1, len(beams)]
    for index in range(len(starts) - 1):
        if index % 512 == 0:
            _check(cancelled)
        a, b = starts[index:index + 2]
        valid = np.flatnonzero(acceptable[a:b]) + a
        if len(valid) and np.max(score[valid]) >= 0.12:
            selected.append(valid[np.argmax(score[valid])])
            continue
        far = np.flatnonzero((radial[a:b] > radius + band) & (confidence[a:b] > 0.35)) + a
        if not len(far):
            continue
        strong = far[confidence[far] >= max(0.35, confidence[far].max() * 0.65)]
        chosen = strong[np.argmin(np.linalg.norm(points[strong] - origins[strong], axis=1))]
        origin, direction = origins[chosen], points[chosen] - origins[chosen]
        local = origin[:2] - center
        quadratic = direction[:2] @ direction[:2]
        linear = 2 * (local @ direction[:2])
        constant = local @ local - radius ** 2
        discriminant = linear ** 2 - 4 * quadratic * constant
        if quadratic > 1e-9 and discriminant >= 0:
            t = (-linear + np.sqrt(discriminant)) / (2 * quadratic)
            if 0 < t < 1:
                openings.append(origin + t * direction)
                external.append(chosen)
    selected = np.asarray(selected, dtype=int)
    if len(selected) < 30:
        raise ValueError("井壁候选过少，请调整参考内径、近场排除距离或回波阈值。")
    result = (points[selected], confidence[selected], np.asarray(openings).reshape(-1, 3))
    if return_external:
        ids = np.asarray(external, dtype=int)
        return (*result, points[ids], confidence[ids])
    return result


def fit_periodic_wall(points, confidence, center, radius, voxel_size=0.05, gap_limit=0.6,
                      openings=None, progress=None, cancelled=None, return_field=False):
    """Robust periodic radial field with bounded, low-confidence interpolation."""
    import numpy as np
    from scipy import sparse
    from scipy.ndimage import distance_transform_edt, convolve, binary_dilation
    from scipy.sparse.linalg import cg

    _check(cancelled)
    points, confidence = np.asarray(points), np.asarray(confidence)
    if voxel_size <= 0 or gap_limit <= 0:
        raise ValueError("网格尺寸和补全跨度必须为正数。")
    # Robust z limits suppress a few tilted multipath endpoints.
    low, high = np.quantile(points[:, 2], [0.005, 0.995])
    if high - low < 1e-5:
        low, high = low - voxel_size / 2, high + voxel_size / 2
    ntheta = int(np.clip(math.ceil(2 * np.pi * radius / voxel_size / 4) * 4, 48, 256))
    nz = int(np.clip(math.ceil((high - low) / voxel_size) + 1, 3, 256))
    dz, arc = (high - low) / (nz - 1), 2 * np.pi * radius / ntheta
    radius_values = np.linalg.norm(points[:, :2] - center, axis=1)

    def grid_indices(xyz):
        theta = np.mod(np.arctan2(xyz[:, 1] - center[1], xyz[:, 0] - center[0]), 2 * np.pi)
        col = np.rint(theta / (2 * np.pi) * ntheta).astype(int) % ntheta
        row = np.clip(np.rint((xyz[:, 2] - low) / dz).astype(int), 0, nz - 1)
        return row, col

    row, col = grid_indices(points)
    flat = row * ntheta + col
    count = nz * ntheta
    values = np.full(count, radius, dtype=float)
    data_weight = np.zeros(count)
    # Weighted cell medians resist internal clutter and repeated outliers.
    order = np.argsort(flat, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(flat[order])) + 1, len(order)]
    for k in range(len(boundaries) - 1):
        if k % 1024 == 0:
            _check(cancelled)
        ids = order[boundaries[k]:boundaries[k + 1]]
        sorted_ids = ids[np.argsort(radius_values[ids])]
        weight = np.maximum(confidence[sorted_ids], 0.001)
        median = radius_values[sorted_ids[np.searchsorted(np.cumsum(weight), weight.sum() / 2)]]
        good = abs(radius_values[ids] - median) < max(0.04, radius * 0.08)
        ids = ids[good]
        slot = flat[ids[0]]
        values[slot] = np.average(radius_values[ids], weights=confidence[ids])
        data_weight[slot] = min(3.0, confidence[ids].sum())
    observed = data_weight.reshape(nz, ntheta) > 0.06

    # Periodic angular Laplacian and natural (open) vertical boundary.
    rows = np.arange(ntheta)
    angular = sparse.coo_matrix((np.r_[np.ones(ntheta), -np.ones(ntheta)],
                                (np.r_[rows, rows], np.r_[rows, (rows + 1) % ntheta])), shape=(ntheta, ntheta)).tocsr()
    vertical = sparse.diags([-np.ones(nz - 1), np.ones(nz - 1)], [0, 1], shape=(nz - 1, nz), format="csr")
    angular_lap = sparse.kron(sparse.eye(nz), angular.T @ angular, format="csr")
    vertical_lap = sparse.kron(vertical.T @ vertical, sparse.eye(ntheta), format="csr")
    scale = max(arc, dz)
    laplacian = angular_lap * (scale / arc) ** 2 + vertical_lap * (scale / max(dz, voxel_size * 0.25)) ** 2
    regularizer = 0.35 * laplacian + 0.12 * (laplacian.T @ laplacian)
    estimate = np.full(count, radius, dtype=float)
    robust = np.ones(count)
    for iteration in range(5):
        _check(cancelled)
        if progress:
            progress(60 + iteration * 5, "拟合连续井壁并抑制内部噪声")
        weight = data_weight * robust
        matrix = regularizer + sparse.diags(weight + 0.0001)
        rhs = weight * values + 0.0001 * radius
        preconditioner = sparse.diags(1 / matrix.diagonal())
        estimate, info = cg(matrix, rhs, x0=estimate, M=preconditioner, rtol=1e-5, atol=1e-7,
                            maxiter=600, callback=lambda _: _check(cancelled))
        if info < 0 or not np.isfinite(estimate).all():
            raise ValueError("井壁曲面拟合失败，请增大网格尺寸。")
        error = abs(estimate - values)
        robust = np.minimum(1.0, max(0.025, radius * 0.035) / np.maximum(error, 1e-8))
    field = estimate.reshape(nz, ntheta)
    supported_observations = observed & (robust.reshape(nz, ntheta) > 0.3)
    distance = distance_transform_edt(~np.tile(supported_observations, (1, 3)), sampling=(dz, arc))[:, ntheta:2 * ntheta]
    support = distance <= max(voxel_size * 1.5, gap_limit / 2)

    # Only repeated far-only returns can reserve a candidate pipe opening.
    # Missing readings alone do not label a hole in the shaft wall.
    opening_grid = np.zeros((nz, ntheta))
    if openings is not None and len(openings):
        inside = (openings[:, 2] >= low) & (openings[:, 2] <= high)
        orow, ocol = grid_indices(openings[inside])
        np.add.at(opening_grid, (orow, ocol), 1)
    tiled = np.tile(opening_grid, (1, 3))
    neighboring = convolve(tiled, np.ones((3, 3)), mode="constant")[:, ntheta:2 * ntheta]
    opening_mask = (opening_grid > 0) & (neighboring >= 4) & ~supported_observations
    expanded = binary_dilation(np.tile(opening_mask, (1, 3)), iterations=1)[:, ntheta:2 * ntheta]
    support &= ~(expanded & ~supported_observations)

    theta, z = np.meshgrid(np.arange(ntheta) / ntheta * 2 * np.pi, np.linspace(low, high, nz))
    vertices = np.column_stack((center[0] + field.ravel() * np.cos(theta.ravel()),
                               center[1] + field.ravel() * np.sin(theta.ravel()), z.ravel()))
    a = np.arange((nz - 1) * ntheta).reshape(nz - 1, ntheta)
    b = np.roll(a, -1, axis=1)
    c, d = a + ntheta, b + ntheta
    valid = support[:-1] & np.roll(support[:-1], -1, axis=1) & support[1:] & np.roll(support[1:], -1, axis=1)
    # theta then z gives outward winding; reverse to face the water cavity.
    faces = np.concatenate((np.column_stack((a[valid], c[valid], b[valid])),
                            np.column_stack((b[valid], c[valid], d[valid]))))
    if not len(faces):
        raise ValueError("连续井壁为空，请增大补全跨度或检查参考内径。")
    certainty = np.exp(-distance / max(voxel_size, gap_limit / 4)) * 0.35
    certainty[supported_observations] = np.clip(data_weight.reshape(nz, ntheta)[supported_observations] / 3, 0.35, 1)
    used, inverse = np.unique(faces, return_inverse=True)
    residual = float(np.median(abs(estimate[data_weight > 0] - values[data_weight > 0])))
    interpolated = float(np.mean(~supported_observations.ravel()[used]))
    note = f"检查井约束；参考内径 {radius * 2:.2f} m；{interpolated:.0%} 网格顶点由邻域补全"
    _check(cancelled)
    if progress and not return_field:
        progress(100, "检查井连续曲面重建完成")
    mesh = SurfaceMesh(vertices[used].astype(np.float32), inverse.reshape(-1, 3).astype(np.int32),
                       certainty.ravel()[used].astype(np.float32), max(arc, dz), len(points), residual, note)
    if return_field:
        return mesh, dict(radii=field, support=support, confidence=certainty, center=center,
                          low=low, high=high, dz=dz, radius=radius)
    return mesh


def fit_connected_pipes(external, confidence, mouths, center, radius, voxel, cancelled=None):
    """Fit finite open cylinders only to clustered, measured exterior walls."""
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.optimize import least_squares

    if len(external) < 30:
        return []
    spacing = max(0.08, min(0.25, radius * 0.25))
    cells = np.floor(mouths / (spacing / 3)).astype(np.int64)
    _, first, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    pairs = cKDTree(mouths[first]).query_pairs(spacing, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(first), len(first)))
    _, groups = connected_components(graph, directed=False)
    labels = groups[inverse]
    pipes = []
    for group in np.unique(labels):
        _check(cancelled)
        ids = np.flatnonzero(labels == group)
        if len(ids) < 30:
            continue
        cloud, weight = external[ids], confidence[ids]
        mouth = np.average(mouths[ids], axis=0, weights=weight)
        theta = math.atan2(mouth[1] - center[1], mouth[0] - center[0])
        tangent = np.array([-math.sin(theta), math.cos(theta), 0])
        if np.ptp(mouths[ids, 2]) > radius * 1.8 or np.ptp(mouths[ids] @ tangent) > radius * 1.8:
            continue
        maximum_radius = min(2.0, radius * 0.8)
        if maximum_radius <= 0.055:
            continue
        initial_radius = float(np.clip(max(np.std(cloud @ tangent), np.std(cloud[:, 2])) * 1.5,
                                       0.055, maximum_radius * 0.9))
        if len(cloud) > 3000:
            subset = np.linspace(0, len(cloud) - 1, 3000).astype(int)
            cloud, weight = cloud[subset], weight[subset]

        def geometry(parameters):
            angle, height, yaw, pitch, pipe_radius = parameters
            origin = np.array([center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), height])
            axis = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
            delta = cloud - origin
            axial = delta @ axis
            perpendicular = delta - axial[:, None] * axis
            return origin, axis, axial, perpendicular, pipe_radius

        def error(parameters):
            _check(cancelled)
            _, _, _, perpendicular, pipe_radius = geometry(parameters)
            return (np.linalg.norm(perpendicular, axis=1) - pipe_radius) * np.sqrt(weight)

        initial = [theta, mouth[2], theta, 0, initial_radius]
        bounds = ([theta - 0.4, mouth[2] - maximum_radius, theta - 0.6, -0.5, 0.05],
                  [theta + 0.4, mouth[2] + maximum_radius, theta + 0.6, 0.5, maximum_radius])
        result = least_squares(error, initial, bounds=bounds, loss="soft_l1", f_scale=0.035, max_nfev=120)
        origin, axis, axial, perpendicular, pipe_radius = geometry(result.x)
        residual = abs(np.linalg.norm(perpendicular, axis=1) - pipe_radius)
        inliers = (residual < max(0.045, pipe_radius * 0.15)) & (axial > -voxel)
        if inliers.mean() < 0.65 or inliers.sum() < 25:
            continue
        low, length = np.quantile(axial[inliers], [0.05, 0.95])
        if length - low < max(0.12, voxel * 2) or length < pipe_radius * 0.5:
            continue
        # A single narrow strip does not determine a pipe cross section.
        basis_a = np.cross(axis, [0, 0, 1])
        basis_a /= np.linalg.norm(basis_a)
        basis_b = np.cross(axis, basis_a)
        phases = np.mod(np.arctan2(perpendicular[inliers] @ basis_b, perpendicular[inliers] @ basis_a), 2 * np.pi)
        if len(np.unique((phases / (2 * np.pi) * 16).astype(int))) < 5:
            continue
        pipes.append(dict(origin=origin, axis=axis, radius=float(pipe_radius), length=float(length),
                          confidence=float(np.median(weight[inliers])), observations=int(inliers.sum())))
    return pipes


def mesh_wall_and_pipes(wall_mesh, wall, pipes, voxel_size, progress=None, cancelled=None):
    """Union of shaft and pipe free-space fields creates a seamless junction."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    from skimage.measure import marching_cubes

    if not pipes:
        return wall_mesh
    extent_points = [wall_mesh.vertices]
    for pipe in pipes:
        end = pipe["origin"] + pipe["length"] * pipe["axis"]
        extent_points.extend([np.array([pipe["origin"] - pipe["radius"], pipe["origin"] + pipe["radius"]]),
                              np.array([end - pipe["radius"], end + pipe["radius"]])])
    xyz = np.vstack(extent_points)
    voxel = voxel_size
    lower, upper = xyz.min(axis=0), xyz.max(axis=0)
    while np.prod(np.ceil((upper - lower) / voxel).astype(np.int64) + 7) > 2_000_000:
        voxel *= 1.15
    lower -= voxel * 3
    shape = np.ceil((upper - lower) / voxel).astype(int) + 4
    total = int(np.prod(shape))
    field = np.empty(total, dtype=np.float32)
    known = np.zeros(total, dtype=bool)
    confidence = np.zeros(total, dtype=np.float32)
    ntheta = wall["radii"].shape[1]
    radius_grid = np.concatenate((wall["radii"], wall["radii"][:, :1]), axis=1)
    support_grid = np.concatenate((wall["support"], wall["support"][:, :1]), axis=1).astype(float)
    certainty_grid = np.concatenate((wall["confidence"], wall["confidence"][:, :1]), axis=1)
    for start in range(0, total, 16384):
        _check(cancelled)
        stop = min(total, start + 16384)
        if progress:
            progress(88 + int(9 * start / total), "融合井筒与连接管道")
        grid = np.column_stack(np.unravel_index(np.arange(start, stop), tuple(shape)))
        points = lower + grid * voxel
        local = points[:, :2] - wall["center"]
        theta = np.mod(np.arctan2(local[:, 1], local[:, 0]), 2 * np.pi)
        coordinates = np.array([(points[:, 2] - wall["low"]) / wall["dz"], theta / (2 * np.pi) * ntheta])
        radius = map_coordinates(radius_grid, coordinates, order=1, mode="nearest")
        shaft = radius - np.linalg.norm(local, axis=1)
        inside_height = (points[:, 2] >= wall["low"]) & (points[:, 2] <= wall["high"])
        support = map_coordinates(support_grid, coordinates, order=1, mode="nearest") > 0.5
        valid = inside_height & support & (abs(shaft) < voxel * 4)
        certainty = map_coordinates(certainty_grid, coordinates, order=1, mode="nearest")
        value = shaft.copy()
        # Extend the shaft field beyond its observed height, but not the known
        # mask. Abruptly changing its sign would create artificial end caps
        # wherever a pipe's support overlaps a shaft height boundary.
        for pipe in pipes:
            delta = points - pipe["origin"]
            axial = delta @ pipe["axis"]
            radial = np.linalg.norm(delta - axial[:, None] * pipe["axis"], axis=1)
            signed = pipe["radius"] - radial
            # Extend the field beyond the measured far end, but not its known
            # mask. This leaves an open end instead of generating a fake cap.
            active = (axial >= -pipe["radius"] * 2) & (axial <= pipe["length"] + voxel * 4)
            pipe_known = active & (axial <= pipe["length"]) & (abs(signed) < voxel * 4)
            valid |= pipe_known
            certainty = np.where(pipe_known, np.maximum(certainty, pipe["confidence"] * 0.5), certainty)
            value = np.where(active, np.maximum(value, signed), value)
        field[start:stop], known[start:stop], confidence[start:stop] = value, valid, certainty
    field, known, confidence = (a.reshape(tuple(shape)) for a in (field, known, confidence))
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(voxel,) * 3, mask=known, allow_degenerate=False)
    cells = np.clip(np.floor(vertices[faces].mean(axis=1) / voxel).astype(int), 0, shape - 2)
    valid_faces = np.ones(len(faces), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                valid_faces &= known[cells[:, 0] + dx, cells[:, 1] + dy, cells[:, 2] + dz]
    faces = faces[valid_faces]
    if not len(faces):
        raise ValueError("井筒与管道融合失败，请增大网格尺寸。")
    used, inverse = np.unique(faces, return_inverse=True)
    vertices = vertices[used]
    certainty = map_coordinates(confidence, (vertices / voxel).T, order=1, mode="nearest")
    vertices += lower
    note = wall_mesh.note + f"；已融合 {len(pipes)} 条有回波支持的支管"
    if voxel > voxel_size * 1.001:
        note += f"；内存预算调整网格至 {voxel * 100:.1f} cm"
    return SurfaceMesh(vertices.astype(np.float32), inverse.reshape(-1, 3).astype(np.int32),
                       certainty.astype(np.float32), voxel, wall_mesh.observation_count,
                       wall_mesh.median_residual, note)


def build_manhole_surface(recording, threshold=32, corrections=(), voxel_size=0.05,
                          diameter=0.0, near_range=0.30, gap_limit=0.6, progress=None, cancelled=None):
    import numpy as np

    points, origins, confidence, beams = _candidate_cloud(recording, threshold, corrections, near_range, progress, cancelled)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(origins).all(axis=1) & np.isfinite(confidence)
    points, origins, confidence, beams = points[valid], origins[valid], confidence[valid], beams[valid]
    if len(points) < 30:
        raise ValueError("有效井壁候选不足，请调整近场排除距离或回波阈值。")
    if progress:
        progress(35, "识别包围探头的主井壁")
    center, radius = fit_reference_wall(points, origins, confidence, diameter, cancelled)
    if progress:
        progress(50, "剔除井内噪点和远处多径回波")
    wall, weights, openings, exterior, exterior_weights = select_enclosing_wall(
        points, origins, confidence, beams, center, radius, cancelled, return_external=True)
    mesh, wall_field = fit_periodic_wall(wall, weights, center, radius, voxel_size, gap_limit, openings,
                                         progress, cancelled, return_field=True)
    if progress:
        progress(85, "识别管口外延的连接管道")
    pipes = fit_connected_pipes(exterior, exterior_weights, openings, center, radius, voxel_size, cancelled)
    mesh = mesh_wall_and_pipes(mesh, wall_field, pipes, voxel_size, progress, cancelled)
    if progress:
        progress(100, "检查井与连接管道重建完成")
    return mesh
