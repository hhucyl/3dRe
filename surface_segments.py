"""Spatial height sections for composite chambers, without primitive fitting.

Boundaries are world Z coordinates, not acquisition order or head depth. A
tilted sweep can contribute observations to several sections. Creases within
a section are still handled by local normal-compatible surface patches.
"""

import math


def validate_boundaries(values):
    values = tuple(float(v) for v in values)
    if len(values) > 16 or not all(math.isfinite(v) for v in values):
        raise ValueError("结构分界最多 16 个，且必须为有限的 Z 坐标。")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("结构分界 Z 必须严格递增，不能重复。")
    return values


def parse_boundaries(text):
    text = text.replace('，', ',').replace('；', ',').replace(';', ',')
    return validate_boundaries(sorted(float(v.strip()) for v in text.split(',') if v.strip()))


def section_ids(points, boundaries):
    import numpy as np
    return np.searchsorted(validate_boundaries(boundaries), np.asarray(points)[:, 2], side='right')


def suggest_boundaries(recording, include_pipes=True):
    """Compatibility entry point for multi-section shape/pipe proposals."""
    from section_detection import propose_sections
    return propose_sections(recording, include_pipes).boundaries


def reconstruct_sections(points, origins, confidence, boundaries, **kwargs):
    """Reconstruct sections independently; never invent transition bridges.

    Local surface patches may be curved or planar. No cylinder/box parameters
    are enforced. No field smoothing or synthetic Z support crosses a boundary.
    All measured observations remain in the point-cloud overlay. Sparse sections
    are reported and kept as observations instead of forcing a spurious surface.
    """
    import numpy as np
    from surface_reconstruction import reconstruct_surface, SurfaceMesh, _check
    boundaries = validate_boundaries(boundaries)
    if not boundaries:
        return reconstruct_surface(points, origins, confidence, **kwargs)
    points, origins, confidence = (np.asarray(v) for v in (points, origins, confidence))
    labels = section_ids(points, boundaries)
    progress, cancelled = kwargs.pop('progress', None), kwargs.get('cancelled')
    free_mask, range_limits = kwargs.pop('free_mask', None), kwargs.pop('range_limits', None)
    synthetic = kwargs.pop('synthetic_mask', None)
    synthetic = np.zeros(len(points), dtype=bool) if synthetic is None else np.asarray(synthetic, dtype=bool)
    vertices, faces, weights, meshes, notes = [], [], [], [], []
    offset = 0
    # Use a common resolution and honor a total memory scale, even when many
    # individual sections would each fit the original requested resolution.
    voxel = float(kwargs.get('voxel_size', 0.05))
    budget = kwargs.get('max_grid_cells', 2_000_000)
    if not math.isfinite(voxel) or voxel <= 0 or budget < 512:
        raise ValueError("网格尺寸或网格预算无效。")
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) < 30:
        raise ValueError("可用壁面点过少，请调整回波阈值。")
    while np.prod(np.ceil(np.ptp(finite, axis=0)/voxel).astype(np.int64)+7) > budget:
        voxel *= 1.15
    kwargs['voxel_size'] = voxel
    count = len(boundaries)+1
    for label in range(count):
        _check(cancelled)
        keep = labels == label
        if (keep & ~synthetic).sum() < 30:
            notes.append(f"段 {label+1} 观测不足，仅保留回波")
            continue

        def report(value, stage):
            if progress:
                progress(30+int(65*(label+max(value, 0)/100)/count), f"结构段 {label+1}/{count}：{stage}")

        try:
            mesh = reconstruct_surface(points[keep], origins[keep], confidence[keep],
                free_mask=np.asarray(free_mask)[keep] if free_mask is not None else None,
                range_limits=np.asarray(range_limits)[keep] if range_limits is not None else None,
                synthetic_mask=synthetic[keep],
                progress=report, **kwargs)
        except ValueError as error:
            # Only known support failures may degrade to points. Parameter and
            # programming errors must propagate instead of silently hiding them.
            if not any(message in str(error) for message in ('壁面有效点过少', '没有形成有效壁面距离场', '有效曲面为空')):
                raise
            notes.append(f"段 {label+1} 支持不足，仅保留回波")
            continue
        # Remove triangles that extrapolate through a declared boundary. This
        # clips support; it never creates a horizontal cap or an inferred ledge.
        triangle_z = mesh.vertices[mesh.faces, 2]
        inside = np.ones(len(mesh.faces), dtype=bool)
        if label:
            inside &= triangle_z.min(axis=1) >= boundaries[label-1]-1e-6
        if label < len(boundaries):
            inside &= triangle_z.max(axis=1) <= boundaries[label]+1e-6
        selected = mesh.faces[inside]
        if not len(selected):
            notes.append(f"段 {label+1} 分界内无有效面，仅保留回波")
            continue
        used, inverse = np.unique(selected, return_inverse=True)
        vertices.append(mesh.vertices[used])
        faces.append(inverse.reshape(-1, 3)+offset)
        weights.append(mesh.confidence[used])
        offset += len(used)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("各结构段均缺少曲面支持；请调整分界或保留点云对照。")
    note = f"分段壁面 {count} 段；段内局部平面/曲面拟合，分界不跨段平滑、不补封口"
    if kwargs.get('z_interpolation'):
        note += "；Z 向插值仅在各段内进行"
    if range_limits is not None:
        note += "；量程边界保持开放"
    if synthetic.any():
        note += f"；空间插值 {int(synthetic.sum()):,} 个辅助点（非实测）"
    if notes:
        note += "；" + "；".join(notes)
    _check(cancelled)
    if progress:
        progress(100, '分段曲面重建完成')
    return SurfaceMesh(np.vstack(vertices), np.vstack(faces).astype(np.int32), np.concatenate(weights),
                       max(m.voxel_size for m in meshes), int((~synthetic).sum()),
                       float(np.median([m.median_residual for m in meshes])), note,
                       sum(m.interpolated_points for m in meshes), sum(m.interpolated_cells for m in meshes), boundaries,
                       int(synthetic.sum()))
