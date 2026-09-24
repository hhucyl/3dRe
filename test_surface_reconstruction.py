import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from bp_reader import RingPoseCorrection, SensorSample, _transform_polar_point
from surface_reconstruction import (
    ReconstructionCancelled, _pose_interpolator, export_mesh, interpolate_sensor,
    reconstruct_surface, extract_wall_observations, interpolate_vertical_samples,
    interpolate_vertical_field, _oriented_samples,
)


def cylinder_observations(open_sector=False):
    angles = np.linspace(0, 2 * np.pi, 140, endpoint=False)
    if open_sector:
        angles = angles[(angles < 0.45 * np.pi) | (angles > 0.85 * np.pi)]
    angle, z = np.meshgrid(angles, np.linspace(0, 1.6, 21))
    points = np.column_stack((np.cos(angle.ravel()), np.sin(angle.ravel()), z.ravel()))
    origins = np.column_stack((np.zeros(angle.size), np.zeros(angle.size), z.ravel()))
    return points, origins, np.ones(len(points))


class SurfaceGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = reconstruct_surface(*cylinder_observations(), voxel_size=0.07)

    def test_cylinder_geometry_and_inward_winding(self):
        mesh = self.mesh
        self.assertGreater(len(mesh.faces), 1000)
        radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
        self.assertLess(np.quantile(abs(radius - 1), 0.95), 0.02)
        triangles = mesh.vertices[mesh.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        centers = triangles.mean(axis=1)
        inward = np.sum(normals[:, :2] * -centers[:, :2], axis=1)
        self.assertGreater(np.mean(inward > 0), 0.98)
        # No automatically generated floor/roof caps.
        self.assertGreater(radius.min(), 0.95)
        self.assertTrue(np.isfinite(mesh.vertices).all())
        self.assertTrue(((mesh.confidence >= 0) & (mesh.confidence <= 1)).all())

    def test_unknown_sector_is_not_closed(self):
        mesh = reconstruct_surface(*cylinder_observations(True), voxel_size=0.07)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        angle = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)
        self.assertFalse(np.any((angle > 0.57 * np.pi) & (angle < 0.73 * np.pi)))

    def test_memory_budget_changes_actual_spacing(self):
        mesh = reconstruct_surface(*cylinder_observations(), voxel_size=0.01, max_grid_cells=35000)
        self.assertGreater(mesh.voxel_size, 0.01)
        self.assertIn("内存预算", mesh.note)

    def test_cancellation_inside_fusion(self):
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls > 4

        with self.assertRaises(ReconstructionCancelled):
            reconstruct_surface(*cylinder_observations(), voxel_size=0.05, cancelled=cancel)

    def test_export_preserves_vertices_and_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            for extension in ("obj", "ply"):
                path = os.path.join(folder, "内壁." + extension)
                export_mesh(self.mesh, path)
                with open(path, encoding="utf-8") as stream:
                    lines = stream.read().splitlines()
                if extension == "obj":
                    self.assertEqual(sum(line.startswith("v ") for line in lines), len(self.mesh.vertices))
                    faces = [list(map(int, line.split()[1:])) for line in lines if line.startswith("f ")]
                    np.testing.assert_array_equal(np.array(faces) - 1, self.mesh.faces)
                else:
                    self.assertIn(f"element face {len(self.mesh.faces)}", lines)
                    self.assertIn("property float confidence", lines)
                    self.assertEqual(len(lines) - lines.index("end_header") - 1, len(self.mesh.vertices) + len(self.mesh.faces))

    def test_invalid_input(self):
        points, origins, weights = cylinder_observations()
        with self.assertRaises(ValueError):
            reconstruct_surface(points, origins, weights, voxel_size=0)
        with self.assertRaises(ValueError):
            reconstruct_surface(points[:3], origins[:3], weights[:3])

    def test_long_pipe_two_sidewalls_remain_straight_and_open(self):
        x, z = np.meshgrid(np.linspace(0, 4, 161), np.linspace(0, 1, 5))
        side = np.column_stack((x.ravel(), np.full(x.size, 0.5), z.ravel()))
        other = side.copy()
        other[:, 1] = -0.5
        points = np.vstack((side, other))
        origins = points.copy()
        origins[:, 0] = -0.5
        origins[:, 1] = 0
        mesh = reconstruct_surface(points, origins, np.ones(len(points)), voxel_size=0.05, connection_radius=0.22)
        self.assertLess(np.quantile(abs(abs(mesh.vertices[:, 1]) - 0.5), 0.95), 0.025)
        self.assertGreater(mesh.vertices[:, 0].max(), 3.95)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        self.assertFalse(np.any(abs(centers[:, 1]) < 0.35), "不能在探测末端或两侧壁之间生成封口")
        between_slices = (centers[:, 2] > 0.08) & (centers[:, 2] < 0.17)
        self.assertGreater(between_slices.sum(), 100)

    def test_non_circular_variable_section_is_preserved(self):
        from scipy.spatial import cKDTree

        z, t = np.meshgrid(np.linspace(0, 1.5, 31), np.linspace(-1, 1, 81))
        width = 0.7 + 0.2 * z.ravel()
        points = np.vstack((
            np.column_stack((width, t.ravel() * 0.5, z.ravel())),
            np.column_stack((-width, t.ravel() * 0.5, z.ravel())),
            np.column_stack((t.ravel() * width, np.full(t.size, 0.5), z.ravel())),
            np.column_stack((t.ravel() * width, np.full(t.size, -0.5), z.ravel())),
        ))
        origins = points.copy()
        origins[:, :2] = 0
        mesh = reconstruct_surface(points, origins, np.ones(len(points)), voxel_size=0.05)
        distances, _ = cKDTree(points).query(mesh.vertices)
        self.assertLess(np.quantile(distances, 0.95), 0.08)
        top = mesh.vertices[(mesh.vertices[:, 2] > 1.3) & (abs(mesh.vertices[:, 1]) < 0.3)]
        bottom = mesh.vertices[(mesh.vertices[:, 2] < 0.2) & (abs(mesh.vertices[:, 1]) < 0.3)]
        self.assertGreater(np.median(abs(top[:, 0])) - np.median(abs(bottom[:, 0])), 0.20)
        # The narrow dimension must remain narrow, rather than fitting a circle.
        self.assertLess(np.quantile(abs(mesh.vertices[:, 1]), 0.99), 0.57)

    def test_multiple_supported_returns_do_not_carve_nearer_wall(self):
        echo = np.full(400, 3.0)
        bins = np.arange(400)
        echo += 180 * np.exp(-((bins - 100) / 2) ** 2) + 150 * np.exp(-((bins - 300) / 2) ** 2)
        raw = echo.astype(np.uint8).tobytes()
        scans = [SimpleNamespace(timestamp=1000 + i * 10, angle_deg=i * 0.45, range_m=4.0) for i in range(64)]
        recording = SimpleNamespace(echo_scans=scans, frame_count=len(scans), sensor_samples=[],
                                    echo_data=lambda scan: raw)
        points, origins, weights, free = extract_wall_observations(recording, return_free_mask=True)
        ranges = np.linalg.norm(points - origins, axis=1)
        self.assertGreater(np.sum(ranges > 2.5), 40, "有邻域支持的第二壁面不能被单峰选择删掉")
        self.assertTrue(np.all(ranges[free] < 1.1), "远回波不能穿过并清除更近的已保留壁面")


class PoseInterpolationTest(unittest.TestCase):
    def test_heading_wrap_and_depth(self):
        recording = SimpleNamespace(
            sensor_samples=[SensorSample(1000, 359, water_height=1), SensorSample(2000, 1, water_height=2)],
            _sensor_times=[1000, 2000],
        )
        depth, heading, weight = interpolate_sensor(recording, 1500)
        self.assertEqual(depth, 1.5)
        self.assertAlmostEqual(heading, 0)
        self.assertEqual(weight, 1)
        self.assertLess(interpolate_sensor(recording, 6000)[2], 0.1)

    def test_rotation_matches_existing_coordinate_convention(self):
        correction = RingPoseCorrection(1000, 0.4, -0.2, 15, -23, 0.15, 12)
        rotation, translation = _pose_interpolator([correction])(1000)
        theta = np.deg2rad(42)
        point = rotation @ (np.array([np.sin(theta), np.cos(theta), 0]) * 2.5) + translation
        point[2] += 1.2
        expected = _transform_polar_point(42, 2.5, 1.2, 0, correction)
        np.testing.assert_allclose(point, expected, atol=1e-10)

    def test_ring_pose_is_continuous(self):
        a = RingPoseCorrection(0, 0, 0, 0, 0, 0, 359)
        b = RingPoseCorrection(1000, 1, 2, 0, 0, 0.4, 1)
        rotation, translation = _pose_interpolator([a, b])(500)
        np.testing.assert_allclose(translation, [0.5, 1, 0.2])
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)


class VerticalInterpolationTest(unittest.TestCase):
    def observations(self):
        x, z = np.meshgrid(np.linspace(0, 2, 81), [0.0, 0.45, 0.9])
        points = np.column_stack((x.ravel(), 0.5 + 0.15 * z.ravel(), z.ravel()))
        origins = points.copy()
        origins[:, 0] = -0.5
        origins[:, 1] = 0
        normals = np.tile([0.0, -1.0, 0.0], (len(points), 1))
        return points, origins, normals, np.ones(len(points)), np.full(len(points), 0.2)

    def test_fills_z_gaps_in_tapered_wall_without_changing_measurements(self):
        points, origins, normals, confidence, radii = self.observations()
        result = interpolate_vertical_samples(points, origins, normals, confidence, radii, 0.05)
        generated = result[-1]
        self.assertGreater(generated, 100)
        np.testing.assert_array_equal(result[0][:len(points)], points)
        self.assertLessEqual(result[3][len(points):].max(), 0.3)
        self.assertGreaterEqual(result[0][:, 2].min(), 0)
        self.assertLessEqual(result[0][:, 2].max(), 0.9)
        baseline = reconstruct_surface(points, origins, confidence, voxel_size=0.05)
        mesh = reconstruct_surface(points, origins, confidence, voxel_size=0.05, z_interpolation=True)
        base_centers = baseline.vertices[baseline.faces].mean(axis=1)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        self.assertFalse(np.any((base_centers[:, 2] > 0.15) & (base_centers[:, 2] < 0.3)))
        self.assertGreater(np.sum((centers[:, 2] > 0.15) & (centers[:, 2] < 0.3)), 100)
        self.assertLess(np.quantile(abs(mesh.vertices[:, 1] - (0.5 + 0.15 * mesh.vertices[:, 2])), 0.95), 0.025)
        self.assertEqual(mesh.observation_count, len(points))
        self.assertGreater(mesh.interpolated_points, 0)
        self.assertIn("非实测", mesh.note)

    def test_maximum_gap_is_respected(self):
        result = interpolate_vertical_samples(*self.observations(), 0.05, max_gap=0.3)
        self.assertEqual(result[-1], 0)

    def test_conflicting_middle_layer_is_not_skipped(self):
        points, origins, normals, confidence, radii = self.observations()
        normals[(points[:, 2] > 0.4) & (points[:, 2] < 0.5)] *= -1
        result = interpolate_vertical_samples(points, origins, normals, confidence, radii, 0.05, max_gap=1.0)
        self.assertEqual(result[-1], 0)

    def test_lateral_jump_is_not_bridged(self):
        points, origins, normals, confidence, radii = self.observations()
        points[points[:, 2] > 0.4, 1] += 1
        result = interpolate_vertical_samples(points, origins, normals, confidence, radii, 0.05)
        # Upper pair can interpolate, but there must be no connector across the jump.
        extra = result[0][len(points):]
        self.assertFalse(np.any(extra[:, 2] < 0.45))

    def test_parallel_pipe_walls_keep_open_ends_after_interpolation(self):
        points, origins, _, confidence, _ = self.observations()
        other = points.copy()
        other[:, 1] *= -1
        mesh = reconstruct_surface(np.vstack((points, other)), np.vstack((origins, origins)),
                                   np.r_[confidence, confidence], voxel_size=0.05, z_interpolation=True)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        self.assertFalse(np.any(abs(centers[:, 1]) < 0.4))
        self.assertLessEqual(mesh.vertices[:, 0].max(), 2.1)

    def test_noisy_thin_slices_form_vertical_walls_instead_of_horizontal_ribbons(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        rng = np.random.default_rng(9)
        x, z = np.meshgrid(np.linspace(0, 2, 201), [0.0, 0.4, 0.8])
        points = np.column_stack((x.ravel(), 0.5 + rng.normal(0, 0.012, x.size), z.ravel()))
        origins = points.copy()
        origins[:, :2] = 0
        confidence = np.ones(len(points))
        _, _, normals, _, _ = _oriented_samples(points, origins, confidence, 0.04, vertical_continuity=True)
        self.assertGreater(np.mean(abs(normals[:, 1]) > 0.95), 0.95)
        mesh = reconstruct_surface(points, origins, confidence, voxel_size=0.04, z_interpolation=True)
        edges = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(mesh.vertices),) * 2)
        _, labels = connected_components(graph, directed=False)
        counts = np.bincount(labels[mesh.faces[:, 0]])
        self.assertGreater(counts.max() / len(mesh.faces), 0.98)
        self.assertLess(np.quantile(abs(mesh.vertices[:, 1] - 0.5), 0.95), 0.03)

    def test_range_clipped_pipe_walls_are_connected_without_end_caps(self):
        x, z = np.meshgrid(np.linspace(0, 4, 201), [0, 0.4, 0.8])
        side = np.column_stack((x.ravel(), np.full(x.size, 0.5), z.ravel()))
        points = np.vstack((side, side * [1, -1, 1]))
        origins = points.copy()
        origins[:, :2] = 0
        limit = 2.2
        keep = np.linalg.norm(points - origins, axis=1) < limit
        points, origins = points[keep], origins[keep]
        mesh = reconstruct_surface(points, origins, np.ones(len(points)), voxel_size=0.04,
                                   z_interpolation=True, range_limits=np.full(len(points), limit))
        self.assertLessEqual(np.linalg.norm(mesh.vertices[:, :2], axis=1).max(), limit + 1e-6)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        self.assertFalse(np.any(abs(centers[:, 1]) < 0.40), "不能把量程边界连成横向端盖")
        self.assertGreater(np.sum((centers[:, 2] > 0.1) & (centers[:, 2] < 0.3)), 100)
        self.assertIn("量程边界保持开放", mesh.note)

    def test_vertical_interpolation_can_be_cancelled(self):
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls > 5

        with self.assertRaises(ReconstructionCancelled):
            interpolate_vertical_samples(*self.observations(), 0.05, cancelled=cancel)


class VerticalFieldTest(unittest.TestCase):
    def columns(self):
        field = np.zeros((1, 1, 15), dtype=np.float32)
        known = np.zeros_like(field, dtype=bool)
        known[..., [1, 5, 9, 13]] = True
        field[..., [1, 5, 9, 13]] = [-0.09, -0.04, 0.02, 0.09]
        confidence = np.full_like(field, 0.6)
        normals = np.zeros((*field.shape, 3))
        normals[..., 1] = -1
        return field, known, confidence, normals

    def test_cubic_field_connects_support_without_extrapolating_or_overshooting(self):
        field, known, confidence, normals = self.columns()
        before = field.copy()
        observed = known.copy()
        count = interpolate_vertical_field(field, known, confidence, normals, 0.05, 0.3, 0.2, 0.3)
        self.assertEqual(count, 9)
        self.assertTrue(known[..., 1:14].all())
        self.assertFalse(known[..., 0].any() or known[..., 14].any())
        np.testing.assert_array_equal(field[observed], before[observed])
        self.assertTrue(np.all(np.diff(field[0, 0, 1:14]) > 0))
        self.assertLessEqual(confidence[known & ~observed].max(), 0.181)

    def test_gap_limit_and_opposite_wall_stop_field_interpolation(self):
        data = self.columns()
        self.assertEqual(interpolate_vertical_field(*data, 0.05, 0.15, 0.2, 0.3), 0)
        data = self.columns()
        data[3][..., [5, 13], 1] = 1
        self.assertEqual(interpolate_vertical_field(*data, 0.05, 0.3, 0.2, 0.3), 0)
        data = self.columns()
        heights = np.zeros_like(data[0])
        heights[..., [1, 5, 9, 13]] = [0, 0.6, 1.2, 1.8]
        self.assertEqual(interpolate_vertical_field(*data, 0.05, 0.3, 0.2, 0.3, source_height=heights), 0)

    def test_recorded_range_boundary_remains_unknown(self):
        data = self.columns()
        margin = np.full_like(data[0], np.nan)
        margin[data[1]] = 0.2
        margin[..., 6:9] = -0.1
        interpolate_vertical_field(*data, 0.05, 0.3, 0.2, 0.3, range_margin=margin)
        self.assertFalse(data[1][..., 6:9].any())
        self.assertTrue(data[1][..., 2:5].all())

    def test_field_stage_can_be_cancelled(self):
        with self.assertRaises(ReconstructionCancelled):
            interpolate_vertical_field(*self.columns(), 0.05, 0.3, 0.2, 0.3, cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main()
