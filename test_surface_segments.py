import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from bp_reader import RingPoseCorrection, SensorSample
from slice_rotation import fit_rotations
from surface_reconstruction import ReconstructionCancelled
from surface_segments import (parse_boundaries, section_ids, reconstruct_sections,
                              suggest_boundaries, validate_boundaries)
import test_slice_rotation


def composite_observations():
    """A narrow cylindrical shaft below a rectangular chamber, with a ledge gap."""
    angle, z = np.meshgrid(np.linspace(0, 2*np.pi, 120, endpoint=False), np.linspace(0, .88, 12))
    cylinder = np.column_stack((.65*np.cos(angle.ravel()), .65*np.sin(angle.ravel()), z.ravel()))
    t, z = np.meshgrid(np.linspace(-1, 1, 75), np.linspace(1.12, 2, 12))
    t, z = t.ravel(), z.ravel()
    box = np.vstack((np.column_stack((t*.9, np.full(len(t), .75), z)),
                     np.column_stack((t*.9, np.full(len(t), -.75), z)),
                     np.column_stack((np.full(len(t), .9), t*.75, z)),
                     np.column_stack((np.full(len(t), -.9), t*.75, z))))
    points = np.vstack((cylinder, box))
    origins = np.column_stack((np.zeros((len(points), 2)), points[:, 2]))
    return points, origins, np.ones(len(points))


class SurfaceSegmentsTest(unittest.TestCase):
    def test_spatial_sections_allow_one_tilted_sweep_to_cross_boundaries(self):
        correction = RingPoseCorrection(1, 0, 0, 0, 30, 0, 0, slice_start=0)
        from surface_reconstruction import _pose_interpolator
        rotation, _ = _pose_interpolator([correction])(1)
        points = np.array([[1, 0, 0], [-1, 0, 0]]) @ rotation.T + [0, 0, 1]
        np.testing.assert_array_equal(section_ids(points, [1]), [0, 1])
        self.assertEqual(parse_boundaries('2.3，0.8'), (.8, 2.3))
        self.assertEqual(parse_boundaries(''), ())
        for boundaries in ((1, 1), (2, 1), (float('nan'),)):
            with self.assertRaises(ValueError):
                validate_boundaries(boundaries)

    def test_cylinder_and_box_keep_shape_without_synthetic_transition(self):
        points, origins, weights = composite_observations()
        before = points.copy()
        mesh = reconstruct_sections(points, origins, weights, [1], voxel_size=.055,
            connection_radius=.35, z_interpolation=True, z_max_gap=.6,
            free_mask=np.zeros(len(points), dtype=bool))
        np.testing.assert_array_equal(points, before)
        triangles = mesh.vertices[mesh.faces]
        self.assertFalse(np.any((triangles[:, :, 2].min(axis=1) < 1) &
                                (triangles[:, :, 2].max(axis=1) > 1)))
        lower = mesh.vertices[mesh.vertices[:, 2] < 1]
        upper = mesh.vertices[mesh.vertices[:, 2] >= 1]
        self.assertGreater(len(lower), 500)
        self.assertGreater(len(upper), 500)
        self.assertLess(np.quantile(abs(np.linalg.norm(lower[:, :2], axis=1)-.65), .95), .035)
        box_error = np.minimum(abs(abs(upper[:, 0])-.9), abs(abs(upper[:, 1])-.75))
        self.assertLess(np.quantile(box_error, .95), .035)
        # The unmeasured ledge must not become a smooth diagonal connection.
        centers = triangles.mean(axis=1)
        self.assertFalse(np.any((centers[:, 2] > .97) & (centers[:, 2] < 1.03)))
        self.assertIn('分段壁面 2 段', mesh.note)
        self.assertIn('Z 向插值仅在各段内', mesh.note)
        self.assertEqual(mesh.boundaries, (1.0,))
        import tempfile
        from pathlib import Path
        from surface_reconstruction import export_mesh
        with tempfile.TemporaryDirectory() as folder:
            for extension in ('ply', 'obj'):
                path = Path(folder)/('segments.'+extension)
                export_mesh(mesh, str(path))
                self.assertIn('structure_boundaries_z_m 1', path.read_text(encoding='utf-8'))

    def test_fitting_does_not_borrow_support_across_sections(self):
        local, origins, headings, ids, initial = test_slice_rotation.SliceRotationTest.pipe_fixture()
        keep = ids < 4
        with self.assertRaisesRegex(ValueError, '缺少可匹配'):
            fit_rotations(local[keep], origins[keep], np.array(headings)[keep], ids[keep], initial[:4],
                          max_gap=.5, boundaries=[.225])

    def test_fitting_reports_section_metrics_separately(self):
        local, origins, headings, ids, initial = test_slice_rotation.SliceRotationTest.pipe_fixture()
        result = fit_rotations(local, origins, headings, ids, initial, max_gap=.5,
                               iterations=3, boundaries=[.525])
        self.assertEqual(result.boundaries, (.525,))
        self.assertEqual({row[0] for row in result.segment_metrics}, {1, 2})
        self.assertTrue(all(row[1] > 0 for row in result.segment_metrics))
        self.assertEqual(result.corrections[0].yaw_correction_deg, 0)

    def test_segmentation_cancel_and_sparse_sections(self):
        data = composite_observations()
        with self.assertRaises(ReconstructionCancelled):
            reconstruct_sections(*data, [1], cancelled=lambda: True)
        mesh = reconstruct_sections(*data, [-1, 1], voxel_size=.09)
        self.assertIn('段 1 观测不足', mesh.note)
        self.assertGreater(len(mesh.faces), 100)

    def test_profile_suggestions_require_persistent_changes(self):
        profiles = [SimpleNamespace(timestamp=i*1000, range_m=3) for i in range(10)]
        samples = [SensorSample(i*1000, 0, water_height=i*.15) for i in range(10)]
        recording = SimpleNamespace(profile_scans=profiles, sensor_samples=samples,
            _sensor_times=[s.timestamp for s in samples],
            profile_distances=lambda p: [1 if p.timestamp < 5000 else 1.6]*800)
        boundaries = suggest_boundaries(recording)
        self.assertEqual(boundaries, (.675,))
        recording.profile_distances = lambda p: [1.6 if p.timestamp == 5000 else 1]*800
        self.assertEqual(suggest_boundaries(recording), ())


if __name__ == '__main__':
    unittest.main()
