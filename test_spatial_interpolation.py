import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from spatial_interpolation import interpolate_spatial
from surface_reconstruction import ReconstructionCancelled, export_mesh
from surface_segments import reconstruct_sections


def wall():
    x, z = np.meshgrid(np.arange(0, 1.01, .06), [0, .24, .48, .72, .96])
    points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    return points, points+[0, 1, 0], np.ones(len(points)), np.ones(len(points), bool), np.full(len(points), 3.)


class SpatialInterpolationTest(unittest.TestCase):
    def test_arbitrary_plane_interpolation_preserves_originals_and_range(self):
        p, o, q, free, limits = wall()
        rotation = Rotation.from_euler('xyz', [35, 27, 18], degrees=True).as_matrix()
        p, o = p @ rotation.T, o @ rotation.T
        result = interpolate_spatial(p, o, q, free, limits)
        self.assertGreater(result.added_count, 50)
        np.testing.assert_array_equal(result.points[:len(p)], p)
        np.testing.assert_array_equal(result.origins[:len(p)], o)
        np.testing.assert_array_equal(result.confidence[:len(p)], q)
        self.assertFalse(result.points.flags.writeable)
        self.assertFalse(result.free_mask[result.synthetic].any())
        extra = result.points[result.synthetic]
        self.assertLess(abs(extra @ rotation[:, 1]).max(), 1e-8)
        self.assertTrue((np.linalg.norm(result.points-result.origins, axis=1) <= result.range_limits).all())
        self.assertTrue((result.confidence[result.synthetic] <= .3).all())

    def test_sections_and_connection_limit_prevent_bridging(self):
        result = interpolate_spatial(*wall(), boundaries=[.36])
        z = result.points[result.synthetic, 2]
        self.assertFalse(np.any((z > .241) & (z < .479)))
        short = interpolate_spatial(*wall(), max_gap=.12)
        self.assertEqual(short.added_count, 0)

    def test_opposite_pipe_walls_remain_separate(self):
        p, _, q, free, limits = wall()
        p[:, 1] = .12
        other = p.copy(); other[:, 1] = -.12
        p = np.vstack((p, other)); o = p.copy(); o[:, 1] = 0
        result = interpolate_spatial(p, o, np.r_[q,q], np.r_[free,free], np.r_[limits,limits], max_gap=.3)
        self.assertTrue(np.all(abs(result.points[result.synthetic, 1]) > .1))

    def test_same_prepared_points_generate_mesh_and_export_counts(self):
        data = interpolate_spatial(*wall())
        mesh = reconstruct_sections(data.points, data.origins, data.confidence, (), free_mask=data.free_mask,
            range_limits=data.range_limits, synthetic_mask=data.synthetic, voxel_size=.06)
        self.assertEqual(mesh.observation_count, len(wall()[0]))
        self.assertEqual(mesh.spatial_interpolated_points, data.added_count)
        self.assertGreater(len(mesh.faces), 100)
        with tempfile.TemporaryDirectory() as folder:
            for extension in ('ply', 'obj'):
                path = Path(folder)/('spatial.'+extension)
                export_mesh(mesh, str(path))
                self.assertIn(f'spatial_interpolated_points {data.added_count}', path.read_text())

    def test_cancellation_invalid_parameters_and_budget(self):
        with self.assertRaises(ReconstructionCancelled):
            interpolate_spatial(*wall(), cancelled=lambda: True)
        with self.assertRaises(ValueError):
            interpolate_spatial(*wall(), spacing=.2, max_gap=.3)
        result = interpolate_spatial(*wall(), max_new_points=20)
        self.assertLessEqual(result.added_count, 20)
        self.assertIn('预算', result.note)


if __name__ == '__main__':
    unittest.main()
