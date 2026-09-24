import unittest

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from manhole_reconstruction import (
    fit_reference_wall, fit_periodic_wall, fit_connected_pipes, mesh_wall_and_pipes,
)


def shaft_samples():
    angle, z = np.meshgrid(np.linspace(0, 2 * np.pi, 160, endpoint=False), np.linspace(0, 2, 41))
    points = np.column_stack((np.cos(angle.ravel()), np.sin(angle.ravel()), z.ravel()))
    return points


class ManholeReconstructionTest(unittest.TestCase):
    def test_reference_circle_rejects_internal_clutter(self):
        rng = np.random.default_rng(5)
        wall = shaft_samples()
        wall[:, :2] += [0.12, -0.08]
        wall[:, :2] += rng.normal(0, 0.008, wall[:, :2].shape)
        angle = rng.uniform(0, 2 * np.pi, 8000)
        radius = rng.uniform(0.10, 0.72, 8000)
        clutter = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0, 2, 8000)))
        points = np.vstack((wall, clutter))
        origins = points.copy()
        origins[:, :2] = 0
        center, radius = fit_reference_wall(points, origins, np.ones(len(points)))
        np.testing.assert_allclose(center, [0.12, -0.08], atol=0.05)
        self.assertAlmostEqual(radius, 1, delta=0.05)

    def test_continuous_wall_bridges_missing_slices_and_rejects_noise(self):
        rng = np.random.default_rng(8)
        points = shaft_samples()
        keep = ~((points[:, 2] > 0.7) & (points[:, 2] < 1.1))
        points = points[keep]
        points[:, :2] *= (1 + rng.normal(0, 0.02, len(points)))[:, None]
        noise = points[::6].copy()
        noise[:, :2] *= 0.72
        points = np.vstack((points, noise))
        mesh = fit_periodic_wall(points, np.ones(len(points)), np.zeros(2), 1, voxel_size=0.06, gap_limit=0.6)
        radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
        self.assertLess(np.quantile(abs(radius - 1), 0.95), 0.05)
        self.assertGreater(radius.min(), 0.90)
        middle = (mesh.vertices[:, 2] > 0.8) & (mesh.vertices[:, 2] < 1.0)
        self.assertGreater(middle.sum(), 100)
        self.assertLess(np.median(mesh.confidence[middle]), 0.35)

    def test_large_missing_area_remains_unknown(self):
        points = shaft_samples()
        points = points[(points[:, 2] < 0.4) | (points[:, 2] > 1.6)]
        mesh = fit_periodic_wall(points, np.ones(len(points)), np.zeros(2), 1, voxel_size=0.06, gap_limit=0.3)
        centers = mesh.vertices[mesh.faces].mean(axis=1)
        self.assertFalse(np.any((centers[:, 2] > 0.8) & (centers[:, 2] < 1.2)))

    def test_pipe_recognition_and_connected_union(self):
        phase, x = np.meshgrid(np.linspace(0, 2 * np.pi, 80, endpoint=False), np.linspace(1, 2, 25))
        pipe_points = np.column_stack((x.ravel(), 0.30 * np.cos(phase.ravel()), 1 + 0.30 * np.sin(phase.ravel())))
        distance = np.linalg.norm(pipe_points[:, :2], axis=1)
        mouths = pipe_points.copy()
        mouths[:, :2] /= distance[:, None]
        pipes = fit_connected_pipes(pipe_points, np.ones(len(pipe_points)), mouths, np.zeros(2), 1, 0.06)
        self.assertEqual(len(pipes), 1)
        self.assertAlmostEqual(pipes[0]["radius"], 0.30, delta=0.03)
        self.assertGreater(pipes[0]["length"], 0.8)
        wall = shaft_samples()
        in_mouth = (wall[:, 0] > 0) & (wall[:, 1] ** 2 + (wall[:, 2] - 1) ** 2 < 0.3 ** 2)
        wall = wall[~in_mouth]
        mesh, field = fit_periodic_wall(wall, np.ones(len(wall)), np.zeros(2), 1, voxel_size=0.06,
                                        openings=mouths, return_field=True)
        network = mesh_wall_and_pipes(mesh, field, pipes, 0.06)
        self.assertGreater(network.vertices[:, 0].max(), 1.8)
        faces = network.faces
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                  shape=(len(network.vertices), len(network.vertices)))
        count, _ = connected_components(graph, directed=False)
        self.assertEqual(count, 1, "井筒与支管必须连成同一网格")
        # No sealing disk across the measured far pipe end.
        centers = network.vertices[faces].mean(axis=1)
        central_end = (centers[:, 0] > 1.7) & (centers[:, 1] ** 2 + (centers[:, 2] - 1) ** 2 < 0.15 ** 2)
        self.assertFalse(central_end.any())

    def test_far_range_sheet_is_not_invented_as_pipe(self):
        y, z = np.meshgrid(np.linspace(-0.3, 0.3, 20), np.linspace(0.7, 1.3, 20))
        points = np.column_stack((np.full(y.size, 3), y.ravel(), z.ravel()))
        mouths = points.copy()
        mouths[:, :2] /= np.linalg.norm(mouths[:, :2], axis=1, keepdims=True)
        pipes = fit_connected_pipes(points, np.ones(len(points)), mouths, np.zeros(2), 1, 0.06)
        self.assertEqual(pipes, [])


if __name__ == "__main__":
    unittest.main()
