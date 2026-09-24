import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from bp_reader import BpRecording, RingPoseCorrection, SensorSample
from slice_rotation import make_slice_corrections, fit_rotations, validate_corrections
from surface_reconstruction import _pose_interpolator, ReconstructionCancelled


class SliceRotationTest(unittest.TestCase):
    def test_partial_sweeps_and_direction_changes_are_covered(self):
        angles = list(np.arange(40, 900, 0.45) % 360) + list(np.arange(179, 70, -0.45))
        scans = [SimpleNamespace(timestamp=i*5, angle_deg=a) for i, a in enumerate(angles)]
        corrections = make_slice_corrections(SimpleNamespace(echo_scans=scans))
        self.assertEqual(corrections[0].slice_start, 0)
        self.assertGreaterEqual(len(corrections), 4)
        self.assertGreaterEqual(corrections[-1].timestamp, scans[-1].timestamp-700)
        validate_corrections(corrections)

    def test_fixed_origin_rotation_preserves_range_and_slice_plane(self):
        a = RingPoseCorrection(500, 0, 0, -12, 18, 0, 7, slice_start=0)
        b = replace(a, timestamp=1500, slice_start=1000, pitch_deg=-8)
        at = _pose_interpolator([a, b])
        for timestamp, correction in ((0, a), (999, a), (1000, b), (2000, b)):
            matrix, translation = at(timestamp)
            expected = Rotation.from_euler('xyz', [correction.roll_deg, correction.pitch_deg, -7], degrees=True).as_matrix()
            np.testing.assert_allclose(matrix, expected)
            np.testing.assert_array_equal(translation, np.zeros(3))
            ray = matrix @ [1.2, 1.6, 0]
            self.assertAlmostEqual(np.linalg.norm(ray), 2)
            self.assertAlmostEqual(np.dot(ray, matrix[:, 2]), 0)
        with self.assertRaises(ValueError):
            validate_corrections([replace(a, local_x_m=0.01)])
        with self.assertRaises(ValueError):
            validate_corrections([replace(a, pitch_deg=float('nan'))])

    def test_cloud_uses_same_rotations_and_interpolated_sensor_as_surface(self):
        r = object.__new__(BpRecording)
        r.sensor_samples = [SensorSample(0, 350, water_height=1), SensorSample(1000, 10, water_height=2)]
        r._sensor_times = [0, 1000]
        r.echo_scans = [SimpleNamespace(timestamp=500, angle_deg=90, range_m=2)]
        r.profile_scans = []
        r.echo_data = lambda scan: bytes([0, 0, 0, 0, 200, 0, 0, 0, 0, 0])
        correction = RingPoseCorrection(500, 0, 0, 15, 20, 0, 6, slice_start=0)
        cloud = r.build_point_cloud(ring_corrections=[correction])
        matrix, origin = _pose_interpolator([correction])(500)
        expected = matrix @ [0.9, 0, 0] + origin + [0, 0, 1.5]
        np.testing.assert_allclose(cloud.points[0][:3], expected)
        zero = replace(correction, pitch_deg=0, roll_deg=0, yaw_correction_deg=0)
        np.testing.assert_allclose(r.build_point_cloud().points, r.build_point_cloud(ring_corrections=[zero]).points)
        r._map = None

    @staticmethod
    def pipe_fixture():
        initial, local, origins, headings, ids = [], [], [], [], []
        for i, z in enumerate(np.linspace(0, 1.2, 9)):
            initial.append(RingPoseCorrection(i*1000+500, 0, 0, 0, 0, 0, 0, slice_start=i*1000))
            true_c = 7*math.sin(i*0.9)
            matrix = Rotation.from_euler('z', -true_c, degrees=True).as_matrix()
            for angle in np.linspace(0, 2*np.pi, 180, endpoint=False):
                direction = np.array([math.sin(angle), math.cos(angle), 0])
                world = matrix @ direction
                distance = 0.7/max(abs(world[1]), 1e-10)
                if distance > 3:
                    continue
                local.append(direction*distance)
                origins.append([0, 0, z])
                headings.append(0)
                ids.append(i)
        return np.array(local), np.array(origins), headings, np.array(ids), tuple(initial)

    def test_open_straight_pipe_alignment_improves_without_moving_origins(self):
        local, origins, headings, ids, initial = self.pipe_fixture()
        result = fit_rotations(local, origins, headings, ids, initial, max_gap=0.4, iterations=5)
        self.assertLess(result.after_m, result.before_m*0.85)
        matrices = Rotation.from_euler('xyz', [[c.roll_deg, c.pitch_deg, -c.yaw_correction_deg] for c in result.corrections], degrees=True).as_matrix()
        points = np.einsum('nij,nj->ni', matrices[ids], local) + origins
        self.assertLess(np.median(abs(abs(points[:, 1])-0.7)), np.median(abs(abs(local[:, 1])-0.7))*0.75)
        np.testing.assert_allclose(np.linalg.norm(points-origins, axis=1), np.linalg.norm(local, axis=1))
        self.assertEqual(result.corrections[0].yaw_correction_deg, 0)
        validate_corrections(result.corrections)

    def test_cancellation_and_insufficient_overlap(self):
        local, origins, headings, ids, initial = self.pipe_fixture()
        with self.assertRaises(ReconstructionCancelled):
            fit_rotations(local, origins, headings, ids, initial, cancelled=lambda: True)
        with self.assertRaisesRegex(ValueError, '缺少可匹配'):
            fit_rotations(local, origins, headings, ids, initial, max_gap=0.01)

    def test_reference_includes_last_partial_sweep(self):
        path = next((Path(__file__).parent/'data').rglob('*473ws0225*102942.bp'), None)
        if not path:
            self.skipTest('参考 BP 不存在')
        with BpRecording(str(path)) as recording:
            corrections = make_slice_corrections(recording)
            self.assertEqual(len(corrections), 27)
            self.assertEqual(corrections[0].slice_start, recording.echo_scans[0].timestamp)
            validate_corrections(corrections)


if __name__ == '__main__':
    unittest.main()
