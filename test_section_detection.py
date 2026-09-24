import unittest
from types import SimpleNamespace
import numpy as np

from bp_reader import SensorSample
from section_detection import profile_features, propose_sections


def pipe_profile(direction=0, second=False):
    theta = (np.arange(800)*.45-direction+180) % 360-180
    ranges = np.ones(800)
    for angles in (theta, (theta+180+180) % 360-180) if second else (theta,):
        inside = abs(angles) < 21
        ranges[inside] = np.minimum(3, .35/np.maximum(abs(np.sin(np.deg2rad(angles[inside]))), 1e-9))
    return ranges


def recording(profiles, heights=None):
    heights = list(np.arange(len(profiles))*.1) if heights is None else heights
    scans = [SimpleNamespace(timestamp=i*1000, range_m=3) for i in range(len(profiles))]
    sensors = [SensorSample(i*1000, 0, water_height=float(h)) for i, h in enumerate(heights)]
    return SimpleNamespace(profile_scans=scans, sensor_samples=sensors, _sensor_times=[s.timestamp for s in sensors],
                           profile_distances=lambda p: profiles[p.timestamp//1000])


class SectionDetectionTest(unittest.TestCase):
    def test_narrow_pipe_band_splits_entry_and_exit_despite_unchanged_quantiles(self):
        circle, pipe = np.ones(800), pipe_profile()
        np.testing.assert_allclose(profile_features(circle, 3)[0], profile_features(pipe, 3)[0])
        r = recording([circle]*4+[pipe]*2+[circle]*4)
        proposal = propose_sections(r)
        self.assertEqual(proposal.boundaries, (.35, .55))
        self.assertIn('出现', proposal.reasons[0])
        self.assertIn('消失', proposal.reasons[1])
        self.assertIn('疑似有管道', proposal.sections[1].state)
        self.assertEqual(propose_sections(r, include_pipes=False).boundaries, ())

    def test_multiple_pipe_bands_and_pipe_count_changes(self):
        circle, pipe, two = np.ones(800), pipe_profile(), pipe_profile(second=True)
        proposal = propose_sections(recording([circle]*3+[pipe]*3+[two]*3+[circle]*3+[pipe]*3+[circle]*3))
        self.assertEqual(proposal.boundaries, (.25, .55, .85, 1.15, 1.45))
        self.assertIn('数量变化', proposal.reasons[1])

    def test_wrap_and_compass_rotation_do_not_change_pipe_evidence(self):
        for direction in (0, 55, 180, 355):
            self.assertEqual(profile_features(pipe_profile(direction), 3)[1], 1)

    def test_cuboid_corner_and_missing_sector_are_not_declared_pipe(self):
        theta = np.deg2rad(np.arange(800)*.45)
        box = np.minimum(.8/np.maximum(abs(np.sin(theta)), 1e-9), 1/np.maximum(abs(np.cos(theta)), 1e-9))
        self.assertEqual(profile_features(box, 3)[1], 0)
        for value in (0, 3, float('nan')):
            missing = np.ones(800)
            missing[100:260] = value
            self.assertEqual(profile_features(missing, 3)[1], -1)
            result = propose_sections(recording([np.ones(800)]*3+[missing]*3))
            self.assertIn('不能判定', result.reasons[0])
            self.assertIn('不确定', result.sections[-1].state)

    def test_single_layer_pipe_outlier_does_not_split(self):
        result = propose_sections(recording([np.ones(800)]*4+[pipe_profile()]+[np.ones(800)]*4))
        self.assertEqual(result.boundaries, ())

    def test_multiple_shape_changes_remain_available(self):
        result = propose_sections(recording([np.ones(800)]*4+[np.full(800, 1.6)]*4+[np.full(800, .8)]*4))
        self.assertEqual(result.boundaries, (.35, .75))

    def test_revisited_heights_do_not_create_acquisition_order_segments(self):
        profiles = [np.ones(800)]*3+[pipe_profile()]*3
        result = propose_sections(recording(profiles+profiles[::-1], list(np.arange(6)*.1)+list(np.arange(6)[::-1]*.1)))
        self.assertEqual(result.boundaries, (.25,))

    def test_large_depth_gap_is_not_a_pipe_transition(self):
        result = propose_sections(recording([np.ones(800)]*3+[pipe_profile()]*3, [0,.1,.2,2,2.1,2.2]))
        self.assertEqual(result.boundaries, ())


if __name__ == '__main__':
    unittest.main()
