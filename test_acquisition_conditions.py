import unittest
from types import SimpleNamespace as NS
import numpy as np

from acquisition_conditions import RangeChecks, common_visibility
from section_detection import propose_sections
from test_section_detection import recording, pipe_profile
import test_slice_rotation
from slice_rotation import fit_rotations, make_slice_corrections
from bp_reader import BpRecording
from surface_reconstruction import extract_wall_observations


def varied_recording(profiles, limits):
    r = recording(profiles)
    for p, limit in zip(r.profile_scans, limits):
        p.range_m = limit
    return r


class AcquisitionConditionsTest(unittest.TestCase):
    def test_range_expansion_and_contraction_do_not_create_pipe_boundary(self):
        theta = (np.arange(800)*.45+180) % 360-180
        physical = np.full(800, 1.2)
        mask = abs(theta) < 17
        physical[mask] = np.minimum(6, .35/np.maximum(abs(np.sin(np.deg2rad(theta[mask]))), 1e-9))
        for limits in ([1.5]*4+[3]*4, [3]*4+[1.5]*4, [1.5]*4+[3]*4+[1.5]*4):
            r = varied_recording([np.minimum(physical, v) for v in limits], limits)
            proposal = propose_sections(r)
            self.assertEqual(proposal.boundaries, ())
            self.assertTrue(any('采集条件变化' in n for n in proposal.acquisition_notes))
        self.assertTrue(any('此前未知' in n for n in propose_sections(r).acquisition_notes))

    def test_real_shape_change_within_common_range_is_retained(self):
        r = varied_recording([np.full(800, .7)]*4+[np.full(800, 1.2)]*5, [1.5]*4+[3]*5)
        p = propose_sections(r)
        self.assertEqual(len(p.boundaries), 1)
        self.assertIn('共同可见范围', p.reasons[0])

    def test_later_changes_in_newly_visible_region_are_retained(self):
        r = varied_recording([np.full(800, 1.4)]*4+[np.full(800, 2.)]*5+[np.full(800, 2.7)]*4,
                            [1.5]*4+[3]*9)
        proposal = propose_sections(r)
        self.assertNotIn(.35, proposal.boundaries)
        self.assertIn(.85, proposal.boundaries)

    def test_true_pipe_within_common_range_is_retained(self):
        r = varied_recording([np.ones(800)]*4+[pipe_profile()]*5, [3]*4+[6]*5)
        p = propose_sections(r)
        self.assertEqual(len(p.boundaries), 1)
        self.assertIn('侧壁', p.reasons[0])

    def test_switch_requires_new_full_sweep_and_rejects_mismatched_profile(self):
        scans = [NS(timestamp=i*10, angle_deg=(i*.45) % 360, range_m=1.5 if i < 20 else 3)
                 for i in range(850)]
        r = NS(echo_scans=scans, profile_scans=[])
        checks = RangeChecks(r)
        self.assertEqual(len(checks.events), 1)
        self.assertTrue(checks.profile_issue(NS(timestamp=3000, range_m=3)))
        self.assertFalse(checks.profile_issue(NS(timestamp=8300, range_m=3)))
        self.assertTrue(checks.profile_issue(NS(timestamp=8300, range_m=1.5)))
        # A range event does not introduce an extra pose or structural slice.
        baseline = NS(echo_scans=[NS(timestamp=s.timestamp, angle_deg=s.angle_deg, range_m=3) for s in scans])
        self.assertEqual(make_slice_corrections(r), make_slice_corrections(baseline))
        self.assertLess(checks.echo_weight(scans[20], 300), checks.echo_weight(scans[24], 300))
        self.assertLess(checks.echo_weight(scans[24], 300), checks.echo_weight(scans[24], 1200))

    def test_short_switch_back_stays_suspect_until_latest_full_turn(self):
        scans = [NS(timestamp=i*10, angle_deg=i*.45 % 360, range_m=3 if 20 <= i < 40 else 1.5)
                 for i in range(850)]
        checks = RangeChecks(NS(echo_scans=scans))
        self.assertEqual(len(checks.events), 2)
        self.assertTrue(checks.profile_issue(NS(timestamp=8000, range_m=1.5)))
        self.assertFalse(checks.profile_issue(NS(timestamp=8490, range_m=1.5)))

    def test_mutual_visibility_rejects_unseen_wall_and_opposing_rays(self):
        points = np.array([[1.,0,0], [1.,0,.1], [2.,0,.1], [-1.,0,.1]])
        origins = np.array([[0.,0,0], [0.,0,.1], [0.,0,.1], [0.,0,.1]])
        np.testing.assert_array_equal(common_visibility(points, origins, np.array([1.5,3,3,3]), 0,
                                                       np.array([1,2,3])), [True,False,False])

    def test_optimizer_refuses_surfaces_without_cross_range_overlap(self):
        local, origins, headings, ids, initial = test_slice_rotation.SliceRotationTest.pipe_fixture()
        # Only one sweep has visible limits; no cross-range multi-sweep plane
        # can be supported. Deliberately incompatible limits exercise the gate.
        limits = np.array([.1+.01*i for i in ids])
        with self.assertRaisesRegex(ValueError, '缺少可匹配'):
            fit_rotations(local, origins, headings, ids, initial, range_limits=limits, iterations=1)

    def test_optimizer_still_aligns_common_walls_with_valid_variable_ranges(self):
        local, origins, headings, ids, initial = test_slice_rotation.SliceRotationTest.pipe_fixture()
        limits = np.where(ids < 4, 1.5, 3.)
        keep = np.linalg.norm(local, axis=1) < limits*.98
        result = fit_rotations(local[keep], origins[keep], np.asarray(headings)[keep], ids[keep], initial,
                               range_limits=limits[keep], observation_weights=np.where(ids[keep] < 4, .9, .7),
                               max_gap=.4, iterations=4)
        self.assertLess(result.after_m, result.before_m*.9)

    def test_replay_does_not_reuse_old_or_mixed_profile_after_switch(self):
        r = object.__new__(BpRecording)
        r.echo_scans = [NS(timestamp=i*10, angle_deg=i*.45 % 360, range_m=1.5 if i < 20 else 3)
                        for i in range(850)]
        r.profile_scans = [NS(timestamp=100, range_m=1.5), NS(timestamp=400, range_m=3),
                           NS(timestamp=8300, range_m=3)]
        r._profile_times = [p.timestamp for p in r.profile_scans]
        r._range_checks = RangeChecks(r)
        self.assertIsNotNone(r.profile_at(150))
        self.assertIsNone(r.profile_at(220))
        self.assertIsNone(r.profile_at(500))
        self.assertEqual(r.profile_at(8390).timestamp, 8300)
        r._map = None

    def test_range_weights_reach_mesh_observations_without_stretching_points(self):
        scans = [NS(timestamp=i*10, angle_deg=i*.45, range_m=1.5 if i < 20 else 3) for i in range(50)]
        from bp_reader import SensorSample
        r = NS(echo_scans=scans, profile_scans=[], frame_count=50,
               sensor_samples=[SensorSample(0, 0, water_height=0), SensorSample(490, 0, water_height=0)],
               _sensor_times=[0, 490])
        def echo(scan):
            values = np.zeros(300, dtype=np.uint8)
            center = int(.8/scan.range_m*300)
            values[center-1:center+2] = [80, 240, 80]
            return values.tobytes()
        r.echo_data = echo
        points, origins, weights, limits = extract_wall_observations(r, denoise=False, return_range_limits=True)
        np.testing.assert_allclose(np.linalg.norm(points-origins, axis=1), .8, atol=.006)
        self.assertLess(weights[20], weights[24]*.5)
        np.testing.assert_array_equal(limits, [s.range_m for s in scans])


if __name__ == '__main__':
    unittest.main()
