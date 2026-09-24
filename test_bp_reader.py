import os
import unittest

from bp_reader import BpRecording, RingPoseCorrection, _transform_polar_point
from rigid_model_optimizer import segment_complete_rings


class BpReaderSmokeTest(unittest.TestCase):
    def test_rigid_pose_translation_and_depth_are_applied(self):
        correction = RingPoseCorrection(
            timestamp=1000,
            local_x_m=0.25,
            local_y_m=-0.5,
            roll_deg=0.0,
            pitch_deg=0.0,
            depth_correction_m=0.1,
            yaw_correction_deg=0.0,
        )
        point = _transform_polar_point(90.0, 2.0, 1.2, 0.0, correction)
        self.assertAlmostEqual(point[0], 2.25, places=6)
        self.assertAlmostEqual(point[1], -0.5, places=6)
        self.assertAlmostEqual(point[2], 1.3, places=6)

    def test_real_recording_index_and_decode(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "2026-8-31",
            "pipe_检测qqq地点-YS01-20260831-043445.bp",
        )
        if not os.path.isfile(path):
            self.skipTest("示例 BP 文件不存在")

        with BpRecording(path) as recording:
            self.assertGreater(recording.frame_count, 1000)
            first = recording.echo_scans[0]
            self.assertAlmostEqual(first.angle_deg, 274.95, places=2)
            self.assertAlmostEqual(first.range_m, 2.0, places=2)
            self.assertEqual(len(recording.echo_data(first)), 670)
            self.assertGreater(len(recording.profile_scans), 0)
            self.assertGreater(len(recording.profile_distances(recording.profile_scans[0])), 700)
            self.assertGreater(len(recording.sensor_samples), 0)
            self.assertIsNone(recording.profile_at(first.timestamp))
            cloud = recording.build_point_cloud(max_points=5000, intensity_threshold=32)
            self.assertGreater(len(cloud.points), 100)
            self.assertLessEqual(len(cloud.points), 5000)
            self.assertAlmostEqual(cloud.min_height, 0.17, places=2)
            self.assertAlmostEqual(cloud.max_height, 0.17, places=2)

    def test_multi_packet_echo_is_joined(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "2026-7-10",
            "pipe_刘店小学-Y03-20260710-101248.bp",
        )
        if not os.path.isfile(path):
            self.skipTest("双包示例 BP 文件不存在")

        with BpRecording(path) as recording:
            first = recording.echo_scans[0]
            self.assertEqual(len(first.packets), 2)
            self.assertEqual(len(recording.echo_data(first)), 2074)
            self.assertAlmostEqual(first.range_m, 6.0, places=2)

    def test_older_footer_variant(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "2026-6-26",
            "pipe_检测地点-01-20260626-015459.bp",
        )
        if not os.path.isfile(path):
            self.skipTest("旧尾部格式示例 BP 文件不存在")

        with BpRecording(path) as recording:
            self.assertGreater(recording.frame_count, 1000)
            self.assertEqual(len(recording.echo_data(recording.echo_scans[0])), 670)

    def test_water_height_creates_vertical_extent(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "2026-7-30",
            "pipe_汤逊湖污水处理厂-WS3-20260730-112046.bp",
        )
        if not os.path.isfile(path):
            self.skipTest("水位变化示例 BP 文件不存在")

        with BpRecording(path) as recording:
            cloud = recording.build_point_cloud(max_points=3000, intensity_threshold=40)
            self.assertGreater(len(cloud.points), 100)
            self.assertGreater(cloud.max_height - cloud.min_height, 2.5)

    def test_real_recording_is_split_into_complete_rings(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "2026-7-24",
            "pipe_汉口北大道-WS330058-20260724-103259.bp",
        )
        if not os.path.isfile(path):
            self.skipTest("完整环扫示例 BP 文件不存在")

        with BpRecording(path) as recording:
            rings = segment_complete_rings(recording)
            self.assertGreaterEqual(len(rings), 10)
            self.assertTrue(all(ring.angular_coverage_deg >= 330.0 for ring in rings))
            self.assertTrue(all(len(ring.indices) >= 700 for ring in rings))


if __name__ == "__main__":
    unittest.main()
