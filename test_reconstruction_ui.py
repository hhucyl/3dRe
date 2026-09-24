"""Offscreen Qt lifecycle checks using real BP files and the real worker."""

import os
import time
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QSettings, QTimer
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication

from main import MainWindow


class ReconstructionUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(["test-reconstruction"])
        font_path = Path('C:/Windows/Fonts/msyh.ttc')
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                cls.app.setFont(QFont(families[0], 9))
        cls.root = Path(__file__).parent / "data"
        cls.path = next(cls.root.rglob("*WS330058-20260724*.bp"), None)
        cls.other = next(cls.root.rglob("*YS01-20260831*.bp"), None)

    def setUp(self):
        if not self.path or not self.other:
            self.skipTest("真实 BP 示例不存在")
        self.window = MainWindow()
        self.temporary = tempfile.TemporaryDirectory()
        self.window.settings = QSettings(str(Path(self.temporary.name) / "settings.ini"), QSettings.IniFormat)

    def tearDown(self):
        if hasattr(self, "window"):
            self.window.close()
            self.wait_until(lambda: self.window._worker is None)
            self.temporary.cleanup()

    def wait_until(self, predicate, timeout=45):
        deadline = time.monotonic() + timeout
        while not predicate():
            self.app.processEvents()
            time.sleep(0.005)
            if time.monotonic() > deadline:
                self.fail("后台任务超时")
        self.app.processEvents()

    def test_progress_responsiveness_mesh_and_cached_toggle(self):
        window = self.window
        window.open_file(str(self.path))
        window.surface_voxel.setValue(9)
        window.z_interpolation_check.setChecked(True)
        window.surface_check.setChecked(True)
        progress = []
        heartbeats = []
        timer = QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: heartbeats.append(time.monotonic()))
        timer.start()
        self.wait_until(lambda: window._active_request is not None and window._active_request["surface"])
        window._worker.progress.connect(lambda generation, value, stage: progress.append(value))
        self.wait_until(lambda: window._worker is None and window._pending_job is None)
        timer.stop()
        self.assertGreater(len(heartbeats), 20)
        self.assertTrue(any(0 < p < 100 for p in progress))
        self.assertIn(100, progress)
        self.assertIsNotNone(window._mesh)
        self.assertIn("Z 向插值", window._mesh.note)
        self.assertTrue(window.export_mesh_button.isEnabled())
        self.assertTrue(window.progress_widget.isHidden())
        mesh = window._mesh
        window.surface_check.setChecked(False)
        self.assertFalse(window.sonar3d._show_surface)
        self.assertFalse(window.export_mesh_button.isEnabled())
        window.surface_check.setChecked(True)
        self.assertIsNone(window._worker)
        self.assertIs(window._mesh, mesh)
        self.assertTrue(window.sonar3d._show_surface)

    def test_cancel_and_file_switch_discard_stale_results(self):
        window = self.window
        window.open_file(str(self.path))
        window.surface_check.setChecked(True)
        window._cancel_reconstruction()
        self.wait_until(lambda: window._worker is None)
        self.assertFalse(window.surface_check.isChecked())
        self.assertIsNone(window._mesh)
        window.surface_check.setChecked(True)
        window.open_file(str(self.other))
        # Switch back to the point view while the prior reconstruction stops.
        window.surface_check.setChecked(False)
        self.wait_until(lambda: window._worker is None and window._pending_job is None)
        self.assertEqual(window.recording.path, str(self.other.resolve()))
        self.assertAlmostEqual(window.sonar3d._cloud.min_height, 0.17)
        self.assertIsNone(window.sonar3d._surface)

    def test_close_cancels_without_destroying_running_thread(self):
        window = self.window
        window.open_file(str(self.path))
        window.surface_check.setChecked(True)
        window.close()
        self.wait_until(lambda: window._worker is None)
        self.assertTrue(window._closing)
        self.assertIsNone(window.recording._map)

    def test_slice_angles_apply_toggle_and_invalidate_mesh_cache(self):
        window = self.window
        window.open_file(str(self.path))
        self.wait_until(lambda: window._worker is None)
        original = window.sonar3d._cloud
        window._edit_slice_angles()
        dialog = window._angle_dialog
        self.assertGreater(dialog.table.rowCount(), 0)
        dialog.offsets[0].setValue(12)
        dialog.offsets[1].setValue(-7)
        dialog.add_offsets()
        raw_key = window._current_mesh_key()
        dialog.submit(False)
        self.wait_until(lambda: window._worker is None)
        self.assertTrue(window.slice_rotation_check.isChecked())
        self.assertTrue(window.sonar3d._model_optimized)
        self.assertNotEqual(raw_key, window._current_mesh_key())
        self.assertNotEqual(original.points, window.sonar3d._cloud.points)
        self.assertEqual(window.slice_optimization.corrections[0].pitch_deg, 12)
        self.assertEqual(window.slice_optimization.corrections[0].local_x_m, 0)
        window.slice_rotation_check.setChecked(False)
        self.wait_until(lambda: window._worker is None)
        self.assertEqual(original.points, window.sonar3d._cloud.points)
        window.open_file(str(self.other))
        self.wait_until(lambda: window._worker is None)
        self.assertIsNone(window._angle_dialog)
        self.assertIsNone(window.slice_optimization)

    def test_reference_automatic_angles_mesh_and_preview(self):
        path = next(self.root.rglob('*473ws0225*102942.bp'), None)
        if not path:
            self.skipTest('参考 BP 不存在')
        window = self.window
        window.open_file(str(path))
        window.surface_voxel.setValue(9)
        window.surface_check.setChecked(True)
        window._edit_slice_angles()
        dialog = window._angle_dialog
        dialog.submit(True)
        self.assertFalse(dialog.table.isEnabled())
        self.wait_until(lambda: window._worker is None and window._pending_job is None, timeout=90)
        self.assertFalse(window._job_failed, window.progress_label.toolTip())
        self.assertIsNotNone(window._mesh)
        self.assertEqual(window._mesh_key, window._current_mesh_key())
        self.assertTrue(window.export_mesh_button.isEnabled())
        self.assertLess(window.slice_optimization.after_m, window.slice_optimization.before_m)
        self.assertTrue(dialog.table.isEnabled())
        window.view_tabs.setCurrentIndex(1)
        window.show()
        self.app.processEvents()
        output = Path(__file__).parent / '.verification'
        output.mkdir(exist_ok=True)
        window.grab().save(str(output/'slice_rotation_main.png'))
        dialog.grab().save(str(output/'slice_rotation_dialog.png'))

    def test_angle_json_roundtrip_and_auto_cancel(self):
        from unittest.mock import patch
        window = self.window
        window.open_file(str(self.path))
        self.wait_until(lambda: window._worker is None)
        window._edit_slice_angles()
        dialog = window._angle_dialog
        dialog.table.cellWidget(0, 3).setValue(8.5)
        dialog.boundaries_edit.setText('0.8, 2.3')
        path = str(Path(self.temporary.name)/'angles.json')
        with patch('slice_rotation_dialog.QFileDialog.getSaveFileName', return_value=(path, '')):
            dialog.save_angles()
        dialog.reset_angles()
        dialog.boundaries_edit.clear()
        with patch('slice_rotation_dialog.QFileDialog.getOpenFileName', return_value=(path, '')):
            dialog.load_angles()
        self.assertEqual(dialog.table.cellWidget(0, 3).value(), 8.5)
        self.assertEqual(dialog.boundaries_edit.text(), '0.8, 2.3')
        dialog.submit(True)
        window._cancel_reconstruction()
        self.wait_until(lambda: window._worker is None)
        self.assertTrue(dialog.table.isEnabled())
        self.assertIsNone(window._mesh)

    def test_spatial_preview_then_mesh_reuses_prepared_points(self):
        from unittest.mock import patch
        window = self.window
        window.open_file(str(self.path))
        window.surface_voxel.setValue(9)
        window.spatial_check.setChecked(True)
        self.wait_until(lambda: window._worker is None and window._pending_job is None, timeout=90)
        self.assertFalse(window._job_failed, window.progress_label.toolTip())
        self.assertFalse(window.surface_check.isChecked())
        self.assertFalse(window.z_interpolation_check.isChecked())
        self.assertIsNone(window.sonar3d._surface)
        self.assertIsNotNone(window._spatial)
        self.assertGreater(window._spatial.added_count, 0)
        self.assertTrue(window.sonar3d._cloud.interpolated)
        self.assertTrue(window.spatial_mesh_button.isEnabled())
        cached = window._spatial
        with patch('spatial_interpolation.prepare_spatial', side_effect=AssertionError('插值缓存应被复用')):
            window._generate_spatial_mesh()
            self.wait_until(lambda: window._worker is None, timeout=90)
        self.assertFalse(window._job_failed, window.progress_label.toolTip())
        self.assertIs(window._spatial, cached)
        self.assertEqual(window._mesh.spatial_interpolated_points, cached.added_count)
        self.assertEqual(window._mesh_key, window._current_mesh_key())
        window.view_tabs.setCurrentIndex(1)
        window.surface_check.setChecked(False)
        window.show(); self.app.processEvents()
        window.grab().save(str(Path(__file__).parent/'.verification/spatial_interpolation_preview.png'))
        window.spatial_gap.setValue(35)
        self.assertFalse(window.spatial_mesh_button.isEnabled())
        window._cancel_reconstruction()
        self.wait_until(lambda: window._worker is None)
        window.spatial_check.setChecked(False)
        self.wait_until(lambda: window._worker is None)
        self.assertFalse(window.sonar3d._cloud.interpolated)
        self.assertFalse(window.spatial_mesh_button.isEnabled())

    def test_structure_boundaries_rebuild_mesh_and_survive_pose_toggle(self):
        from surface_segments import parse_boundaries
        path = next(self.root.rglob('*473ws0225*102942.bp'), None)
        if not path:
            self.skipTest('参考 BP 不存在')
        window = self.window
        window.open_file(str(path))
        window.surface_voxel.setValue(9)
        window.surface_check.setChecked(True)
        self.wait_until(lambda: window._worker is None and window._pending_job is None)
        old_key = window._mesh_key
        window._edit_slice_angles()
        dialog = window._angle_dialog
        self.assertTrue(dialog.include_pipes.isChecked())
        dialog.suggest_sections()
        self.assertTrue(dialog._proposal_text)
        self.assertFalse(dialog.section_report.isHidden())
        dialog.boundaries_edit.setText('1.0, 2.0')
        dialog.submit(True)
        self.wait_until(lambda: window._worker is None and window._pending_job is None, timeout=90)
        self.assertFalse(window._job_failed, window.progress_label.toolTip())
        self.assertNotEqual(window._mesh_key, old_key)
        self.assertEqual(window._mesh_key, window._current_mesh_key())
        self.assertIn('分段壁面 3 段', window._mesh.note)
        self.assertEqual(window.slice_optimization.boundaries, (1.0, 2.0))
        self.assertEqual(parse_boundaries(dialog.boundaries_edit.text()), (1.0, 2.0))
        window.slice_rotation_check.setChecked(False)
        self.wait_until(lambda: window._worker is None)
        self.assertIn('分段壁面 3 段', window._mesh.note)
        self.assertEqual(window._mesh_key, window._current_mesh_key())
        window.view_tabs.setCurrentIndex(1)
        window.show()
        self.app.processEvents()
        output = Path(__file__).parent / '.verification'
        window.grab().save(str(output/'segmented_surface_main.png'))
        dialog.grab().save(str(output/'segmented_surface_dialog.png'))


if __name__ == "__main__":
    unittest.main()
