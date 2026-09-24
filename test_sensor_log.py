import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtCore import QSettings, QPoint, Qt
from PyQt5.QtTest import QTest

from bp_reader import BpRecording, SensorSample, load_sensor_log
from main import MainWindow
from sensor_log_widget import SensorLogWidget


class SensorLogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test-sensor-log'])
        font = QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
        families = QFontDatabase.applicationFontFamilies(font)
        if families:
            cls.app.setFont(QFont(families[0], 9))

    def test_parser_preserves_all_original_fields_and_zero_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'scan.txt'
            fields = dict(currTime=2000, comPassAngle=361.2, waterHeight=.3, waterSpeed=0,
                          waterDirection=2, airHeight=3.519, waterBottomHeight=346.44098,
                          extra={'value': None})
            path.write_text('\ufeff'+json.dumps(fields)+'\n[]\ninvalid\n'+
                            json.dumps(dict(currTime=1000, waterHeight=0)), encoding='utf-8')
            samples = load_sensor_log(str(path.with_suffix('.bp')))
            self.assertEqual([s.timestamp for s in samples], [1000, 2000])
            self.assertEqual(dict(samples[1].raw_fields), fields)
            self.assertAlmostEqual(samples[1].compass_deg, 1.2)

    def test_nearest_record_seek_endpoints_and_missing_log(self):
        r = object.__new__(BpRecording)
        r.path = 'sample.bp'
        r.sensor_samples = [SensorSample(1000, 20, water_height=.1), SensorSample(2000, 40, water_height=.2)]
        r._sensor_times = [1000, 2000]
        widget = SensorLogWidget()
        widget.set_recording(r)
        for stamp, matched in ((1100, 1000), (1900, 2000), (1200, 1000)):
            widget.show_frame(r, stamp)
            self.assertEqual(widget._sample.timestamp, matched)
            self.assertIn(f'{(matched-stamp)/1000:+.3f} s', widget.timing.text())
        widget.show_frame(r, 3000)
        self.assertIn('超出 TXT 时间范围', widget.timing.text())
        r.sensor_samples = []
        r._sensor_times = []
        widget.set_recording(r)
        widget.show_frame(r, 1000)
        self.assertEqual(widget.table.rowCount(), 0)
        self.assertIn('没有可用', widget.timing.text())
        widget.close()
        r._map = None

    def test_radar_playback_seek_and_export_page_include_txt(self):
        path = next((Path(__file__).parent/'data').rglob('*473ws0225*102942.bp'), None)
        if path is None:
            self.skipTest('参考 BP 不存在')
        window = MainWindow()
        with tempfile.TemporaryDirectory() as folder:
            window.settings = QSettings(str(Path(folder)/'settings.ini'), QSettings.IniFormat)
            try:
                with patch.object(window, '_build_3d_cloud'):
                    window.open_file(str(path))
                window.show()
                self.app.processEvents()
                r = window.recording
                window._render_index(1000, rebuild=True)
                self.assertIs(window.sensor_log._sample, r.sensor_at(r.echo_scans[1000].timestamp))
                window.playing = True
                window.play_start_index = 1000
                with patch.object(window, 'clock') as timer:
                    timer.elapsed.return_value = 2500
                    window._play_tick()
                self.assertGreater(window.current_index, 1000)
                self.assertIs(window.sensor_log._sample, r.sensor_at(r.echo_scans[window.current_index].timestamp))
                window.playing = False
                window._render_index(300, rebuild=True)
                self.assertIs(window.sensor_log._sample, r.sensor_at(r.echo_scans[300].timestamp))
                self.assertEqual(window.sensor_log.table.rowCount(), 8)
                self.assertIs(window.view_tabs.currentWidget(), window.radar_tab)
                self.assertTrue(window.sensor_log.isVisible())
                window._render_index(len(r.echo_scans)//2, rebuild=True)
                self.app.processEvents()
                window.measure_button.click()
                center = window.radar._content_rect().center()
                radius = window.radar._echo_radius()
                for fraction in (-.25, .25):
                    QTest.mouseClick(window.radar, Qt.LeftButton,
                                     pos=QPoint(round(center.x()+fraction*radius), round(center.y())))
                self.assertIsNotNone(window.radar.measured_distance_m)
                output = Path(__file__).parent/'.verification'/'radar_txt_sync.png'
                output.parent.mkdir(exist_ok=True)
                self.assertTrue(window.grab().save(str(output)))
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
