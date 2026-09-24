import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from main import MainWindow
from radar_widget import RadarWidget


class RadarMeasurementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test-radar-measurement'])

    @staticmethod
    def click_at_fraction(widget, fraction):
        center = widget._content_rect().center()
        radius = widget._echo_radius()
        point = QPoint(round(center.x()+fraction*radius), round(center.y()))
        QTest.mouseClick(widget, Qt.LeftButton, pos=point)

    def test_two_clicks_measure_echo_scale_and_survive_resize(self):
        radar = RadarWidget()
        radar.resize(700, 630)
        radar.show()
        self.app.processEvents()
        radar.add_scan(90, 2, bytes(200))
        self.click_at_fraction(radar, .25)
        self.assertEqual(len(radar._measurement_points), 0)  # tool off
        radar.set_measurement_enabled(True)
        self.click_at_fraction(radar, .25)
        self.assertIsNone(radar.measured_distance_m)
        self.click_at_fraction(radar, .75)
        self.assertAlmostEqual(radar.measured_distance_m, 1.0, delta=.012)
        measured = radar.measured_distance_m
        radar.resize(800, 710)
        radar.set_north_up(True)
        self.app.processEvents()
        self.assertEqual(radar.measured_distance_m, measured)
        radar.grab()  # Exercise actual paint path and label drawing.
        radar.close()

    def test_outside_click_new_measurement_clear_and_range_change(self):
        radar = RadarWidget()
        radar.resize(680, 680)
        radar.show()
        self.app.processEvents()
        radar.set_measurement_enabled(True)
        QTest.mouseClick(radar, Qt.LeftButton, pos=QPoint(5, 5))
        self.assertEqual(len(radar._measurement_points), 0)
        self.click_at_fraction(radar, 0)
        self.click_at_fraction(radar, .5)
        self.assertAlmostEqual(radar.measured_distance_m, 1.0, delta=.01)  # initial 2 m
        self.click_at_fraction(radar, -.5)
        self.assertEqual(len(radar._measurement_points), 1)
        radar.clear_measurement()
        self.assertEqual(len(radar._measurement_points), 0)
        self.click_at_fraction(radar, .25)
        self.click_at_fraction(radar, .75)
        radar.add_scan(0, 3, bytes(200))
        self.assertIsNone(radar.measured_distance_m)
        self.assertEqual(len(radar._measurement_points), 0)
        radar.close()

    def test_main_window_controls_and_replay_clear(self):
        window = MainWindow()
        window.show()
        self.app.processEvents()
        window.measure_button.click()
        self.assertTrue(window.radar._measurement_enabled)
        radar = window.radar
        self.click_at_fraction(radar, 0)
        self.assertIn('起点', window.measurement_label.text())
        self.click_at_fraction(radar, .5)
        self.assertAlmostEqual(radar.measured_distance_m, 1.0, delta=.012)
        self.assertIn('两点直线距离', window.measurement_label.text())
        window.clear_measurement_button.click()
        self.assertIn('请在雷达图内点击起点', window.measurement_label.text())
        window.measure_button.click()
        self.assertFalse(radar._measurement_enabled)
        window.close()


if __name__ == '__main__':
    unittest.main()
