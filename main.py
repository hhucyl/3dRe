"""Qt5 desktop replay application for PipeSonar .bp recordings."""

from __future__ import annotations

import bisect
import os
import sys
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import QElapsedTimer, QSettings, QSize, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from bp_reader import BpFormatError, BpRecording
from radar_widget import RadarWidget
from sensor_log_widget import SensorLogWidget
from slice_rotation import SliceRotationResult, make_slice_corrections
from reconstruction_worker import ReconstructionWorker
from sonar3d_widget import Sonar3DWidget


def _format_duration(milliseconds: int) -> str:
    seconds = max(0, milliseconds // 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _format_timestamp(milliseconds: int) -> str:
    if milliseconds <= 0:
        return "--"
    try:
        return datetime.fromtimestamp(milliseconds / 1000.0).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except (ValueError, OSError, OverflowError):
        return str(milliseconds)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PipeSonar BP 雷达重绘")
        self.resize(1180, 880)
        self.setMinimumSize(QSize(900, 680))

        self.recording: Optional[BpRecording] = None
        self.current_index = 0
        self.playing = False
        self.speed = 1.0
        self.play_start_index = 0
        self.clock = QElapsedTimer()
        self.profile_timestamp = -1
        self.slice_optimization: Optional[SliceRotationResult] = None
        self._angle_dialog = None
        self._auto_fit = False
        self._slice_fit_gap = 0.5
        self._worker = None
        self._pending_job = None
        self._active_request = None
        self._generation = 0
        self._closing = False
        self._mesh = None
        self._mesh_key = None
        self._spatial = None
        self._spatial_key = None
        self._job_failed = False
        self.settings = QSettings("CTG", "PipeSonarBpReplay")

        self.radar = RadarWidget()
        self.sonar3d = Sonar3DWidget()
        self._build_ui()
        self._apply_style()

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._play_tick)
        self.rebuild_timer = QTimer(self)
        self.rebuild_timer.setSingleShot(True)
        self.rebuild_timer.setInterval(350)
        self.rebuild_timer.timeout.connect(self._build_3d_cloud)

        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            QTimer.singleShot(0, lambda: self.open_file(sys.argv[1]))

    def _build_ui(self) -> None:
        toolbar = QToolBar("文件", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("打开 BP", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.choose_file)
        toolbar.addAction(open_action)
        export_action = QAction("导出图片", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_image)
        toolbar.addAction(export_action)
        toolbar.addSeparator()

        self.file_label = QLabel("尚未打开文件")
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.file_label.setMinimumWidth(180)
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        toolbar.addWidget(self.file_label)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)
        self.view_tabs = QTabWidget()
        self.radar_page = QSplitter(Qt.Horizontal)
        self.radar_page.setChildrenCollapsible(False)
        self.sensor_log = SensorLogWidget()
        self.radar_page.addWidget(self.radar)
        self.radar_page.addWidget(self.sensor_log)
        self.radar_page.setStretchFactor(0, 1)
        self.radar_page.setStretchFactor(1, 0)
        self.radar_page.setSizes([800, 310])
        self.radar_tab = QWidget()
        radar_layout = QVBoxLayout(self.radar_tab)
        radar_layout.setContentsMargins(0, 0, 0, 0)
        radar_layout.setSpacing(6)
        radar_layout.addWidget(self.radar_page, 1)
        measurement_controls = QHBoxLayout()
        self.measure_button = QPushButton('测量距离')
        self.measure_button.setCheckable(True)
        self.measure_button.setToolTip('在二维雷达内依次点击两点；第三次点击开始新测量。')
        self.measure_button.toggled.connect(self.radar.set_measurement_enabled)
        self.measure_button.toggled.connect(self._update_measurement_label)
        measurement_controls.addWidget(self.measure_button)
        self.clear_measurement_button = QPushButton('清除测量')
        self.clear_measurement_button.clicked.connect(self.radar.clear_measurement)
        measurement_controls.addWidget(self.clear_measurement_button)
        self.measurement_label = QLabel('点击“测量距离”，再在雷达图内依次点击两点')
        self.measurement_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        measurement_controls.addWidget(self.measurement_label, 1)
        radar_layout.addLayout(measurement_controls)
        self.radar.measurementChanged.connect(self._update_measurement_label)
        self.view_tabs.addTab(self.radar_tab, "二维雷达")
        three_d_page = QWidget()
        three_d_layout = QVBoxLayout(three_d_page)
        three_d_layout.setContentsMargins(0, 0, 0, 0)
        three_d_layout.setSpacing(6)
        three_d_layout.addWidget(self.sonar3d, 1)
        cloud_controls = QHBoxLayout()
        cloud_controls.addWidget(QLabel("回波阈值"))
        self.cloud_threshold = QSpinBox()
        self.cloud_threshold.setRange(0, 255)
        self.cloud_threshold.setValue(32)
        cloud_controls.addWidget(self.cloud_threshold)
        cloud_controls.addWidget(QLabel("点数上限"))
        self.cloud_limit = QSpinBox()
        self.cloud_limit.setRange(10, 200)
        self.cloud_limit.setValue(60)
        self.cloud_limit.setSuffix(" 千")
        cloud_controls.addWidget(self.cloud_limit)
        cloud_controls.addWidget(QLabel("点大小"))
        self.cloud_point_size = QDoubleSpinBox()
        self.cloud_point_size.setRange(0.5, 5.0)
        self.cloud_point_size.setSingleStep(0.2)
        self.cloud_point_size.setValue(1.7)
        self.cloud_point_size.valueChanged.connect(self.sonar3d.set_point_size)
        cloud_controls.addWidget(self.cloud_point_size)
        self.cloud_contour_check = QCheckBox("显示轮廓线")
        self.cloud_contour_check.setChecked(True)
        self.cloud_contour_check.toggled.connect(self.sonar3d.set_show_contours)
        cloud_controls.addWidget(self.cloud_contour_check)
        self.slice_rotation_check = QCheckBox("切片姿态校正")
        self.slice_rotation_check.setChecked(False)
        self.slice_rotation_check.setToolTip("绕各圈探头中心调整 a/b/c；固定 XY，保留实测水位。取消可对照原始切片。")
        self.slice_rotation_check.toggled.connect(self._slice_rotation_toggled)
        self.slice_angles_button = QPushButton("调整 a / b / c…")
        self.slice_angles_button.setEnabled(False)
        self.slice_angles_button.clicked.connect(self._edit_slice_angles)
        rebuild_cloud = QPushButton("重新生成三维叠加")
        rebuild_cloud.clicked.connect(self._build_3d_cloud)
        cloud_controls.addWidget(rebuild_cloud)
        cloud_controls.addStretch(1)
        three_d_layout.addLayout(cloud_controls)

        surface_controls = QHBoxLayout()
        self.surface_check = QCheckBox("三维曲面重建")
        self.surface_check.setToolTip("从原始回波逐束定位，融合壁面法向与有符号距离场，生成连续三角网格。")
        self.surface_check.toggled.connect(self._surface_toggled)
        surface_controls.addWidget(self.surface_check)
        surface_controls.addWidget(QLabel("网格尺寸"))
        self.surface_voxel = QDoubleSpinBox()
        self.surface_voxel.setRange(1.0, 50.0)
        self.surface_voxel.setDecimals(1)
        self.surface_voxel.setValue(5.0)
        self.surface_voxel.setSuffix(" cm")
        self.surface_voxel.setToolTip("尺寸越小细节越多；大范围记录会自动调整尺寸以限制内存占用。")
        self.surface_voxel.valueChanged.connect(self._surface_settings_changed)
        self.cloud_threshold.valueChanged.connect(self._surface_settings_changed)
        surface_controls.addWidget(self.surface_voxel)
        self.surface_points = QCheckBox("叠加回波点")
        self.surface_points.setToolTip("在曲面上叠加原始回波点；若同时勾选显示轮廓线，也会叠加原始轮廓。")
        self.surface_points.toggled.connect(self.sonar3d.set_surface_points)
        surface_controls.addWidget(self.surface_points)
        self.surface_confidence = QCheckBox("置信度着色")
        self.surface_confidence.setToolTip("青色表示较高融合权重，橙色表示较低融合权重；不是测量精度。")
        self.surface_confidence.toggled.connect(self.sonar3d.set_confidence_colors)
        surface_controls.addWidget(self.surface_confidence)
        angle_controls = QHBoxLayout()
        angle_controls.addWidget(self.slice_rotation_check)
        angle_controls.addWidget(self.slice_angles_button)
        self.slice_note = QLabel("固定探头 XY · 保留实测水位 · 每圈旋转")
        self.slice_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        angle_controls.addWidget(self.slice_note, 1)
        three_d_layout.addLayout(angle_controls)
        self.export_mesh_button = QPushButton("导出模型")
        self.export_mesh_button.setEnabled(False)
        self.export_mesh_button.clicked.connect(self.export_surface)
        surface_controls.addWidget(self.export_mesh_button)
        surface_controls.addStretch(1)
        three_d_layout.addLayout(surface_controls)

        wall_controls = QHBoxLayout()
        self.local_denoise_check = QCheckBox("局部连续性去噪")
        self.local_denoise_check.setChecked(True)
        self.local_denoise_check.setToolTip("根据相邻回波和局部线/面结构降低孤立噪声权重；不使用圆井、固定内径或圆管拟合。可取消勾选以对照细节。")
        self.local_denoise_check.toggled.connect(self._surface_settings_changed)
        wall_controls.addWidget(self.local_denoise_check)
        wall_controls.addWidget(QLabel("近场排除"))
        self.wall_near = QDoubleSpinBox()
        self.wall_near.setRange(0, 200)
        self.wall_near.setValue(0)
        self.wall_near.setSuffix(" cm")
        self.wall_near.setToolTip("默认 0，不按固定距离删除近处结构。确认探头附近存在混响时再增大；始终排除最前端两个采样。")
        self.wall_near.valueChanged.connect(self._surface_settings_changed)
        wall_controls.addWidget(self.wall_near)
        wall_controls.addWidget(QLabel("局部连接半径"))
        self.surface_support = QDoubleSpinBox()
        self.surface_support.setRange(5, 100)
        self.surface_support.setValue(20)
        self.surface_support.setSuffix(" cm")
        self.surface_support.setToolTip("限制局部曲面融合范围。增大可连接邻近断面，减小可保留窄间隙和细节；未知区域和量程末端保持开放。")
        self.surface_support.valueChanged.connect(self._surface_settings_changed)
        wall_controls.addWidget(self.surface_support)
        wall_controls.addStretch(1)
        three_d_layout.addLayout(wall_controls)

        z_controls = QHBoxLayout()
        self.z_interpolation_check = QCheckBox("Z 向连续插值")
        self.z_interpolation_check.setChecked(True)
        self.z_interpolation_check.setToolTip("先补充上下相容壁面，再沿 Z 对距离场作保形三次插值，连接断裂面片；量程末端保持开放。可取消勾选对照。")
        z_controls.addWidget(self.z_interpolation_check)
        z_controls.addWidget(QLabel("最大层距"))
        self.z_max_gap = QDoubleSpinBox()
        self.z_max_gap.setRange(5, 200)
        self.z_max_gap.setValue(50)
        self.z_max_gap.setSuffix(" cm")
        self.z_max_gap.setToolTip("允许插值的上下观测间最大 Z 距离；横向匹配范围使用局部连接半径。")
        self.z_max_gap.setEnabled(True)
        self.z_max_gap.valueChanged.connect(self._surface_settings_changed)
        z_controls.addWidget(self.z_max_gap)
        z_controls.addWidget(QLabel("插值权重"))
        self.z_weight = QDoubleSpinBox()
        self.z_weight.setRange(0.05, 0.50)
        self.z_weight.setDecimals(2)
        self.z_weight.setSingleStep(0.05)
        self.z_weight.setValue(0.30)
        self.z_weight.setToolTip("辅助点和补全距离场相对于上下支持的置信度比例；默认 0.30。权重影响融合与着色，补全跨度由最大层距控制。")
        self.z_weight.setEnabled(True)
        self.z_weight.valueChanged.connect(self._surface_settings_changed)
        z_controls.addWidget(self.z_weight)
        self.z_interpolation_check.toggled.connect(self.z_max_gap.setEnabled)
        self.z_interpolation_check.toggled.connect(self.z_weight.setEnabled)
        self.z_interpolation_check.toggled.connect(self._surface_settings_changed)
        z_controls.addStretch(1)
        three_d_layout.addLayout(z_controls)

        spatial_controls = QHBoxLayout()
        self.spatial_check = QCheckBox("空间插值")
        self.spatial_check.setToolTip("先生成三维插值点云，紫色为非实测辅助点；准备好后可从插值点生成网格。")
        spatial_controls.addWidget(self.spatial_check)
        spatial_controls.addWidget(QLabel("插值间距"))
        self.spatial_spacing = QDoubleSpinBox()
        self.spatial_spacing.setRange(1, 25)
        self.spatial_spacing.setValue(5)
        self.spatial_spacing.setSuffix(" cm")
        spatial_controls.addWidget(self.spatial_spacing)
        spatial_controls.addWidget(QLabel("最大连接"))
        self.spatial_gap = QDoubleSpinBox()
        self.spatial_gap.setRange(5, 100)
        self.spatial_gap.setValue(30)
        self.spatial_gap.setSuffix(" cm")
        self.spatial_gap.setToolTip("仅连接距离以内的相容壁面；至少为插值间距的两倍。开口附近可减小以限制补点。")
        spatial_controls.addWidget(self.spatial_gap)
        self.spatial_mesh_button = QPushButton("从插值点生成网格")
        self.spatial_mesh_button.setEnabled(False)
        self.spatial_mesh_button.clicked.connect(self._generate_spatial_mesh)
        spatial_controls.addWidget(self.spatial_mesh_button)
        for widget in (self.spatial_spacing, self.spatial_gap):
            widget.setEnabled(False)
            widget.valueChanged.connect(self._surface_settings_changed)
        self.spatial_check.toggled.connect(self._spatial_toggled)
        spatial_controls.addStretch(1)
        three_d_layout.addLayout(spatial_controls)

        self.progress_widget = QWidget()
        progress_layout = QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(4, 0, 4, 0)
        self.progress_label = QLabel("准备重建")
        self.progress_label.setMinimumWidth(190)
        progress_layout.addWidget(self.progress_label)
        self.reconstruction_progress = QProgressBar()
        self.reconstruction_progress.setRange(0, 100)
        progress_layout.addWidget(self.reconstruction_progress, 1)
        self.cancel_reconstruction = QPushButton("取消")
        self.cancel_reconstruction.clicked.connect(self._cancel_reconstruction)
        progress_layout.addWidget(self.cancel_reconstruction)
        self.progress_widget.hide()
        three_d_layout.addWidget(self.progress_widget)
        self.view_tabs.addTab(three_d_page, "三维水位叠加")
        layout.addWidget(self.view_tabs, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        controls.addWidget(self.stop_button)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderPressed.connect(self._pause_for_seek)
        self.position_slider.sliderReleased.connect(self._seek_released)
        controls.addWidget(self.position_slider, 1)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(105)
        controls.addWidget(self.time_label)

        controls.addWidget(QLabel("倍速"))
        self.speed_combo = QComboBox()
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        controls.addWidget(self.speed_combo)
        layout.addLayout(controls)

        options = QHBoxLayout()
        options.addWidget(QLabel("增益"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.2, 8.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(1.7)
        self.gain_spin.setSuffix("×")
        self.gain_spin.valueChanged.connect(self.radar.set_gain)
        options.addWidget(self.gain_spin)

        options.addWidget(QLabel("色板"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItem("声呐绿", "green")
        self.palette_combo.addItem("琥珀", "amber")
        self.palette_combo.addItem("热力", "thermal")
        self.palette_combo.currentIndexChanged.connect(
            lambda: self.radar.set_palette(self.palette_combo.currentData())
        )
        options.addWidget(self.palette_combo)

        self.profile_check = QCheckBox("轮廓叠加")
        self.profile_check.setChecked(True)
        self.profile_check.toggled.connect(self.radar.set_show_profile)
        options.addWidget(self.profile_check)
        self.north_check = QCheckBox("北向朝上（使用同名 TXT 航向）")
        self.north_check.toggled.connect(self.radar.set_north_up)
        options.addWidget(self.north_check)
        options.addStretch(1)
        self.frame_label = QLabel("帧 --")
        self.frame_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.frame_label.setMinimumWidth(120)
        options.addWidget(self.frame_label)
        layout.addLayout(options)

        self.setCentralWidget(central)
        status = QStatusBar()
        self.setStatusBar(status)
        self.status_info = QLabel("就绪")
        self.status_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.cursor_info = QLabel("")
        status.addWidget(self.status_info, 1)
        status.addPermanentWidget(self.cursor_info)
        self.radar.cursorChanged.connect(self._cursor_changed)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #101c24; color: #dbe9eb; font-family: "Microsoft YaHei UI"; }
            QToolBar { background: #152630; border: 0; border-bottom: 1px solid #29404a; spacing: 8px; padding: 7px; }
            QToolButton, QPushButton { background: #1e7780; border: 0; border-radius: 4px; padding: 7px 14px; color: white; }
            QToolButton:hover, QPushButton:hover { background: #278f99; }
            QPushButton:checked { background: #916b20; color: #fff3a6; }
            QToolButton:disabled, QPushButton:disabled { background: #263940; color: #71838a; }
            QComboBox, QDoubleSpinBox, QSpinBox { background: #172a34; border: 1px solid #35515c; border-radius: 3px; padding: 5px; }
            QProgressBar { border: 1px solid #35515c; border-radius: 3px; background: #172a34; text-align: center; }
            QProgressBar::chunk { background: #238d96; }
            QSlider::groove:horizontal { height: 5px; background: #2b434c; border-radius: 2px; }
            QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #66d4ce; border-radius: 7px; }
            QTabWidget::pane { border: 1px solid #29404a; }
            QTabBar::tab { background: #14262f; padding: 7px 18px; border: 1px solid #29404a; }
            QTabBar::tab:selected { background: #1e7780; color: white; }
            QStatusBar { background: #0d171d; border-top: 1px solid #29404a; }
            """
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._closing = True
        self.rebuild_timer.stop()
        self.timer.stop()
        self._pending_job = None
        if self._worker is not None:
            self._generation += 1
            self._worker.requestInterruption()
            self.setEnabled(False)
            self.status_info.setText("正在停止后台重建并关闭……")
            event.ignore()
            return
        if self.recording:
            self.recording.close()
        event.accept()

    def choose_file(self) -> None:
        initial = self.settings.value("lastDirectory", os.path.join(os.path.dirname(__file__), "data"))
        path, _ = QFileDialog.getOpenFileName(self, "打开 PipeSonar 回放文件", str(initial), "PipeSonar BP (*.bp);;所有文件 (*)")
        if path:
            self.open_file(path)

    def open_file(self, path: str) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            new_recording = BpRecording(path)
        except (OSError, BpFormatError, ValueError) as error:
            QMessageBox.critical(self, "无法打开", f"无法读取文件：\n{path}\n\n{error}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if self.recording:
            self.recording.close()
        self.rebuild_timer.stop()
        self._generation += 1
        self._pending_job = None
        if self._worker:
            self._worker.requestInterruption()
        self._mesh = None
        self._mesh_key = None
        self.export_mesh_button.setEnabled(False)
        self.recording = new_recording
        self.sensor_log.set_recording(new_recording)
        self._spatial = None
        self._spatial_key = None
        self.spatial_mesh_button.setEnabled(False)
        self.current_index = 0
        self.playing = False
        self.timer.stop()
        self.profile_timestamp = -1
        self.slice_optimization = None
        self._auto_fit = False
        if self._angle_dialog is not None:
            self._angle_dialog.close()
            self._angle_dialog.deleteLater()
            self._angle_dialog = None
        self.slice_note.setText("固定探头 XY · 保留实测水位 · 每圈旋转")
        self.slice_angles_button.setEnabled(bool(new_recording.sensor_samples))
        self.radar.clear()
        self.sonar3d.clear()
        self.sonar3d.set_structure_boundaries(())

        self.settings.setValue("lastDirectory", os.path.dirname(path))
        self.file_label.setText(os.path.basename(path))
        self.file_label.setToolTip(path)
        self.position_slider.setRange(0, new_recording.frame_count - 1)
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.play_button.setText("播放")

        sidecar = "有航向数据" if new_recording.sensor_samples else "无同名 TXT 航向数据"
        self.slice_rotation_check.setEnabled(bool(new_recording.sensor_samples))
        if not new_recording.sensor_samples and self.slice_rotation_check.isChecked():
            self.slice_rotation_check.blockSignals(True)
            self.slice_rotation_check.setChecked(False)
            self.slice_rotation_check.blockSignals(False)
        self.status_info.setText(
            f"{new_recording.frame_count:,} 个强度帧，{len(new_recording.profile_scans):,} 个轮廓，{sidecar}"
        )
        self._render_index(0, rebuild=True)
        self._build_3d_cloud()

    def _build_3d_cloud(self) -> None:
        if not self.recording or self._closing:
            return
        self.rebuild_timer.stop()
        self._generation += 1
        self._job_failed = False
        self.sonar3d.set_surface(None)
        self.export_mesh_button.setEnabled(False)
        self.spatial_mesh_button.setEnabled(False)
        request = dict(
            path=self.recording.path, threshold=self.cloud_threshold.value(),
            max_points=self.cloud_limit.value() * 1000, attitude=self.slice_rotation_check.isChecked(),
            surface=self.surface_check.isChecked(), voxel=self.surface_voxel.value() / 100,
            optimization=self.slice_optimization,
            angles=self.slice_optimization.corrections if self.slice_rotation_check.isChecked() and self.slice_optimization else (),
            boundaries=self.slice_optimization.boundaries if self.slice_optimization else (),
            auto_fit=self._auto_fit,
            fit_gap=self._slice_fit_gap,
            denoise=self.local_denoise_check.isChecked(), near=self.wall_near.value() / 100,
            support=self.surface_support.value() / 100,
            z_interpolation=self.z_interpolation_check.isChecked(), z_max_gap=self.z_max_gap.value() / 100,
            z_weight=self.z_weight.value(),
            spatial=self.spatial_check.isChecked(), spatial_spacing=self.spatial_spacing.value()/100,
            spatial_gap=self.spatial_gap.value()/100,
        )
        if request["spatial"] and not request["auto_fit"] and self._spatial_key == self._request_spatial_key(request):
            request["cached_spatial"] = self._spatial
        self._auto_fit = False
        if not request["auto_fit"] and self._mesh_key == self._request_mesh_key(request):
            request["cached_mesh"] = self._mesh
        self._pending_job = (self._generation, request)
        self.progress_widget.show()
        self.cancel_reconstruction.setEnabled(True)
        if self._worker is not None:
            self._worker.requestInterruption()
            self.progress_label.setText("正在停止旧任务……")
            self.reconstruction_progress.setRange(0, 0)
        else:
            self._start_pending_job()

    @staticmethod
    def _request_mesh_key(request):
        return tuple(request[name] for name in ("path", "threshold", "attitude", "voxel", "denoise", "near", "support",
                                                "z_interpolation", "z_max_gap", "z_weight", "angles", "boundaries",
                                                "spatial", "spatial_spacing", "spatial_gap"))

    @staticmethod
    def _request_spatial_key(request):
        return tuple(request[name] for name in ("path", "threshold", "attitude", "denoise", "near", "support",
                                                "angles", "boundaries", "spatial_spacing", "spatial_gap"))

    def _current_spatial_key(self):
        if not self.recording:
            return None
        return (self.recording.path, self.cloud_threshold.value(), self.slice_rotation_check.isChecked(),
                self.local_denoise_check.isChecked(), self.wall_near.value()/100, self.surface_support.value()/100,
                self.slice_optimization.corrections if self.slice_rotation_check.isChecked() and self.slice_optimization else (),
                self.slice_optimization.boundaries if self.slice_optimization else (),
                self.spatial_spacing.value()/100, self.spatial_gap.value()/100)

    def _current_mesh_key(self):
        if not self.recording:
            return None
        return (self.recording.path, self.cloud_threshold.value(), self.slice_rotation_check.isChecked(),
                self.surface_voxel.value() / 100, self.local_denoise_check.isChecked(),
                self.wall_near.value() / 100, self.surface_support.value() / 100,
                self.z_interpolation_check.isChecked(), self.z_max_gap.value() / 100, self.z_weight.value(),
                self.slice_optimization.corrections if self.slice_rotation_check.isChecked() and self.slice_optimization else (),
                self.slice_optimization.boundaries if self.slice_optimization else (),
                self.spatial_check.isChecked(), self.spatial_spacing.value()/100, self.spatial_gap.value()/100)

    def _start_pending_job(self):
        if self._closing or self._pending_job is None:
            return
        generation, request = self._pending_job
        self._pending_job = None
        self._active_request = dict(request)
        worker = ReconstructionWorker(generation, request, self)
        self._worker = worker
        worker.progress.connect(self._reconstruction_progress)
        worker.cloudReady.connect(self._cloud_ready)
        worker.meshReady.connect(self._mesh_ready)
        worker.spatialReady.connect(self._spatial_ready)
        worker.failed.connect(self._reconstruction_failed)
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _reconstruction_progress(self, generation, value, stage):
        if generation != self._generation or self._closing:
            return
        self.progress_label.setText(stage)
        self.progress_label.setToolTip(stage)
        self.status_info.setText(stage + "……")
        self.reconstruction_progress.setRange(0, 0 if value < 0 else 100)
        if value >= 0:
            self.reconstruction_progress.setValue(value)

    def _cloud_ready(self, generation, cloud, optimization):
        if generation != self._generation or self._closing:
            return
        if optimization:
            self.slice_optimization = optimization
            self._active_request["angles"] = optimization.corrections
            self._active_request["boundaries"] = optimization.boundaries
            self.slice_note.setText(optimization.note)
            self.slice_note.setToolTip(optimization.note)
            if self._angle_dialog is not None and self._active_request.get("auto_fit"):
                self._angle_dialog.set_result(optimization)
        if self._active_request.get("spatial") and self._spatial is not None:
            self._spatial_key = self._request_spatial_key(self._active_request)
        self.sonar3d.set_cloud(cloud, model_optimized=bool(optimization))
        self.sonar3d.set_structure_boundaries(self._active_request.get("boundaries", ()))
        self.status_info.setText(
            f"三维叠加完成：{len(cloud.points):,} 个回波点，{len(cloud.contours):,} 个轮廓层，"
            f"Z {cloud.min_height:.3f}–{cloud.max_height:.3f} m"
            + ("；"+cloud.interpolation_note if cloud.interpolation_note else "")
        )

    def _spatial_ready(self, generation, observations):
        if generation == self._generation and not self._closing:
            self._spatial = observations
            self._spatial_key = None  # Filled after corrected angles arrive with cloudReady.

    def _mesh_ready(self, generation, mesh):
        if generation != self._generation or self._closing or not self.surface_check.isChecked():
            return
        self._mesh = mesh
        self._mesh_key = self._request_mesh_key(self._active_request)
        self.sonar3d.set_surface(mesh)
        self.sonar3d.set_show_surface(True)
        self.export_mesh_button.setEnabled(True)
        self.reconstruction_progress.setRange(0, 100)
        self.reconstruction_progress.setValue(100)
        self._show_mesh_status()

    def _show_mesh_status(self):
        mesh = self._mesh
        self.status_info.setText(
            f"曲面重建完成：{len(mesh.vertices):,} 个顶点，{len(mesh.faces):,} 个三角面，"
            f"网格 {mesh.voxel_size * 100:.1f} cm，壁面观测 {mesh.observation_count:,} 条"
            + (f"；{mesh.note}" if mesh.note else "")
        )

    def _reconstruction_failed(self, generation, message):
        if generation != self._generation or self._closing:
            return
        self._job_failed = True
        if self._angle_dialog is not None:
            self._angle_dialog.note.setText("处理失败：" + message)
        self.surface_check.blockSignals(True)
        self.surface_check.setChecked(False)
        self.surface_check.blockSignals(False)
        self.sonar3d.set_surface(None)
        self.export_mesh_button.setEnabled(False)
        self.progress_label.setText("重建失败，可调整参数后重试")
        self.progress_label.setToolTip(message)
        self.status_info.setText("三维重建失败：" + message)

    def _worker_finished(self):
        worker = self._worker
        self._worker = None
        if worker:
            worker.deleteLater()
        if self._closing:
            QTimer.singleShot(0, self.close)
        elif self._pending_job:
            self._start_pending_job()
        else:
            if self._angle_dialog is not None:
                self._angle_dialog.set_busy(False)
            self.spatial_mesh_button.setEnabled(self.spatial_check.isChecked() and self._spatial is not None
                and self._spatial_key == self._current_spatial_key())
            self.cancel_reconstruction.setEnabled(False)
            if not self._job_failed:
                self.progress_widget.hide()
            else:
                self.reconstruction_progress.setRange(0, 100)
                self.reconstruction_progress.setValue(0)

    def _cancel_reconstruction(self):
        self.rebuild_timer.stop()
        self._generation += 1
        self._pending_job = None
        if self._worker:
            self._worker.requestInterruption()
        self.surface_check.blockSignals(True)
        self.surface_check.setChecked(False)
        self.surface_check.blockSignals(False)
        self.sonar3d.set_show_surface(False)
        self.export_mesh_button.setEnabled(False)
        self.cancel_reconstruction.setEnabled(False)
        self.progress_label.setText("正在取消……")
        self.status_info.setText("已取消重建，保留当前回波点云")

    def _surface_toggled(self, enabled):
        self.sonar3d.set_show_surface(enabled)
        self.export_mesh_button.setEnabled(False)
        if not self.recording:
            return
        if self._worker is None and not enabled:
            self.rebuild_timer.stop()
            self.status_info.setText("已恢复三维回波点云显示")
            return
        if self._worker is None and self._mesh is not None and self._mesh_key == self._current_mesh_key():
            self.sonar3d.set_surface(self._mesh)
            self.export_mesh_button.setEnabled(True)
            self._show_mesh_status()
            return
        self._build_3d_cloud()

    def _surface_settings_changed(self, _value):
        if (self.surface_check.isChecked() or self.spatial_check.isChecked()) and self.recording:
            self.export_mesh_button.setEnabled(False)
            self.spatial_mesh_button.setEnabled(False)
            if self._worker is not None:
                self._generation += 1
                self._pending_job = None
                self._worker.requestInterruption()
            self.rebuild_timer.start()

    def _spatial_toggled(self, enabled):
        for widget in (self.spatial_spacing, self.spatial_gap):
            widget.setEnabled(enabled)
        if enabled:
            self.z_interpolation_check.setChecked(False)
        self.z_interpolation_check.setEnabled(not enabled)
        self.surface_check.blockSignals(True)
        self.surface_check.setChecked(False)
        self.surface_check.blockSignals(False)
        self.sonar3d.set_show_surface(False)
        self.spatial_mesh_button.setEnabled(False)
        if self.recording:
            self._build_3d_cloud()

    def _generate_spatial_mesh(self):
        if not self.spatial_mesh_button.isEnabled() or self._spatial_key != self._current_spatial_key():
            return
        if self.surface_check.isChecked():
            self._build_3d_cloud()
        else:
            self.surface_check.setChecked(True)

    def export_surface(self):
        if (self._mesh is None or not self.recording or not self.export_mesh_button.isEnabled()
                or self._mesh_key != self._current_mesh_key()):
            return
        path, chosen = QFileDialog.getSaveFileName(
            self, "导出三维内壁模型", os.path.splitext(self.recording.path)[0] + "_surface.ply",
            "PLY 模型 (*.ply);;OBJ 模型 (*.obj)",
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".obj" if "OBJ" in chosen else ".ply"
        try:
            from surface_reconstruction import export_mesh
            export_mesh(self._mesh, path)
            self.status_info.setText(f"已导出三维模型：{path}")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "模型导出失败", str(error))

    def _slice_rotation_toggled(self, enabled: bool) -> None:
        if self.recording:
            if enabled and self.slice_optimization is None:
                self.slice_optimization = SliceRotationResult(make_slice_corrections(self.recording))
            self._build_3d_cloud()

    def _edit_slice_angles(self):
        if not self.recording:
            return
        if self.slice_optimization is None:
            self.slice_optimization = SliceRotationResult(make_slice_corrections(self.recording))
        if self._angle_dialog is None:
            from slice_rotation_dialog import SliceRotationDialog
            self._angle_dialog = SliceRotationDialog(self.recording, self.slice_optimization.corrections, self)
            self._angle_dialog.set_result(self.slice_optimization)
            self._angle_dialog.applied.connect(self._apply_slice_angles)
        self._angle_dialog.show()
        self._angle_dialog.raise_()
        self._angle_dialog.activateWindow()

    def _apply_slice_angles(self, corrections, automatic, fit_gap=0.5, boundaries=()):
        self.slice_optimization = SliceRotationResult(tuple(corrections), boundaries=tuple(boundaries))
        self._auto_fit = bool(automatic)
        self._slice_fit_gap = fit_gap
        if self._angle_dialog is not None:
            self._angle_dialog.set_busy(bool(automatic))
        self.slice_rotation_check.blockSignals(True)
        self.slice_rotation_check.setChecked(True)
        self.slice_rotation_check.blockSignals(False)
        self._build_3d_cloud()

    def toggle_play(self) -> None:
        if not self.recording:
            return
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.play_button.setText("播放")
        else:
            if self.current_index >= self.recording.frame_count - 1:
                self.current_index = 0
                self.radar.clear()
            self.playing = True
            self.play_start_index = self.current_index
            self.clock.restart()
            self.timer.start()
            self.play_button.setText("暂停")

    def stop(self) -> None:
        self.playing = False
        self.timer.stop()
        self.play_button.setText("播放")
        if self.recording:
            self.current_index = 0
            self._render_index(0, rebuild=True)

    def _speed_changed(self) -> None:
        self.speed = float(self.speed_combo.currentData())
        if self.playing:
            self.play_start_index = self.current_index
            self.clock.restart()

    def _pause_for_seek(self) -> None:
        self._was_playing_before_seek = self.playing
        if self.playing:
            self.toggle_play()

    def _seek_released(self) -> None:
        self._render_index(self.position_slider.value(), rebuild=True)
        if getattr(self, "_was_playing_before_seek", False):
            self.toggle_play()

    def _play_tick(self) -> None:
        if not self.recording or not self.playing:
            return
        scans = self.recording.echo_scans
        start_time = scans[self.play_start_index].timestamp
        target_time = start_time + int(self.clock.elapsed() * self.speed)
        target_index = min(
            len(scans) - 1,
            bisect.bisect_right(self.recording.echo_times, target_time) - 1,
        )
        if target_index <= self.current_index:
            return

        # Render each intervening ray to preserve a complete radar image.  A
        # cap keeps the UI responsive after large clock jumps.
        last = min(target_index, self.current_index + 500)
        for index in range(self.current_index + 1, last + 1):
            self._render_index(index, update_controls=False)
        self._update_controls()
        if self.current_index >= len(scans) - 1:
            self.playing = False
            self.timer.stop()
            self.play_button.setText("播放")

    def _render_index(self, index: int, rebuild: bool = False, update_controls: bool = True) -> None:
        if not self.recording:
            return
        index = max(0, min(index, self.recording.frame_count - 1))
        if rebuild:
            self.radar.clear()
            self.profile_timestamp = -1
            # Reconstruct at most the current revolution for a useful seek preview.
            start = index
            current_angle = self.recording.echo_scans[index].angle_deg
            while start > 0 and index - start < 900:
                previous = self.recording.echo_scans[start - 1].angle_deg
                if previous > 315.0 and current_angle < 45.0:
                    break
                current_angle = previous
                start -= 1
            for rebuild_index in range(start, index):
                self._draw_scan(rebuild_index)

        self._draw_scan(index)
        self.current_index = index
        if update_controls:
            self._update_controls()

    def _draw_scan(self, index: int) -> None:
        assert self.recording is not None
        scan = self.recording.echo_scans[index]
        sensor = self.recording.sensor_at(scan.timestamp)
        heading = sensor.compass_deg if sensor else 0.0
        self.radar.add_scan(scan.angle_deg, scan.range_m, self.recording.echo_data(scan), heading)
        self.sonar3d.set_current_height(sensor.water_height if sensor else None)
        profile = self.recording.profile_at(scan.timestamp)
        if profile is None:
            self.radar.set_profile([])
            self.profile_timestamp = None
        if profile and profile.timestamp != self.profile_timestamp:
            self.radar.set_profile(self.recording.profile_distances(profile))
            self.profile_timestamp = profile.timestamp

    def _update_controls(self) -> None:
        if not self.recording:
            return
        scan = self.recording.echo_scans[self.current_index]
        first_time = self.recording.echo_scans[0].timestamp
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(self.current_index)
        self.position_slider.blockSignals(False)
        self.time_label.setText(
            f"{_format_duration(scan.timestamp - first_time)} / {_format_duration(self.recording.duration_ms)}"
        )
        sensor = self.recording.sensor_at(scan.timestamp)
        heading_text = f"，航向 {sensor.compass_deg:.2f}°" if sensor else ""
        self.sensor_log.show_frame(self.recording, scan.timestamp)
        self.frame_label.setText(
            f"帧 {self.current_index + 1:,}/{self.recording.frame_count:,}  |  "
            f"{_format_timestamp(scan.timestamp)}  |  量程 {scan.range_m:g} m  |  "
            f"探头 {scan.angle_deg:.2f}°{heading_text}"
        )

    def _cursor_changed(self, distance: float, angle: float) -> None:
        self.cursor_info.setText("" if distance < 0 else f"光标：{distance:.3f} m / {angle:.2f}°")

    def _update_measurement_label(self) -> None:
        distance = self.radar.measured_distance_m
        if distance is not None:
            self.measurement_label.setText(f'两点直线距离：{distance:.3f} m（单击第三点开始新测量）')
        elif self.radar._measurement_points:
            self.measurement_label.setText('已选起点，请点击终点')
        elif self.measure_button.isChecked():
            self.measurement_label.setText('请在雷达图内点击起点')
        else:
            self.measurement_label.setText('点击“测量距离”，再在雷达图内依次点击两点')

    def export_image(self) -> None:
        if not self.recording:
            QMessageBox.information(self, "导出图片", "请先打开一个 .bp 文件。")
            return
        default_name = os.path.splitext(self.recording.path)[0] + "_radar.png"
        path, _ = QFileDialog.getSaveFileName(self, "导出当前雷达图", default_name, "PNG 图片 (*.png);;JPEG 图片 (*.jpg)")
        if not path:
            return
        pixmap = self.view_tabs.currentWidget().grab()
        if not pixmap.save(path):
            QMessageBox.warning(self, "导出失败", f"无法写入：\n{path}")
        else:
            self.status_info.setText(f"已导出：{path}")


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("PipeSonar BP 雷达重绘")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
