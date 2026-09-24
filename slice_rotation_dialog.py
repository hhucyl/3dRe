"""Editable per-sweep a/b/c table with explicit apply and fitting actions."""

import json
from dataclasses import replace
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
                            QTableWidgetItem, QDoubleSpinBox, QPushButton, QHeaderView,
                            QFileDialog, QMessageBox, QLineEdit, QCheckBox, QPlainTextEdit)
from slice_rotation import validate_corrections
from surface_segments import parse_boundaries, validate_boundaries
from surface_reconstruction import interpolate_sensor
from acquisition_conditions import RangeChecks


class SliceRotationDialog(QDialog):
    applied = pyqtSignal(object, bool, float, object)

    def __init__(self, recording, corrections, parent=None):
        super().__init__(parent)
        self.setWindowTitle("切片姿态校正 · 固定探头位置")
        self.resize(900, 730)
        self.path = recording.path
        self.recording = recording
        self.base = tuple(corrections)
        layout = QVBoxLayout(self)
        hint = QLabel("每圈绕探头中心旋转；XY 固定，Z 保留实测 waterHeight。\n"
                      "a：绕 Y 轴俯仰；b：绕 X 轴翻滚；c：叠加到罗盘的顺时针航向修正（度）。\n"
                      "自动拟合使用重叠壁面的局部平面与角度连续性；首圈 c 锚定，绝对方向需手动标定。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        common = QHBoxLayout()
        common.addWidget(QLabel("全圈增量"))
        self.offsets = []
        for name, limit in (("a", 85), ("b", 85), ("c", 180)):
            common.addWidget(QLabel(name))
            spin = self.angle_spin(limit)
            self.offsets.append(spin)
            common.addWidget(spin)
        add = QPushButton("加到所有圈")
        add.clicked.connect(self.add_offsets)
        common.addWidget(add)
        reset = QPushButton("全部归零")
        reset.clicked.connect(self.reset_angles)
        common.addWidget(reset)
        layout.addLayout(common)
        self.table = QTableWidget(len(corrections), 6)
        self.table.setHorizontalHeaderLabels(["圈", "起始 / s", "探头 Z / m", "俯仰 a / °", "翻滚 b / °", "航向修正 c / °"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().hide()
        for row, c in enumerate(corrections):
            for col, value in enumerate((str(row+1), f"{(c.slice_start-recording.echo_scans[0].timestamp)/1000:.2f}",
                                         f"{interpolate_sensor(recording, c.timestamp)[0]:.3f}")):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
            for col, value in enumerate((c.pitch_deg, c.roll_deg, c.yaw_correction_deg), 3):
                spin = self.angle_spin(180 if col == 5 else 85)
                spin.setValue(value)
                self.table.setCellWidget(row, col, spin)
        layout.addWidget(self.table, 1)
        fit_controls = QHBoxLayout()
        fit_controls.addWidget(QLabel("自动拟合：最大匹配层距"))
        self.fit_gap = QDoubleSpinBox()
        self.fit_gap.setRange(5, 200)
        self.fit_gap.setValue(50)
        self.fit_gap.setSuffix(" cm")
        self.fit_gap.setToolTip("按探头水位筛选可匹配切片；与曲面 Z 插值参数独立。每次角度搜索为当前初值 ±20°。")
        fit_controls.addWidget(self.fit_gap)
        fit_controls.addStretch(1)
        layout.addLayout(fit_controls)
        structure = QHBoxLayout()
        structure.addWidget(QLabel("结构分界 Z / m"))
        self.boundaries_edit = QLineEdit()
        self.boundaries_edit.setPlaceholderText("例如 0.80, 2.30；留空不分段")
        self.boundaries_edit.setToolTip("圆柱井筒/方形井室等交界的空间 Z，逗号分隔。按旋转后的回波点划段，一圈可跨多段。")
        structure.addWidget(self.boundaries_edit, 1)
        self.suggest_button = QPushButton("建议分界")
        self.suggest_button.clicked.connect(self.suggest_sections)
        structure.addWidget(self.suggest_button)
        layout.addLayout(structure)
        self.include_pipes = QCheckBox("建议分段时纳入管道出现 / 消失")
        self.include_pipes.setChecked(True)
        self.include_pipes.setToolTip("结合远距扇区、双侧近似平行直线和跨层持续性；缺回波或达到量程不单独作为管道证据。")
        layout.addWidget(self.include_pipes)
        self.section_report = QPlainTextEdit()
        self.section_report.setReadOnly(True)
        self.section_report.setMaximumHeight(120)
        self.section_report.hide()
        layout.addWidget(self.section_report)
        self._proposal_text = ""
        acquisition_report = RangeChecks(recording).report()
        if acquisition_report:
            self.section_report.setPlainText('\n'.join(acquisition_report))
            self.section_report.show()
        self.boundaries_edit.textChanged.connect(self._boundaries_changed)
        section_hint = QLabel("段内拟合局部平面或曲面；结构分界不跨段平滑、不自动补台阶。棱边允许法向突变。")
        section_hint.setWordWrap(True)
        layout.addWidget(section_hint)
        self.note = QLabel("修改后点击“应用到三维”；自动拟合完成后会回填本表。")
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.note)
        buttons = QHBoxLayout()
        self.edit_buttons = [add, reset, self.suggest_button]
        for label, action in (("读取角度", self.load_angles), ("保存角度", self.save_angles),
                              ("自动拟合并应用", lambda: self.submit(True)),
                              ("应用到三维", lambda: self.submit(False)), ("关闭", self.close)):
            button = QPushButton(label)
            button.clicked.connect(action)
            buttons.addWidget(button)
            if label != "关闭":
                self.edit_buttons.append(button)
        layout.addLayout(buttons)

    def set_busy(self, busy):
        self.table.setEnabled(not busy)
        self.fit_gap.setEnabled(not busy)
        self.boundaries_edit.setEnabled(not busy)
        self.include_pipes.setEnabled(not busy)
        for widget in self.offsets + self.edit_buttons:
            widget.setEnabled(not busy)

    @staticmethod
    def angle_spin(limit):
        spin = QDoubleSpinBox()
        spin.setRange(-limit, limit)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setSuffix("°")
        spin.setKeyboardTracking(False)
        return spin

    def corrections(self):
        values = tuple(replace(c, pitch_deg=self.table.cellWidget(i, 3).value(),
                               roll_deg=self.table.cellWidget(i, 4).value(),
                               yaw_correction_deg=self.table.cellWidget(i, 5).value())
                       for i, c in enumerate(self.base))
        validate_corrections(values)
        return values

    def set_result(self, result):
        # Retain full solver precision until the user next edits/applies.
        self.base = result.corrections
        self.boundaries_edit.setText(', '.join(f'{v:g}' for v in result.boundaries))
        for i, c in enumerate(result.corrections):
            for col, value in enumerate((c.pitch_deg, c.roll_deg, c.yaw_correction_deg), 3):
                self.table.cellWidget(i, col).setValue(value)
        self.note.setText(result.note)

    def submit(self, automatic):
        try:
            boundaries = parse_boundaries(self.boundaries_edit.text())
        except ValueError as error:
            QMessageBox.warning(self, "结构分界无效", str(error))
            return
        self.applied.emit(self.corrections(), automatic, self.fit_gap.value()/100, boundaries)
        self.note.setText("正在后台拟合；可在主窗口取消。" if automatic else "已应用，正在重新生成三维。")

    def suggest_sections(self):
        from section_detection import propose_sections
        proposal = propose_sections(self.recording, self.include_pipes.isChecked())
        boundaries = proposal.boundaries
        if boundaries:
            self.boundaries_edit.setText(', '.join(f'{v:g}' for v in boundaries))
            self.note.setText(f"已建议 {len(boundaries)+1} 段；综合井室形状与管道证据。候选高度按探头水位估计，请检查后应用。")
        else:
            self.note.setText("没有足够证据建议分界，保留当前输入；可根据实际井筒/井室交界手动填写 Z。")
        rows = [f"段 {i+1} · Z {section.low:g}–{section.high:g} m · {section.state} · {section.levels} 层支持"
                for i, section in enumerate(proposal.sections)]
        rows.extend(f"分界 {height:g} m：{reason}" for height, reason in zip(boundaries, proposal.reasons))
        rows.extend(proposal.acquisition_notes)
        self._proposal_text = '\n'.join(rows)
        self.section_report.setPlainText(self._proposal_text)
        self.section_report.setToolTip(self._proposal_text)
        self.section_report.show()

    def _boundaries_changed(self):
        if self._proposal_text:
            self.section_report.setPlainText("分界已改变；之前的管道证据仅供参考，重新点击建议分界可刷新。")

    def add_offsets(self):
        for row in range(self.table.rowCount()):
            for col, offset in enumerate(self.offsets, 3):
                spin = self.table.cellWidget(row, col)
                spin.setValue(spin.value()+offset.value())
        for spin in self.offsets:
            spin.setValue(0)

    def reset_angles(self):
        for row in range(self.table.rowCount()):
            for col in range(3, 6):
                self.table.cellWidget(row, col).setValue(0)

    def save_angles(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存切片角度", str(Path(self.path).with_suffix('.angles.json')), "JSON (*.json)")
        if not path:
            return
        try:
            values = self.corrections()
            boundaries = parse_boundaries(self.boundaries_edit.text())
            payload = {"version": 2, "recording": Path(self.path).name,
                       "acquisition_events": [dict(timestamp=e.timestamp, before_m=e.before_m,
                                                   after_m=e.after_m, source=e.source)
                                              for e in RangeChecks(self.recording).events],
                       "boundaries": boundaries,
                       "convention": "Rz(-(compass+c)) Ry(a) Rx(b); fixed XY; measured waterHeight",
                       "slices": [{"start": c.slice_start, "a": c.pitch_deg, "b": c.roll_deg, "c": c.yaw_correction_deg} for c in values]}
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            self.note.setText("角度已保存："+path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "保存失败", str(error))

    def load_angles(self):
        path, _ = QFileDialog.getOpenFileName(self, "读取切片角度", str(Path(self.path).parent), "JSON (*.json)")
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
            rows = payload['slices']
            if (payload['version'] not in (1, 2) or payload['recording'] != Path(self.path).name or
                    [r['start'] for r in rows] != [c.slice_start for c in self.base]):
                raise ValueError("角度文件与当前 BP 的切片不匹配。")
            values = tuple(replace(c, pitch_deg=float(r['a']), roll_deg=float(r['b']), yaw_correction_deg=float(r['c']))
                           for c, r in zip(self.base, rows))
            validate_corrections(values)
            boundaries = validate_boundaries(payload['boundaries']) if payload['version'] == 2 else ()
            from slice_rotation import SliceRotationResult
            self.set_result(SliceRotationResult(values, "已读取角度与结构分界，点击应用到三维。", boundaries=boundaries))
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "读取失败", str(error))
