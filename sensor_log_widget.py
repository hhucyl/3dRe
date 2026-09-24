"""Time-aligned display of the original same-name TXT record."""
import json
import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTableWidget,
                            QTableWidgetItem, QHeaderView, QAbstractItemView)


def format_time(timestamp):
    try:
        return datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    except (ValueError, OverflowError, OSError):
        return str(timestamp)


class SensorLogWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        title = QLabel('TXT 同步记录')
        title.setStyleSheet('font-size: 15px; font-weight: bold; color: #66d4ce;')
        layout.addWidget(title)
        self.source = QLabel()
        self.timing = QLabel()
        for label in (self.source, self.timing):
            label.setWordWrap(True)
            label.setTextFormat(Qt.PlainText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(label)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['TXT 字段', '原始值'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setWordWrap(True)
        layout.addWidget(self.table, 1)
        hint = QLabel('按最近 TXT 时间戳匹配；原值显示，不插值。\n方向代码及未确认单位的字段保留原样。')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #9bb0b8;')
        layout.addWidget(hint)
        self._sample = None
        self._path = None
        self.set_recording(None)

    def set_recording(self, recording):
        self._path = os.path.splitext(recording.path)[0]+'.txt' if recording else None
        self._sample = None
        self.table.setRowCount(0)
        self.source.setText(os.path.basename(self._path) if self._path else '尚未打开 BP 文件')
        self.source.setToolTip(self._path or '')
        self.timing.setText('等待回放' if recording and recording.sensor_samples else
                            '没有可用的同名 TXT 记录' if recording else '打开 BP 后随回放同步显示')

    def show_frame(self, recording, timestamp):
        sample = recording.sensor_at(timestamp)
        if sample is None:
            self._sample = None
            self.table.setRowCount(0)
            self.timing.setText(f'声呐帧：{format_time(timestamp)}\n没有可用的同名 TXT 记录')
            return
        delta = (sample.timestamp-timestamp)/1000
        outside = timestamp < recording.sensor_samples[0].timestamp or timestamp > recording.sensor_samples[-1].timestamp
        status = '超出 TXT 时间范围，显示最近端点' if outside else '最近时间戳匹配'
        self.timing.setText(f'声呐帧：{format_time(timestamp)}\nTXT 记录：{format_time(sample.timestamp)}\n'
                            f'TXT − 声呐：{delta:+.3f} s\n{status}')
        if sample is self._sample:
            return
        self._sample = sample
        fields = sample.raw_fields or (('currTime', sample.timestamp), ('currDateStr', sample.date_text),
                                       ('comPassAngle', sample.compass_deg), ('waterHeight', sample.water_height))
        labels = {'currTime': '时间戳 / ms', 'currDateStr': '记录日期', 'comPassAngle': '罗盘角 / °',
                  'waterHeight': '探头水深 / m'}
        self.table.setRowCount(len(fields))
        for row, (key, value) in enumerate(fields):
            name = labels.get(key, key)
            if key in labels:
                name += '\n'+key
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            for col, text in enumerate((name, rendered)):
                item = QTableWidgetItem(text)
                item.setToolTip(f'{key}: {rendered}')
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()
