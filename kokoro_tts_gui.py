import sys
import os
import json
import re
import time
import gc
import math
import struct
import tempfile
import wave
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget,
                             QTableWidgetItem, QProgressBar, QDoubleSpinBox, QScrollArea,
                             QCheckBox, QMessageBox, QGridLayout, QSplitter, QSplitterHandle,
                             QTabWidget, QTextEdit, QSpinBox, QHeaderView, QStyledItemDelegate,
                             QStyle, QStyleOptionViewItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QEvent, QTimer
from PyQt5.QtGui import QCursor
import soundfile as sf
import numpy as np
import torch
import psutil
from kokoro_onnx import Kokoro

# Translations dictionary for English only
TRANSLATIONS = {
    "window_title": "Kokoro TTS & Split GUI v2.0",
    "tab_split": "Text Splitting",
    "tab_tts": "TTS Processing",
    "tab_custom_mix": "Voice Custom Mix",
    "tab_custom_mix_1": "Voice Custom Mix 1",
    "tab_custom_mix_2": "Voice Custom Mix 2",
    "tab_custom_mix_3": "Voice Custom Mix 3",
    "tab_custom_mix_4": "Voice Custom Mix 4",
    "tab_custom_mix_5": "Voice Custom Mix 5",
    "split_input_file_label": "Input Text File:",
    "split_input_file_placeholder": "Select a text file...",
    "split_browse_button": "Browse...",
    "split_parts_label": "Number of Parts:",
    "split_word_label": "Split Before Word/Tag:",
    "split_word_placeholder": "e.g. [voice=custom_mix] or Chapter",
    "split_button": "Split Text File",
    "load_split_to_tts_button": "Load Split Files to TTS",
    "split_log_label": "Status:",
    "split_log_placeholder": "Status messages will appear here...",
    "tts_input_file_label": "Input Text File:",
    "tts_input_file_placeholder": "Select a text file...",
    "tts_browse_button": "Browse...",
    "tts_split_files_button": "Load Split Files...",
    "tts_output_file_label": "Output Audio File:",
    "tts_output_file_placeholder": "Output file (e.g. output.wav)",
    "tts_speed_label": "Speed:",
    "tts_max_threads_label": "Maximum Threads:",
    "tts_config_label": "Configuration:",
    "tts_save_config_button": "Save Configuration",
    "tts_load_config_button": "Load Configuration",
    "tts_add_task_button": "Add Task",
    "tts_clear_all_button": "Delete All Tasks",
    "tts_clear_all_title": "Delete all tasks?",
    "tts_clear_all_question": "Really delete all tasks?!",
    "tts_processes_label": "Processes:",
    "tts_table_headers": ["Process ID", "Input File", "Output File", "Progress", "Status", "Time", "Action", "Delete"],
    "tts_cancel_button": "Cancel",
    "tts_restart_button": "Restart",
    "tts_pause_button": "Pause",
    "tts_resume_button": "Resume",
    "tts_overall_label": "Batch Progress:",
    "tts_overall_done": "{done} of {total} done",
    "tts_overall_eta": "ETA: {eta}",
    "tts_overall_eta_unknown": "ETA: --",
    "tts_overall_all_done": "All tasks completed",
    "tts_overall_elapsed": "Elapsed: {time}",
    "log_all_done": "All tasks finished.",
    "tts_delete_button": "Delete",
    "voice_selection_label": "Voice Selection and Weights:",
    "error_invalid_input_file": "Please select a valid input file.",
    "error_invalid_parts": "Number of parts must be greater than 0.",
    "error_no_split_word": "Please specify a split word or tag.",
    "error_no_split_files": "No split files available.",
    "error_invalid_output_file": "Please specify an output filename.",
    "error_output_not_wav": "The output file must be a .wav file.",
    "error_no_active_voices": "Please enable at least one voice and set a weight > 0.",
    "error_split_file_pattern": "The selected file does not match the pattern 'Name_XXX.txt'.",
    "error_no_split_files_found": "No split files found for base name '{}'.",
    "success_split": "Text file successfully split into {} parts.",
    "success_tasks_added": "{} tasks for split files added.",
    "log_file_read": "File read: {} ({} characters)",
    "log_split_points": "Found split points: {} occurrences of '{}'",
    "log_no_split_word": "Warning: No occurrences of '{}' found. Splitting by character count.",
    "log_actual_splits": "Actual split points: {}",
    "log_part_saved": "Saved: {} ({} characters)",
    "log_split_success": "✅ Successfully split into {} files.",
    "log_config_saved": "Configuration saved to: {}",
    "log_config_loaded": "Configuration loaded from: {}",
    "log_last_config_loaded": "Last configuration loaded.",
    "log_config_save_warning": "Warning: Could not save last configuration: {}",
    "log_config_load_warning": "Warning: Could not load last configuration: {}",
    "log_max_threads_changed": "Maximum threads changed to: {}",
    "log_task_added": "Task added for {} -> {}",
    "log_thread_started": "New thread started for process {}, active threads: {}",
    "log_thread_check": "Checking start: {} active threads, max_threads={}, queue={}",
    "log_thread_finished": "[Process {}] Thread finished, was_canceled={}",
    "log_process_init": "[Process {}] Initializing Kokoro...",
    "log_file_parsed": "[Process {}] Input file parsed, {} entries found.",
    "log_total_entries": "[Process {}] Total number of entries: {}",
    "log_custom_pause": "[Process {}][{}] Adding custom pause of {} seconds.",
    "log_process_voice": "[Process {}][{}] Processing with voice '{}'",
    "log_process_custom_mix": "[Process {}][{}] Processing with custom voice mix (VOICEPACK)",
    "log_process_custom_mix_1": "[Process {}][{}] Processing with custom voice mix 1 (VOICEPACK_1)",
    "log_process_custom_mix_2": "[Process {}][{}] Processing with custom voice mix 2 (VOICEPACK_2)",
    "log_process_custom_mix_3": "[Process {}][{}] Processing with custom voice mix 3 (VOICEPACK_3)",
    "log_process_custom_mix_4": "[Process {}][{}] Processing with custom voice mix 4 (VOICEPACK_4)",
    "log_process_custom_mix_5": "[Process {}][{}] Processing with custom voice mix 5 (VOICEPACK_5)",
    "log_voice_not_found": "[Process {}] ⚠️ Voice '{}' not found, using VOICEPACK.",
    "log_generate_text": "[Process {}] → Generating text: \"{}...\"",
    "log_sample_rate": "[Process {}] Sample rate from kokoro.create: {}, Samples length: {}",
    "log_sample_rate_warning": "[Process {}] ⚠️ Warning: Sample rate {} differs from {}",
    "log_memory_usage": "[Process {}] Memory usage: {:.2f} MB",
    "log_process_canceled": "[Process {}] ❌ Process canceled.",
    "log_process_completed": "[Process {}] ✅ Processing completed, file written: {}",
    "log_memory_freed": "[Process {}] Memory freed. Memory usage: {:.2f} MB",
    "log_cleanup_warning": "[Process {}] Warning during cleanup: {}",
    "log_error": "[Process {}] ❌ Error: {}",
    "log_no_write_access": "[Process {}] ❌ Error: No write access to directory {}",
    "log_pending_cleanup": "Waiting for cleanup of processes: {}",
    "log_thread_removed": "[Process {}] Removing finished thread (isRunning: {})",
    "log_custom_speed": "[Process {}][{}] Setting custom speed to {}",
    "log_speed_clamped": "[Process {}][{}] Speed {:.2f} outside allowed range, clamped to {:.2f}",
    "log_voicepack_fallback": "[Process {}] ⚠️ Voice mix '{}' has no active voices, using the active mix instead.",
    "log_no_active_mix": "[Process {}] ❌ Error: No active voice mix configured (all weights are 0).",
}

class CursorHeaderView(QHeaderView):
    """QHeaderView that shows the split double-arrow cursor when the mouse
    hovers near a section boundary (column separator line).

    The cursor is applied to the *viewport* (the widget that is actually under
    the mouse) and AFTER the base class handling, so nothing can override it.
    As an extra guarantee a CursorSupervisor polls the global mouse position.
    """

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def _update_cursor(self, pos):
        if QApplication.mouseButtons():
            return  # dragging a section - keep Qt's own cursor
        viewport = self.viewport()
        x = pos.x()
        near_border = False
        for i in range(1, self.count()):
            if abs(self.sectionViewportPosition(i) - x) <= 5:
                near_border = True
                break
        if near_border:
            if viewport.cursor().shape() != Qt.SplitHCursor:
                viewport.setCursor(Qt.SplitHCursor)
        else:
            if viewport.cursor().shape() != Qt.ArrowCursor:
                viewport.unsetCursor()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._update_cursor(event.pos())

    def leaveEvent(self, event):
        self.viewport().unsetCursor()
        super().leaveEvent(event)


class CursorSplitterHandle(QSplitterHandle):
    """QSplitterHandle that reliably shows the vertical double-arrow cursor
    while the mouse is over the splitter bar."""

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)

    def enterEvent(self, event):
        if not QApplication.mouseButtons():
            self.setCursor(Qt.SplitVCursor)
        super().enterEvent(event)

    def mouseMoveEvent(self, event):
        if not QApplication.mouseButtons() and self.cursor().shape() != Qt.SplitVCursor:
            self.setCursor(Qt.SplitVCursor)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)


class CursorSplitter(QSplitter):
    """QSplitter that uses CursorSplitterHandle so the resize cursor is always
    shown when hovering the bar between the panes."""

    def createHandle(self):
        return CursorSplitterHandle(self.orientation(), self)


class CursorSupervisor(QObject):
    """Guarantees the double-arrow resize cursors.

    Three independent layers make sure the cursor appears even when the
    platform swallows hover events or reports mouse positions in a different
    coordinate system (e.g. high-DPI scaling):
      1. an application-wide event filter that uses widget-local coordinates,
      2. a timer that polls the global mouse position,
      3. an application-wide *override* cursor as the final fallback.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._headers = []
        self._handles = []
        self._override_active = False
        self._override_shape = None
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._check)
        self._timer.start()

    def register_header(self, header):
        if header not in self._headers:
            self._headers.append(header)

    def register_handle(self, handle):
        if handle not in self._handles:
            self._handles.append(handle)

    # ---------- Layer 1: app-wide event filter (widget-local coords) ----------
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove:
            for header in self._headers:
                viewport = header.viewport()
                if obj is header or (viewport is not None and obj is viewport):
                    self._apply_header_cursor(header, event.pos())
            for handle in self._handles:
                if obj is handle:
                    self._apply_handle_cursor(handle, handle.rect().contains(event.pos()))
        elif event.type() in (QEvent.Leave, QEvent.HoverLeave):
            for header in self._headers:
                viewport = header.viewport()
                if obj is header or (viewport is not None and obj is viewport):
                    viewport.unsetCursor()
            for handle in self._handles:
                if obj is handle:
                    handle.unsetCursor()
        return False

    # ---------- Layer 2: timer polling (global coords) ----------
    def _check(self, gpos=None):
        if gpos is None:
            gpos = QCursor.pos()
        if QApplication.mouseButtons():
            return  # dragging - Qt manages its own cursor
        near_header = False
        near_handle = False
        for header in self._headers:
            viewport = header.viewport()
            if not header.isVisible() or viewport is None or not viewport.isVisible():
                continue
            pos = viewport.mapFromGlobal(gpos)
            if viewport.rect().contains(pos):
                self._apply_header_cursor(header, pos)
                x = pos.x()
                if any(abs(header.sectionViewportPosition(i) - x) <= 5
                       for i in range(1, header.count())):
                    near_header = True
            elif viewport.cursor().shape() != Qt.ArrowCursor:
                viewport.unsetCursor()
        for handle in self._handles:
            if not handle.isVisible():
                continue
            over = handle.rect().contains(handle.mapFromGlobal(gpos))
            self._apply_handle_cursor(handle, over)
            near_handle = near_handle or over
        # ---------- Layer 3: override cursor (final fallback) ----------
        if near_header or near_handle:
            shape = Qt.SplitHCursor if near_header else Qt.SplitVCursor
            if not self._override_active or self._override_shape != shape:
                if self._override_active:
                    QApplication.restoreOverrideCursor()
                QApplication.setOverrideCursor(shape)
                self._override_active = True
                self._override_shape = shape
        elif self._override_active:
            QApplication.restoreOverrideCursor()
            self._override_active = False
            self._override_shape = None

    def _apply_header_cursor(self, header, pos):
        x = pos.x()
        viewport = header.viewport()
        near = any(abs(header.sectionViewportPosition(i) - x) <= 5
                   for i in range(1, header.count()))
        if near:
            if viewport.cursor().shape() != Qt.SplitHCursor:
                viewport.setCursor(Qt.SplitHCursor)
        else:
            if viewport.cursor().shape() != Qt.ArrowCursor:
                viewport.unsetCursor()

    def _apply_handle_cursor(self, handle, over):
        if over:
            if handle.cursor().shape() != Qt.SplitVCursor:
                handle.setCursor(Qt.SplitVCursor)
        else:
            if handle.cursor().shape() != Qt.ArrowCursor:
                handle.unsetCursor()


class PathDelegate(QStyledItemDelegate):
    """Draw file paths so that the *tail* stays visible.

    The text is elided on the left, therefore a narrow column shows only the
    file name and the last directory level (".../last_dir/file.txt"). Widening
    the column gradually reveals more of the directory structure until the
    complete path (up to the home directory) is displayed.
    """

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        full = opt.text
        if full:
            rect = opt.rect.adjusted(4, 0, -4, 0)
            opt.text = opt.fontMetrics.elidedText(full, Qt.ElideLeft, max(20, rect.width()))
            opt.textElideMode = Qt.ElideLeft
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)


class TTSThread(QThread):
    """Thread for processing TTS tasks using the Kokoro ONNX model."""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # process_id, progress
    time_signal = pyqtSignal(int, str)  # process_id, time_info
    work_time_signal = pyqtSignal(int, float)  # process_id, work seconds (pauses excluded)
    status_signal = pyqtSignal(int, str)  # process_id, status
    finished_signal = pyqtSignal(int, bool)  # process_id, was_canceled
    error_signal = pyqtSignal(int, str)

    def __init__(self, process_id, input_file, output_file, speed, voice_weights, voice_weights_1, voice_weights_2, voice_weights_3, voice_weights_4, voice_weights_5):
        super().__init__()
        self.process_id = process_id
        self.input_file = input_file
        self.output_file = output_file
        self.default_speed = speed
        self.voice_weights = voice_weights
        self.voice_weights_1 = voice_weights_1
        self.voice_weights_2 = voice_weights_2
        self.voice_weights_3 = voice_weights_3
        self.voice_weights_4 = voice_weights_4
        self.voice_weights_5 = voice_weights_5
        self._stop = False
        self._was_canceled = False
        self._paused = False
        self.start_time = None
        self.pause_start_time = None
        self.total_pause_duration = 0
        self.kokoro = None
        self.audio_file = None
        self.last_time_update = 0
        self._cleaned = False

    def cleanup(self):
        """Release all resources used by the thread."""
        if self._cleaned:
            return
        self._cleaned = True
        try:
            if self.audio_file is not None:
                self.audio_file.close()
                self.audio_file = None
            if self.kokoro is not None:
                self.kokoro = None
            gc.collect()
            memory_usage = psutil.Process().memory_info().rss / 1024**2
            self.log_signal.emit(TRANSLATIONS["log_memory_freed"].format(self.process_id, memory_usage))
        except Exception as e:
            self.log_signal.emit(TRANSLATIONS["log_cleanup_warning"].format(self.process_id, str(e)))

    def stop(self):
        """Request the thread to stop.

        Cleanup intentionally happens inside the worker thread (see run())
        so that model/audio resources are never touched from the GUI thread
        while the worker is still using them (this prevented crashes/races
        when canceling during onnxruntime inference or file writes).
        """
        self._stop = True
        self._was_canceled = True

    def pause(self):
        """Pause or resume the thread."""
        if not self._paused:
            self._paused = True
            self.pause_start_time = time.time()
            self.status_signal.emit(self.process_id, "Paused")
        else:
            if self.pause_start_time is not None:
                pause_duration = time.time() - self.pause_start_time
                self.total_pause_duration += pause_duration
                self.pause_start_time = None
            self._paused = False
            self.status_signal.emit(self.process_id, "Running")

    def format_time(self, seconds):
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def update_time(self, current_time, status="processing"):
        """Update the elapsed time if one second has passed."""
        if current_time - self.last_time_update >= 1.0:
            elapsed_time = (current_time - self.start_time) - self.total_pause_duration
            time_info = f"Time: {self.format_time(elapsed_time)} ({status})"
            self.time_signal.emit(self.process_id, time_info)
            self.work_time_signal.emit(self.process_id, elapsed_time)
            self.last_time_update = current_time

    def run(self):
        """Execute the TTS processing task."""
        try:
            self.start_time = time.time()
            self.last_time_update = self.start_time
            self.log_signal.emit(TRANSLATIONS["log_process_init"].format(self.process_id))
            try:
                self.kokoro = Kokoro("kokoro.onnx", "voices-v1.0.bin")
            except Exception as e:
                raise RuntimeError(f"Could not load Kokoro model files: {e}") from e
            self.status_signal.emit(self.process_id, "Running")
            self.progress_signal.emit(self.process_id, 5)  # Indicate processing has started
            self.update_time(self.start_time, "processing")  # Initial time update

            # Compute the three voice packs. If a mix has no active voices, it
            # falls back to the first mix that does have active voices, so that
            # "default"/"custom_mix" texts never get a zero vector (silence).
            def build_voicepack(weights):
                wsum = sum(w for w in weights.values() if w > 0)
                if wsum == 0:
                    return None
                return sum(
                    self.kokoro.voices[v] * (w / wsum)
                    for v, w in weights.items() if w > 0
                )

            packs = {
                "custom_mix": build_voicepack(self.voice_weights),
                "custom_mix_1": build_voicepack(self.voice_weights_1),
                "custom_mix_2": build_voicepack(self.voice_weights_2),
                "custom_mix_3": build_voicepack(self.voice_weights_3),
                "custom_mix_4": build_voicepack(self.voice_weights_4),
                "custom_mix_5": build_voicepack(self.voice_weights_5),
            }
            active_pack = next((p for p in packs.values() if p is not None), None)
            if active_pack is None:
                raise RuntimeError(TRANSLATIONS["log_no_active_mix"].format(self.process_id))
            for name, pack in packs.items():
                if pack is None:
                    self.log_signal.emit(
                        TRANSLATIONS["log_voicepack_fallback"].format(self.process_id, name)
                    )
                    packs[name] = active_pack
            VOICEPACK = packs["custom_mix"]
            VOICEPACK_1 = packs["custom_mix_1"]
            VOICEPACK_2 = packs["custom_mix_2"]
            VOICEPACK_3 = packs["custom_mix_3"]
            VOICEPACK_4 = packs["custom_mix_4"]
            VOICEPACK_5 = packs["custom_mix_5"]

            def parse_text_file(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                entries = []
                buffer = []
                current_voice = None
                current_speed = self.default_speed
                for line in lines:
                    line = line.strip()
                    if line.startswith("[voice=") and line.endswith("]"):
                        if buffer:
                            entries.append(("text", current_voice or "default", " ".join(buffer), current_speed))
                            buffer = []
                        current_voice = line[len("[voice="):-1]
                    elif line.startswith("[pause=") and line.endswith("]"):
                        if buffer:
                            entries.append(("text", current_voice or "default", " ".join(buffer), current_speed))
                            buffer = []
                        pause_duration = float(line[len("[pause="):-1])
                        entries.append(("pause", pause_duration))
                    elif line.startswith("[speed=") and line.endswith("]"):
                        if buffer:
                            entries.append(("text", current_voice or "default", " ".join(buffer), current_speed))
                            buffer = []
                        current_speed = float(line[len("[speed="):-1])
                        self.log_signal.emit(TRANSLATIONS["log_custom_speed"].format(self.process_id, len(entries) + 1, current_speed))
                    elif line:
                        buffer.append(line)
                if buffer:
                    entries.append(("text", current_voice or "default", " ".join(buffer), current_speed))
                return entries

            entries = parse_text_file(self.input_file)
            self.log_signal.emit(TRANSLATIONS["log_file_parsed"].format(self.process_id, len(entries)))

            total_entries = len(entries)
            self.log_signal.emit(TRANSLATIONS["log_total_entries"].format(self.process_id, total_entries))

            sample_rate = 24000
            output_dir = os.path.dirname(self.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            if output_dir and not os.access(output_dir, os.W_OK):
                self.log_signal.emit(TRANSLATIONS["log_no_write_access"].format(self.process_id, output_dir))
                self.error_signal.emit(self.process_id, TRANSLATIONS["log_no_write_access"].format(self.process_id, output_dir))
                return

            self.audio_file = sf.SoundFile(self.output_file, mode='w', samplerate=sample_rate, channels=1)

            for i, entry in enumerate(entries):
                if self._stop:
                    self.log_signal.emit(TRANSLATIONS["log_process_canceled"].format(self.process_id))
                    return

                if total_entries > 0:
                    self.progress_signal.emit(self.process_id, int(i * 100 / total_entries))

                while self._paused and not self._stop:
                    self.msleep(100)
                    current_time = time.time()
                    self.update_time(current_time, "paused")
                if self._stop:
                    break

                current_time = time.time()
                self.update_time(current_time, "processing")

                entry_type = entry[0]
                if entry_type == "pause":
                    custom_pause_duration = entry[1]
                    self.log_signal.emit(TRANSLATIONS["log_custom_pause"].format(self.process_id, i+1, custom_pause_duration))
                    silence = np.zeros(int(custom_pause_duration * sample_rate), dtype=np.float32)
                    self.audio_file.write(silence)
                    del silence
                    gc.collect()
                elif entry_type == "text":
                    voice, text, speed = entry[1], entry[2], entry[3]
                    # Clamp speed to the range supported by kokoro-onnx.
                    clamped_speed = max(0.5, min(2.0, speed))
                    if abs(clamped_speed - speed) > 1e-9:
                        self.log_signal.emit(
                            TRANSLATIONS["log_speed_clamped"].format(self.process_id, i + 1, speed, clamped_speed)
                        )
                        speed = clamped_speed

                    if voice == "custom_mix":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK
                    elif voice == "custom_mix_1":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix_1"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK_1
                    elif voice == "custom_mix_2":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix_2"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK_2
                    elif voice == "custom_mix_3":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix_3"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK_3
                    elif voice == "custom_mix_4":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix_4"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK_4
                    elif voice == "custom_mix_5":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix_5"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK_5
                    elif voice == "default":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK
                    else:
                        self.log_signal.emit(TRANSLATIONS["log_process_voice"].format(self.process_id, i+1, voice))
                        try:
                            actual_voice = self.kokoro.voices[voice]
                        except KeyError:
                            self.log_signal.emit(TRANSLATIONS["log_voice_not_found"].format(self.process_id, voice))
                            actual_voice = VOICEPACK

                    text = text.strip()
                    if not text:
                        continue

                    # Pause/stop check BEFORE synthesis.
                    while self._paused and not self._stop:
                        self.msleep(100)
                        self.update_time(time.time(), "paused")
                    if self._stop:
                        break

                    self.log_signal.emit(TRANSLATIONS["log_generate_text"].format(self.process_id, text[:40]))
                    with torch.no_grad():
                        samples, sr = self.kokoro.create(text, voice=actual_voice, speed=speed, lang="en-us")
                    self.log_signal.emit(TRANSLATIONS["log_sample_rate"].format(self.process_id, sr, len(samples)))
                    if sr != sample_rate:
                        self.log_signal.emit(TRANSLATIONS["log_sample_rate_warning"].format(self.process_id, sr, sample_rate))

                    # Pause/stop check AFTER synthesis, BEFORE writing the audio -
                    # this makes Pause effective even for a single text entry.
                    while self._paused and not self._stop:
                        self.msleep(100)
                        self.update_time(time.time(), "paused")
                    if self._stop:
                        break
                    self.audio_file.write(samples)
                    del samples
                    gc.collect()

                    memory_usage = psutil.Process().memory_info().rss / 1024**2
                    self.log_signal.emit(TRANSLATIONS["log_memory_usage"].format(self.process_id, memory_usage))

                current_time = time.time()
                self.update_time(current_time, "processing")

            if self._stop:
                self.log_signal.emit(TRANSLATIONS["log_process_canceled"].format(self.process_id))
                return

            current_time = time.time()
            elapsed_time = (current_time - self.start_time) - self.total_pause_duration
            self.log_signal.emit(TRANSLATIONS["log_process_completed"].format(self.process_id, self.output_file))
            self.progress_signal.emit(self.process_id, 100)
            self.status_signal.emit(self.process_id, "Completed")
            self.time_signal.emit(self.process_id, f"Time: {self.format_time(elapsed_time)} (completed)")
            self.work_time_signal.emit(self.process_id, elapsed_time)

        except Exception as e:
            error_msg = TRANSLATIONS["log_error"].format(self.process_id, str(e))
            self.log_signal.emit(error_msg)
            self.error_signal.emit(self.process_id, str(e))
            self.status_signal.emit(self.process_id, f"Error: {str(e)}")
            self.progress_signal.emit(self.process_id, 0)
            self.time_signal.emit(self.process_id, "Time: --:--:-- (error)")
            self.work_time_signal.emit(self.process_id,
                max(0.0, (time.time() - self.start_time) - self.total_pause_duration))
        finally:
            self.cleanup()
            memory_usage = psutil.Process().memory_info().rss / 1024**2
            self.log_signal.emit(TRANSLATIONS["log_memory_freed"].format(self.process_id, memory_usage))
            self.finished_signal.emit(self.process_id, self._was_canceled)

class MainWindow(QMainWindow):
    """Main window for the Kokoro TTS & Split GUI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TRANSLATIONS["window_title"])
        self.setGeometry(100, 100, 1200, 800)

        self.last_split_files = []

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.split_tab = QWidget()
        self.split_layout = QVBoxLayout()
        self.split_tab.setLayout(self.split_layout)
        self.tab_widget.addTab(self.split_tab, TRANSLATIONS["tab_split"])

        self.tts_tab = QWidget()
        self.tts_layout = QVBoxLayout()
        self.tts_tab.setLayout(self.tts_layout)
        self.tab_widget.addTab(self.tts_tab, TRANSLATIONS["tab_tts"])

        self.custom_mix_tab = QWidget()
        self.custom_mix_layout = QVBoxLayout()
        self.custom_mix_tab.setLayout(self.custom_mix_layout)
        self.tab_widget.addTab(self.custom_mix_tab, TRANSLATIONS["tab_custom_mix"])

        self.custom_mix_1_tab = QWidget()
        self.custom_mix_1_layout = QVBoxLayout()
        self.custom_mix_1_tab.setLayout(self.custom_mix_1_layout)
        self.tab_widget.addTab(self.custom_mix_1_tab, TRANSLATIONS["tab_custom_mix_1"])

        self.custom_mix_2_tab = QWidget()
        self.custom_mix_2_layout = QVBoxLayout()
        self.custom_mix_2_tab.setLayout(self.custom_mix_2_layout)
        self.tab_widget.addTab(self.custom_mix_2_tab, TRANSLATIONS["tab_custom_mix_2"])

        self.custom_mix_3_tab = QWidget()
        self.custom_mix_3_layout = QVBoxLayout()
        self.custom_mix_3_tab.setLayout(self.custom_mix_3_layout)
        self.tab_widget.addTab(self.custom_mix_3_tab, TRANSLATIONS["tab_custom_mix_3"])

        self.custom_mix_4_tab = QWidget()
        self.custom_mix_4_layout = QVBoxLayout()
        self.custom_mix_4_tab.setLayout(self.custom_mix_4_layout)
        self.tab_widget.addTab(self.custom_mix_4_tab, TRANSLATIONS["tab_custom_mix_4"])

        self.custom_mix_5_tab = QWidget()
        self.custom_mix_5_layout = QVBoxLayout()
        self.custom_mix_5_tab.setLayout(self.custom_mix_5_layout)
        self.tab_widget.addTab(self.custom_mix_5_tab, TRANSLATIONS["tab_custom_mix_5"])

        self.available_voices = []
        self.models_loaded = False
        self.tts_tasks = []
        self.tts_threads = {}
        self.tts_process_counter = 0
        self.tts_task_queue = []
        self.tts_pending_cleanup = set()
        self.tts_global_paused = False
        self.tts_task_start_times = {}
        self.tts_progress_values = {}
        self.tts_finished_durations = []
        self.tts_work_times = {}  # process_id -> work seconds (pauses excluded)

        self.init_split_tab()
        self.init_tts_tab()
        try:
            self.init_custom_mix_tab()
            self.models_loaded = True
        except (FileNotFoundError, RuntimeError) as e:
            self.models_loaded = False
            QMessageBox.critical(
                self, "Error",
                f"{e}\n\nPlace 'kokoro.onnx' and 'voices-v1.0.bin' next to the script.\n"
                "The TTS and voice-mix tabs are disabled until the model files are available."
            )
            for idx in range(1, self.tab_widget.count()):
                self.tab_widget.setTabEnabled(idx, False)
        except Exception as e:
            self.models_loaded = False
            QMessageBox.critical(self, "Error", f"Unexpected error while loading Kokoro voices: {e}")
            for idx in range(1, self.tab_widget.count()):
                self.tab_widget.setTabEnabled(idx, False)

        self.init_custom_mix_1_tab()
        self.init_custom_mix_2_tab()
        self.init_custom_mix_3_tab()
        self.init_custom_mix_4_tab()
        self.init_custom_mix_5_tab()
        if self.models_loaded:
            self.tab_widget.setCurrentWidget(self.tts_tab)
        else:
            self.tab_widget.setCurrentWidget(self.split_tab)

        self.load_last_configuration()

    def init_split_tab(self):
        """Initialize the Text Splitting tab."""
        self.split_input_file_edit = QLineEdit()
        self.split_input_file_edit.setPlaceholderText(TRANSLATIONS["split_input_file_placeholder"])
        split_input_file_button = QPushButton(TRANSLATIONS["split_browse_button"])
        split_input_file_button.clicked.connect(self.browse_split_input_file)

        split_input_file_layout = QHBoxLayout()
        split_input_file_layout.addWidget(QLabel(TRANSLATIONS["split_input_file_label"]))
        split_input_file_layout.addWidget(self.split_input_file_edit)
        split_input_file_layout.addWidget(split_input_file_button)
        self.split_layout.addLayout(split_input_file_layout)

        self.split_parts_spin = QSpinBox()
        self.split_parts_spin.setRange(1, 1000)
        self.split_parts_spin.setValue(999)  # Changed default from 10 to 999

        split_parts_layout = QHBoxLayout()
        split_parts_layout.addWidget(QLabel(TRANSLATIONS["split_parts_label"]))
        split_parts_layout.addWidget(self.split_parts_spin)
        split_parts_layout.addStretch()
        self.split_layout.addLayout(split_parts_layout)

        self.split_word_edit = QLineEdit()
        self.split_word_edit.setPlaceholderText(TRANSLATIONS["split_word_placeholder"])
        self.split_word_edit.setText("[voice=")  # Changed default from [voice=custom_mix] to [voice=

        split_word_layout = QHBoxLayout()
        split_word_layout.addWidget(QLabel(TRANSLATIONS["split_word_label"]))
        split_word_layout.addWidget(self.split_word_edit)
        split_word_layout.addStretch()
        self.split_layout.addLayout(split_word_layout)

        self.split_button = QPushButton(TRANSLATIONS["split_button"])
        self.split_button.clicked.connect(self.split_text_file)
        self.split_layout.addWidget(self.split_button)

        self.load_split_to_tts_button = QPushButton(TRANSLATIONS["load_split_to_tts_button"])
        self.load_split_to_tts_button.clicked.connect(self.load_split_files_to_tts)
        self.load_split_to_tts_button.setEnabled(False)
        self.split_layout.addWidget(self.load_split_to_tts_button)

        self.split_log_text = QTextEdit()
        self.split_log_text.setReadOnly(True)
        self.split_log_text.setPlaceholderText(TRANSLATIONS["split_log_placeholder"])
        self.split_layout.addWidget(QLabel(TRANSLATIONS["split_log_label"]))
        self.split_layout.addWidget(self.split_log_text)

        self.split_layout.addStretch()

    def init_tts_tab(self):
        """Initialize the TTS Processing tab."""
        tts_splitter = CursorSplitter(Qt.Vertical)
        self.tts_layout.addWidget(tts_splitter)

        tts_upper_widget = QWidget()
        tts_upper_layout = QVBoxLayout()
        tts_upper_widget.setLayout(tts_upper_layout)
        tts_scroll_widget = QWidget()
        tts_scroll_layout = QVBoxLayout()
        tts_scroll_widget.setLayout(tts_scroll_layout)
        tts_scroll_area = QScrollArea()
        tts_scroll_area.setWidgetResizable(True)
        tts_scroll_area.setWidget(tts_scroll_widget)
        tts_upper_layout.addWidget(tts_scroll_area)

        self.tts_input_file_edit = QLineEdit()
        self.tts_input_file_edit.setPlaceholderText(TRANSLATIONS["tts_input_file_placeholder"])
        tts_input_file_button = QPushButton(TRANSLATIONS["tts_browse_button"])
        tts_input_file_button.clicked.connect(self.browse_tts_input_file)
        tts_split_files_button = QPushButton(TRANSLATIONS["tts_split_files_button"])
        tts_split_files_button.clicked.connect(self.browse_tts_split_files)

        tts_input_layout = QHBoxLayout()
        tts_input_layout.addWidget(QLabel(TRANSLATIONS["tts_input_file_label"]))
        tts_input_layout.addWidget(self.tts_input_file_edit)
        tts_input_layout.addWidget(tts_input_file_button)
        tts_input_layout.addWidget(tts_split_files_button)
        tts_scroll_layout.addLayout(tts_input_layout)

        self.tts_output_file_edit = QLineEdit()
        self.tts_output_file_edit.setPlaceholderText(TRANSLATIONS["tts_output_file_placeholder"])
        tts_output_file_button = QPushButton(TRANSLATIONS["tts_browse_button"])
        tts_output_file_button.clicked.connect(self.browse_tts_output_file)

        tts_output_layout = QHBoxLayout()
        tts_output_layout.addWidget(QLabel(TRANSLATIONS["tts_output_file_label"]))
        tts_output_layout.addWidget(self.tts_output_file_edit)
        tts_output_layout.addWidget(tts_output_file_button)
        tts_scroll_layout.addLayout(tts_output_layout)

        tts_params_grid = QGridLayout()
        self.tts_speed_spin = QDoubleSpinBox()
        self.tts_speed_spin.setRange(0.5, 2.0)  # kokoro-onnx supports 0.5–2.0
        self.tts_speed_spin.setValue(0.9)
        self.tts_speed_spin.setSingleStep(0.1)
        tts_params_grid.addWidget(QLabel(TRANSLATIONS["tts_speed_label"]), 0, 0)
        tts_params_grid.addWidget(self.tts_speed_spin, 0, 1)

        self.tts_max_threads_spin = QSpinBox()
        self.tts_max_threads_spin.setRange(0, 8)  # Changed to allow 0 threads
        self.tts_max_threads_spin.setValue(1)  # Changed default from 2 to 1
        self.tts_max_threads_spin.valueChanged.connect(self.on_max_threads_changed)
        tts_params_grid.addWidget(QLabel(TRANSLATIONS["tts_max_threads_label"]), 1, 0)
        tts_params_grid.addWidget(self.tts_max_threads_spin, 1, 1)
        tts_scroll_layout.addLayout(tts_params_grid)

        tts_scroll_layout.addWidget(QLabel(TRANSLATIONS["tts_config_label"]))
        config_layout = QHBoxLayout()
        save_config_button = QPushButton(TRANSLATIONS["tts_save_config_button"])
        save_config_button.clicked.connect(self.save_tts_configuration)
        load_config_button = QPushButton(TRANSLATIONS["tts_load_config_button"])
        load_config_button.clicked.connect(self.load_tts_configuration)
        config_layout.addWidget(save_config_button)
        config_layout.addWidget(load_config_button)
        tts_scroll_layout.addLayout(config_layout)

        tts_scroll_layout.addStretch()

        tts_lower_widget = QWidget()
        tts_lower_layout = QVBoxLayout()
        tts_lower_widget.setLayout(tts_lower_layout)
        tts_button_row = QHBoxLayout()
        self.tts_add_task_button = QPushButton(TRANSLATIONS["tts_add_task_button"])
        self.tts_add_task_button.setFixedHeight(30)
        self.tts_add_task_button.setMinimumWidth(130)
        self.tts_add_task_button.clicked.connect(self.add_tts_task)
        tts_button_row.addWidget(self.tts_add_task_button)
        self.tts_pause_all_button = QPushButton(TRANSLATIONS["tts_pause_button"])
        self.tts_pause_all_button.setFixedHeight(30)
        self.tts_pause_all_button.setMinimumWidth(90)
        self.tts_pause_all_button.setEnabled(False)
        self.tts_pause_all_button.clicked.connect(self.toggle_pause_all_tasks)
        tts_button_row.addWidget(self.tts_pause_all_button)
        self.tts_clear_all_button = QPushButton(TRANSLATIONS["tts_clear_all_button"])
        self.tts_clear_all_button.setFixedHeight(20)
        self.tts_clear_all_button.setFixedWidth(90)
        self.tts_clear_all_button.setStyleSheet("font-size: 8pt; padding: 0px 4px;")
        self.tts_clear_all_button.clicked.connect(self.delete_all_tts_tasks)
        tts_button_row.addWidget(self.tts_clear_all_button)
        tts_button_row.addStretch()
        tts_lower_layout.addLayout(tts_button_row)
        # Overall batch progress bar + ETA for the whole task list
        overall_row = QHBoxLayout()
        overall_row.addWidget(QLabel(TRANSLATIONS["tts_overall_label"]))
        self.tts_overall_progress = QProgressBar()
        self.tts_overall_progress.setRange(0, 100)
        self.tts_overall_progress.setValue(0)
        self.tts_overall_progress.setFixedWidth(220)
        overall_row.addWidget(self.tts_overall_progress)
        self.tts_overall_status_label = QLabel(TRANSLATIONS["tts_overall_done"].format(done=0, total=0))
        overall_row.addWidget(self.tts_overall_status_label)
        self.tts_overall_eta_label = QLabel(TRANSLATIONS["tts_overall_eta_unknown"])
        overall_row.addWidget(self.tts_overall_eta_label)
        self.tts_overall_elapsed_label = QLabel(
            TRANSLATIONS["tts_overall_elapsed"].format(time="0:00"))
        overall_row.addWidget(self.tts_overall_elapsed_label)
        overall_row.addStretch()
        tts_lower_layout.addLayout(overall_row)
        tts_lower_layout.addWidget(QLabel(TRANSLATIONS["tts_processes_label"]))

        self.tts_process_table = QTableWidget()
        self.tts_process_table.setColumnCount(8)
        self.tts_process_table.setHorizontalHeaderLabels(TRANSLATIONS["tts_table_headers"])
        self.tts_process_table.setSelectionMode(QTableWidget.NoSelection)
        self.tts_process_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tts_process_table.setAlternatingRowColors(True)
        # Visible header section borders + interactive resize/move handles
        # (the double-arrow cursor appears when hovering a section border).
        self.tts_process_table.setStyleSheet(
            "QHeaderView::section { background-color: #ececec; border: 1px solid #a0a0a0;"
            " padding: 4px; font-weight: bold; }"
        )
        # Custom header subclass that reliably shows the split double-arrow
        # cursor when the mouse hovers a column separator line.
        header = CursorHeaderView(Qt.Horizontal)
        self.tts_process_table.setHorizontalHeader(header)
        header.setSectionsClickable(True)
        header.setHighlightSections(True)
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionsMovable(True)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(60)
        # Initial column widths; user adjustments are NOT overwritten later
        # (resizeColumnsToContents on every task addition was removed).
        self.tts_process_table.setColumnWidth(0, 60)
        self.tts_process_table.setColumnWidth(1, 200)
        self.tts_process_table.setColumnWidth(2, 200)
        self.tts_process_table.setColumnWidth(3, 100)
        self.tts_process_table.setColumnWidth(4, 90)
        self.tts_process_table.setColumnWidth(5, 200)
        self.tts_process_table.setColumnWidth(6, 230)
        self.tts_process_table.setColumnWidth(7, 60)
        # Elide file paths on the left: by default only the file name and the
        # last directory level are visible; widening the column reveals more of
        # the directory structure up to the full path.
        self._path_delegate = PathDelegate(self.tts_process_table)
        self.tts_process_table.setItemDelegateForColumn(1, self._path_delegate)
        self.tts_process_table.setItemDelegateForColumn(2, self._path_delegate)
        tts_lower_layout.addWidget(self.tts_process_table)

        tts_splitter.addWidget(tts_upper_widget)
        tts_splitter.addWidget(tts_lower_widget)
        tts_splitter.setSizes([400, 400])
        tts_splitter.setStretchFactor(0, 0)
        tts_splitter.setStretchFactor(1, 1)
        tts_splitter.setHandleWidth(8)
        tts_splitter.setChildrenCollapsible(True)
        tts_splitter.setStyleSheet("QSplitter::handle { background-color: #c0c0c0; }")
        # Guaranteed double-arrow cursors (works even where the platform
        # swallows hover/mousemove events): polls the global mouse position.
        self._cursor_supervisor = CursorSupervisor(self)
        self._cursor_supervisor.register_header(header)
        self._cursor_supervisor.register_handle(tts_splitter.handle(0))

    def init_custom_mix_tab(self):
        """Initialize the Voice Custom Mix tab."""
        if not os.path.exists("kokoro.onnx") or not os.path.exists("voices-v1.0.bin"):
            raise FileNotFoundError("Kokoro model files are missing.")
        try:
            kokoro_temp = Kokoro("kokoro.onnx", "voices-v1.0.bin")
        except Exception as e:
            raise RuntimeError(f"Could not load Kokoro model: {e}") from e
        self.available_voices = sorted(kokoro_temp.voices.keys())
        del kokoro_temp
        gc.collect()
        self.custom_mix_voice_checkboxes = {}
        self.custom_mix_voice_spins = {}
        default_weights = {voice: 0.0 for voice in self.available_voices}
        default_enabled = {voice: False for voice in self.available_voices}
        self.custom_mix_layout.addWidget(QLabel(TRANSLATIONS["voice_selection_label"]))
        voice_grid = QGridLayout()
        for idx, voice in enumerate(self.available_voices):
            row = idx // 3
            col = idx % 3
            voice_layout = QHBoxLayout()
            checkbox = QCheckBox(voice)
            checkbox.setChecked(default_enabled.get(voice, False))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setValue(default_weights.get(voice, 0.0))
            spin.setSingleStep(0.05)
            spin.setEnabled(checkbox.isChecked())
            checkbox.stateChanged.connect(lambda state, s=spin: s.setEnabled(state == Qt.Checked))
            self.custom_mix_voice_checkboxes[voice] = checkbox
            self.custom_mix_voice_spins[voice] = spin
            voice_layout.addWidget(checkbox)
            voice_layout.addWidget(spin)
            voice_grid.addLayout(voice_layout, row, col)
        self.custom_mix_layout.addLayout(voice_grid)

        self.custom_mix_layout.addWidget(QLabel(TRANSLATIONS["tts_config_label"]))
        config_layout = QHBoxLayout()
        save_config_button = QPushButton(TRANSLATIONS["tts_save_config_button"])
        save_config_button.clicked.connect(lambda: self.save_voice_mix_configuration("custom_mix"))
        load_config_button = QPushButton(TRANSLATIONS["tts_load_config_button"])
        load_config_button.clicked.connect(lambda: self.load_voice_mix_configuration("custom_mix"))
        config_layout.addWidget(save_config_button)
        config_layout.addWidget(load_config_button)
        self.custom_mix_layout.addLayout(config_layout)

        self.custom_mix_layout.addStretch()

    def init_custom_mix_1_tab(self):
        """Initialize the Voice Custom Mix 1 tab."""
        self.custom_mix_1_voice_checkboxes = {}
        self.custom_mix_1_voice_spins = {}
        default_weights = {voice: 0.0 for voice in self.available_voices}
        default_enabled = {voice: False for voice in self.available_voices}
        self.custom_mix_1_layout.addWidget(QLabel(TRANSLATIONS["voice_selection_label"]))
        voice_grid = QGridLayout()
        for idx, voice in enumerate(self.available_voices):
            row = idx // 3
            col = idx % 3
            voice_layout = QHBoxLayout()
            checkbox = QCheckBox(voice)
            checkbox.setChecked(default_enabled.get(voice, False))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setValue(default_weights.get(voice, 0.0))
            spin.setSingleStep(0.05)
            spin.setEnabled(checkbox.isChecked())
            checkbox.stateChanged.connect(lambda state, s=spin: s.setEnabled(state == Qt.Checked))
            self.custom_mix_1_voice_checkboxes[voice] = checkbox
            self.custom_mix_1_voice_spins[voice] = spin
            voice_layout.addWidget(checkbox)
            voice_layout.addWidget(spin)
            voice_grid.addLayout(voice_layout, row, col)
        self.custom_mix_1_layout.addLayout(voice_grid)

        self.custom_mix_1_layout.addWidget(QLabel(TRANSLATIONS["tts_config_label"]))
        config_layout = QHBoxLayout()
        save_config_button = QPushButton(TRANSLATIONS["tts_save_config_button"])
        save_config_button.clicked.connect(lambda: self.save_voice_mix_configuration("custom_mix_1"))
        load_config_button = QPushButton(TRANSLATIONS["tts_load_config_button"])
        load_config_button.clicked.connect(lambda: self.load_voice_mix_configuration("custom_mix_1"))
        config_layout.addWidget(save_config_button)
        config_layout.addWidget(load_config_button)
        self.custom_mix_1_layout.addLayout(config_layout)

        self.custom_mix_1_layout.addStretch()

    def init_custom_mix_2_tab(self):
        """Initialize the Voice Custom Mix 2 tab."""
        self.custom_mix_2_voice_checkboxes = {}
        self.custom_mix_2_voice_spins = {}
        default_weights = {voice: 0.0 for voice in self.available_voices}
        default_enabled = {voice: False for voice in self.available_voices}
        self.custom_mix_2_layout.addWidget(QLabel(TRANSLATIONS["voice_selection_label"]))
        voice_grid = QGridLayout()
        for idx, voice in enumerate(self.available_voices):
            row = idx // 3
            col = idx % 3
            voice_layout = QHBoxLayout()
            checkbox = QCheckBox(voice)
            checkbox.setChecked(default_enabled.get(voice, False))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setValue(default_weights.get(voice, 0.0))
            spin.setSingleStep(0.05)
            spin.setEnabled(checkbox.isChecked())
            checkbox.stateChanged.connect(lambda state, s=spin: s.setEnabled(state == Qt.Checked))
            self.custom_mix_2_voice_checkboxes[voice] = checkbox
            self.custom_mix_2_voice_spins[voice] = spin
            voice_layout.addWidget(checkbox)
            voice_layout.addWidget(spin)
            voice_grid.addLayout(voice_layout, row, col)
        self.custom_mix_2_layout.addLayout(voice_grid)

        self.custom_mix_2_layout.addWidget(QLabel(TRANSLATIONS["tts_config_label"]))
        config_layout = QHBoxLayout()
        save_config_button = QPushButton(TRANSLATIONS["tts_save_config_button"])
        save_config_button.clicked.connect(lambda: self.save_voice_mix_configuration("custom_mix_2"))
        load_config_button = QPushButton(TRANSLATIONS["tts_load_config_button"])
        load_config_button.clicked.connect(lambda: self.load_voice_mix_configuration("custom_mix_2"))
        config_layout.addWidget(save_config_button)
        config_layout.addWidget(load_config_button)
        self.custom_mix_2_layout.addLayout(config_layout)

        self.custom_mix_2_layout.addStretch()

    def init_custom_mix_3_tab(self):
        """Initialize the Voice Custom Mix 3 tab."""
        self.custom_mix_3_voice_checkboxes = {}
        self.custom_mix_3_voice_spins = {}
        default_weights = {voice: 0.0 for voice in self.available_voices}
        default_enabled = {voice: False for voice in self.available_voices}
        self.custom_mix_3_layout.addWidget(QLabel(TRANSLATIONS["voice_selection_label"]))
        voice_grid = QGridLayout()
        for idx, voice in enumerate(self.available_voices):
            row = idx // 3
            col = idx % 3
            voice_layout = QHBoxLayout()
            checkbox = QCheckBox(voice)
            checkbox.setChecked(default_enabled.get(voice, False))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setValue(default_weights.get(voice, 0.0))
            spin.setSingleStep(0.05)
            spin.setEnabled(checkbox.isChecked())
            checkbox.stateChanged.connect(lambda state, s=spin: s.setEnabled(state == Qt.Checked))
            self.custom_mix_3_voice_checkboxes[voice] = checkbox
            self.custom_mix_3_voice_spins[voice] = spin
            voice_layout.addWidget(checkbox)
            voice_layout.addWidget(spin)
            voice_grid.addLayout(voice_layout, row, col)
        self.custom_mix_3_layout.addLayout(voice_grid)

        self.custom_mix_3_layout.addWidget(QLabel(TRANSLATIONS["tts_config_label"]))
        config_layout = QHBoxLayout()
        save_config_button = QPushButton(TRANSLATIONS["tts_save_config_button"])
        save_config_button.clicked.connect(lambda: self.save_voice_mix_configuration("custom_mix_3"))
        load_config_button = QPushButton(TRANSLATIONS["tts_load_config_button"])
        load_config_button.clicked.connect(lambda: self.load_voice_mix_configuration("custom_mix_3"))
        config_layout.addWidget(save_config_button)
        config_layout.addWidget(load_config_button)
        self.custom_mix_3_layout.addLayout(config_layout)

        self.custom_mix_3_layout.addStretch()

    def init_custom_mix_4_tab(self):
        """Initialize the Voice Custom Mix 4 tab."""
        self.custom_mix_4_voice_checkboxes = {}
        self.custom_mix_4_voice_spins = {}
        default_weights = {voice: 0.0 for voice in self.available_voices}
        default_enabled = {voice: False for voice in self.available_voices}
        self.custom_mix_4_layout.addWidget(QLabel(TRANSLATIONS["voice_selection_label"]))
        voice_grid = QGridLayout()
        for idx, voice in enumerate(self.available_voices):
            row = idx // 3
            col = idx % 3
            voice_layout = QHBoxLayout()
            checkbox = QCheckBox(voice)
            checkbox.setChecked(default_enabled.get(voice, False))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setValue(default_weights.get(voice, 0.0))
            spin.setSingleStep(0.05)
            spin.setEnabled(checkbox.isChecked())
            checkbox.stateChanged.connect(lambda state, s=spin: s.setEnabled(state == Qt.Checked))
            self.custom_mix_4_voice_checkboxes[voice] = checkbox
            self.custom_mix_4_voice_spins[voice] = spin
            voice_layout.addWidget(checkbox)
            voice_layout.addWidget(spin)
            voice_grid.addLayout(voice_layout, row, col)
        self.custom_mix_4_layout.addLayout(voice_grid)

        self.custom_mix_4_layout.addWidget(QLabel(TRANSLATIONS["tts_config_label"]))
        config_layout = QHBoxLayout()
        save_config_button = QPushButton(TRANSLATIONS["tts_save_config_button"])
        save_config_button.clicked.connect(lambda: self.save_voice_mix_configuration("custom_mix_4"))
        load_config_button = QPushButton(TRANSLATIONS["tts_load_config_button"])
        load_config_button.clicked.connect(lambda: self.load_voice_mix_configuration("custom_mix_4"))
        config_layout.addWidget(save_config_button)
        config_layout.addWidget(load_config_button)
        self.custom_mix_4_layout.addLayout(config_layout)

        self.custom_mix_4_layout.addStretch()

    def init_custom_mix_5_tab(self):
        """Initialize the Voice Custom Mix 5 tab."""
        self.custom_mix_5_voice_checkboxes = {}
        self.custom_mix_5_voice_spins = {}
        default_weights = {voice: 0.0 for voice in self.available_voices}
        default_enabled = {voice: False for voice in self.available_voices}
        self.custom_mix_5_layout.addWidget(QLabel(TRANSLATIONS["voice_selection_label"]))
        voice_grid = QGridLayout()
        for idx, voice in enumerate(self.available_voices):
            row = idx // 3
            col = idx % 3
            voice_layout = QHBoxLayout()
            checkbox = QCheckBox(voice)
            checkbox.setChecked(default_enabled.get(voice, False))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setValue(default_weights.get(voice, 0.0))
            spin.setSingleStep(0.05)
            spin.setEnabled(checkbox.isChecked())
            checkbox.stateChanged.connect(lambda state, s=spin: s.setEnabled(state == Qt.Checked))
            self.custom_mix_5_voice_checkboxes[voice] = checkbox
            self.custom_mix_5_voice_spins[voice] = spin
            voice_layout.addWidget(checkbox)
            voice_layout.addWidget(spin)
            voice_grid.addLayout(voice_layout, row, col)
        self.custom_mix_5_layout.addLayout(voice_grid)

        self.custom_mix_5_layout.addWidget(QLabel(TRANSLATIONS["tts_config_label"]))
        config_layout = QHBoxLayout()
        save_config_button = QPushButton(TRANSLATIONS["tts_save_config_button"])
        save_config_button.clicked.connect(lambda: self.save_voice_mix_configuration("custom_mix_5"))
        load_config_button = QPushButton(TRANSLATIONS["tts_load_config_button"])
        load_config_button.clicked.connect(lambda: self.load_voice_mix_configuration("custom_mix_5"))
        config_layout.addWidget(save_config_button)
        config_layout.addWidget(load_config_button)
        self.custom_mix_5_layout.addLayout(config_layout)

        self.custom_mix_5_layout.addStretch()

    def on_max_threads_changed(self, value):
        """Handle changes to the maximum threads setting."""
        self.split_log_text.append(TRANSLATIONS["log_max_threads_changed"].format(value))
        self.start_tts_queued_tasks()

    def save_tts_configuration(self):
        """Save the TTS configuration to a JSON file, including voice weights."""
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "configs/", "JSON Files (*.json)")
        if file_name:
            if not file_name.endswith(".json"):
                file_name += ".json"
            config = {
                "speed": self.tts_speed_spin.value(),
                "voice_weights": {voice: self.custom_mix_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled": {voice: self.custom_mix_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
                "voice_weights_1": {voice: self.custom_mix_1_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled_1": {voice: self.custom_mix_1_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
                "voice_weights_2": {voice: self.custom_mix_2_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled_2": {voice: self.custom_mix_2_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
                "voice_weights_3": {voice: self.custom_mix_3_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled_3": {voice: self.custom_mix_3_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
                "voice_weights_4": {voice: self.custom_mix_4_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled_4": {voice: self.custom_mix_4_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
                "voice_weights_5": {voice: self.custom_mix_5_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled_5": {voice: self.custom_mix_5_voice_checkboxes[voice].isChecked() for voice in self.available_voices}
            }
            config_dir = os.path.dirname(file_name)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            try:
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4)
                self.split_log_text.append(TRANSLATIONS["log_config_saved"].format(file_name))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save configuration: {str(e)}")

    def load_tts_configuration(self):
        """Load a TTS configuration from a JSON file, including voice weights."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "configs/", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.tts_speed_spin.setValue(config.get("speed", 0.9))
                for voice in self.available_voices:
                    weight = config.get("voice_weights", {}).get(voice, 0.0)
                    enabled = config.get("voice_enabled", {}).get(voice, False)
                    self.custom_mix_voice_spins[voice].setValue(weight)
                    self.custom_mix_voice_checkboxes[voice].setChecked(enabled)
                    self.custom_mix_voice_spins[voice].setEnabled(enabled)
                    weight_1 = config.get("voice_weights_1", {}).get(voice, 0.0)
                    enabled_1 = config.get("voice_enabled_1", {}).get(voice, False)
                    self.custom_mix_1_voice_spins[voice].setValue(weight_1)
                    self.custom_mix_1_voice_checkboxes[voice].setChecked(enabled_1)
                    self.custom_mix_1_voice_spins[voice].setEnabled(enabled_1)
                    weight_2 = config.get("voice_weights_2", {}).get(voice, 0.0)
                    enabled_2 = config.get("voice_enabled_2", {}).get(voice, False)
                    self.custom_mix_2_voice_spins[voice].setValue(weight_2)
                    self.custom_mix_2_voice_checkboxes[voice].setChecked(enabled_2)
                    self.custom_mix_2_voice_spins[voice].setEnabled(enabled_2)
                    weight_3 = config.get("voice_weights_3", {}).get(voice, 0.0)
                    enabled_3 = config.get("voice_enabled_3", {}).get(voice, False)
                    self.custom_mix_3_voice_spins[voice].setValue(weight_3)
                    self.custom_mix_3_voice_checkboxes[voice].setChecked(enabled_3)
                    self.custom_mix_3_voice_spins[voice].setEnabled(enabled_3)
                    weight_4 = config.get("voice_weights_4", {}).get(voice, 0.0)
                    enabled_4 = config.get("voice_enabled_4", {}).get(voice, False)
                    self.custom_mix_4_voice_spins[voice].setValue(weight_4)
                    self.custom_mix_4_voice_checkboxes[voice].setChecked(enabled_4)
                    self.custom_mix_4_voice_spins[voice].setEnabled(enabled_4)
                    weight_5 = config.get("voice_weights_5", {}).get(voice, 0.0)
                    enabled_5 = config.get("voice_enabled_5", {}).get(voice, False)
                    self.custom_mix_5_voice_spins[voice].setValue(weight_5)
                    self.custom_mix_5_voice_checkboxes[voice].setChecked(enabled_5)
                    self.custom_mix_5_voice_spins[voice].setEnabled(enabled_5)
                self.split_log_text.append(TRANSLATIONS["log_config_loaded"].format(file_name))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load configuration: {str(e)}")

    def save_voice_mix_configuration(self, mix_type):
        """Save the voice mix configuration for a specific mix to a JSON file."""
        file_name, _ = QFileDialog.getSaveFileName(self, f"Save {mix_type} Configuration", "configs/", "JSON Files (*.json)")
        if file_name:
            if not file_name.endswith(".json"):
                file_name += ".json"
            if mix_type == "custom_mix":
                checkboxes = self.custom_mix_voice_checkboxes
                spins = self.custom_mix_voice_spins
            elif mix_type == "custom_mix_1":
                checkboxes = self.custom_mix_1_voice_checkboxes
                spins = self.custom_mix_1_voice_spins
            elif mix_type == "custom_mix_2":
                checkboxes = self.custom_mix_2_voice_checkboxes
                spins = self.custom_mix_2_voice_spins
            elif mix_type == "custom_mix_3":
                checkboxes = self.custom_mix_3_voice_checkboxes
                spins = self.custom_mix_3_voice_spins
            elif mix_type == "custom_mix_4":
                checkboxes = self.custom_mix_4_voice_checkboxes
                spins = self.custom_mix_4_voice_spins
            elif mix_type == "custom_mix_5":
                checkboxes = self.custom_mix_5_voice_checkboxes
                spins = self.custom_mix_5_voice_spins
            else:
                return
            config = {
                "voice_weights": {voice: spins[voice].value() for voice in self.available_voices},
                "voice_enabled": {voice: checkboxes[voice].isChecked() for voice in self.available_voices}
            }
            config_dir = os.path.dirname(file_name)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            try:
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4)
                self.split_log_text.append(TRANSLATIONS["log_config_saved"].format(file_name))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save configuration: {str(e)}")

    def load_voice_mix_configuration(self, mix_type):
        """Load a voice mix configuration for a specific mix from a JSON file."""
        file_name, _ = QFileDialog.getOpenFileName(self, f"Load {mix_type} Configuration", "configs/", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if mix_type == "custom_mix":
                    checkboxes = self.custom_mix_voice_checkboxes
                    spins = self.custom_mix_voice_spins
                elif mix_type == "custom_mix_1":
                    checkboxes = self.custom_mix_1_voice_checkboxes
                    spins = self.custom_mix_1_voice_spins
                elif mix_type == "custom_mix_2":
                    checkboxes = self.custom_mix_2_voice_checkboxes
                    spins = self.custom_mix_2_voice_spins
                elif mix_type == "custom_mix_3":
                    checkboxes = self.custom_mix_3_voice_checkboxes
                    spins = self.custom_mix_3_voice_spins
                elif mix_type == "custom_mix_4":
                    checkboxes = self.custom_mix_4_voice_checkboxes
                    spins = self.custom_mix_4_voice_spins
                elif mix_type == "custom_mix_5":
                    checkboxes = self.custom_mix_5_voice_checkboxes
                    spins = self.custom_mix_5_voice_spins
                else:
                    return
                for voice in self.available_voices:
                    weight = config.get("voice_weights", {}).get(voice, 0.0)
                    enabled = config.get("voice_enabled", {}).get(voice, False)
                    spins[voice].setValue(weight)
                    checkboxes[voice].setChecked(enabled)
                    spins[voice].setEnabled(enabled)
                self.split_log_text.append(TRANSLATIONS["log_config_loaded"].format(file_name))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load configuration: {str(e)}")

    def save_last_configuration(self):
        """Save the last TTS configuration to a JSON file."""
        config = {
            "speed": self.tts_speed_spin.value(),
            "voice_weights": {voice: self.custom_mix_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled": {voice: self.custom_mix_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
            "voice_weights_1": {voice: self.custom_mix_1_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled_1": {voice: self.custom_mix_1_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
            "voice_weights_2": {voice: self.custom_mix_2_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled_2": {voice: self.custom_mix_2_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
            "voice_weights_3": {voice: self.custom_mix_3_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled_3": {voice: self.custom_mix_3_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
            "voice_weights_4": {voice: self.custom_mix_4_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled_4": {voice: self.custom_mix_4_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
            "voice_weights_5": {voice: self.custom_mix_5_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled_5": {voice: self.custom_mix_5_voice_checkboxes[voice].isChecked() for voice in self.available_voices}
        }
        try:
            os.makedirs("configs", exist_ok=True)
            with open("configs/last_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.split_log_text.append(TRANSLATIONS["log_config_save_warning"].format(str(e)))

    def load_last_configuration(self):
        """Load the last TTS configuration from a JSON file."""
        config_file = "configs/last_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.tts_speed_spin.setValue(config.get("speed", 0.9))
                for voice in self.available_voices:
                    weight = config.get("voice_weights", {}).get(voice, 0.0)
                    enabled = config.get("voice_enabled", {}).get(voice, False)
                    self.custom_mix_voice_spins[voice].setValue(weight)
                    self.custom_mix_voice_checkboxes[voice].setChecked(enabled)
                    self.custom_mix_voice_spins[voice].setEnabled(enabled)
                    weight_1 = config.get("voice_weights_1", {}).get(voice, 0.0)
                    enabled_1 = config.get("voice_enabled_1", {}).get(voice, False)
                    self.custom_mix_1_voice_spins[voice].setValue(weight_1)
                    self.custom_mix_1_voice_checkboxes[voice].setChecked(enabled_1)
                    self.custom_mix_1_voice_spins[voice].setEnabled(enabled_1)
                    weight_2 = config.get("voice_weights_2", {}).get(voice, 0.0)
                    enabled_2 = config.get("voice_enabled_2", {}).get(voice, False)
                    self.custom_mix_2_voice_spins[voice].setValue(weight_2)
                    self.custom_mix_2_voice_checkboxes[voice].setChecked(enabled_2)
                    self.custom_mix_2_voice_spins[voice].setEnabled(enabled_2)
                    weight_3 = config.get("voice_weights_3", {}).get(voice, 0.0)
                    enabled_3 = config.get("voice_enabled_3", {}).get(voice, False)
                    self.custom_mix_3_voice_spins[voice].setValue(weight_3)
                    self.custom_mix_3_voice_checkboxes[voice].setChecked(enabled_3)
                    self.custom_mix_3_voice_spins[voice].setEnabled(enabled_3)
                    weight_4 = config.get("voice_weights_4", {}).get(voice, 0.0)
                    enabled_4 = config.get("voice_enabled_4", {}).get(voice, False)
                    self.custom_mix_4_voice_spins[voice].setValue(weight_4)
                    self.custom_mix_4_voice_checkboxes[voice].setChecked(enabled_4)
                    self.custom_mix_4_voice_spins[voice].setEnabled(enabled_4)
                    weight_5 = config.get("voice_weights_5", {}).get(voice, 0.0)
                    enabled_5 = config.get("voice_enabled_5", {}).get(voice, False)
                    self.custom_mix_5_voice_spins[voice].setValue(weight_5)
                    self.custom_mix_5_voice_checkboxes[voice].setChecked(enabled_5)
                    self.custom_mix_5_voice_spins[voice].setEnabled(enabled_5)
                self.split_log_text.append(TRANSLATIONS["log_last_config_loaded"])
            except Exception as e:
                self.split_log_text.append(TRANSLATIONS["log_config_load_warning"].format(str(e)))

    def closeEvent(self, event):
        """Handle the window close event."""
        try:
            self.save_last_configuration()
        except Exception:
            pass
        for thread in self.tts_threads.values():
            thread.stop()
        for thread in self.tts_threads.values():
            thread.wait(15000)
        for thread in self.tts_threads.values():
            thread.cleanup()
        self.tts_threads.clear()
        gc.collect()
        super().closeEvent(event)

    def browse_split_input_file(self):
        """Open a file dialog to select an input text file for splitting."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Text File", "", "Text Files (*.txt)")
        if file_name:
            self.split_input_file_edit.setText(file_name)
            self.split_log_text.append(f"Selected file: {file_name}")

    def split_text_file(self):
        """Split the input text file into multiple parts."""
        input_file = self.split_input_file_edit.text()
        num_parts = self.split_parts_spin.value()
        split_word = self.split_word_edit.text().strip()

        if not input_file or not os.path.exists(input_file):
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_invalid_input_file"])
            self.split_log_text.append(TRANSLATIONS["error_invalid_input_file"])
            return
        if num_parts < 1:
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_invalid_parts"])
            self.split_log_text.append(TRANSLATIONS["error_invalid_parts"])
            return
        if not split_word:
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_no_split_word"])
            self.split_log_text.append(TRANSLATIONS["error_no_split_word"])
            return

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                text = f.read()

            self.split_log_text.append(TRANSLATIONS["log_file_read"].format(input_file, len(text)))

            split_positions = [m.start() for m in re.finditer(re.escape(split_word), text)]
            if not split_positions:
                self.split_log_text.append(TRANSLATIONS["log_no_split_word"].format(split_word))
                reply = QMessageBox.warning(
                    self, "Split Warning",
                    f"No occurrences of '{split_word}' were found in the text.\n\n"
                    "Splitting by character count would ignore the split rule.\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    self.split_log_text.append(f"Splitting canceled: '{split_word}' not found in text.")
                    return
                part_size = len(text) // num_parts
                split_positions = [i * part_size for i in range(num_parts)]
            else:
                split_positions.insert(0, 0)

            self.split_log_text.append(TRANSLATIONS["log_split_points"].format(len(split_positions) - 1, split_word))

            total_length = len(text)
            ideal_part_size = total_length // num_parts
            actual_splits = [0]

            for i in range(1, num_parts):
                target_pos = i * ideal_part_size
                closest_pos = min(split_positions, key=lambda x: abs(x - target_pos) if x > actual_splits[-1] else float('inf'))
                if closest_pos > actual_splits[-1]:
                    actual_splits.append(closest_pos)
                else:
                    for pos in split_positions:
                        if pos > actual_splits[-1]:
                            actual_splits.append(pos)
                            break
                    else:
                        break

            actual_splits.append(len(text))
            actual_splits = sorted(list(set(actual_splits)))
            if len(actual_splits) - 1 < num_parts:
                self.split_log_text.append(f"Warning: Only {len(actual_splits) - 1} parts possible due to insufficient split points.")
                num_parts = len(actual_splits) - 1

            self.split_log_text.append(TRANSLATIONS["log_actual_splits"].format(actual_splits))

            input_dir = os.path.dirname(input_file)
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            self.last_split_files = []

            for i in range(num_parts):
                start_pos = actual_splits[i]
                end_pos = actual_splits[i + 1]
                part_text = text[start_pos:end_pos].strip()

                output_file = os.path.join(input_dir, f"{input_name}_{i+1:03d}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(part_text)

                self.last_split_files.append(output_file)
                self.split_log_text.append(TRANSLATIONS["log_part_saved"].format(output_file, len(part_text)))

            self.split_log_text.append(TRANSLATIONS["log_split_success"].format(num_parts))
            self.load_split_to_tts_button.setEnabled(True)
            QMessageBox.information(self, "Success", TRANSLATIONS["success_split"].format(num_parts))

        except Exception as e:
            self.split_log_text.append(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error splitting file: {str(e)}")

    def load_split_files_to_tts(self):
        """Load split files to the TTS tab."""
        if not self.last_split_files:
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_no_split_files"])
            self.split_log_text.append(TRANSLATIONS["error_no_split_files"])
            return

        self.tab_widget.setCurrentWidget(self.tts_tab)
        self.load_tts_split_files(self.last_split_files[0])

    def browse_tts_input_file(self):
        """Open a file dialog to select an input text file for TTS."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Text File", "", "Text Files (*.txt)")
        if file_name:
            self.tts_input_file_edit.setText(file_name)
            output_file = os.path.splitext(file_name)[0] + '.wav'
            self.tts_output_file_edit.setText(output_file)

    def browse_tts_split_files(self):
        """Open a file dialog to select the first split text file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select First Split Text File", "", "Text Files (*.txt)")
        if file_name:
            self.load_tts_split_files(file_name)

    def load_tts_split_files(self, first_file):
        """Load a series of split text files for TTS processing."""
        directory = os.path.dirname(first_file)
        filename = os.path.basename(first_file)
        base_name_match = re.match(r'(.+)_(\d{3})\.txt$', filename, re.IGNORECASE)

        if not base_name_match:
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_split_file_pattern"])
            self.split_log_text.append(TRANSLATIONS["error_split_file_pattern"])
            return

        base_name = base_name_match.group(1)
        start_number = int(base_name_match.group(2))

        split_files = []
        for i in range(start_number, 1000):
            file_path = os.path.join(directory, f"{base_name}_{i:03d}.txt")
            if os.path.exists(file_path):
                split_files.append(file_path)
            else:
                break

        if not split_files:
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_no_split_files_found"].format(base_name))
            self.split_log_text.append(TRANSLATIONS["error_no_split_files_found"].format(base_name))
            return

        self.split_log_text.append(f"Found split files: {split_files}")

        for input_file in split_files:
            output_file = os.path.splitext(input_file)[0] + '.wav'
            speed = self.tts_speed_spin.value()
            voice_weights = {
                voice: self.custom_mix_voice_spins[voice].value() if self.custom_mix_voice_checkboxes[voice].isChecked() else 0.0
                for voice in self.available_voices
            }
            voice_weights_1 = {
                voice: self.custom_mix_1_voice_spins[voice].value() if self.custom_mix_1_voice_checkboxes[voice].isChecked() else 0.0
                for voice in self.available_voices
            }
            voice_weights_2 = {
                voice: self.custom_mix_2_voice_spins[voice].value() if self.custom_mix_2_voice_checkboxes[voice].isChecked() else 0.0
                for voice in self.available_voices
            }
            voice_weights_3 = {
                voice: self.custom_mix_3_voice_spins[voice].value() if self.custom_mix_3_voice_checkboxes[voice].isChecked() else 0.0
                for voice in self.available_voices
            }
            voice_weights_4 = {
                voice: self.custom_mix_4_voice_spins[voice].value() if self.custom_mix_4_voice_checkboxes[voice].isChecked() else 0.0
                for voice in self.available_voices
            }
            voice_weights_5 = {
                voice: self.custom_mix_5_voice_spins[voice].value() if self.custom_mix_5_voice_checkboxes[voice].isChecked() else 0.0
                for voice in self.available_voices
            }

            if not os.path.exists(input_file):
                QMessageBox.warning(self, "Error", f"Input file not found: {input_file}")
                self.split_log_text.append(f"Error: Input file not found: {input_file}")
                continue
            if all(weight == 0.0 for weight in voice_weights.values()) and \
               all(weight == 0.0 for weight in voice_weights_1.values()) and \
               all(weight == 0.0 for weight in voice_weights_2.values()) and \
               all(weight == 0.0 for weight in voice_weights_3.values()) and \
               all(weight == 0.0 for weight in voice_weights_4.values()) and \
               all(weight == 0.0 for weight in voice_weights_5.values()):
                QMessageBox.warning(self, "Error", TRANSLATIONS["error_no_active_voices"])
                self.split_log_text.append(TRANSLATIONS["error_no_active_voices"])
                continue

            self.tts_process_counter += 1
            task = {
                "process_id": self.tts_process_counter,
                "input_file": input_file,
                "output_file": output_file,
                "speed": speed,
                "voice_weights": voice_weights,
                "voice_weights_1": voice_weights_1,
                "voice_weights_2": voice_weights_2,
                "voice_weights_3": voice_weights_3,
                "voice_weights_4": voice_weights_4,
                "voice_weights_5": voice_weights_5
            }
            self.tts_tasks.append(task)
            self.tts_task_queue.append(task)
            self.add_tts_task_to_table(task)
            self.split_log_text.append(TRANSLATIONS["log_task_added"].format(input_file, output_file))

        self.start_tts_queued_tasks()
        self.tts_input_file_edit.clear()
        self.tts_output_file_edit.clear()
        QMessageBox.information(self, "Success", TRANSLATIONS["success_tasks_added"].format(len(split_files)))

    def browse_tts_output_file(self):
        """Open a file dialog to select an output audio file."""
        file_name, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "Audio Files (*.wav)")
        if file_name:
            if not file_name.lower().endswith('.wav'):
                file_name += '.wav'
            self.tts_output_file_edit.setText(file_name)

    def add_tts_task(self):
        """Add a new TTS task to the queue."""
        input_file = self.tts_input_file_edit.text()
        output_file = self.tts_output_file_edit.text()
        speed = self.tts_speed_spin.value()
        voice_weights = {
            voice: self.custom_mix_voice_spins[voice].value() if self.custom_mix_voice_checkboxes[voice].isChecked() else 0.0
            for voice in self.available_voices
        }
        voice_weights_1 = {
            voice: self.custom_mix_1_voice_spins[voice].value() if self.custom_mix_1_voice_checkboxes[voice].isChecked() else 0.0
            for voice in self.available_voices
        }
        voice_weights_2 = {
            voice: self.custom_mix_2_voice_spins[voice].value() if self.custom_mix_2_voice_checkboxes[voice].isChecked() else 0.0
            for voice in self.available_voices
        }
        voice_weights_3 = {
            voice: self.custom_mix_3_voice_spins[voice].value() if self.custom_mix_3_voice_checkboxes[voice].isChecked() else 0.0
            for voice in self.available_voices
        }
        voice_weights_4 = {
            voice: self.custom_mix_4_voice_spins[voice].value() if self.custom_mix_4_voice_checkboxes[voice].isChecked() else 0.0
            for voice in self.available_voices
        }
        voice_weights_5 = {
            voice: self.custom_mix_5_voice_spins[voice].value() if self.custom_mix_5_voice_checkboxes[voice].isChecked() else 0.0
            for voice in self.available_voices
        }

        if not input_file or not os.path.exists(input_file):
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_invalid_input_file"])
            self.split_log_text.append(TRANSLATIONS["error_invalid_input_file"])
            return
        if not output_file:
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_invalid_output_file"])
            self.split_log_text.append(TRANSLATIONS["error_invalid_output_file"])
            return
        if not output_file.lower().endswith('.wav'):
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_output_not_wav"])
            self.split_log_text.append(TRANSLATIONS["error_output_not_wav"])
            return
        if all(weight == 0.0 for weight in voice_weights.values()) and \
           all(weight == 0.0 for weight in voice_weights_1.values()) and \
           all(weight == 0.0 for weight in voice_weights_2.values()) and \
           all(weight == 0.0 for weight in voice_weights_3.values()) and \
           all(weight == 0.0 for weight in voice_weights_4.values()) and \
           all(weight == 0.0 for weight in voice_weights_5.values()):
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_no_active_voices"])
            self.split_log_text.append(TRANSLATIONS["error_no_active_voices"])
            return

        # New batch -> restart the total working-time counter
        if not self.tts_threads and not self.tts_task_queue and self.tts_process_table.rowCount() == 0:
            self.tts_work_times.clear()
        self.tts_process_counter += 1
        task = {
            "process_id": self.tts_process_counter,
            "input_file": input_file,
            "output_file": output_file,
            "speed": speed,
            "voice_weights": voice_weights,
            "voice_weights_1": voice_weights_1,
            "voice_weights_2": voice_weights_2,
            "voice_weights_3": voice_weights_3,
            "voice_weights_4": voice_weights_4,
            "voice_weights_5": voice_weights_5
        }
        self.tts_tasks.append(task)
        self.tts_task_queue.append(task)
        self.add_tts_task_to_table(task)
        self.tts_input_file_edit.clear()
        self.tts_output_file_edit.clear()
        self.start_tts_queued_tasks()

    def add_tts_task_to_table(self, task):
        """Add a TTS task to the process table."""
        row = self.tts_process_table.rowCount()
        self.tts_process_table.insertRow(row)
        self.tts_process_table.setItem(row, 0, QTableWidgetItem(str(task["process_id"])))
        input_item = QTableWidgetItem(task["input_file"])
        input_item.setToolTip(task["input_file"])
        self.tts_process_table.setItem(row, 1, input_item)
        output_item = QTableWidgetItem(task["output_file"])
        output_item.setToolTip(task["output_file"])
        self.tts_process_table.setItem(row, 2, output_item)
        progress_bar = QProgressBar()
        progress_bar.setValue(0)
        self.tts_process_table.setCellWidget(row, 3, progress_bar)
        self.tts_process_table.setItem(row, 4, QTableWidgetItem("Waiting"))
        self.tts_process_table.setItem(row, 5, QTableWidgetItem("Time: --:--:-- (approx. --:--:-- remaining)"))

        action_widget = QWidget()
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_widget.setLayout(action_layout)

        cancel_button = QPushButton(TRANSLATIONS["tts_cancel_button"])
        cancel_button.clicked.connect(lambda: self.cancel_tts_task(task["process_id"]))
        cancel_button.setEnabled(True)
        action_layout.addWidget(cancel_button)

        restart_button = QPushButton(TRANSLATIONS["tts_restart_button"])
        restart_button.clicked.connect(lambda: self.restart_tts_task(task["process_id"]))
        restart_button.setEnabled(False)
        action_layout.addWidget(restart_button)

        self.tts_process_table.setCellWidget(row, 6, action_widget)

        delete_button = QPushButton(TRANSLATIONS["tts_delete_button"])
        delete_button.clicked.connect(lambda: self.delete_tts_task(task["process_id"]))
        delete_button.setEnabled(True)
        self.tts_process_table.setCellWidget(row, 7, delete_button)
        self._update_overall_progress()

    def start_tts_queued_tasks(self):
        """Start new threads from the queue if possible."""
        if self.tts_pending_cleanup:
            self.split_log_text.append(TRANSLATIONS["log_pending_cleanup"].format(self.tts_pending_cleanup))
            return

        for process_id, thread in list(self.tts_threads.items()):
            if not thread.isRunning():
                self.split_log_text.append(TRANSLATIONS["log_thread_removed"].format(process_id, thread.isRunning()))
                thread.cleanup()
                thread.wait()
                del self.tts_threads[process_id]
                if process_id in self.tts_pending_cleanup:
                    self.tts_pending_cleanup.remove(process_id)

        max_threads = self.tts_max_threads_spin.value()
        active_threads = len(self.tts_threads)
        self.split_log_text.append(TRANSLATIONS["log_thread_check"].format(active_threads, max_threads, len(self.tts_task_queue)))

        if max_threads == 0:
            self.split_log_text.append("No new threads started (max_threads=0).")
            return

        while self.tts_task_queue and active_threads < max_threads:
            task = self.tts_task_queue.pop(0)
            thread = TTSThread(
                task["process_id"],
                task["input_file"],
                task["output_file"],
                task["speed"],
                task["voice_weights"],
                task["voice_weights_1"],
                task["voice_weights_2"],
                task["voice_weights_3"],
                task["voice_weights_4"],
                task["voice_weights_5"]
            )
            thread.log_signal.connect(self.update_tts_log)
            thread.progress_signal.connect(self.update_tts_progress)
            thread.time_signal.connect(self.update_tts_time)
            thread.work_time_signal.connect(self.update_tts_work_time)
            thread.status_signal.connect(self.update_tts_task_status)
            thread.finished_signal.connect(self.on_tts_finished)
            thread.error_signal.connect(self.on_tts_error)
            self.tts_threads[task["process_id"]] = thread
            self.tts_task_start_times[task["process_id"]] = time.time()
            self.tts_progress_values[task["process_id"]] = 0
            self.update_tts_task_status(task["process_id"], "Running")
            thread.start()
            if self.tts_global_paused:
                thread.pause()
            active_threads += 1
            self.split_log_text.append(TRANSLATIONS["log_thread_started"].format(task["process_id"], active_threads))
        self.update_pause_all_button_state()
        self._update_overall_progress()

    def update_tts_log(self, message):
        """Update the log with a new message."""
        self.split_log_text.append(message)
        self.split_log_text.verticalScrollBar().setValue(self.split_log_text.verticalScrollBar().maximum())

    def update_tts_progress(self, process_id, progress):
        """Update the progress bar for a task."""
        self.tts_progress_values[process_id] = progress
        for row in range(self.tts_process_table.rowCount()):
            if self.tts_process_table.item(row, 0) and self.tts_process_table.item(row, 0).text() == str(process_id):
                progress_bar = self.tts_process_table.cellWidget(row, 3)
                progress_bar.setValue(progress)
                break
        self._update_overall_progress()

    def update_tts_time(self, process_id, time_info):
        """Update the time information for a task."""
        for row in range(self.tts_process_table.rowCount()):
            if self.tts_process_table.item(row, 0) and self.tts_process_table.item(row, 0).text() == str(process_id):
                self.tts_process_table.setItem(row, 5, QTableWidgetItem(time_info))
                break

    def update_tts_work_time(self, process_id, work_seconds):
        """Track the accumulated working time of a task (pauses excluded)."""
        self.tts_work_times[process_id] = work_seconds
        self._update_overall_progress()

    def update_tts_task_status(self, process_id, status):
        """Update the status of a task in the process table."""
        for row in range(self.tts_process_table.rowCount()):
            if self.tts_process_table.item(row, 0) and self.tts_process_table.item(row, 0).text() == str(process_id):
                self.tts_process_table.setItem(row, 4, QTableWidgetItem(status))
                action_widget = self.tts_process_table.cellWidget(row, 6)
                cancel_button = action_widget.layout().itemAt(0).widget()
                restart_button = action_widget.layout().itemAt(1).widget()
                delete_button = self.tts_process_table.cellWidget(row, 7)

                if status in ["Running", "Paused"]:
                    cancel_button.setEnabled(True)
                    restart_button.setEnabled(False)
                    delete_button.setEnabled(True)
                elif status == "Canceling...":
                    cancel_button.setEnabled(True)
                    restart_button.setEnabled(False)
                    delete_button.setEnabled(False)
                else:
                    cancel_button.setEnabled(False)
                    restart_button.setEnabled(True)
                    delete_button.setEnabled(True)
                    current_time = self.tts_process_table.item(row, 5).text()
                    if "remaining" in current_time:
                        elapsed = current_time.split(" (")[0]
                        self.tts_process_table.setItem(row, 5, QTableWidgetItem(elapsed))
                break
        self._update_overall_progress()

    def cancel_tts_task(self, process_id):
        """Cancel a TTS task (running or still queued)."""
        if process_id in self.tts_threads:
            thread = self.tts_threads[process_id]
            thread.stop()
            self.update_tts_task_status(process_id, "Canceling...")
            self.tts_pending_cleanup.add(process_id)
        elif any(t["process_id"] == process_id for t in self.tts_task_queue):
            self.tts_task_queue = [t for t in self.tts_task_queue if t["process_id"] != process_id]
            self.tts_tasks = [t for t in self.tts_tasks if t["process_id"] != process_id]
            self.remove_tts_task_row(process_id)
            self.tts_progress_values.pop(process_id, None)
            self.tts_work_times.pop(process_id, None)
            self._update_overall_progress()

    def toggle_pause_all_tasks(self):
        """Pause or resume ALL running TTS tasks with a single button."""
        if self.tts_global_paused:
            self.resume_all_tasks()
        else:
            self.pause_all_tasks()

    def pause_all_tasks(self):
        """Pause every currently running TTS thread."""
        running = [t for t in self.tts_threads.values() if t.isRunning()]
        if not running:
            return
        self.tts_global_paused = True
        for thread in running:
            thread.pause()
        self.tts_pause_all_button.setText(TRANSLATIONS["tts_resume_button"])

    def resume_all_tasks(self):
        """Resume every paused TTS thread."""
        running = [t for t in self.tts_threads.values() if t.isRunning()]
        self.tts_global_paused = False
        for thread in running:
            thread.pause()  # toggles the per-thread pause state back
        self.tts_pause_all_button.setText(TRANSLATIONS["tts_pause_button"])
        self.update_pause_all_button_state()

    def update_pause_all_button_state(self):
        """Sync the global pause/resume button with the actual thread state."""
        has_running = any(t.isRunning() for t in self.tts_threads.values())
        if self.tts_global_paused and not has_running:
            self.tts_global_paused = False
            self.tts_pause_all_button.setText(TRANSLATIONS["tts_pause_button"])
        self.tts_pause_all_button.setEnabled(bool(self.tts_threads))

    def restart_tts_task(self, process_id):
        """Restart a TTS task."""
        for task in self.tts_tasks:
            if task["process_id"] == process_id:
                self.tts_process_counter += 1
                new_task = {
                    "process_id": self.tts_process_counter,
                    "input_file": task["input_file"],
                    "output_file": task["output_file"],
                    "speed": task["speed"],
                    "voice_weights": task["voice_weights"],
                    "voice_weights_1": task["voice_weights_1"],
                    "voice_weights_2": task["voice_weights_2"],
                    "voice_weights_3": task["voice_weights_3"],
                    "voice_weights_4": task["voice_weights_4"],
                    "voice_weights_5": task["voice_weights_5"]
                }
                self.tts_tasks.append(new_task)
                self.tts_task_queue.append(new_task)
                self.add_tts_task_to_table(new_task)
                self.start_tts_queued_tasks()
                return
        QMessageBox.warning(self, "Error", f"Could not restart process {process_id}: Task not found.")

    def delete_tts_task(self, process_id):
        """Delete a TTS task from the table."""
        if process_id in self.tts_threads:
            thread = self.tts_threads[process_id]
            thread.stop()
            thread.wait()
            thread.cleanup()
            del self.tts_threads[process_id]
            if process_id in self.tts_pending_cleanup:
                self.tts_pending_cleanup.remove(process_id)
        self.tts_tasks = [task for task in self.tts_tasks if task["process_id"] != process_id]
        self.tts_task_queue = [task for task in self.tts_task_queue if task["process_id"] != process_id]
        self.tts_task_start_times.pop(process_id, None)
        self.tts_progress_values.pop(process_id, None)
        self.tts_work_times.pop(process_id, None)
        self.remove_tts_task_row(process_id)
        self.start_tts_queued_tasks()
        self._update_overall_progress()

    def remove_tts_task_row(self, process_id):
        """Remove the table row that belongs to the given process id.

        The scroll position is restored afterwards so the viewport stays
        stable (deleting e.g. the very last task no longer jumps to the top).
        """
        table = self.tts_process_table
        vbar = table.verticalScrollBar()
        saved_value = vbar.value()
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and item.text() == str(process_id):
                table.removeRow(row)
                break
        # Restore the scroll value; if it exceeds the new range (last row
        # deleted) it clamps to the bottom, keeping the view in place.
        vbar.setValue(saved_value)

    def delete_all_tts_tasks(self):
        """Stop and delete ALL TTS tasks (after confirmation)."""
        if not self.tts_tasks and not self.tts_threads:
            QMessageBox.information(self, "Info", "No tasks to delete.")
            return
        reply = QMessageBox.question(
            self, TRANSLATIONS["tts_clear_all_title"],
            TRANSLATIONS["tts_clear_all_question"],
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        for thread in self.tts_threads.values():
            thread.stop()
        for thread in self.tts_threads.values():
            thread.wait(15000)
        for thread in self.tts_threads.values():
            thread.cleanup()
        self.tts_threads.clear()
        self.tts_pending_cleanup.clear()
        self.tts_tasks = []
        self.tts_task_queue = []
        self.tts_process_table.setRowCount(0)
        self.split_log_text.append("All tasks deleted.")
        self.tts_global_paused = False
        self.tts_pause_all_button.setText(TRANSLATIONS["tts_pause_button"])
        self.update_pause_all_button_state()
        self.tts_task_start_times.clear()
        self.tts_progress_values.clear()
        self.tts_finished_durations.clear()
        self.tts_work_times.clear()
        self._update_overall_progress()
        gc.collect()

    def on_tts_finished(self, process_id, was_canceled):
        """Handle the completion of a TTS thread."""
        if process_id in self.tts_threads:
            thread = self.tts_threads[process_id]
            thread.wait()
            self.split_log_text.append(TRANSLATIONS["log_thread_finished"].format(process_id, was_canceled))
            status = "Canceled" if was_canceled else "Completed"
            self.update_tts_task_status(process_id, status)
            start = self.tts_task_start_times.pop(process_id, None)
            if start and not was_canceled:
                self.tts_finished_durations.append(time.time() - start)
                if len(self.tts_finished_durations) > 30:
                    self.tts_finished_durations.pop(0)
            self.tts_progress_values.pop(process_id, None)
            thread.cleanup()
            del self.tts_threads[process_id]
            if process_id in self.tts_pending_cleanup:
                self.tts_pending_cleanup.remove(process_id)
            self.start_tts_queued_tasks()
            self._update_overall_progress()
            if not self.tts_threads and not self.tts_task_queue:
                self.split_log_text.append(TRANSLATIONS["log_all_done"])
                if not was_canceled:
                    self._play_all_done_sound()

    def on_tts_error(self, process_id, error_message):
        """Handle errors in a TTS thread."""
        self.split_log_text.append(TRANSLATIONS["log_error"].format(process_id, error_message))
        self.update_tts_task_status(process_id, "Error")
        if process_id in self.tts_threads:
            thread = self.tts_threads[process_id]
            thread.wait()
            thread.cleanup()
            del self.tts_threads[process_id]
            if process_id in self.tts_pending_cleanup:
                self.tts_pending_cleanup.remove(process_id)
        self.tts_task_start_times.pop(process_id, None)
        self.tts_progress_values.pop(process_id, None)
        self.start_tts_queued_tasks()
        self._update_overall_progress()
        if not self.tts_threads and not self.tts_task_queue:
            self.split_log_text.append(TRANSLATIONS["log_all_done"])
            self._play_all_done_sound()

    def _update_overall_progress(self):
        """Update the overall batch progress bar and the ETA label."""
        table = self.tts_process_table
        total = table.rowCount()
        done = 0
        for row in range(total):
            item = table.item(row, 4)
            if item is not None:
                status = item.text()
                if status == "Completed" or status.startswith("Error") or status == "Canceled":
                    done += 1
        self.tts_overall_progress.setValue(int(done * 100 / total) if total > 0 else 0)
        self.tts_overall_status_label.setText(
            TRANSLATIONS["tts_overall_done"].format(done=done, total=total))

        remaining = total - done
        running = [pid for pid, t in self.tts_threads.items() if t.isRunning()]
        eta = None
        if remaining > 0 and running:
            avg_seconds = None
            if self.tts_finished_durations:
                avg_seconds = sum(self.tts_finished_durations) / len(self.tts_finished_durations)
            else:
                estimates = []
                now = time.time()
                for pid in running:
                    start = self.tts_task_start_times.get(pid)
                    progress = self.tts_progress_values.get(pid, 0)
                    if start and progress >= 10:
                        estimates.append((now - start) / (progress / 100.0))
                if estimates:
                    avg_seconds = sum(estimates) / len(estimates)
            if avg_seconds:
                eta = remaining * avg_seconds / max(1, len(running))
        if remaining <= 0 and total > 0:
            self.tts_overall_eta_label.setText(TRANSLATIONS["tts_overall_all_done"])
        elif eta is not None:
            self.tts_overall_eta_label.setText(
                TRANSLATIONS["tts_overall_eta"].format(eta=self._format_eta(eta)))
        else:
            self.tts_overall_eta_label.setText(TRANSLATIONS["tts_overall_eta_unknown"])
        total_work = sum(self.tts_work_times.values())
        self.tts_overall_elapsed_label.setText(
            TRANSLATIONS["tts_overall_elapsed"].format(time=self._format_eta(total_work)))

    @staticmethod
    def _format_eta(seconds):
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _play_all_done_sound(self):
        """Play a short chime when the whole batch is finished."""
        try:
            from PyQt5.QtMultimedia import QSound
            wav = self._ensure_done_chime_wav()
            if wav:
                QSound.play(wav)
                return
        except Exception:
            pass
        QApplication.beep()

    def _ensure_done_chime_wav(self):
        """Create (once) a short two-tone chime WAV and return its path."""
        if getattr(self, "_done_chime_wav", None) and os.path.exists(self._done_chime_wav):
            return self._done_chime_wav
        path = os.path.join(tempfile.gettempdir(), "kokoro_done_chime.wav")
        try:
            rate = 44100

            def tone(freq, dur, volume=0.35):
                n = int(rate * dur)
                return [volume * math.sin(2 * math.pi * freq * i / rate) for i in range(n)]

            samples = tone(880.0, 0.18) + tone(1174.66, 0.22)  # A5 -> D6
            fade = int(rate * 0.02)  # fade-out to avoid a click
            for i in range(fade):
                samples[-1 - i] *= i / fade
            with wave.open(path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                data = b"".join(
                    struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
                w.writeframes(data)
            self._done_chime_wav = path
            return path
        except Exception:
            return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
