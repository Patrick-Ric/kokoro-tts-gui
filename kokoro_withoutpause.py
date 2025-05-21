import sys
import os
import json
import re
import time
import gc
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget,
                             QTableWidgetItem, QProgressBar, QDoubleSpinBox, QScrollArea,
                             QCheckBox, QMessageBox, QGridLayout, QSplitter, QTabWidget,
                             QTextEdit, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import soundfile as sf
import numpy as np
import torch
import psutil
from kokoro_onnx import Kokoro

# Translations dictionary for English only
TRANSLATIONS = {
    "window_title": "Kokoro TTS & Split GUI",
    "tab_split": "Text Splitting",
    "tab_tts": "TTS Processing",
    "tab_custom_mix": "Voice Custom Mix",
    "tab_custom_mix_1": "Voice Custom Mix 1",
    "tab_custom_mix_2": "Voice Custom Mix 2",
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
    "tts_pause_duration_label": "Pause Duration After Sentences (seconds):",
    "tts_speed_label": "Speed:",
    "tts_max_threads_label": "Maximum Threads:",
    "tts_config_label": "Configuration:",
    "tts_save_config_button": "Save Configuration",
    "tts_load_config_button": "Load Configuration",
    "tts_add_task_button": "Add Task",
    "tts_processes_label": "Processes:",
    "tts_table_headers": ["Process ID", "Input File", "Output File", "Progress", "Status", "Time", "Action", "Delete"],
    "tts_cancel_button": "Cancel",
    "tts_restart_button": "Restart",
    "tts_pause_button": "Pause",
    "tts_resume_button": "Resume",
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
}

class TTSThread(QThread):
    """Thread for processing TTS tasks using the Kokoro ONNX model."""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # process_id, progress
    time_signal = pyqtSignal(int, str)  # process_id, time_info
    status_signal = pyqtSignal(int, str)  # process_id, status
    finished_signal = pyqtSignal(int, bool)  # process_id, was_canceled
    error_signal = pyqtSignal(int, str)

    def __init__(self, process_id, input_file, output_file, sentence_pause_duration, speed, voice_weights, voice_weights_1, voice_weights_2):
        super().__init__()
        self.process_id = process_id
        self.input_file = input_file
        self.output_file = output_file
        self.sentence_pause_duration = sentence_pause_duration
        self.speed = speed
        self.voice_weights = voice_weights
        self.voice_weights_1 = voice_weights_1
        self.voice_weights_2 = voice_weights_2
        self._stop = False
        self._was_canceled = False
        self._paused = False
        self.start_time = None
        self.pause_start_time = None
        self.total_pause_duration = 0
        self.kokoro = None
        self.audio_file = None
        self.last_time_update = 0

    def cleanup(self):
        """Release all resources used by the thread."""
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
        """Stop the thread and clean up resources."""
        self._stop = True
        self._was_canceled = True
        self.cleanup()

    def pause(self):
        """Pause or resume the thread."""
        if not self._paused:
            self._paused = True
            self.pause_start_time = time.time()
            self.status_signal.emit(self.process_id, TRANSLATIONS["tts_pause_button"])
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
            self.last_time_update = current_time

    def run(self):
        """Execute the TTS processing task."""
        try:
            self.start_time = time.time()
            self.last_time_update = self.start_time
            self.log_signal.emit(TRANSLATIONS["log_process_init"].format(self.process_id))
            self.kokoro = Kokoro("kokoro.onnx", "voices-v1.0.bin")
            self.status_signal.emit(self.process_id, "Running")
            self.progress_signal.emit(self.process_id, 50)  # Indicate processing has started
            self.update_time(self.start_time, "processing")  # Initial time update

            # Compute VOICEPACK for custom_mix
            weight_sum = sum(weight for weight in self.voice_weights.values() if weight > 0)
            if weight_sum == 0:
                weight_sum = 1.0
            VOICEPACK = sum(
                self.kokoro.voices[voice] * (weight / weight_sum)
                for voice, weight in self.voice_weights.items() if weight > 0
            )

            # Compute VOICEPACK_1 for custom_mix_1
            weight_sum_1 = sum(weight for weight in self.voice_weights_1.values() if weight > 0)
            if weight_sum_1 == 0:
                weight_sum_1 = 1.0
            VOICEPACK_1 = sum(
                self.kokoro.voices[voice] * (weight / weight_sum_1)
                for voice, weight in self.voice_weights_1.items() if weight > 0
            )

            # Compute VOICEPACK_2 for custom_mix_2
            weight_sum_2 = sum(weight for weight in self.voice_weights_2.values() if weight > 0)
            if weight_sum_2 == 0:
                weight_sum_2 = 1.0
            VOICEPACK_2 = sum(
                self.kokoro.voices[voice] * (weight / weight_sum_2)
                for voice, weight in self.voice_weights_2.items() if weight > 0
            )

            def parse_text_file(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                entries = []
                buffer = []
                current_voice = None
                for line in lines:
                    line = line.strip()
                    if line.startswith("[voice=") and line.endswith("]"):
                        if buffer:
                            entries.append(("voice", current_voice or "default", " ".join(buffer)))
                            buffer = []
                        current_voice = line[len("[voice="):-1]
                    elif line.startswith("[pause=") and line.endswith("]"):
                        if buffer:
                            entries.append(("voice", current_voice or "default", " ".join(buffer)))
                            buffer = []
                        pause_duration = float(line[len("[pause="):-1])
                        entries.append(("pause", pause_duration))
                    elif line:
                        buffer.append(line)
                if buffer:
                    entries.append(("voice", current_voice or "default", " ".join(buffer)))
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

            for i, (entry_type, *args) in enumerate(entries):
                if self._stop:
                    self.log_signal.emit(TRANSLATIONS["log_process_canceled"].format(self.process_id))
                    return

                while self._paused and not self._stop:
                    self.msleep(100)
                    current_time = time.time()
                    self.update_time(current_time, "paused")

                current_time = time.time()
                self.update_time(current_time, "processing")

                if entry_type == "pause":
                    custom_pause_duration = args[0]
                    self.log_signal.emit(TRANSLATIONS["log_custom_pause"].format(self.process_id, i+1, custom_pause_duration))
                    silence = np.zeros(int(custom_pause_duration * sample_rate), dtype=np.float32)
                    self.audio_file.write(silence)
                    del silence
                    gc.collect()
                else:
                    voice, text = args
                    if voice == "custom_mix":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK
                    elif voice == "custom_mix_1":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix_1"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK_1
                    elif voice == "custom_mix_2":
                        self.log_signal.emit(TRANSLATIONS["log_process_custom_mix_2"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK_2
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

                    self.log_signal.emit(TRANSLATIONS["log_generate_text"].format(self.process_id, text[:40]))
                    with torch.no_grad():
                        samples, sr = self.kokoro.create(text, voice=actual_voice, speed=self.speed, lang="en-us")
                    self.log_signal.emit(TRANSLATIONS["log_sample_rate"].format(self.process_id, sr, len(samples)))
                    if sr != sample_rate:
                        self.log_signal.emit(TRANSLATIONS["log_sample_rate_warning"].format(self.process_id, sr, sample_rate))
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

        except Exception as e:
            error_msg = TRANSLATIONS["log_error"].format(self.process_id, str(e))
            self.log_signal.emit(error_msg)
            self.error_signal.emit(self.process_id, str(e))
            self.status_signal.emit(self.process_id, f"Error: {str(e)}")
            self.progress_signal.emit(self.process_id, 0)
            self.time_signal.emit(self.process_id, "Time: --:--:-- (error)")
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

        self.init_split_tab()
        self.init_tts_tab()
        self.init_custom_mix_tab()
        self.init_custom_mix_1_tab()
        self.init_custom_mix_2_tab()
        self.tab_widget.setCurrentWidget(self.tts_tab)

        self.tts_tasks = []
        self.tts_threads = {}
        self.tts_process_counter = 0
        self.tts_task_queue = []
        self.tts_pending_cleanup = set()

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
        tts_splitter = QSplitter(Qt.Vertical)
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
        self.tts_pause_duration_spin = QDoubleSpinBox()
        self.tts_pause_duration_spin.setRange(0.0, 10.0)
        self.tts_pause_duration_spin.setValue(1.0)
        self.tts_pause_duration_spin.setSingleStep(0.1)
        tts_params_grid.addWidget(QLabel(TRANSLATIONS["tts_pause_duration_label"]), 0, 0)
        tts_params_grid.addWidget(self.tts_pause_duration_spin, 0, 1)

        self.tts_speed_spin = QDoubleSpinBox()
        self.tts_speed_spin.setRange(0.1, 2.0)
        self.tts_speed_spin.setValue(0.9)
        self.tts_speed_spin.setSingleStep(0.1)
        tts_params_grid.addWidget(QLabel(TRANSLATIONS["tts_speed_label"]), 0, 2)
        tts_params_grid.addWidget(self.tts_speed_spin, 0, 3)

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
        self.tts_add_task_button = QPushButton(TRANSLATIONS["tts_add_task_button"])
        self.tts_add_task_button.clicked.connect(self.add_tts_task)
        tts_lower_layout.addWidget(self.tts_add_task_button)
        tts_lower_layout.addWidget(QLabel(TRANSLATIONS["tts_processes_label"]))

        self.tts_process_table = QTableWidget()
        self.tts_process_table.setColumnCount(8)
        self.tts_process_table.setHorizontalHeaderLabels(TRANSLATIONS["tts_table_headers"])
        self.tts_process_table.setSelectionMode(QTableWidget.NoSelection)
        self.tts_process_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tts_process_table.horizontalHeader().setStretchLastSection(True)
        tts_lower_layout.addWidget(self.tts_process_table)

        tts_splitter.addWidget(tts_upper_widget)
        tts_splitter.addWidget(tts_lower_widget)
        tts_splitter.setSizes([400, 400])

    def init_custom_mix_tab(self):
        """Initialize the Voice Custom Mix tab."""
        if not os.path.exists("kokoro.onnx") or not os.path.exists("voices-v1.0.bin"):
            raise FileNotFoundError("Kokoro model files are missing.")
        kokoro_temp = Kokoro("kokoro.onnx", "voices-v1.0.bin")
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
                "pause_duration": self.tts_pause_duration_spin.value(),
                "voice_weights": {voice: self.custom_mix_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled": {voice: self.custom_mix_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
                "voice_weights_1": {voice: self.custom_mix_1_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled_1": {voice: self.custom_mix_1_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
                "voice_weights_2": {voice: self.custom_mix_2_voice_spins[voice].value() for voice in self.available_voices},
                "voice_enabled_2": {voice: self.custom_mix_2_voice_checkboxes[voice].isChecked() for voice in self.available_voices}
            }
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
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
                self.tts_pause_duration_spin.setValue(config.get("pause_duration", 1.0))
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
            else:
                return
            config = {
                "voice_weights": {voice: spins[voice].value() for voice in self.available_voices},
                "voice_enabled": {voice: checkboxes[voice].isChecked() for voice in self.available_voices}
            }
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
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
            "pause_duration": self.tts_pause_duration_spin.value(),
            "voice_weights": {voice: self.custom_mix_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled": {voice: self.custom_mix_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
            "voice_weights_1": {voice: self.custom_mix_1_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled_1": {voice: self.custom_mix_1_voice_checkboxes[voice].isChecked() for voice in self.available_voices},
            "voice_weights_2": {voice: self.custom_mix_2_voice_spins[voice].value() for voice in self.available_voices},
            "voice_enabled_2": {voice: self.custom_mix_2_voice_checkboxes[voice].isChecked() for voice in self.available_voices}
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
                self.tts_pause_duration_spin.setValue(config.get("pause_duration", 1.0))
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
                self.split_log_text.append(TRANSLATIONS["log_last_config_loaded"])
            except Exception as e:
                self.split_log_text.append(TRANSLATIONS["log_config_load_warning"].format(str(e)))

    def closeEvent(self, event):
        """Handle the window close event."""
        self.save_last_configuration()
        for thread in self.tts_threads.values():
            thread.stop()
            thread.wait()
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
            sentence_pause_duration = self.tts_pause_duration_spin.value()
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

            if not os.path.exists(input_file):
                QMessageBox.warning(self, "Error", f"Input file not found: {input_file}")
                self.split_log_text.append(f"Error: Input file not found: {input_file}")
                continue
            if all(weight == 0.0 for weight in voice_weights.values()) and \
               all(weight == 0.0 for weight in voice_weights_1.values()) and \
               all(weight == 0.0 for weight in voice_weights_2.values()):
                QMessageBox.warning(self, "Error", TRANSLATIONS["error_no_active_voices"])
                self.split_log_text.append(TRANSLATIONS["error_no_active_voices"])
                continue

            self.tts_process_counter += 1
            task = {
                "process_id": self.tts_process_counter,
                "input_file": input_file,
                "output_file": output_file,
                "sentence_pause_duration": sentence_pause_duration,
                "speed": speed,
                "voice_weights": voice_weights,
                "voice_weights_1": voice_weights_1,
                "voice_weights_2": voice_weights_2
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
        sentence_pause_duration = self.tts_pause_duration_spin.value()
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
           all(weight == 0.0 for weight in voice_weights_2.values()):
            QMessageBox.warning(self, "Error", TRANSLATIONS["error_no_active_voices"])
            self.split_log_text.append(TRANSLATIONS["error_no_active_voices"])
            return

        self.tts_process_counter += 1
        task = {
            "process_id": self.tts_process_counter,
            "input_file": input_file,
            "output_file": output_file,
            "sentence_pause_duration": sentence_pause_duration,
            "speed": speed,
            "voice_weights": voice_weights,
            "voice_weights_1": voice_weights_1,
            "voice_weights_2": voice_weights_2
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
        self.tts_process_table.setItem(row, 1, QTableWidgetItem(task["input_file"]))
        self.tts_process_table.setItem(row, 2, QTableWidgetItem(task["output_file"]))
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

        pause_button = QPushButton(TRANSLATIONS["tts_pause_button"])
        pause_button.clicked.connect(lambda: self.pause_tts_task(task["process_id"]))
        pause_button.setEnabled(True)
        action_layout.addWidget(pause_button)

        self.tts_process_table.setCellWidget(row, 6, action_widget)

        delete_button = QPushButton(TRANSLATIONS["tts_delete_button"])
        delete_button.clicked.connect(lambda: self.delete_tts_task(task["process_id"], row))
        delete_button.setEnabled(True)
        self.tts_process_table.setCellWidget(row, 7, delete_button)

        self.tts_process_table.resizeColumnsToContents()

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
                task["sentence_pause_duration"],
                task["speed"],
                task["voice_weights"],
                task["voice_weights_1"],
                task["voice_weights_2"]
            )
            thread.log_signal.connect(self.update_tts_log)
            thread.progress_signal.connect(self.update_tts_progress)
            thread.time_signal.connect(self.update_tts_time)
            thread.status_signal.connect(self.update_tts_task_status)
            thread.finished_signal.connect(self.on_tts_finished)
            thread.error_signal.connect(self.on_tts_error)
            self.tts_threads[task["process_id"]] = thread
            self.update_tts_task_status(task["process_id"], "Running")
            thread.start()
            active_threads += 1
            self.split_log_text.append(TRANSLATIONS["log_thread_started"].format(task["process_id"], active_threads))

    def update_tts_log(self, message):
        """Update the log with a new message."""
        self.split_log_text.append(message)
        self.split_log_text.verticalScrollBar().setValue(self.split_log_text.verticalScrollBar().maximum())

    def update_tts_progress(self, process_id, progress):
        """Update the progress bar for a task."""
        for row in range(self.tts_process_table.rowCount()):
            if self.tts_process_table.item(row, 0) and self.tts_process_table.item(row, 0).text() == str(process_id):
                progress_bar = self.tts_process_table.cellWidget(row, 3)
                progress_bar.setValue(progress)
                break

    def update_tts_time(self, process_id, time_info):
        """Update the time information for a task."""
        for row in range(self.tts_process_table.rowCount()):
            if self.tts_process_table.item(row, 0) and self.tts_process_table.item(row, 0).text() == str(process_id):
                self.tts_process_table.setItem(row, 5, QTableWidgetItem(time_info))
                break

    def update_tts_task_status(self, process_id, status):
        """Update the status of a task in the process table."""
        for row in range(self.tts_process_table.rowCount()):
            if self.tts_process_table.item(row, 0) and self.tts_process_table.item(row, 0).text() == str(process_id):
                self.tts_process_table.setItem(row, 4, QTableWidgetItem(status))
                action_widget = self.tts_process_table.cellWidget(row, 6)
                cancel_button = action_widget.layout().itemAt(0).widget()
                restart_button = action_widget.layout().itemAt(1).widget()
                pause_button = action_widget.layout().itemAt(2).widget()
                delete_button = self.tts_process_table.cellWidget(row, 7)

                if status in ["Running", TRANSLATIONS["tts_pause_button"]]:
                    cancel_button.setEnabled(True)
                    restart_button.setEnabled(False)
                    pause_button.setEnabled(True)
                    pause_button.setText(TRANSLATIONS["tts_resume_button"] if status == TRANSLATIONS["tts_pause_button"] else TRANSLATIONS["tts_pause_button"])
                    delete_button.setEnabled(True)
                elif status == "Canceling...":
                    cancel_button.setEnabled(True)
                    restart_button.setEnabled(False)
                    pause_button.setEnabled(False)
                    delete_button.setEnabled(False)
                else:
                    cancel_button.setEnabled(False)
                    restart_button.setEnabled(True)
                    pause_button.setEnabled(False)
                    pause_button.setText(TRANSLATIONS["tts_pause_button"])
                    delete_button.setEnabled(True)
                    current_time = self.tts_process_table.item(row, 5).text()
                    if "remaining" in current_time:
                        elapsed = current_time.split(" (")[0]
                        self.tts_process_table.setItem(row, 5, QTableWidgetItem(elapsed))
                break

    def cancel_tts_task(self, process_id):
        """Cancel a TTS task."""
        if process_id in self.tts_threads:
            thread = self.tts_threads[process_id]
            thread.stop()
            self.update_tts_task_status(process_id, "Canceling...")
            self.tts_pending_cleanup.add(process_id)

    def pause_tts_task(self, process_id):
        """Pause or resume a TTS task."""
        if process_id in self.tts_threads:
            thread = self.tts_threads[process_id]
            thread.pause()

    def restart_tts_task(self, process_id):
        """Restart a TTS task."""
        for task in self.tts_tasks:
            if task["process_id"] == process_id:
                self.tts_process_counter += 1
                new_task = {
                    "process_id": self.tts_process_counter,
                    "input_file": task["input_file"],
                    "output_file": task["output_file"],
                    "sentence_pause_duration": task["sentence_pause_duration"],
                    "speed": task["speed"],
                    "voice_weights": task["voice_weights"],
                    "voice_weights_1": task["voice_weights_1"],
                    "voice_weights_2": task["voice_weights_2"]
                }
                self.tts_tasks.append(new_task)
                self.tts_task_queue.append(new_task)
                self.add_tts_task_to_table(new_task)
                self.start_tts_queued_tasks()
                return
        QMessageBox.warning(self, "Error", f"Could not restart process {process_id}: Task not found.")

    def delete_tts_task(self, process_id, row):
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
        self.tts_process_table.removeRow(row)
        self.start_tts_queued_tasks()

    def on_tts_finished(self, process_id, was_canceled):
        """Handle the completion of a TTS thread."""
        if process_id in self.tts_threads:
            thread = self.tts_threads[process_id]
            thread.wait()
            self.split_log_text.append(TRANSLATIONS["log_thread_finished"].format(process_id, was_canceled))
            status = "Canceled" if was_canceled else "Completed"
            self.update_tts_task_status(process_id, status)
            thread.cleanup()
            del self.tts_threads[process_id]
            if process_id in self.tts_pending_cleanup:
                self.tts_pending_cleanup.remove(process_id)
            self.start_tts_queued_tasks()

    def on_tts_error(self, process_id, error_message):
        """Handle errors in a TTS thread."""
        self.split_log_text.append(TRANSLATIONS["log_error"].format(process_id, error_message))
        self.update_tts_task_status(process_id, "Error")
        if process_id in self.tts_threads:
            thread = self.tts_threads[process_id]
            thread.cleanup()
            thread.wait()
            del self.tts_threads[process_id]
            if process_id in self.tts_pending_cleanup:
                self.tts_pending_cleanup.remove(process_id)
        self.start_tts_queued_tasks()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())