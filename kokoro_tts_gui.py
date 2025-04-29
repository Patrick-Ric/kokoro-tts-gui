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
                             QTextEdit, QSpinBox, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import soundfile as sf
import numpy as np
import torch
import psutil
from kokoro_onnx import Kokoro

# Translations dictionary for multilingual support
TRANSLATIONS = {
    "en": {
        "window_title": "Kokoro TTS & Split GUI",
        "tab_split": "Text Splitting",
        "tab_tts": "TTS Processing",
        "tab_help": "Help",
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
        "tts_voice_selection_label": "Voice Selection and Weights:",
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
        "language_label": "Language:",
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
        "log_total_sentences": "[Process {}] Total number of sentences: {}",
        "log_custom_pause": "[Process {}][{}] Adding custom pause of {} seconds.",
        "log_process_voice": "[Process {}][{}] Processing with voice '{}'",
        "log_process_custom_mix": "[Process {}][{}] Processing with custom voice mix (VOICEPACK)",
        "log_voice_not_found": "[Process {}] ⚠️ Voice '{}' not found, using VOICEPACK.",
        "log_sentences_found": "[Process {}][{}] Found sentences: {}",
        "log_generate_sentence": "[Process {}] → Generating sentence: \"{}...\"",
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
    },
    "de": {
        "window_title": "Kokoro TTS & Split GUI",
        "tab_split": "Textaufteilung",
        "tab_tts": "TTS-Verarbeitung",
        "tab_help": "Hilfe",
        "split_input_file_label": "Eingabe-Textdatei:",
        "split_input_file_placeholder": "Wählen Sie eine Textdatei aus...",
        "split_browse_button": "Durchsuchen...",
        "split_parts_label": "Anzahl der Teile:",
        "split_word_label": "Teilung vor Wort/Tag:",
        "split_word_placeholder": "z.B. [voice=custom_mix] oder Chapter",
        "split_button": "Textdatei teilen",
        "load_split_to_tts_button": "Gesplittete Dateien in TTS laden",
        "split_log_label": "Status:",
        "split_log_placeholder": "Statusmeldungen erscheinen hier...",
        "tts_input_file_label": "Eingabe-Textdatei:",
        "tts_input_file_placeholder": "Wählen Sie eine Textdatei aus...",
        "tts_browse_button": "Durchsuchen...",
        "tts_split_files_button": "Gesplittete Dateien laden...",
        "tts_output_file_label": "Ausgabe-Audiodatei:",
        "tts_output_file_placeholder": "Ausgabedatei (z.B. output.wav)",
        "tts_pause_duration_label": "Pausendauer nach Sätzen (Sekunden):",
        "tts_speed_label": "Geschwindigkeit:",
        "tts_max_threads_label": "Maximale Threads:",
        "tts_voice_selection_label": "Voice-Auswahl und Gewichtungen:",
        "tts_config_label": "Konfiguration:",
        "tts_save_config_button": "Konfiguration speichern",
        "tts_load_config_button": "Konfiguration laden",
        "tts_add_task_button": "Aufgabe hinzufügen",
        "tts_processes_label": "Prozesse:",
        "tts_table_headers": ["Prozess-ID", "Eingabedatei", "Ausgabedatei", "Fortschritt", "Status", "Zeit", "Aktion", "Löschen"],
        "tts_cancel_button": "Abbrechen",
        "tts_restart_button": "Neu starten",
        "tts_pause_button": "Pausieren",
        "tts_resume_button": "Fortführen",
        "tts_delete_button": "Löschen",
        "language_label": "Sprache:",
        "error_invalid_input_file": "Bitte eine gültige Eingabedatei auswählen.",
        "error_invalid_parts": "Die Anzahl der Teile muss größer als 0 sein.",
        "error_no_split_word": "Bitte ein Aufteilungswort oder -tag angeben.",
        "error_no_split_files": "Keine gesplitteten Dateien verfügbar.",
        "error_invalid_output_file": "Bitte einen Ausgabedateinamen angeben.",
        "error_output_not_wav": "Die Ausgabedatei muss eine .wav-Datei sein.",
        "error_no_active_voices": "Bitte mindestens eine Stimme aktivieren und eine Gewichtung > 0 setzen.",
        "error_split_file_pattern": "Die ausgewählte Datei entspricht nicht dem Muster 'Name_XXX.txt'.",
        "error_no_split_files_found": "Keine gesplitteten Dateien für Basisname '{}' gefunden.",
        "success_split": "Textdatei wurde erfolgreich in {} Teile aufgeteilt.",
        "success_tasks_added": "{} Aufgaben für gesplittete Dateien hinzugefügt.",
        "log_file_read": "Datei eingelesen: {} ({} Zeichen)",
        "log_split_points": "Gefundene Aufteilungspunkte: {} Vorkommen von '{}'",
        "log_no_split_word": "Warnung: Keine Vorkommen von '{}' gefunden. Teile nach Zeichenanzahl.",
        "log_actual_splits": "Tatsächliche Split-Punkte: {}",
        "log_part_saved": "Gespeichert: {} ({} Zeichen)",
        "log_split_success": "✅ Erfolgreich geteilt in {} Dateien.",
        "log_config_saved": "Konfiguration gespeichert unter: {}",
        "log_config_loaded": "Konfiguration geladen von: {}",
        "log_last_config_loaded": "Letzte Konfiguration geladen.",
        "log_config_save_warning": "Warnung: Konnte letzte Konfiguration nicht speichern: {}",
        "log_config_load_warning": "Warnung: Konnte letzte Konfiguration nicht laden: {}",
        "log_max_threads_changed": "Maximale Threads geändert auf: {}",
        "log_task_added": "Aufgabe hinzugefügt für {} -> {}",
        "log_thread_started": "Neuer Thread gestartet für Prozess {}, aktive Threads: {}",
        "log_thread_check": "Start prüfe: {} aktive Threads, max_threads={}, Warteschlange={}",
        "log_thread_finished": "[Prozess {}] Thread beendet, was_canceled={}",
        "log_process_init": "[Prozess {}] Initialisiere Kokoro...",
        "log_file_parsed": "[Prozess {}] Eingabedatei geparst, {} Einträge gefunden.",
        "log_total_sentences": "[Prozess {}] Gesamtzahl der Sätze: {}",
        "log_custom_pause": "[Prozess {}][{}] Füge benutzerdefinierte Pause von {} Sekunden ein.",
        "log_process_voice": "[Prozess {}][{}] Verarbeite mit Stimme '{}'",
        "log_process_custom_mix": "[Prozess {}][{}] Verarbeite mit gemischter Stimme (VOICEPACK)",
        "log_voice_not_found": "[Prozess {}] ⚠️ Stimme '{}' nicht gefunden, verwende VOICEPACK.",
        "log_sentences_found": "[Prozess {}][{}] Gefundene Sätze: {}",
        "log_generate_sentence": "[Prozess {}] → Generiere Satz: \"{}...\"",
        "log_sample_rate": "[Prozess {}] Abtastrate von kokoro.create: {}, Samples Länge: {}",
        "log_sample_rate_warning": "[Prozess {}] ⚠️ Warnung: Abtastrate {} unterscheidet sich von {}",
        "log_memory_usage": "[Prozess {}] Speicherverbrauch: {:.2f} MB",
        "log_process_canceled": "[Prozess {}] ❌ Prozess abgebrochen.",
        "log_process_completed": "[Prozess {}] ✅ Verarbeitung abgeschlossen, Datei geschrieben: {}",
        "log_memory_freed": "[Prozess {}] Speicher freigegeben. Speicherverbrauch: {:.2f} MB",
        "log_cleanup_warning": "[Prozess {}] Warnung beim Cleanup: {}",
        "log_error": "[Prozess {}] ❌ Allgemeiner Fehler: {}",
        "log_no_write_access": "[Prozess {}] ❌ Fehler: Kein Schreibzugriff auf Verzeichnis {}",
        "log_pending_cleanup": "Warte auf Cleanup von Prozessen: {}",
        "log_thread_removed": "[Prozess {}] Entferne abgeschlossenen Thread (isRunning: {})",
    }
}

class TTSThread(QThread):
    """Thread for processing TTS tasks using the Kokoro ONNX model."""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # process_id, progress
    time_signal = pyqtSignal(int, str)  # process_id, time_info
    status_signal = pyqtSignal(int, str)  # process_id, status
    finished_signal = pyqtSignal(int, bool)  # process_id, was_canceled
    error_signal = pyqtSignal(int, str)

    def __init__(self, process_id, input_file, output_file, sentence_pause_duration, speed, voice_weights, language="en"):
        super().__init__()
        self.process_id = process_id
        self.input_file = input_file
        self.output_file = output_file
        self.sentence_pause_duration = sentence_pause_duration
        self.speed = speed
        self.voice_weights = voice_weights
        self.language = language
        self._stop = False
        self._was_canceled = False
        self._paused = False
        self.start_time = None
        self.pause_start_time = None
        self.total_pause_duration = 0
        self.kokoro = None
        self.audio_file = None

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
            self.log_signal.emit(TRANSLATIONS[self.language]["log_memory_freed"].format(self.process_id, memory_usage))
        except Exception as e:
            self.log_signal.emit(TRANSLATIONS[self.language]["log_cleanup_warning"].format(self.process_id, str(e)))

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
            self.status_signal.emit(self.process_id, TRANSLATIONS[self.language]["tts_pause_button"])
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

    def run(self):
        """Execute the TTS processing task."""
        try:
            self.start_time = time.time()
            self.log_signal.emit(TRANSLATIONS[self.language]["log_process_init"].format(self.process_id))
            self.kokoro = Kokoro("kokoro.onnx", "voices-v1.0.bin")
            weight_sum = sum(weight for weight in self.voice_weights.values() if weight > 0)
            if weight_sum == 0:
                weight_sum = 1.0
            VOICEPACK = sum(
                self.kokoro.voices[voice] * (weight / weight_sum)
                for voice, weight in self.voice_weights.items() if weight > 0
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

            def split_sentences(text):
                sentences = re.split(r'(?<=[.!?])\s+', text.strip())
                return [s for s in sentences if s]

            entries = parse_text_file(self.input_file)
            self.log_signal.emit(TRANSLATIONS[self.language]["log_file_parsed"].format(self.process_id, len(entries)))

            total_sentences = 0
            for entry_type, *args in entries:
                if entry_type == "voice":
                    _, text = args
                    sentences = split_sentences(text)
                    total_sentences += len(sentences)
            self.log_signal.emit(TRANSLATIONS[self.language]["log_total_sentences"].format(self.process_id, total_sentences))

            sample_rate = 24000
            output_dir = os.path.dirname(self.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            if output_dir and not os.access(output_dir, os.W_OK):
                self.log_signal.emit(TRANSLATIONS[self.language]["log_no_write_access"].format(self.process_id, output_dir))
                self.error_signal.emit(self.process_id, TRANSLATIONS[self.language]["log_no_write_access"].format(self.process_id, output_dir))
                return

            self.audio_file = sf.SoundFile(self.output_file, mode='w', samplerate=sample_rate, channels=1)
            processed_sentences = 0

            for i, (entry_type, *args) in enumerate(entries):
                if self._stop:
                    self.log_signal.emit(TRANSLATIONS[self.language]["log_process_canceled"].format(self.process_id))
                    return

                while self._paused and not self._stop:
                    self.msleep(100)

                if entry_type == "pause":
                    custom_pause_duration = args[0]
                    self.log_signal.emit(TRANSLATIONS[self.language]["log_custom_pause"].format(self.process_id, i+1, custom_pause_duration))
                    silence = np.zeros(int(custom_pause_duration * sample_rate), dtype=np.float32)
                    self.audio_file.write(silence)
                    del silence
                    gc.collect()
                else:
                    voice, text = args
                    if voice == "default" or voice == "custom_mix":
                        self.log_signal.emit(TRANSLATIONS[self.language]["log_process_custom_mix"].format(self.process_id, i+1))
                        actual_voice = VOICEPACK
                    else:
                        self.log_signal.emit(TRANSLATIONS[self.language]["log_process_voice"].format(self.process_id, i+1, voice))
                        try:
                            actual_voice = self.kokoro.voices[voice]
                        except KeyError:
                            self.log_signal.emit(TRANSLATIONS[self.language]["log_voice_not_found"].format(self.process_id, voice))
                            actual_voice = VOICEPACK

                    sentences = split_sentences(text)
                    self.log_signal.emit(TRANSLATIONS[self.language]["log_sentences_found"].format(self.process_id, i+1, len(sentences)))
                    for sentence in sentences:
                        if self._stop:
                            self.log_signal.emit(TRANSLATIONS[self.language]["log_process_canceled"].format(self.process_id))
                            return

                        while self._paused and not self._stop:
                            self.msleep(100)

                        self.log_signal.emit(TRANSLATIONS[self.language]["log_generate_sentence"].format(self.process_id, sentence[:40]))
                        with torch.no_grad():
                            samples, sr = self.kokoro.create(sentence, voice=actual_voice, speed=self.speed, lang="en-us")
                        self.log_signal.emit(TRANSLATIONS[self.language]["log_sample_rate"].format(self.process_id, sr, len(samples)))
                        if sr != sample_rate:
                            self.log_signal.emit(TRANSLATIONS[self.language]["log_sample_rate_warning"].format(self.process_id, sr, sample_rate))
                        self.audio_file.write(samples)
                        del samples
                        if self.sentence_pause_duration > 0:
                            pause = np.zeros(int(self.sentence_pause_duration * sample_rate), dtype=np.float32)
                            self.audio_file.write(pause)
                            del pause
                        gc.collect()

                        processed_sentences += 1
                        progress = int((processed_sentences / total_sentences) * 100) if total_sentences > 0 else 0
                        self.progress_signal.emit(self.process_id, progress)

                        memory_usage = psutil.Process().memory_info().rss / 1024**2
                        self.log_signal.emit(TRANSLATIONS[self.language]["log_memory_usage"].format(self.process_id, memory_usage))

                        elapsed_time = (time.time() - self.start_time) - self.total_pause_duration
                        if processed_sentences > 0:
                            avg_time_per_sentence = elapsed_time / processed_sentences
                            remaining_sentences = total_sentences - processed_sentences
                            estimated_remaining_time = remaining_sentences * avg_time_per_sentence
                            time_info = f"Time: {self.format_time(elapsed_time)} (approx. {self.format_time(estimated_remaining_time)} remaining)"
                        else:
                            time_info = f"Time: {self.format_time(elapsed_time)} (approx. --:--:-- remaining)"
                        self.time_signal.emit(self.process_id, time_info)

            if self._stop:
                self.log_signal.emit(TRANSLATIONS[self.language]["log_process_canceled"].format(self.process_id))
                return

            self.log_signal.emit(TRANSLATIONS[self.language]["log_process_completed"].format(self.process_id, self.output_file))

        except Exception as e:
            error_msg = TRANSLATIONS[self.language]["log_error"].format(self.process_id, str(e))
            self.log_signal.emit(error_msg)
            self.error_signal.emit(self.process_id, str(e))
        finally:
            self.cleanup()
            memory_usage = psutil.Process().memory_info().rss / 1024**2
            self.log_signal.emit(TRANSLATIONS[self.language]["log_memory_freed"].format(self.process_id, memory_usage))
            self.finished_signal.emit(self.process_id, self._was_canceled)

class MainWindow(QMainWindow):
    """Main window for the Kokoro TTS & Split GUI."""
    def __init__(self):
        super().__init__()
        self.language = "en"  # Default language
        self.setWindowTitle(TRANSLATIONS[self.language]["window_title"])
        self.setGeometry(100, 100, 1200, 800)

        # Last split files (for passing between tabs)
        self.last_split_files = []

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Deutsch"])
        self.language_combo.currentIndexChanged.connect(self.change_language)
        language_layout = QHBoxLayout()
        language_layout.addWidget(QLabel(TRANSLATIONS[self.language]["language_label"]))
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        main_layout.addLayout(language_layout)

        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Text Splitting
        self.split_tab = QWidget()
        self.split_layout = QVBoxLayout()
        self.split_tab.setLayout(self.split_layout)
        self.tab_widget.addTab(self.split_tab, TRANSLATIONS[self.language]["tab_split"])

        # Tab 2: TTS Processing
        self.tts_tab = QWidget()
        self.tts_layout = QVBoxLayout()
        self.tts_tab.setLayout(self.tts_layout)
        self.tab_widget.addTab(self.tts_tab, TRANSLATIONS[self.language]["tab_tts"])

        # Tab 3: Help
        self.help_tab = QWidget()
        self.help_layout = QVBoxLayout()
        self.help_tab.setLayout(self.help_layout)
        self.tab_widget.addTab(self.help_tab, TRANSLATIONS[self.language]["tab_help"])

        # Initialize UI
        self.init_split_tab()
        self.init_tts_tab()
        self.init_help_tab()
        self.tab_widget.setCurrentWidget(self.tts_tab)  # Set TTS Processing as default tab

        # TTS initialization
        self.tts_tasks = []
        self.tts_threads = {}
        self.tts_process_counter = 0
        self.tts_task_queue = []
        self.tts_pending_cleanup = set()

        # Load last configuration
        self.load_last_configuration()

    def init_split_tab(self):
        """Initialize the Text Splitting tab."""
        # Input text file
        self.split_input_file_edit = QLineEdit()
        self.split_input_file_edit.setPlaceholderText(TRANSLATIONS[self.language]["split_input_file_placeholder"])
        split_input_file_button = QPushButton(TRANSLATIONS[self.language]["split_browse_button"])
        split_input_file_button.clicked.connect(self.browse_split_input_file)

        split_input_file_layout = QHBoxLayout()
        split_input_file_layout.addWidget(QLabel(TRANSLATIONS[self.language]["split_input_file_label"]))
        split_input_file_layout.addWidget(self.split_input_file_edit)
        split_input_file_layout.addWidget(split_input_file_button)
        self.split_layout.addLayout(split_input_file_layout)

        # Number of parts
        self.split_parts_spin = QSpinBox()
        self.split_parts_spin.setRange(1, 1000)
        self.split_parts_spin.setValue(10)

        split_parts_layout = QHBoxLayout()
        split_parts_layout.addWidget(QLabel(TRANSLATIONS[self.language]["split_parts_label"]))
        split_parts_layout.addWidget(self.split_parts_spin)
        split_parts_layout.addStretch()
        self.split_layout.addLayout(split_parts_layout)

        # Split word
        self.split_word_edit = QLineEdit()
        self.split_word_edit.setPlaceholderText(TRANSLATIONS[self.language]["split_word_placeholder"])
        self.split_word_edit.setText("[voice=custom_mix]")

        split_word_layout = QHBoxLayout()
        split_word_layout.addWidget(QLabel(TRANSLATIONS[self.language]["split_word_label"]))
        split_word_layout.addWidget(self.split_word_edit)
        split_word_layout.addStretch()
        self.split_layout.addLayout(split_word_layout)

        # Split button
        self.split_button = QPushButton(TRANSLATIONS[self.language]["split_button"])
        self.split_button.clicked.connect(self.split_text_file)
        self.split_layout.addWidget(self.split_button)

        # Load split files to TTS button
        self.load_split_to_tts_button = QPushButton(TRANSLATIONS[self.language]["load_split_to_tts_button"])
        self.load_split_to_tts_button.clicked.connect(self.load_split_files_to_tts)
        self.load_split_to_tts_button.setEnabled(False)
        self.split_layout.addWidget(self.load_split_to_tts_button)

        # Status log
        self.split_log_text = QTextEdit()
        self.split_log_text.setReadOnly(True)
        self.split_log_text.setPlaceholderText(TRANSLATIONS[self.language]["split_log_placeholder"])
        self.split_layout.addWidget(QLabel(TRANSLATIONS[self.language]["split_log_label"]))
        self.split_layout.addWidget(self.split_log_text)

        self.split_layout.addStretch()

    def init_tts_tab(self):
        """Initialize the TTS Processing tab."""
        # Splitter for TTS tab
        tts_splitter = QSplitter(Qt.Vertical)
        self.tts_layout.addWidget(tts_splitter)

        # Upper half: Input fields and voice selection
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

        # Input text file
        self.tts_input_file_edit = QLineEdit()
        self.tts_input_file_edit.setPlaceholderText(TRANSLATIONS[self.language]["tts_input_file_placeholder"])
        tts_input_file_button = QPushButton(TRANSLATIONS[self.language]["tts_browse_button"])
        tts_input_file_button.clicked.connect(self.browse_tts_input_file)
        tts_split_files_button = QPushButton(TRANSLATIONS[self.language]["tts_split_files_button"])
        tts_split_files_button.clicked.connect(self.browse_tts_split_files)

        tts_input_layout = QHBoxLayout()
        tts_input_layout.addWidget(QLabel(TRANSLATIONS[self.language]["tts_input_file_label"]))
        tts_input_layout.addWidget(self.tts_input_file_edit)
        tts_input_layout.addWidget(tts_input_file_button)
        tts_input_layout.addWidget(tts_split_files_button)
        tts_scroll_layout.addLayout(tts_input_layout)

        # Output audio file
        self.tts_output_file_edit = QLineEdit()
        self.tts_output_file_edit.setPlaceholderText(TRANSLATIONS[self.language]["tts_output_file_placeholder"])
        tts_output_file_button = QPushButton(TRANSLATIONS[self.language]["tts_browse_button"])
        tts_output_file_button.clicked.connect(self.browse_tts_output_file)

        tts_output_layout = QHBoxLayout()
        tts_output_layout.addWidget(QLabel(TRANSLATIONS[self.language]["tts_output_file_label"]))
        tts_output_layout.addWidget(self.tts_output_file_edit)
        tts_output_layout.addWidget(tts_output_file_button)
        tts_scroll_layout.addLayout(tts_output_layout)

        # Pause duration, speed, and max threads
        tts_params_grid = QGridLayout()
        self.tts_pause_duration_spin = QDoubleSpinBox()
        self.tts_pause_duration_spin.setRange(0.0, 10.0)
        self.tts_pause_duration_spin.setValue(1.0)
        self.tts_pause_duration_spin.setSingleStep(0.1)
        tts_params_grid.addWidget(QLabel(TRANSLATIONS[self.language]["tts_pause_duration_label"]), 0, 0)
        tts_params_grid.addWidget(self.tts_pause_duration_spin, 0, 1)

        self.tts_speed_spin = QDoubleSpinBox()
        self.tts_speed_spin.setRange(0.1, 2.0)
        self.tts_speed_spin.setValue(0.9)
        self.tts_speed_spin.setSingleStep(0.1)
        tts_params_grid.addWidget(QLabel(TRANSLATIONS[self.language]["tts_speed_label"]), 0, 2)
        tts_params_grid.addWidget(self.tts_speed_spin, 0, 3)

        self.tts_max_threads_spin = QSpinBox()
        self.tts_max_threads_spin.setRange(1, 8)
        self.tts_max_threads_spin.setValue(2)
        self.tts_max_threads_spin.valueChanged.connect(self.on_max_threads_changed)
        tts_params_grid.addWidget(QLabel(TRANSLATIONS[self.language]["tts_max_threads_label"]), 1, 0)
        tts_params_grid.addWidget(self.tts_max_threads_spin, 1, 1)
        tts_scroll_layout.addLayout(tts_params_grid)

        # Voice selection and weights
        if not os.path.exists("kokoro.onnx") or not os.path.exists("voices-v1.0.bin"):
            raise FileNotFoundError("Kokoro model files are missing.")
        kokoro_temp = Kokoro("kokoro.onnx", "voices-v1.0.bin")
        self.available_voices = sorted(kokoro_temp.voices.keys())
        del kokoro_temp
        gc.collect()
        self.tts_voice_checkboxes = {}
        self.tts_voice_spins = {}
        default_weights = {
            "am_adam": 0.0, "am_michael": 0.4, "am_echo": 0.2, "am_onyx": 0.4,
            "am_bella": 0.0, "am_sarah": 0.0, "br_charlie": 0.0, "br_nova": 0.0
        }
        default_enabled = {
            "am_adam": False, "am_michael": True, "am_echo": True, "am_onyx": True,
            "am_bella": False, "am_sarah": False, "br_charlie": False, "br_nova": False
        }
        tts_scroll_layout.addWidget(QLabel(TRANSLATIONS[self.language]["tts_voice_selection_label"]))
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
            self.tts_voice_checkboxes[voice] = checkbox
            self.tts_voice_spins[voice] = spin
            voice_layout.addWidget(checkbox)
            voice_layout.addWidget(spin)
            voice_grid.addLayout(voice_layout, row, col)
        tts_scroll_layout.addLayout(voice_grid)

        # Configuration
        tts_scroll_layout.addWidget(QLabel(TRANSLATIONS[self.language]["tts_config_label"]))
        config_layout = QHBoxLayout()
        save_config_button = QPushButton(TRANSLATIONS[self.language]["tts_save_config_button"])
        save_config_button.clicked.connect(self.save_configuration)
        load_config_button = QPushButton(TRANSLATIONS[self.language]["tts_load_config_button"])
        load_config_button.clicked.connect(self.load_configuration)
        config_layout.addWidget(save_config_button)
        config_layout.addWidget(load_config_button)
        tts_scroll_layout.addLayout(config_layout)

        tts_scroll_layout.addStretch()

        # Lower half: Process table
        tts_lower_widget = QWidget()
        tts_lower_layout = QVBoxLayout()
        tts_lower_widget.setLayout(tts_lower_layout)
        self.tts_add_task_button = QPushButton(TRANSLATIONS[self.language]["tts_add_task_button"])
        self.tts_add_task_button.clicked.connect(self.add_tts_task)
        tts_lower_layout.addWidget(self.tts_add_task_button)
        tts_lower_layout.addWidget(QLabel(TRANSLATIONS[self.language]["tts_processes_label"]))

        self.tts_process_table = QTableWidget()
        self.tts_process_table.setColumnCount(8)
        self.tts_process_table.setHorizontalHeaderLabels(TRANSLATIONS[self.language]["tts_table_headers"])
        self.tts_process_table.setSelectionMode(QTableWidget.NoSelection)
        self.tts_process_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tts_process_table.horizontalHeader().setStretchLastSection(True)
        tts_lower_layout.addWidget(self.tts_process_table)

        tts_splitter.addWidget(tts_upper_widget)
        tts_splitter.addWidget(tts_lower_widget)
        tts_splitter.setSizes([400, 400])

    def init_help_tab(self):
        """Initialize the Help tab with documentation."""
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h1>Kokoro TTS & Split GUI</h1>
            <p>This GUI application facilitates text-to-speech (TTS) processing using the Kokoro ONNX model and text file splitting for large inputs.</p>

            <h2>Features</h2>
            <ul>
                <li><b>Text Splitting</b>: Split large text files into smaller parts based on a specified word or tag.</li>
                <li><b>TTS Processing</b>: Convert text files to audio (WAV) using customizable voice mixes, pause durations after each sentence, and reading speed.</li>
                <li><b>Multithreading</b>: Process multiple TTS tasks concurrently with configurable thread limits, Maximum Threads=1 is recommended.</li>
                <li><b>Multilingual Support</b>: Switch between English and German interfaces.</li>
                <li><b>Configuration Management</b>: Save and load settings for quick reuse.</li>
            </ul>

            <h2>Text Splitting Tab</h2>
            <p>Allows splitting a text file into multiple parts:</p>
            <ul>
                <li><b>Input Text File</b>: Select the text file to split.</li>
                <li><b>Number of Parts</b>: Specify how many parts to create (1–1000).</li>
                <li><b>Split Before Word/Tag</b>: Define a word or tag (e.g., "[voice=custom_mix]") to split before.</li>
                <li><b>Split Text File</b>: Execute the splitting process.</li>
                <li><b>Load Split Files to TTS</b>: Transfer split files to the TTS tab for processing.</li>
                <li><b>Note</b>: For very long texts such as audio books, this offers the option of splitting a long document into many smaller ones, so that if an error is detected later in the text or audio, only this section of the text needs to be corrected and recalculated instead of the entire audio book. Tools such as fre:ac can be used to merge the audios after everything is finished.</li>
            </ul>

            <h2>TTS Processing Tab</h2>
            <p>Converts text to audio with customizable settings:</p>
            <ul>
                <li><b>Input Text File</b>: Select a single text file or load split files.</li>
                <li><b>Output Audio File</b>: Specify the output WAV file.</li>
                <li><b>Pause Duration</b>: Set pause duration between sentences (0–10 seconds).</li>
                <li><b>Speed</b>: Adjust speech speed (0.1–2.0).</li>
                <li><b>Maximum Threads</b>: Set the number of concurrent tasks (1–8), 1 is recommended.</li>
                <li><b>Voice Selection</b>: Enable voices and set weights (0.0–1.0) for a custom mix.</li>
                <li><b>Configuration</b>: Save or load settings.</li>
                <li><b>Add Task</b>: Add a TTS task to the queue.</li>
                <li><b>Process Table</b>: Monitor tasks with options to pause, cancel, restart, or delete.</li>
            </ul>

            <h2>Usage</h2>
            <ol>
                <li>Split a large text file in the "Text Splitting" tab if needed.</li>
                <li>Load split files or select a single file in the "TTS Processing" tab.</li>
                <li>Configure voice weights, pause duration, and speed.</li>
                <li>Add tasks to the queue and monitor progress in the process table.</li>
                <li><b>Note</b>: Within the text file you can use control commands such as [voice=custom_mix] [voice=af_heart] [pause=1.2] always at the beginning and alone in a line.</li>
            </ol>

            <h2>Requirements</h2>
            <ul>
                <li>Python 3.9–3.12</li>
                <li>Dependencies: PyQt5, numpy, torch, soundfile, psutil, kokoro-onnx, phonemizer-fork</li>
                <li>Kokoro model files: kokoro.onnx, voices-v1.0.bin</li>
            </ul>

            <p>For more details, see the <a href="https://github.com/your-repo/kokoro-tts-gui">GitHub repository</a>.</p>
        """)
        self.help_layout.addWidget(help_text)

    def change_language(self):
        """Change the UI language based on the selected option."""
        language_map = {0: "en", 1: "de"}
        self.language = language_map[self.language_combo.currentIndex()]
        self.update_ui_language()
        self.save_last_configuration()

    def update_ui_language(self):
        """Update all UI elements to the current language."""
        self.setWindowTitle(TRANSLATIONS[self.language]["window_title"])
        self.tab_widget.setTabText(0, TRANSLATIONS[self.language]["tab_split"])
        self.tab_widget.setTabText(1, TRANSLATIONS[self.language]["tab_tts"])
        self.tab_widget.setTabText(2, TRANSLATIONS[self.language]["tab_help"])

        # Update Split Tab
        self.split_layout.itemAt(0).layout().itemAt(0).widget().setText(TRANSLATIONS[self.language]["split_input_file_label"])
        self.split_input_file_edit.setPlaceholderText(TRANSLATIONS[self.language]["split_input_file_placeholder"])
        self.split_layout.itemAt(0).layout().itemAt(2).widget().setText(TRANSLATIONS[self.language]["split_browse_button"])
        self.split_layout.itemAt(1).layout().itemAt(0).widget().setText(TRANSLATIONS[self.language]["split_parts_label"])
        self.split_layout.itemAt(2).layout().itemAt(0).widget().setText(TRANSLATIONS[self.language]["split_word_label"])
        self.split_word_edit.setPlaceholderText(TRANSLATIONS[self.language]["split_word_placeholder"])
        self.split_button.setText(TRANSLATIONS[self.language]["split_button"])
        self.load_split_to_tts_button.setText(TRANSLATIONS[self.language]["load_split_to_tts_button"])
        self.split_layout.itemAt(5).widget().setText(TRANSLATIONS[self.language]["split_log_label"])
        self.split_log_text.setPlaceholderText(TRANSLATIONS[self.language]["split_log_placeholder"])

        # Update TTS Tab
        # Access the scroll layout through the QSplitter
        tts_upper_widget = self.tts_layout.itemAt(0).widget().widget(0)  # Upper widget in QSplitter
        tts_scroll_layout = tts_upper_widget.layout().itemAt(0).widget().widget().layout()  # QScrollArea's widget layout
        tts_lower_widget = self.tts_layout.itemAt(0).widget().widget(1)  # Lower widget in QSplitter

        # Update input file section
        tts_scroll_layout.itemAt(0).layout().itemAt(0).widget().setText(TRANSLATIONS[self.language]["tts_input_file_label"])
        self.tts_input_file_edit.setPlaceholderText(TRANSLATIONS[self.language]["tts_input_file_placeholder"])
        tts_scroll_layout.itemAt(0).layout().itemAt(2).widget().setText(TRANSLATIONS[self.language]["tts_browse_button"])
        tts_scroll_layout.itemAt(0).layout().itemAt(3).widget().setText(TRANSLATIONS[self.language]["tts_split_files_button"])

        # Update output file section
        tts_scroll_layout.itemAt(1).layout().itemAt(0).widget().setText(TRANSLATIONS[self.language]["tts_output_file_label"])
        self.tts_output_file_edit.setPlaceholderText(TRANSLATIONS[self.language]["tts_output_file_placeholder"])
        tts_scroll_layout.itemAt(1).layout().itemAt(2).widget().setText(TRANSLATIONS[self.language]["tts_browse_button"])

        # Update parameters section
        tts_scroll_layout.itemAt(2).layout().itemAt(0).widget().setText(TRANSLATIONS[self.language]["tts_pause_duration_label"])
        tts_scroll_layout.itemAt(2).layout().itemAt(2).widget().setText(TRANSLATIONS[self.language]["tts_speed_label"])
        tts_scroll_layout.itemAt(2).layout().itemAt(4).widget().setText(TRANSLATIONS[self.language]["tts_max_threads_label"])

        # Update voice selection and configuration labels
        tts_scroll_layout.itemAt(3).widget().setText(TRANSLATIONS[self.language]["tts_voice_selection_label"])
        tts_scroll_layout.itemAt(5).widget().setText(TRANSLATIONS[self.language]["tts_config_label"])
        tts_scroll_layout.itemAt(6).layout().itemAt(0).widget().setText(TRANSLATIONS[self.language]["tts_save_config_button"])
        tts_scroll_layout.itemAt(6).layout().itemAt(1).widget().setText(TRANSLATIONS[self.language]["tts_load_config_button"])

        # Update lower widget (process table section)
        tts_lower_layout = tts_lower_widget.layout()
        tts_lower_layout.itemAt(0).widget().setText(TRANSLATIONS[self.language]["tts_add_task_button"])
        tts_lower_layout.itemAt(1).widget().setText(TRANSLATIONS[self.language]["tts_processes_label"])
        self.tts_process_table.setHorizontalHeaderLabels(TRANSLATIONS[self.language]["tts_table_headers"])

        # Update process table buttons
        for row in range(self.tts_process_table.rowCount()):
            action_widget = self.tts_process_table.cellWidget(row, 6)
            action_widget.layout().itemAt(0).widget().setText(TRANSLATIONS[self.language]["tts_cancel_button"])
            action_widget.layout().itemAt(1).widget().setText(TRANSLATIONS[self.language]["tts_restart_button"])
            pause_button = action_widget.layout().itemAt(2).widget()
            status = self.tts_process_table.item(row, 4).text()
            pause_button.setText(TRANSLATIONS[self.language]["tts_resume_button"] if status == TRANSLATIONS[self.language]["tts_pause_button"] else TRANSLATIONS[self.language]["tts_pause_button"])
            self.tts_process_table.cellWidget(row, 7).setText(TRANSLATIONS[self.language]["tts_delete_button"])

    def on_max_threads_changed(self, value):
        """Handle changes to the maximum threads setting."""
        self.split_log_text.append(TRANSLATIONS[self.language]["log_max_threads_changed"].format(value))
        self.start_tts_queued_tasks()

    def save_configuration(self):
        """Save the current configuration to a JSON file."""
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "configs/", "JSON Files (*.json)")
        if file_name:
            if not file_name.endswith(".json"):
                file_name += ".json"
            config = {
                "language": self.language,
                "split_parts": self.split_parts_spin.value(),
                "split_word": self.split_word_edit.text(),
                "tts_pause_duration": self.tts_pause_duration_spin.value(),
                "tts_speed": self.tts_speed_spin.value(),
                "tts_max_threads": self.tts_max_threads_spin.value(),
                "tts_voice_weights": {voice: self.tts_voice_spins[voice].value() for voice in self.available_voices},
                "tts_voice_enabled": {voice: self.tts_voice_checkboxes[voice].isChecked() for voice in self.available_voices}
            }
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            try:
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4)
                self.split_log_text.append(TRANSLATIONS[self.language]["log_config_saved"].format(file_name))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save configuration: {str(e)}")

    def load_configuration(self):
        """Load a configuration from a JSON file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "configs/", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.language = config.get("language", "en")
                self.language_combo.setCurrentIndex(0 if self.language == "en" else 1)
                self.split_parts_spin.setValue(config.get("split_parts", 10))
                self.split_word_edit.setText(config.get("split_word", "[voice=custom_mix]"))
                self.tts_pause_duration_spin.setValue(config.get("tts_pause_duration", 1.0))
                self.tts_speed_spin.setValue(config.get("tts_speed", 0.9))
                self.tts_max_threads_spin.setValue(config.get("tts_max_threads", 2))
                for voice in self.available_voices:
                    weight = config.get("tts_voice_weights", {}).get(voice, 0.0)
                    enabled = config.get("tts_voice_enabled", {}).get(voice, False)
                    self.tts_voice_spins[voice].setValue(weight)
                    self.tts_voice_checkboxes[voice].setChecked(enabled)
                    self.tts_voice_spins[voice].setEnabled(enabled)
                self.split_log_text.append(TRANSLATIONS[self.language]["log_config_loaded"].format(file_name))
                self.update_ui_language()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load configuration: {str(e)}")

    def save_last_configuration(self):
        """Save the last configuration to a JSON file."""
        config = {
            "language": self.language,
            "split_parts": self.split_parts_spin.value(),
            "split_word": self.split_word_edit.text(),
            "tts_pause_duration": self.tts_pause_duration_spin.value(),
            "tts_speed": self.tts_speed_spin.value(),
            "tts_max_threads": self.tts_max_threads_spin.value(),
            "tts_voice_weights": {voice: self.tts_voice_spins[voice].value() for voice in self.available_voices},
            "tts_voice_enabled": {voice: self.tts_voice_checkboxes[voice].isChecked() for voice in self.available_voices}
        }
        try:
            os.makedirs("configs", exist_ok=True)
            with open("configs/last_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.split_log_text.append(TRANSLATIONS[self.language]["log_config_save_warning"].format(str(e)))

    def load_last_configuration(self):
        """Load the last configuration from a JSON file."""
        config_file = "configs/last_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.language = config.get("language", "en")
                self.language_combo.setCurrentIndex(0 if self.language == "en" else 1)
                self.split_parts_spin.setValue(config.get("split_parts", 10))
                self.split_word_edit.setText(config.get("split_word", "[voice=custom_mix]"))
                self.tts_pause_duration_spin.setValue(config.get("tts_pause_duration", 1.0))
                self.tts_speed_spin.setValue(config.get("tts_speed", 0.9))
                self.tts_max_threads_spin.setValue(config.get("tts_max_threads", 2))
                for voice in self.available_voices:
                    weight = config.get("tts_voice_weights", {}).get(voice, 0.0)
                    enabled = config.get("tts_voice_enabled", {}).get(voice, False)
                    self.tts_voice_spins[voice].setValue(weight)
                    self.tts_voice_checkboxes[voice].setChecked(enabled)
                    self.tts_voice_spins[voice].setEnabled(enabled)
                self.split_log_text.append(TRANSLATIONS[self.language]["log_last_config_loaded"])
                self.update_ui_language()
            except Exception as e:
                self.split_log_text.append(TRANSLATIONS[self.language]["log_config_load_warning"].format(str(e)))

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

        # Validation
        if not input_file or not os.path.exists(input_file):
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_invalid_input_file"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_invalid_input_file"])
            return
        if num_parts < 1:
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_invalid_parts"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_invalid_parts"])
            return
        if not split_word:
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_no_split_word"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_no_split_word"])
            return

        try:
            # Read text file
            with open(input_file, "r", encoding="utf-8") as f:
                text = f.read()

            self.split_log_text.append(TRANSLATIONS[self.language]["log_file_read"].format(input_file, len(text)))

            # Find split points
            split_positions = [m.start() for m in re.finditer(re.escape(split_word), text)]
            if not split_positions:
                self.split_log_text.append(TRANSLATIONS[self.language]["log_no_split_word"].format(split_word))
                part_size = len(text) // num_parts
                split_positions = [i * part_size for i in range(num_parts)]
            else:
                split_positions.insert(0, 0)

            self.split_log_text.append(TRANSLATIONS[self.language]["log_split_points"].format(len(split_positions) - 1, split_word))

            # Create approximately equal parts
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

            self.split_log_text.append(TRANSLATIONS[self.language]["log_actual_splits"].format(actual_splits))

            # Write split files
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
                self.split_log_text.append(TRANSLATIONS[self.language]["log_part_saved"].format(output_file, len(part_text)))

            self.split_log_text.append(TRANSLATIONS[self.language]["log_split_success"].format(num_parts))
            self.load_split_to_tts_button.setEnabled(True)
            QMessageBox.information(self, "Success", TRANSLATIONS[self.language]["success_split"].format(num_parts))

        except Exception as e:
            self.split_log_text.append(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error splitting file: {str(e)}")

    def load_split_files_to_tts(self):
        """Load split files to the TTS tab."""
        if not self.last_split_files:
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_no_split_files"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_no_split_files"])
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
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_split_file_pattern"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_split_file_pattern"])
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
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_no_split_files_found"].format(base_name))
            self.split_log_text.append(TRANSLATIONS[self.language]["error_no_split_files_found"].format(base_name))
            return

        self.split_log_text.append(f"Found split files: {split_files}")

        for input_file in split_files:
            output_file = os.path.splitext(input_file)[0] + '.wav'
            sentence_pause_duration = self.tts_pause_duration_spin.value()
            speed = self.tts_speed_spin.value()
            voice_weights = {
                voice: self.tts_voice_spins[voice].value() if self.tts_voice_checkboxes[voice].isChecked() else 0.0
                for voice in self.available_voices
            }

            if not os.path.exists(input_file):
                QMessageBox.warning(self, "Error", f"Input file not found: {input_file}")
                self.split_log_text.append(f"Error: Input file not found: {input_file}")
                continue
            if all(weight == 0.0 for weight in voice_weights.values()):
                QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_no_active_voices"])
                self.split_log_text.append(TRANSLATIONS[self.language]["error_no_active_voices"])
                continue

            self.tts_process_counter += 1
            task = {
                "process_id": self.tts_process_counter,
                "input_file": input_file,
                "output_file": output_file,
                "sentence_pause_duration": sentence_pause_duration,
                "speed": speed,
                "voice_weights": voice_weights
            }
            self.tts_tasks.append(task)
            self.tts_task_queue.append(task)
            self.add_tts_task_to_table(task)
            self.split_log_text.append(TRANSLATIONS[self.language]["log_task_added"].format(input_file, output_file))

        self.start_tts_queued_tasks()
        self.tts_input_file_edit.clear()
        self.tts_output_file_edit.clear()
        QMessageBox.information(self, "Success", TRANSLATIONS[self.language]["success_tasks_added"].format(len(split_files)))

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
            voice: self.tts_voice_spins[voice].value() if self.tts_voice_checkboxes[voice].isChecked() else 0.0
            for voice in self.available_voices
        }

        if not input_file or not os.path.exists(input_file):
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_invalid_input_file"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_invalid_input_file"])
            return
        if not output_file:
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_invalid_output_file"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_invalid_output_file"])
            return
        if not output_file.lower().endswith('.wav'):
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_output_not_wav"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_output_not_wav"])
            return
        if all(weight == 0.0 for weight in voice_weights.values()):
            QMessageBox.warning(self, "Error", TRANSLATIONS[self.language]["error_no_active_voices"])
            self.split_log_text.append(TRANSLATIONS[self.language]["error_no_active_voices"])
            return

        self.tts_process_counter += 1
        task = {
            "process_id": self.tts_process_counter,
            "input_file": input_file,
            "output_file": output_file,
            "sentence_pause_duration": sentence_pause_duration,
            "speed": speed,
            "voice_weights": voice_weights
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

        cancel_button = QPushButton(TRANSLATIONS[self.language]["tts_cancel_button"])
        cancel_button.clicked.connect(lambda: self.cancel_tts_task(task["process_id"]))
        cancel_button.setEnabled(True)
        action_layout.addWidget(cancel_button)

        restart_button = QPushButton(TRANSLATIONS[self.language]["tts_restart_button"])
        restart_button.clicked.connect(lambda: self.restart_tts_task(task["process_id"]))
        restart_button.setEnabled(False)
        action_layout.addWidget(restart_button)

        pause_button = QPushButton(TRANSLATIONS[self.language]["tts_pause_button"])
        pause_button.clicked.connect(lambda: self.pause_tts_task(task["process_id"]))
        pause_button.setEnabled(True)
        action_layout.addWidget(pause_button)

        self.tts_process_table.setCellWidget(row, 6, action_widget)

        delete_button = QPushButton(TRANSLATIONS[self.language]["tts_delete_button"])
        delete_button.clicked.connect(lambda: self.delete_tts_task(task["process_id"], row))
        delete_button.setEnabled(True)
        self.tts_process_table.setCellWidget(row, 7, delete_button)

        self.tts_process_table.resizeColumnsToContents()

    def start_tts_queued_tasks(self):
        """Start new threads from the queue if possible."""
        if self.tts_pending_cleanup:
            self.split_log_text.append(TRANSLATIONS[self.language]["log_pending_cleanup"].format(self.tts_pending_cleanup))
            return

        # Check for finished threads
        for process_id, thread in list(self.tts_threads.items()):
            if not thread.isRunning():
                self.split_log_text.append(TRANSLATIONS[self.language]["log_thread_removed"].format(process_id, thread.isRunning()))
                thread.cleanup()
                thread.wait()
                del self.tts_threads[process_id]
                if process_id in self.tts_pending_cleanup:
                    self.tts_pending_cleanup.remove(process_id)

        max_threads = self.tts_max_threads_spin.value()
        active_threads = len(self.tts_threads)
        self.split_log_text.append(TRANSLATIONS[self.language]["log_thread_check"].format(active_threads, max_threads, len(self.tts_task_queue)))

        while self.tts_task_queue and active_threads < max_threads:
            task = self.tts_task_queue.pop(0)
            thread = TTSThread(
                task["process_id"],
                task["input_file"],
                task["output_file"],
                task["sentence_pause_duration"],
                task["speed"],
                task["voice_weights"],
                self.language
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
            self.split_log_text.append(TRANSLATIONS[self.language]["log_thread_started"].format(task["process_id"], active_threads))

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

                if status in ["Running", TRANSLATIONS[self.language]["tts_pause_button"]]:
                    cancel_button.setEnabled(True)
                    restart_button.setEnabled(False)
                    pause_button.setEnabled(True)
                    pause_button.setText(TRANSLATIONS[self.language]["tts_resume_button"] if status == TRANSLATIONS[self.language]["tts_pause_button"] else TRANSLATIONS[self.language]["tts_pause_button"])
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
                    pause_button.setText(TRANSLATIONS[self.language]["tts_pause_button"])
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
                    "voice_weights": task["voice_weights"]
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
            self.split_log_text.append(TRANSLATIONS[self.language]["log_thread_finished"].format(process_id, was_canceled))
            status = "Canceled" if was_canceled else "Completed"
            self.update_tts_task_status(process_id, status)
            thread.cleanup()
            del self.tts_threads[process_id]
            if process_id in self.tts_pending_cleanup:
                self.tts_pending_cleanup.remove(process_id)
            self.start_tts_queued_tasks()

    def on_tts_error(self, process_id, error_message):
        """Handle errors in a TTS thread."""
        self.split_log_text.append(TRANSLATIONS[self.language]["log_error"].format(process_id, error_message))
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
