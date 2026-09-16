# Kokoro TTS GUI (Audiobook Maker) v2.1

A graphical user interface (GUI) for text-to-speech (TTS) processing and text file splitting using the Kokoro ONNX model ideal for audiobook production. This tool allows users to split large text files into smaller parts and convert them to audio with customizable voice mixes, pause durations after each sentence, and reading speed.
Additional control characters such as flexible pause times and different voices can be used within the text (see TTS Processing below).

## Features
- **Text Splitting**: Split large text files into multiple parts based on a specified word or tag (e.g.,`Chapter`, `[voice=custom_mix]`).
- **TTS Processing**: Convert text files to WAV audio files using the Kokoro ONNX model with configurable voices, pauses, and speeds.
- **Voice Mixing**: Mix up to 6 different self-mixable voices (custom_mix and custom_mix_1 through custom_mix_5) directly in the GUI and activate them via control commands in the text file.
- **Multithreading**: Process multiple TTS tasks concurrently with adjustable thread limits (shared model instance — loaded only once, ~310 MB).
- **Configuration Management**: Save and load settings for quick reuse.
- **Phoneme Mode**: Wrap text in `$$...$$` to pass it directly to Kokoro as a phoneme string (G2P skipped). A single `$` is ignored and left untouched.
- **Safety**: The app asks for confirmation before quitting while tasks are running or queued, and it runs from any working directory (model paths are script-relative).
- **CPU by default**: runs on any PC without a GPU (tested). NVIDIA GPU acceleration via `pip install "kokoro-onnx[gpu]"` should work automatically but is untested — feedback welcome.


## Screenshots
![TTS Processing Tab](screenshots/TTS_GUI2a.png)
![Text Splitting Tab](screenshots/TTS_GUI1.png)
![Voice Mixing Tab](screenshots/Kokoro_GUI_Mix.png)
## Requirements
- **Python**: Version 3.10–3.13 (required: `kokoro-onnx>=0.4.7` needs Python ≥ 3.10 and supports only up to 3.13. With Python 3.9 pip silently installs an old `kokoro-onnx 0.1.x` that cannot read `voices-v1.0.bin`; with Python 3.14+ pip cannot install a working `kokoro-onnx` at all. On rolling-release distros with Python 3.14 as default, create the venv with an older Python, e.g. `uv venv --python 3.13 venv` after `uv python install 3.13`.)
- **Dependencies**:
  ```bash
  pip install PyQt5 numpy soundfile psutil kokoro-onnx phonemizer-fork
  ```
  (No `torch` needed — ONNX inference only.)
- **Kokoro Model Files** (not included in the repository — place them next to the script, **no renaming needed**; you can find them here:
  https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0):
  - Model: `kokoro-v1.0.onnx` (~310 MB, f32 version) — also accepted: `kokoro.onnx` or any `kokoro*.onnx`
  - Voices: `voices-v1.0.bin` (26.9 MB) — also accepted: any `voices*.bin`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Patrick-Ric/kokoro-tts-gui.git
   cd kokoro-tts-gui
   ```
2. Run the setup script (copy & paste — picks a working Python 3.10–3.13
   automatically; on Python 3.14+ systems it uses 3.13 via `uv`):
   ```bash
   bash setup.sh
   ```
   Manual alternative (without the script):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Notes: modern Linux distributions (Manjaro, Ubuntu 23.04+, Fedora,
   Debian 12+) block system-wide `pip installs`
   (`externally-managed-environment`) — always use the venv.
   `uv venv` creates no `pip` by default — then either use
   `uv venv --seed ...` or install with
   `uv pip install --python venv/bin/python -r requirements.txt`.
3. Place the model file (`kokoro-v1.0.onnx`) and `voices-v1.0.bin` next to the script (any `kokoro*.onnx` / `voices*.bin` name works).
4. Run the application (the venv is still active from step 2; after a
   restart, activate it again with `source venv/bin/activate`):
   ```bash
   source venv/bin/activate  # only needed after a restart
   python kokoro_tts_gui.py
   ```

## Usage

### 1. Text Splitting
- Go to the "Text Splitting" tab.
- Select an input text file and specify the number of parts and a split word/tag (e.g., `[voice=custom_mix]`).
- Click "Split Text File" to create split files.
- Use "Load Split Files to TTS" to transfer them to the TTS tab by selecting the first split part (e.g. with the ending `_001.txt`). All split parts are then read in by the GUI and processing and the work is started immediately. You can also start from any later part (e.g. `_005.txt`) — that file and all following parts are then loaded automatically.
- **Note**: Splitting is optional. You can also process a whole book as a single file — just add it directly in the "TTS Processing" tab via "Add Task" (e.g. `MyBook.txt` → `MyBook.wav`). Splitting is only recommended for very long texts such as audiobooks: if an error is detected later in the text or audio, only that text-section needs to be corrected and re-synthesized instead of the entire book.
- You can find a more detailed Text Splitting explanation here:
  https://github.com/Patrick-Ric/kokoro-tts-gui/issues/2

### 2. TTS Processing
- In the "TTS Processing" tab, select an input text file or load split files.
- Specify an output WAV file, speed, and voice weights.
- Click "Add Task" to queue the task.
- Monitor progress in the process table, where you can pause, cancel, restart, or delete tasks.
- **Note**: Within the text file you can use control commands such as
  ```
  [voice=custom_mix]
  [voice=custom_mix_1] ... [voice=custom_mix_5]
  [voice=af_heart]
  [pause=1.2]
  [pause=2.34]
  ```
  always at the beginning and alone in a line.
- **Phoneme mode**: everything between a pair of `$$` markers is sent directly to Kokoro as phonemes (`is_phonemes=True`), e.g. `$$hɛˈloʊ wɜːld$$`. If a `$$` section is never closed, the rest of the text is treated as phonemes and a warning is logged.
- Voicemix is the mixed voice from the GUI (Voice Selection and Weights) and it can be activated by the control command `[voice=custom_mix]` within the text file. Now up to 6 different self-mixable voices (custom_mix and custom_mix_1 through custom_mix_5) can be configured directly in the GUI.

### 3. Example Text File with Control Commands
```text
[pause=1.5]
[voice=af_heart]
Chapter 1. INTRODUCTION TO ARTIFICIAL INTELLIGENCE AND SPEECH SYNTHESIS.

[pause=2.0]
[voice=custom_mix]
Artificial intelligence allows computers to learn from experience and perform human-like tasks. 
[pause=0.8]
Neural speech synthesis transforms written text into natural-sounding speech with remarkable realism.

[pause=2.15]
[voice=custom_mix_1]
By blending different voice models, you can create unique narrators tailored to specific stories or topics.
[pause=0.75]
Control tags give you precise command over pauses, pacing, and speaker switching throughout your audiobook.
```

### 4. Configuration
- Save your settings with "Save Configuration" for reuse.
- Load previous settings with "Load Configuration".

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For issues or questions, open an issue on GitHub.
