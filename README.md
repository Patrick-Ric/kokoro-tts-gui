# Kokoro TTS GUI (Audiobook Maker) v2.0

A graphical user interface (GUI) for text-to-speech (TTS) processing and text file splitting using the Kokoro ONNX model ideal for audiobook production. This tool allows users to split large text files into smaller parts and convert them to audio with customizable voice mixes, pause durations after each sentence, and reading speed.
Additional control characters such as flexible pause times and different voices can be used within the text (see TTS Processing below).

## Features
- **Text Splitting**: Split large text files into multiple parts based on a specified word or tag (e.g.,`Chapter`, `[voice=custom_mix]`).
- **TTS Processing**: Convert text files to WAV audio files using the Kokoro ONNX model with configurable voices, pauses, and speeds.
- **Voice Mixing**: Mix up to 6 different self-mixable voices (custom_mix and custom_mix_1 through custom_mix_5) directly in the GUI and activate them via control commands in the text file.
- **Multithreading**: Process multiple TTS tasks concurrently with adjustable thread limits.
- **Multilingual Support**: Switch between English and German interfaces.
- **Configuration Management**: Save and load settings for quick reuse.
- **Help Documentation**: Built-in help tab with usage instructions.

## Screenshots
![TTS Processing Tab](screenshots/TTS_GUI2a.png)
![Text Splitting Tab](screenshots/TTS_GUI1.png)
![Text Splitting Tab](screenshots/Kokoro_GUI_Mix.png)
## Requirements
- **Python**: Version 3.9–3.12
- **Dependencies**:
  ```bash
  pip install PyQt5 numpy torch soundfile psutil kokoro-onnx phonemizer-fork
  ```
- **Kokoro Model Files**:
  - `kokoro.onnx`
  - `voices-v1.0.bin`
  *(Note: These files are not included in the repository, they must be in the same 
folder as the gui, you can find them here 
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0 
kokoro-v1.0.onnx (~310 MB, that is the f32-Version), voices-v1.0.bin (26.9 MB).
BUT the GUI looks for filename "kokoro.onnx" therefore the name must be **renamed** in kokoro.onnx )*

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Patrick-Ric/kokoro-tts-gui.git
   cd kokoro-tts-gui
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `kokoro.onnx` and `voices-v1.0.bin` in the project directory.
4. Run the application:
   ```bash
   python kokoro_tts_gui.py
   ```

## Usage

### 1. Text Splitting
- Go to the "Text Splitting" tab.
- Select an input text file and specify the number of parts and a split word/tag (e.g., `[voice=custom_mix]`).
- Click "Split Text File" to create split files.
- Use "Load Split Files to TTS" to transfer them to the TTS tab by selecting the first split part with the ending `_001.txt`. All split parts are then read in by the GUI and processing and the work is started immediately.
- **Note**: For very long texts such as audio books, this offers the option of splitting a long document into many smaller ones, so that if an error is detected later in the text or audio, only this section of the text needs to be corrected and recalculated instead of the entire audio book.
- You can find a more detailed Text Splitting explanation here:
  https://github.com/Patrick-Ric/kokoro-tts-gui/issues/2

### 2. TTS Processing
- In the "TTS Processing" tab, select an input text file or load split files.
- Specify an output WAV file, pause duration, speed, and voice weights.
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
- Voicemix is the mixed voice from the GUI (Voice Selection and Weights) and it can be activated by the control command `[voice=custom_mix]` within the text file. Now up to 6 different self-mixable voices (custom_mix and custom_mix_1 through custom_mix_5) can be configured directly in the GUI.

### 3. Example Text File with Control Commands
```text
[pause=1.5]
[voice=af_heart]
Chapter 1. INTRODUCTION TO ARTIFICIAL INTELLIGENCE AND SPEECH SYNTHESIS

[pause=2.0]
[voice=custom_mix]
Artificial intelligence allows computers to learn from experience and perform human-like tasks. 
[pause=0.8]
Neural speech synthesis transforms written text into natural-sounding speech with remarkable realism.

[pause=2.0]
[voice=custom_mix_1]
By blending different voice models, you can create unique narrators tailored to specific stories or topics.
[pause=0.8]
Control tags give you precise command over pauses, pacing, and speaker switching throughout your audiobook.
```

### 4. Configuration
- Save your settings with "Save Configuration" for reuse.
- Load previous settings with "Load Configuration".

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For issues or questions, open an issue on GitHub.
