#!/usr/bin/env bash
# Setup script for Kokoro TTS GUI.
# Copy & paste setup (run once after cloning):
#   bash setup.sh
# Picks a working Python automatically (3.10-3.13 required by kokoro-onnx):
#   - system Python 3.10-3.13 -> plain venv
#   - otherwise (e.g. Python 3.9 or 3.14+) -> venv with Python 3.13 via uv
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Kokoro TTS GUI setup ==="

# 1. Find a suitable system Python (3.10 - 3.13).
SYS_PY=""
for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        if "$py" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' 2>/dev/null; then
            SYS_PY="$py"
            echo "Found suitable Python: $("$py" --version 2>&1)"
            break
        else
            echo "Skipping $py ($("$py" --version 2>&1)): needs 3.10-3.13"
        fi
    fi
done

have_pkg_mgr() { command -v "$1" >/dev/null 2>&1; }

# 2. Create the venv.
if [ -d venv ]; then
    echo "venv/ already exists - reusing it."
elif [ -n "$SYS_PY" ]; then
    echo "Creating venv with system Python..."
    if ! "$SYS_PY" -m venv venv 2>/dev/null; then
        echo "The 'venv' module is missing. Install it, then re-run setup.sh:"
        if have_pkg_mgr pacman; then
            echo "  sudo pacman -S python-virtualenv"
        elif have_pkg_mgr apt-get; then
            PYV="python3-venv"
            "$SYS_PY" -c 'import sys; print("sudo apt install python%d.%d-venv" % sys.version_info[:2])' 2>/dev/null || echo "  sudo apt install $PYV"
        elif have_pkg_mgr dnf; then
            echo "  sudo dnf install python3-devel"
        elif have_pkg_mgr zypper; then
            echo "  sudo zypper install python3-venv"
        else
            echo "  Install the 'venv' module for your distribution, then re-run setup.sh"
        fi
        exit 1
    fi
else
    echo "No Python 3.10-3.13 found - falling back to Python 3.13 via uv."
    if ! command -v uv >/dev/null 2>&1; then
        # Try unattended install via the detected package manager first.
        if have_pkg_mgr pacman && sudo pacman -S --noconfirm uv; then
            :
        elif have_pkg_mgr apt-get && sudo apt-get update && sudo apt-get install -y uv; then
            :
        elif have_pkg_mgr dnf && sudo dnf install -y uv; then
            :
        elif have_pkg_mgr zypper && sudo zypper --non-interactive install uv; then
            :
        elif have_pkg_mgr brew && brew install uv; then
            :
        else
            echo "Installing uv into ~/.local/bin ..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
        command -v uv >/dev/null 2>&1 || {
            echo "ERROR: 'uv' is still missing. Install it manually, then re-run setup.sh:"
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        }
    fi
    uv python install 3.13
    # --seed installs pip into the venv too; otherwise 'pip' still resolves
    # to /usr/bin/pip (externally-managed-environment error).
    uv venv --seed --python 3.13 venv
fi

# 3. Install dependencies (CPU-only - no NVIDIA/CUDA downloads).
# A 'uv venv' without --seed contains no pip -> then install via 'uv pip'.
echo "Installing dependencies..."
if [ -x venv/bin/pip ]; then
    venv/bin/pip install --upgrade pip
    venv/bin/pip install -r requirements.txt
elif command -v uv >/dev/null 2>&1; then
    uv pip install --python venv/bin/python -r requirements.txt
else
    venv/bin/python -m pip install -r requirements.txt
fi

# 4. Verify kokoro-onnx version.
KOKORO_VER="$(venv/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("kokoro-onnx"))' 2>/dev/null || echo "missing")"
echo "kokoro-onnx version installed: $KOKORO_VER"

# 5. Check model files.
MISSING=0
if ! ls kokoro*.onnx >/dev/null 2>&1; then
    echo "NOTE: no kokoro*.onnx found next to the script."
    MISSING=1
fi
if ! ls voices*.bin >/dev/null 2>&1; then
    echo "NOTE: no voices*.bin found next to the script."
    MISSING=1
fi
if [ "$MISSING" = "1" ]; then
    echo "Download them here (no renaming needed):"
    echo "  https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
fi

echo ""
echo "Done. Start the GUI with:"
echo "  source venv/bin/activate"
echo "  python kokoro_tts_gui.py"
