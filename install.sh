#!/bin/bash
# PTT — Push-to-Talk Transcription installer
# Run: curl -fsSL https://raw.githubusercontent.com/leonideang/ptt/main/install.sh | bash
set -e

echo ""
echo "  PTT — Push-to-Talk Transcription"
echo "  ================================="
echo ""

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "  Error: PTT requires Apple Silicon (M1/M2/M3/M4)."
    exit 1
fi

# Check/install uv
if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone or update
PTT_DIR="$HOME/.local/share/ptt"
if [ -d "$PTT_DIR/.git" ]; then
    echo "  Updating PTT..."
    git -C "$PTT_DIR" pull --quiet
else
    echo "  Downloading PTT..."
    mkdir -p "$(dirname "$PTT_DIR")"
    git clone --quiet https://github.com/leonideang/ptt.git "$PTT_DIR"
fi

# Create ptt command
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/ptt" << 'WRAPPER'
#!/bin/bash
UV_HTTP_TIMEOUT=300 exec uv run --script "$HOME/.local/share/ptt/ptt.py" "$@"
WRAPPER
chmod +x "$HOME/.local/bin/ptt"

# Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    SHELL_RC="$HOME/.zshrc"
    [ -f "$HOME/.bashrc" ] && [ ! -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.bashrc"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    export PATH="$HOME/.local/bin:$PATH"
    echo "  Added ~/.local/bin to PATH in $(basename "$SHELL_RC")"
fi

echo ""
echo "  Installed! First run downloads the AI model (~1.5 GB)."
echo ""
echo "  Start PTT:            ptt"
echo "  Auto-start on login:  ptt --install"
echo ""
echo "  Accessibility permission needed:"
echo "  System Settings → Privacy & Security → Accessibility"
echo "  → add Terminal (or your terminal app)"
echo ""
