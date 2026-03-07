#!/bin/bash
# PTT — Push-to-Talk Transcription
# Usage: ./run.sh [--lang en|sv] [--model MODEL] [--key alt_r|alt]
cd "$(dirname "$0")"
UV_HTTP_TIMEOUT=300 exec uv run ptt.py "$@"
