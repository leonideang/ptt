class Ptt < Formula
  desc "Push-to-Talk transcription for macOS — hold a key, speak, release"
  homepage "https://github.com/leonideang/ptt"
  url "https://github.com/leonideang/ptt.git", branch: "main"
  version "1.0.0"
  license "MIT"

  depends_on :macos
  depends_on :arch => :arm64

  resource "uv" do
    # uv is installed separately — we just check for it
  end

  def install
    libexec.install "ptt.py"

    (bin/"ptt").write <<~SH
      #!/bin/bash
      # PTT — Push-to-Talk Transcription
      # Requires: uv (brew install uv)
      if ! command -v uv &> /dev/null; then
        echo "Error: uv is required. Install with: brew install uv"
        exit 1
      fi
      UV_HTTP_TIMEOUT=300 exec uv run --script "#{libexec}/ptt.py" "$@"
    SH
  end

  def caveats
    <<~EOS
      PTT requires:
        1. Apple Silicon (M1/M2/M3/M4)
        2. uv — install with: brew install uv
        3. Accessibility permission for your terminal app:
           System Settings → Privacy & Security → Accessibility

      First launch downloads the Whisper model (~1.5 GB).

      Start PTT:
        ptt

      Auto-start on login:
        ptt --install

      Text polish (optional):
        Hold ⌘ + hotkey to clean up spoken text via Groq.
        Get a free API key at https://console.groq.com
        Add it via Settings or:
          security add-generic-password -a groq -s ptt -w "gsk_your_key"
    EOS
  end

  test do
    assert_match "PTT", shell_output("#{bin}/ptt --help 2>&1", 0)
  end
end
