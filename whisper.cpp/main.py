import sounddevice as sd
import numpy as np
import keyboard
import subprocess
import os
import time
from scipy.io.wavfile import write
import tempfile

# ========================== CONFIGURE THESE ==========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WHISPER_CLI_PATH = os.path.join(SCRIPT_DIR, "build", "bin", "Release", "whisper-cli.exe")
MODEL_PATH       = os.path.join(SCRIPT_DIR,  "models", "ggml-base.en.bin")
AUDIO_FILE       = os.path.join(SCRIPT_DIR, "recording.wav")

# =====================================================================

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

def record_audio():
    print("Hold **SPACE** to speak... (release SPACE to stop)")

    keyboard.wait('space')
    print("🎙️  Recording... (release SPACE to stop)")

    frames = []
    start_time = time.time()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
            while keyboard.is_pressed('space'):
                data, overflowed = stream.read(CHUNK_SIZE)
                if overflowed:
                    print(" Audio overflow!")
                frames.append(data.copy())

                if time.time() - start_time > 60:
                    print(" Max recording time (60s) reached")
                    break
    except Exception as e:
        print(f"Recording error: {e}")
        return np.array([], dtype=np.int16)

    print(" Recording finished.")

    if not frames:
        print("No audio recorded.")
        return np.array([], dtype=np.int16)

    audio = np.concatenate(frames, axis=0).flatten()

    # Ignore very short recordings
    if len(audio) < SAMPLE_RATE * 0.3:
        print("Recording too short.")
        return np.array([], dtype=np.int16)

    return audio


def save_and_transcribe(audio):
    if len(audio) == 0:
        return "(no audio recorded)"

    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_wav_path = tmp.name

    # Save audio to temp file
    write(temp_wav_path, SAMPLE_RATE, audio)

    print("Running whisper-cli...")

    cmd = [
        WHISPER_CLI_PATH,
        "-m", MODEL_PATH,
        "-f", temp_wav_path,
        "-nt",
        "--print-progress", "false"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)

        # Clean up temp file
        os.remove(temp_wav_path)

        if result.returncode != 0:
            print(f"Whisper failed (code {result.returncode})")
            if result.stderr:
                print("Error:", result.stderr.strip())
            return "(transcription failed)"

        text = result.stdout.strip()
        return text if text else "(no speech detected)"

    except FileNotFoundError:
        print("❌ Could not find whisper-cli.exe - check WHISPER_CLI_PATH")
        return "(whisper-cli not found)"
    except Exception as e:
        print(f"Error running whisper: {e}")

        # Ensure temp file is deleted even on error
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

        return "(transcription error)"


def main():
    print("=== Simple Voice Transcription (using recording.wav) ===\n")
 

    # if not os.path.exists(WHISPER_CLI_PATH):
    #     print("ERROR: whisper-cli.exe not found!")
    #     print("Please update WHISPER_CLI_PATH at the top of the script.")
    #     return

    print("Hold SPACE to record → Release to transcribe (Ctrl+C to quit)\n")

    while True:
        try:
            audio = record_audio()
            text = save_and_transcribe(audio)

            print("\n🧠 Transcription:")
            print(text)
            print("-" * 70)
            print("Ready for next recording...\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()