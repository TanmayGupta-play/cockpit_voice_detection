# inference/stream_pipeline.py
import os
import sys
import time
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.vadinference import is_speech
from inference.triggerinference import predict_trigger
from inference.asrinference import transcribe_audio

SAMPLE_RATE = 16000
RECORD_SECONDS = 1  # you can increase to 2 seconds if needed

def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    print("🎙 Listening...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    # Save to a temporary .wav file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(temp_file.name, sample_rate, (audio * 32767).astype(np.int16))
    return temp_file.name

def stream_pipeline():
    print("🔁 Starting real-time voice pipeline... Press Ctrl+C to stop.")

    try:
        while True:
            audio_path = record_audio()

            # Step 1: VAD
            vad_detected, vad_conf = is_speech(audio_path)
            if not vad_detected:
                print("🔇 No speech detected.")
                os.remove(audio_path)
                continue

            print(f"🗣 Speech detected (Confidence: {vad_conf:.2f})")

            # Step 2: Trigger Detection
            trigger_detected, trigger_conf = predict_trigger(audio_path)
            os.remove(audio_path)

            if not trigger_detected:
                print("❌ Trigger not detected.")
                continue

            print(f"✅ Trigger word detected! (Confidence: {trigger_conf:.2f})")

            # Step 3: Ask for ASR input audio path
            command_audio = input("🎤 Please provide path to command .wav file: ").strip()

            if not os.path.exists(command_audio):
                print(f"❌ File not found: {command_audio}")
                continue

            # Step 4: Transcribe
            transcript = transcribe_audio(command_audio)
            print(f"📝 Transcribed Command: {transcript}\n")
            print("🔁 Listening again...\n")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

# Entry point
if __name__ == "__main__":
    stream_pipeline()
