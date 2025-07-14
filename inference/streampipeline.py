# inference/stream_pipeline.py
import os
import sys
import torch
import sounddevice as sd
import numpy as np
import tempfile
import wave

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.vadinference import is_speech
from inference.triggerinference import predict_trigger
from inference.asrinference import transcribe_audio

SAMPLE_RATE = 16000
VAD_DURATION = 2     # seconds for VAD + Trigger detection
ASR_DURATION = 5     # seconds for command capture
CHANNELS = 1

def record_audio(duration, filename=None):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()

    if filename:
        # Save as WAV file
        audio_int16 = np.int16(audio * 32767)
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return filename
    else:
        return audio

def main():
    print("🎧 Listening for trigger word... (Ctrl+C to stop)\n")
    while True:
        # 1. Record 2s for VAD + trigger
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            vad_path = tmpfile.name
        record_audio(VAD_DURATION, filename=vad_path)

        # 2. VAD
        speech_detected, conf = is_speech(vad_path)
        if not speech_detected:
            print("🔇 No speech detected.")
            continue

        # 3. Trigger Word Detection
        trigger_detected, trigger_conf = predict_trigger(vad_path)
        if trigger_detected:
            print(f"\n🚀 Trigger detected! (Confidence: {trigger_conf:.2f})")
            print("🎤 Speak your command (5s)...")

            # 4. Record 5s for ASR
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as cmdfile:
                asr_path = cmdfile.name
            record_audio(ASR_DURATION, filename=asr_path)

            # 5. ASR Transcription
            transcript = transcribe_audio(asr_path)
            print(f"📝 Transcribed Command: {transcript}\n")
        else:
            print("❌ No trigger word detected.")

if __name__ == "__main__":
    main()