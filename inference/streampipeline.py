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
VAD_DURATION = 2    # seconds for VAD + Trigger detection
ASR_DURATION = 5    # seconds for command capture
CHANNELS = 1

VAD_THRESHOLD = 0.6       # tune this
TRIGGER_THRESHOLD = 0.5   # tune this

def record_audio(duration, filename=None):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()

    if filename:
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
        # Step 1: Record 2s audio for VAD + Trigger
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            vad_path = tmpfile.name
        record_audio(VAD_DURATION, filename=vad_path)

        # Step 2: VAD → Check if any human voice exists
        speech_detected, vad_conf = is_speech(vad_path)
        if not speech_detected or vad_conf < VAD_THRESHOLD:
            print(f"🔇 No speech detected. (Confidence: {vad_conf:.2f})")
            continue
        print(f"🗣️ Speech detected. (Confidence: {vad_conf:.2f})")

        # Step 3: Trigger Word Detection
        trigger_detected, trigger_conf = predict_trigger(vad_path)
        if not trigger_detected or trigger_conf < TRIGGER_THRESHOLD:
            print(f"❌ No trigger word detected. (Confidence: {trigger_conf:.2f})\n")
            continue

        print(f"\n🚀 Trigger detected! (Confidence: {trigger_conf:.2f})")
        print("🎤 Speak your command (5s)...")

        # Step 4: Record 5s audio for ASR
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as cmdfile:
            asr_path = cmdfile.name
        record_audio(ASR_DURATION, filename=asr_path)

        # Step 5: Transcribe
        transcript = transcribe_audio(asr_path)
        print(f"📝 Transcribed Command: {transcript}\n")

if __name__ == "__main__":
    main()