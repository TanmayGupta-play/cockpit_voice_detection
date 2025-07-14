# inference/pipeline.py
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.vadinference import is_speech
from inference.triggerinference import predict_trigger
from inference.asrinference import transcribe_audio

def run_pipeline(trigger_audio_path):
    print(f"\n🎧 Processing Trigger Audio: {trigger_audio_path}")

    # 1. Voice Activity Detection
    speech_detected, speech_conf = is_speech(trigger_audio_path)
    print(f"🔊 Voice Activity Detected: {speech_detected} (Confidence: {speech_conf:.2f})")

    if not speech_detected:
        print("⛔ No speech detected. Exiting pipeline.")
        return

    # 2. Trigger Word Detection
    trigger_detected, trigger_conf = predict_trigger(trigger_audio_path)
    print(f"🚀 Trigger Detected: {bool(trigger_detected)} (Confidence: {trigger_conf:.2f})")

    if not trigger_detected:
        print("⛔ Trigger word not detected. Exiting pipeline.")
        return

    # 3. Ask for ASR command file
    command_audio = input("🎤 Trigger confirmed! Please enter path to the command .wav file: ").strip()

    if not os.path.exists(command_audio):
        print(f"❌ File not found: {command_audio}")
        return

    # 4. Transcribe Command
    transcript = transcribe_audio(command_audio)
    print(f"📝 Transcribed Command: {transcript}")

# Entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VAD → Trigger → Ask for ASR .wav pipeline")
    parser.add_argument("audio_path", help="Path to trigger word audio (.wav)")
    args = parser.parse_args()

    run_pipeline(args.audio_path)