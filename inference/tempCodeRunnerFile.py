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