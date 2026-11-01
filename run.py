#!/usr/bin/env python3
"""VoiceOps Official Run Command — Batch Voice & Text Support Evaluation.

Accepts a directory of customer audio files (.wav), runs the complete VoiceOps
agent pipeline on each, and outputs:
- Transcribed customer query (via Gnani STT)
- Grounded agent reply text (via Step 3 RAG & Tools)
- Synthesized agent spoken reply audio (via Gnani TTS)
- Document chunks / section citations used

Usage:
    python run.py --input-dir <audio_folder> --output-dir <results_folder>
    python run.py --interactive   # Live conversational text-and-speech console
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.voice_agent import VoiceOpsAgent


def process_audio_batch(input_dir: str, output_dir: str):
    """Batch process a directory of customer .wav audio clips."""
    os.makedirs(output_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))

    if not audio_files:
        print(f"⚠️ No .wav audio files found in: {input_dir}")
        return

    print(f"[INFO] Discovered {len(audio_files)} audio file(s) in {input_dir}")
    print(f"[INFO] Destination output directory: {output_dir}\n")

    agent = VoiceOpsAgent()
    batch_results = []

    for idx, audio_path in enumerate(audio_files):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_wav = os.path.join(output_dir, f"{base_name}_reply.wav")

        print(f"[{idx+1}/{len(audio_files)}] Processing {os.path.basename(audio_path)}...")
        result = agent.voice_turn(
            audio_path=audio_path,
            output_wav_path=out_wav,
            language_code="en-IN",
        )

        result["input_audio"] = audio_path
        batch_results.append(result)

        print(f"  Transcript : \"{result['transcript']}\"")
        print(f"  Response   : \"{result['reply_text'][:120]}...\"")
        print(f"  Citations  : {result['retrieved_sources']}")
        print(f"  Audio File : {out_wav}\n")

    summary_file = os.path.join(output_dir, "batch_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Batch processing completed. Summary stored in {summary_file}")


def interactive_mode():
    """Interactive console for testing text and synthesizing speech replies."""
    print("=" * 65)
    print("📞 ZentroMart AI Voice Support — Interactive Console")
    print("=" * 65)
    print("Type your message to test agent behavior. (Type 'exit' to quit)\n")

    agent = VoiceOpsAgent()
    turn = 1

    while True:
        try:
            msg = input(f"Customer Turn {turn} > ").strip()
            if not msg:
                continue
            if msg.lower() in ("exit", "quit"):
                print("Ending session. Goodbye!")
                break

            reply, tools = agent.agent.respond(msg)
            print(f"\nAI Agent > {reply}")
            if tools:
                print(f"  [Tools Executed: {', '.join(tools)}]")

            # Synthesize speech output
            out_file = f"sample_outputs/interactive_turn_{turn}.wav"
            os.makedirs("sample_outputs", exist_ok=True)
            spoken = agent.tts.synthesize(
                text=reply,
                voice="Kaveri",
                model="timbre-v2.5",
                language="hi-IN" if any(ch in reply for ch in "अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह") else "en-IN",
                output_file=out_file,
            )
            print(f"  [Audio synthesized to {out_file}]\n")
            turn += 1

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="VoiceOps Agent 2026 Batch Runner")
    parser.add_argument("--input-dir", type=str, help="Directory containing customer input .wav audio files")
    parser.add_argument("--output-dir", type=str, default="sample_outputs", help="Directory to save agent replies and audio")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive text/speech console")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.input_dir:
        process_audio_batch(args.input_dir, args.output_dir)
    else:
        print("[INFO] Use --input-dir <dir> for batch audio processing, or --interactive for live console mode.")
        interactive_mode()


if __name__ == "__main__":
    main()

