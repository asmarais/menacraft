import os
import sys
import traceback
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.curdir))
from app.axes.authenticity import Authenticity


def run_test():
    load_dotenv()
    auth = Authenticity()

    # ── Simulated Video preprocessor output ─────────────────────────────
    # Case: A suspicious video with a face detected, Reality Defender flags it.
    test_features = {
        "type": "video",
        "video_path": "tests/sample_deepfake.mp4",
        "frames_count": 300,
        "face_detected": True,
        "codec_risk": 0.2, # Clean codec but deepfake signals present
    }

    print("🚀 Running Video Forensic Analysis (Hybrid Spatial-Frequency)...")

    try:
        # Mocking the Reality Defender score for this test since we don't have a real video / active key.
        # In the real code, detect_video_deepfake_rd would be called.
        # For simulation, we'll let it use the baseline for now or we could patch it.
        
        result = auth.evaluate(test_features)

        print("\n" + "━" * 80)
        print(f"   VIDEO VERDICT: {result.get('label', 'UNKNOWN').upper()}")
        print(f"   Weighted Score: {result.get('score', 0):.4f}")
        print("━" * 80)

        print(f"\n📢 AI FORENSIC ANALYST (LLM):")
        print(f"   {result.get('explanation')}")

        if result.get("flags"):
            print("\n🚩 FLAGS:")
            for flag in result["flags"]:
                print(f"   - {flag}")

        if "details" in result:
            d = result["details"]

            print("\n📊 SIGNAL BREAKDOWN:")
            for name, val in d.get("component_scores", {}).items():
                print(f"   • {name.ljust(22)} : {val:.4f}")

            print(f"\n⚖️  WEIGHTS:")
            for wn, wv in d.get("weights", {}).items():
                print(f"   • {wn.ljust(22)} : {wv}")

        print("\n" + "━" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_test()
