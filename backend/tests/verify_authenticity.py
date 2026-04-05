import os
import sys
import json
import traceback
from dotenv import load_dotenv

# Add search path for the app
sys.path.append(os.path.abspath(os.curdir))

# Import the Authenticity class
from app.axes.authenticity import Authenticity

def run_test():
    # Ensure environment variables are loaded
    load_dotenv()
    
    # 1. Initialize Authenticity handler
    auth = Authenticity()

    # 2. Define the test image
    test_features = {
        "type": "image",
        "path": "tests/test_image.png"
    }

    # Verify if the file exists locally
    if not os.path.exists(test_features["path"]):
        print(f"❌ Error: {test_features['path']} not found. Please provide a valid path.")
        return

    print(f"🚀 Running Deep Forensic Analysis for: {test_features['path']}...")

    try:
        # 3. Call evaluate
        result = auth.evaluate(test_features)

        # 4. Print beautiful report
        print("\n" + "━" * 80)
        print(f"   AUTHENTICITY VERDICT: {result.get('label', 'UNKNOWN').upper()}")
        print(f"   Weighted Conflict Score: {result.get('score', 0):.4f}")
        print("━" * 80)
        
        print(f"\n📢 AI FORENSIC ANALYST SUMMARY (LLM):")
        print(f"   {result.get('explanation')}")

        if result.get("flags"):
            print("\n🚩 FORENSIC FLAGS:")
            for flag in result["flags"]:
                print(f"   - {flag}")

        if "details" in result:
            details = result["details"]
            
            print("\n📊 SENSOR AXIS BREAKDOWN:")
            scores = details.get("component_scores", {})
            for name, score_val in scores.items():
                if isinstance(score_val, (int, float)):
                    display_score = f"{score_val:.4f}"
                else:
                    display_score = str(score_val)
                print(f"   • {name.ljust(15)} : {display_score}")
            
            weights = details.get("weights", {})
            print(f"   • Face Detection  : {'DETECTED' if weights.get('face_detected') else 'MINIMAL / NONE'}")


        print("\n" + "━" * 80)

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
