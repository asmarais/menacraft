import os
import sys
import traceback
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.curdir))
from app.axes.authenticity import Authenticity


def run_test():
    load_dotenv()
    auth = Authenticity()

    # ── Simulated URL/article preprocessor output ─────────────────────────
    test_features = {
        "type": "url",

        # Scraped article body text
        "body_text": (
            "Le Président de la République par intérim, "
            "Sur proposition de la haute instance pour la réalisation des objectifs "
            "de la révolution, de la réforme politique et de la transition démocratique, "
            "Vu la loi organique n° 93-80 du 26 juillet 1993, relative à l'installation "
            "des organisations non gouvernementales en Tunisie, "
            "Vu la loi n° 59-154 du 7 novembre 1959, relative aux associations, "
            "Vu la loi n° 68-8 du 8 mars 1968, portant organisation de la cour des comptes, "
            "ensemble les textes qui l'ont modifié ou complété, "
            "Vu le décret-loi n° 2011-6 du 18 février 2011, portant création de la haute "
            "instance pour la réalisation des objectifs de la révolution, de la réforme "
            "politique et de la transition démocratique."
        ),
        "word_count": 110,

        # Burstiness from text features
        "burstiness_ratio":          0.65,
        "sentence_length_variance":  11.0,
        "avg_sentence_length":       21.0,

        # No featured image for this test
        "image_features": None,
    }

    print("🚀 Running URL/Article Forensic Analysis (no image)...")

    try:
        result = auth.evaluate(test_features)

        print("\n" + "━" * 80)
        print(f"   URL VERDICT: {result.get('label', 'UNKNOWN').upper()}")
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

            print("\n📑 ARTICLE STATS:")
            ds = d.get("article_stats", {})
            for k, v in ds.items():
                print(f"   • {k.ljust(22)} : {v}")

        print("\n" + "━" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_test()
