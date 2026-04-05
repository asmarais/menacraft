import os
import sys
import traceback
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.curdir))
from app.axes.authenticity import Authenticity


def run_test():
    load_dotenv()
    auth = Authenticity()

    # ── Real Tunisian legal decree text ─────────────────────────────────────
    test_features = {
        "type": "document",

        # Signal 1 input: clean_text for RoBERTa
        "clean_text": (
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
            "politique et de la transition démocratique, "
            "Vu le décret-loi n° 2011-14 du 23 mars 2011, portant organisation provisoire "
            "des pouvoirs publics, "
            "Vu le décret n° 70-118 du 11 avril 1970, portant organisation des services "
            "du Premier ministère, ensemble les textes qui l'ont modifié ou complété, "
            "Vu la délibération du conseil des ministres, "
            "Prend le décret-loi dont la teneur suit :"
        ),
        "word_count": 148,

        # Signal 2 inputs: burstiness analysis
        "burstiness_ratio":          0.72,   # high variance → human legal writing
        "sentence_length_variance":  12.3,   # high variance → natural
        "avg_sentence_length":       22.0,   # long legal sentences

        # Signal 3 inputs: metadata & layout anomalies
        "metadata_anomalies":  [],
        "layout_anomalies":    [],
        "font_consistency_score": 0.9,  # consistent PDF

        # Raw metadata dict
        "metadata": {
            "author":            "République Tunisienne",
            "creation_date":     "2011-09-24",
            "modification_date": "2011-09-24",
            "Producer":          "Microsoft Word",
        },
    }

    print("🚀 Running 3-Signal Document Forensic Analysis...")

    try:
        result = auth.evaluate(test_features)

        print("\n" + "━" * 80)
        print(f"   DOCUMENT VERDICT: {result.get('label', 'UNKNOWN').upper()}")
        print(f"   Weighted Score:   {result.get('score', 0):.4f}")
        print("━" * 80)

        print(f"\n📢 AI FORENSIC ANALYST (LLM):")
        print(f"   {result.get('explanation')}")

        if result.get("flags"):
            print("\n🚩 FORENSIC FLAGS:")
            for flag in result["flags"]:
                print(f"   - {flag}")

        if "details" in result:
            d = result["details"]

            print("\n📊 SIGNAL AXIS BREAKDOWN:")
            for name, val in d.get("component_scores", {}).items():
                print(f"   • {name.ljust(20)} : {val:.4f}")

            print(f"\n⚖️  WEIGHTS USED:")
            for wn, wv in d.get("weights", {}).items():
                print(f"   • {wn.ljust(20)} : {wv}")

            sd = d.get("signal_details", {})
            print("\n📡 SIGNAL 1 — RoBERTa AI Text:")
            rob = sd.get("roberta", {})
            print(f"   • Score  : {rob.get('ai_generated_score', 'N/A')}")
            print(f"   • Verdict: {rob.get('verdict', 'N/A')}")
            if rob.get("error"):
                print(f"   • Error  : {rob['error']}")

            print("\n📡 SIGNAL 2 — Burstiness:")
            bst = sd.get("burstiness", {})
            print(f"   • AI likelihood : {bst.get('ai_likelihood', 'N/A')}")
            print(f"   • Burst AI      : {bst.get('burst_ai', 'N/A')}")
            print(f"   • Variance AI   : {bst.get('variance_ai', 'N/A')}")
            print(f"   • Length AI     : {bst.get('length_ai', 'N/A')}")
            print(f"   • Raw burstiness: {bst.get('raw_burstiness', 'N/A')}")

            print("\n📡 SIGNAL 3 — Metadata Anomalies:")
            mta = sd.get("metadata", {})
            print(f"   • Risk score    : {mta.get('risk_score', 'N/A')}")
            print(f"   • Rules fired   : {mta.get('triggered_rules', [])}")

            print("\n📑 DOCUMENT STATS:")
            ds = d.get("document_stats", {})
            for k, v in ds.items():
                print(f"   • {k.ljust(20)} : {v}")

        print("\n" + "━" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_test()
