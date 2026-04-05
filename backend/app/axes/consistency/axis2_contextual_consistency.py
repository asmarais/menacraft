"""
AXIS 2 — Contextual Consistency Detection
==========================================
Detects misleading context: caption-image mismatch,
claim vs article inconsistency, fake news classification.

Tools used (all free):
  - CLIP (OpenAI)              → image ↔ caption similarity (local, no limit)
    - BLIP                       → auto-generate caption from image (HuggingFace)
    - BART/Roberta NLI chain     → claim vs article entailment (HuggingFace)
  - sentence-transformers      → semantic text similarity (local, no limit)
    - Fake news model chain      → classify text as real/fake (HuggingFace)

Setup:
  pip install transformers torch sentence-transformers Pillow requests python-dotenv
  pip install git+https://github.com/openai/CLIP.git
"""

import os
import json
import re
from datetime import datetime
import requests
import torch
from PIL import Image
from sentence_transformers import util
from dotenv import load_dotenv

try:
    from .sentence_model_singleton import get_sentence_model
except ImportError:
    from .sentence_model_singleton import get_sentence_model

# Load environment variables from .env file and override stale shell values.
load_dotenv(override=True)

# ─────────────────────────────────────────────
# CONFIGURATION — API keys loaded from .env
# ─────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token")
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
HF_ROUTER_ALT_BASE = "https://router.huggingface.co/models"

MODEL_BLIP_CAPTION = "Salesforce/blip-image-captioning-base"
MODEL_NLI_PRIMARY = "facebook/bart-large-mnli"
MODEL_NLI_FALLBACK = "roberta-large-mnli"
MODEL_NLI_FALLBACK_2 = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL_FAKENEWS_PRIMARY = "Falconsai/fake_news_detection"
MODEL_FAKENEWS_FALLBACK = "jy46604790/Fake-News-Bert-Detect"

NLI_CHUNK_SIZE = 512
NLI_CHUNK_OVERLAP = 64
NLI_MAX_CHUNKS = 12


_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None


def _hf_url(model_name: str) -> str:
    return f"{HF_API_BASE}/{model_name}"


def _hf_router_urls_from_legacy(url: str) -> list[str]:
    legacy_prefix = f"{HF_API_BASE}/"
    if not url.startswith(legacy_prefix):
        return []

    model_name = url[len(legacy_prefix):]
    return [
        f"{HF_ROUTER_BASE}/{model_name}",
        f"{HF_ROUTER_ALT_BASE}/{model_name}",
    ]


def _auth_headers(extra_headers: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _load_clip():
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is None:
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
            ) from exc
        _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=_CLIP_DEVICE)
    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def _safe_post(
    url: str,
    tool_name: str,
    *,
    json_payload=None,
    data=None,
    extra_headers: dict | None = None,
    timeout: int = 45,
) -> tuple[object | None, dict | None]:
    def _post_once(target_url: str):
        return requests.post(
            target_url,
            headers=_auth_headers(extra_headers),
            json=json_payload,
            data=data,
            timeout=timeout,
        )

    try:
        response = _post_once(url)
    except requests.exceptions.RequestException as exc:
        return None, {
            "tool": tool_name,
            "error": "Network error while calling HuggingFace API",
            "details": str(exc),
        }

    if response.status_code == 410 and "no longer supported" in response.text.lower():
        router_response = None
        for router_url in _hf_router_urls_from_legacy(url):
            try:
                candidate = _post_once(router_url)
            except requests.exceptions.RequestException:
                continue
            if router_response is None:
                router_response = candidate
            if candidate.status_code < 400:
                response = candidate
                break
            if candidate.status_code in (401, 403):
                response = candidate
                break
        else:
            if router_response is not None:
                response = router_response

    if response.status_code >= 400:
        preview = response.text.strip().replace("\n", " ")[:240]
        return None, {
            "tool": tool_name,
            "error": "HTTP error from HuggingFace API",
            "status_code": response.status_code,
            "response_preview": preview,
        }

    try:
        return response.json(), None
    except (requests.exceptions.JSONDecodeError, ValueError):
        preview = response.text.strip().replace("\n", " ")[:240]
        return None, {
            "tool": tool_name,
            "error": "Non-JSON response from upstream API",
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "response_preview": preview,
        }


def _has_real_token(token_value: str, placeholder: str) -> bool:
    return bool(token_value and token_value.strip() and token_value.strip() != placeholder)

# ══════════════════════════════════════════════
# 2A. IMAGE ↔ CAPTION SIMILARITY — CLIP (local)
# ══════════════════════════════════════════════
def check_image_caption_clip(image_path: str, caption: str) -> dict:
    """
    Uses OpenAI CLIP to measure semantic similarity between an image and a caption.
    Score near 1.0 = strong match, near 0.0 = mismatch.
    Runs 100% locally — no API needed.

    Install: pip install git+https://github.com/openai/CLIP.git
    """
    try:
        import clip
        model, preprocess, device = _load_clip()
    except ImportError:
        return {"error": "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"}

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # Normalize and compute cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features  /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()

    # CLIP cosine similarity: >0.25 = good match, <0.15 = likely mismatch
    verdict = "CAPTION MATCHES IMAGE" if similarity > 0.20 else "CAPTION MISMATCH DETECTED"

    return {
        "tool": "CLIP (OpenAI)",
        "similarity_score": round(similarity, 4),
        "verdict": verdict,
        "caption": caption,
        "threshold_used": 0.20,
        "note": "Scores >0.20 suggest the caption matches the image content.",
    }


# ══════════════════════════════════════════════
# 2B. AUTO-CAPTION FROM IMAGE — BLIP (HuggingFace)
# ══════════════════════════════════════════════
def generate_caption_blip(image_path: str) -> dict:
    """
    Uses BLIP (Salesforce) via HuggingFace Inference API to auto-generate
    a caption for the image. Compare this to the provided caption to spot
    misleading descriptions.

    Free with HuggingFace token.
    """
    if not _has_real_token(HF_TOKEN, "your_hf_token"):
        return {
            "tool": "BLIP (Salesforce)",
            "error": "HF_TOKEN is missing or still set to placeholder value.",
        }

    if not os.path.exists(image_path):
        return {"tool": "BLIP (Salesforce)", "error": f"Image file not found: {image_path}"}

    with open(image_path, "rb") as f:
        image_data = f.read()

    result, err = _safe_post(
        _hf_url(MODEL_BLIP_CAPTION),
        tool_name="BLIP (Salesforce)",
        data=image_data,
        extra_headers={"Content-Type": "application/octet-stream"},
    )
    if err:
        return err

    generated_caption = ""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        generated_caption = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        generated_caption = result.get("generated_text", "")

    if generated_caption:
        return {
            "tool": "BLIP (Salesforce)",
            "auto_generated_caption": generated_caption,
            "usage": "Compare this caption to the human-provided one. Large differences suggest misleading context.",
        }
    else:
        return {"tool": "BLIP", "error": str(result)}


# ══════════════════════════════════════════════
# 2C. CAPTION vs AUTO-CAPTION TEXT SIMILARITY
# ══════════════════════════════════════════════
def compare_captions(provided_caption: str, auto_caption: str) -> dict:
    """
    Computes semantic similarity between the provided caption and the
    BLIP-generated caption using sentence-transformers (local, no API).
    Score near 1.0 = captions agree, near 0.0 = they describe different things.

    Install: pip install sentence-transformers
    """
    model = get_sentence_model()
    emb1 = model.encode(provided_caption, convert_to_tensor=True)
    emb2 = model.encode(auto_caption, convert_to_tensor=True)
    score = float(util.cos_sim(emb1, emb2))

    verdict = "CAPTIONS CONSISTENT" if score > 0.55 else "CAPTIONS INCONSISTENT — possible misleading context"

    return {
        "tool": "sentence-transformers (all-MiniLM-L6-v2)",
        "provided_caption": provided_caption,
        "auto_caption": auto_caption,
        "similarity_score": round(score, 4),
        "verdict": verdict,
    }


# ══════════════════════════════════════════════
# 2D. CLAIM vs ARTICLE CONSISTENCY — NLI DeBERTa
# ══════════════════════════════════════════════
def check_claim_consistency_nli(claim: str, article_text: str) -> dict:
    """
    Natural Language Inference with automatic model fallback.

    Primary model : facebook/bart-large-mnli
    Fallback model: roberta-large-mnli

    Aggregation rule: if any chunk contradicts, final verdict is contradiction.
    """
    if not _has_real_token(HF_TOKEN, "your_hf_token"):
        return {
            "tool": "NLI",
            "error": "HF_TOKEN is missing or still set to placeholder value.",
        }

    chunks = _chunk_text_for_nli(article_text)
    if not chunks:
        return {
            "tool": "NLI",
            "error": "Article text is empty after normalization.",
        }

    last_result: dict | None = None
    for model_name in (MODEL_NLI_PRIMARY, MODEL_NLI_FALLBACK, MODEL_NLI_FALLBACK_2):
        current = _run_nli_on_chunks(model_name=model_name, claim=claim, chunks=chunks)
        if "error" not in current:
            return current
        last_result = current

    return last_result or {
        "tool": "NLI",
        "error": "NLI fallback chain failed without a detailed error.",
    }


def _run_nli_on_chunks(model_name: str, claim: str, chunks: list[str]) -> dict:
    aggregated_scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    chunk_labels: list[str] = []
    contradiction_indices: list[int] = []
    chunk_results: list[dict] = []
    errors: list[dict] = []

    for idx, chunk in enumerate(chunks):
        payload = {"inputs": {"text": chunk, "text_pair": claim}}
        result, err = _safe_post(
            _hf_url(model_name),
            tool_name=f"NLI ({model_name})",
            json_payload=payload,
        )
        if err and _is_zero_shot_payload_error(err):
            zero_shot_payload = {
                "inputs": f"Premise: {chunk}\nHypothesis: {claim}",
                "parameters": {
                    "candidate_labels": ["entailment", "neutral", "contradiction"],
                    "multi_label": False,
                },
            }
            result, err = _safe_post(
                _hf_url(model_name),
                tool_name=f"NLI ({model_name})",
                json_payload=zero_shot_payload,
            )

        if err:
            err["chunk_index"] = idx
            errors.append(err)
            continue

        normalized_scores = _extract_nli_scores(result)
        for label, score in normalized_scores.items():
            aggregated_scores[label] = max(aggregated_scores[label], score)

        if not normalized_scores:
            errors.append(
                {
                    "chunk_index": idx,
                    "error": "No valid NLI labels found in response",
                }
            )
            continue

        top_label = max(normalized_scores, key=normalized_scores.get)
        chunk_labels.append(top_label)
        if top_label == "contradiction":
            contradiction_indices.append(idx)

        chunk_results.append(
            {
                "chunk_index": idx,
                "nli_label": top_label,
                "scores": {k: round(v, 3) for k, v in normalized_scores.items()},
                "excerpt": chunk[:140],
            }
        )

    if not chunk_results:
        return {
            "tool": f"NLI ({model_name})",
            "error": "All chunked NLI requests failed",
            "chunk_count": len(chunks),
            "errors": errors,
        }

    if contradiction_indices:
        final_label = "contradiction"
    elif "entailment" in chunk_labels:
        final_label = "entailment"
    else:
        final_label = "neutral"

    verdict_map = {
        "entailment": "CONSISTENT - Article supports the claim",
        "contradiction": "INCONSISTENT - Article contradicts the claim",
        "neutral": "NEUTRAL - Article does not clearly confirm or deny the claim",
    }

    return {
        "tool": f"NLI ({model_name})",
        "claim": claim,
        "nli_label": final_label,
        "verdict": verdict_map.get(final_label, final_label),
        "scores": {k: round(v, 3) for k, v in aggregated_scores.items()},
        "chunk_count": len(chunks),
        "processed_chunks": len(chunk_results),
        "contradiction_chunk_count": len(contradiction_indices),
        "contradiction_chunk_indices": contradiction_indices,
        "chunk_results": chunk_results,
        "errors": errors,
    }


def _is_zero_shot_payload_error(err: dict) -> bool:
    message = str(err.get("response_preview", "")).lower()
    return "zero-shot-classification expects" in message


def _extract_nli_scores(result: object) -> dict[str, float]:
    scores: dict[str, float] = {}

    if isinstance(result, list):
        # Standard text-classification response shape.
        for item in result:
            if not isinstance(item, dict):
                continue
            label = _normalize_nli_label(str(item.get("label", "")))
            if label not in {"entailment", "neutral", "contradiction"}:
                continue
            scores[label] = float(item.get("score", 0.0))
        return scores

    if isinstance(result, dict):
        # Zero-shot response shape with parallel labels/scores arrays.
        labels = result.get("labels")
        values = result.get("scores")
        if isinstance(labels, list) and isinstance(values, list):
            for label, value in zip(labels, values):
                normalized = _normalize_nli_label(str(label))
                if normalized not in {"entailment", "neutral", "contradiction"}:
                    continue
                try:
                    scores[normalized] = float(value)
                except (TypeError, ValueError):
                    continue
    return scores


def _chunk_text_for_nli(
    text: str,
    chunk_size: int = NLI_CHUNK_SIZE,
    overlap: int = NLI_CHUNK_OVERLAP,
    max_chunks: int = NLI_MAX_CHUNKS,
) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(normalized), step):
        chunk = normalized[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if len(chunks) >= max_chunks or start + chunk_size >= len(normalized):
            break
    return chunks


def _normalize_nli_label(label: str) -> str:
    lowered = label.strip().lower()

    # Common textual variants
    if "contrad" in lowered:
        return "contradiction"
    if "entail" in lowered or "consistent" in lowered or "support" in lowered:
        return "entailment"
    if "neutral" in lowered or "unrelated" in lowered or "uncertain" in lowered:
        return "neutral"

    # Common numeric label variants used by some NLI models
    if lowered in {"label_0", "0", "class_0"}:
        return "contradiction"
    if lowered in {"label_1", "1", "class_1"}:
        return "neutral"
    if lowered in {"label_2", "2", "class_2"}:
        return "entailment"

    return lowered


def assess_reverse_image_reuse(reverse_image_hits: list[dict], claim: str, article_text: str = "") -> dict:
    """
    Consumes reverse image search hits from any provider and derives reuse/context mismatch signals.

    Expected hit fields (best effort):
      - published_at or original_publication_date
      - publisher or source
      - context, title, or snippet
      - url
    """
    if not reverse_image_hits:
        return {
            "tool": "Reverse Image Evidence",
            "error": "No reverse image hits provided.",
        }

    normalized_hits: list[dict] = []
    for hit in reverse_image_hits:
        if not isinstance(hit, dict):
            continue
        published_at = str(hit.get("published_at") or hit.get("original_publication_date") or "").strip()
        publisher = str(hit.get("publisher") or hit.get("source") or "unknown publisher").strip()
        context = str(hit.get("context") or hit.get("title") or hit.get("snippet") or "").strip()
        url = str(hit.get("url") or "").strip()
        parsed_date = _parse_date(published_at)
        normalized_hits.append(
            {
                "published_at": published_at,
                "parsed_date": parsed_date,
                "publisher": publisher,
                "context": context,
                "url": url,
            }
        )

    if not normalized_hits:
        return {
            "tool": "Reverse Image Evidence",
            "error": "Reverse image hits are not in expected format.",
        }

    dated_hits = [h for h in normalized_hits if h["parsed_date"] is not None]
    if dated_hits:
        origin = min(dated_hits, key=lambda h: h["parsed_date"])
    else:
        origin = normalized_hits[0]

    claimed_context = " ".join(
        part for part in [claim.strip(), article_text.strip()[:900]] if part
    ).strip()
    original_context = origin.get("context", "")

    similarity = None
    mismatch_score = 0.0
    if claimed_context and original_context:
        model = get_sentence_model()
        emb_claimed = model.encode(claimed_context, convert_to_tensor=True)
        emb_origin = model.encode(original_context, convert_to_tensor=True)
        similarity = float(util.cos_sim(emb_claimed, emb_origin))
        mismatch_score = max(0.0, min(1.0, (0.55 - similarity) / 0.55))

    claim_year = _extract_year(claim)
    origin_year = origin["parsed_date"].year if origin.get("parsed_date") else None
    date_conflict = bool(claim_year and origin_year and claim_year != origin_year)
    context_conflict = similarity is not None and similarity < 0.45
    reused_likely = date_conflict or context_conflict or len(normalized_hits) >= 2

    return {
        "tool": "Reverse Image Evidence",
        "verdict": "CONTENT LIKELY REUSED FROM DIFFERENT EVENT" if reused_likely else "NO STRONG REUSE SIGNAL",
        "reused_likely": reused_likely,
        "original_publication_date": origin.get("published_at") or "unknown",
        "original_publisher": origin.get("publisher", "unknown publisher"),
        "original_context": original_context,
        "top_match_url": origin.get("url", ""),
        "context_similarity": round(similarity, 4) if similarity is not None else None,
        "context_mismatch_score": round(mismatch_score, 4),
        "date_conflict": date_conflict,
        "claim_year": claim_year,
        "origin_year": origin_year,
        "match_count": len(normalized_hits),
    }


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None

    candidates = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m", "%Y"]
    for fmt in candidates:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _extract_year(text: str) -> int | None:
    match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text or "")
    if match:
        return int(match.group(1))
    return None


# ══════════════════════════════════════════════
# 2E. FAKE NEWS CLASSIFICATION — HuggingFace
# ══════════════════════════════════════════════
def classify_fake_news(text: str) -> dict:
    """
    Classifies text as fake/real using verified model chain.

    Primary  : Falconsai/fake_news_detection
    Fallback : jy46604790/Fake-News-Bert-Detect
    """
    if not _has_real_token(HF_TOKEN, "your_hf_token"):
        return {
            "tool": "Fake News Classifier",
            "error": "HF_TOKEN is missing or still set to placeholder value.",
        }

    last_result: dict | None = None
    for model_name in (MODEL_FAKENEWS_PRIMARY, MODEL_FAKENEWS_FALLBACK):
        current = _run_fake_news_model(model_name=model_name, text=text)
        if "error" not in current:
            return current
        last_result = current

    return last_result or {
        "tool": "Fake News Classifier",
        "error": "Fake-news fallback chain failed without a detailed error.",
    }


def _run_fake_news_model(model_name: str, text: str) -> dict:
    result, err = _safe_post(
        _hf_url(model_name),
        tool_name=f"Fake News Classifier ({model_name})",
        json_payload={"inputs": text[:512]},
    )
    if err:
        return err

    if isinstance(result, list) and result and isinstance(result[0], list):
        scores = result[0]
    elif isinstance(result, list):
        scores = result
    else:
        return {
            "tool": f"Fake News Classifier ({model_name})",
            "error": f"Unexpected response format: {str(result)[:200]}",
        }

    label_map = {}
    for item in scores:
        if isinstance(item, dict):
            label_map[str(item.get("label", "")).upper()] = float(item.get("score", 0.0))

    fake_score = (
        label_map.get("FAKE")
        or label_map.get("LABEL_0")
        or label_map.get("FALSE")
        or 0.0
    )
    real_score = (
        label_map.get("REAL")
        or label_map.get("LABEL_1")
        or label_map.get("TRUE")
        or 0.0
    )

    verdict = "FAKE NEWS DETECTED" if fake_score > 0.5 else "LIKELY REAL"
    return {
        "tool": f"Fake News Classifier ({model_name})",
        "fake_score": round(fake_score, 3),
        "real_score": round(real_score, 3),
        "verdict": verdict,
        "confidence": f"{max(fake_score, real_score) * 100:.1f}%",
    }


# ══════════════════════════════════════════════
# COMBINED RUNNER — Axis 2
# ══════════════════════════════════════════════
def run_axis2(
    image_path: str = None,
    provided_caption: str = None,
    claim: str = None,
    article_text: str = None,
    reverse_image_hits: list[dict] | None = None,
) -> dict:
    """
    Runs all Axis 2 checks.

    Args:
      image_path:        path to local image file
      provided_caption:  the caption/description provided with the image
      claim:             a specific claim to verify against article_text
      article_text:      the article or post body text
    reverse_image_hits: list of reverse-image matches from any search provider

    Returns:
      dict with all results and verdicts
    """
    results = {}

    # Image + caption checks
    if image_path and provided_caption:
        print(f"\n[Axis 2] Checking caption-image consistency...")

        print("  → CLIP similarity (local)...")
        results["clip_similarity"] = check_image_caption_clip(image_path, provided_caption)

        if _has_real_token(HF_TOKEN, "your_hf_token"):
            print("  → Generating auto-caption with BLIP...")
            blip_result = generate_caption_blip(image_path)
            results["blip_caption"] = blip_result

            auto_cap = blip_result.get("auto_generated_caption", "")
            if auto_cap:
                print("  → Comparing captions semantically...")
                results["caption_comparison"] = compare_captions(provided_caption, auto_cap)

    if reverse_image_hits:
        print("  → Evaluating reverse image search evidence...")
        results["reverse_image_search"] = assess_reverse_image_reuse(
            reverse_image_hits,
            claim or "",
            article_text or "",
        )

    # Claim vs article NLI
    if claim and article_text:
        print(f"\n[Axis 2] Checking claim consistency against article...")
        if _has_real_token(HF_TOKEN, "your_hf_token"):
            print("  → NLI entailment check (BART/Roberta fallback chain)...")
            results["claim_nli"] = check_claim_consistency_nli(claim, article_text)

    # Fake news classification
    if article_text:
        print(f"\n[Axis 2] Classifying article as real/fake...")
        if _has_real_token(HF_TOKEN, "your_hf_token"):
            print("  → Fake news classifier (primary/fallback chain)...")
            results["fake_news"] = classify_fake_news(article_text)

    print("\n── AXIS 2 RESULTS ──")
    print(json.dumps(results, indent=2))
    return results


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_axis2(
        image_path="test_image.jpg",
        provided_caption="Thousands march in Paris climate protest, 2024",
        claim="The protest took place in Paris in 2024.",
        article_text="""
            Hundreds of activists gathered in front of the Eiffel Tower on Saturday
            to demand stronger climate action from the French government.
            Organizers estimated over 5,000 participants attended the rally in central Paris.
            The event is part of a growing global movement demanding net-zero emissions by 2035.
        """,
    )
