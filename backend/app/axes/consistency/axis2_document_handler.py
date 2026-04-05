# axis2_document_handler.py
import asyncio
import html
import os
import re
import tempfile
from dataclasses import dataclass

import numpy as np
import requests

try:
    from .burstiness_analyzer import DocumentBurstinessAnalyzer
    from .fact_check_client import FactCheckResult, GoogleFactCheckClient
    from .axis2_contextual_consistency import run_axis2
    from .sentence_model_singleton import get_sentence_model
except ImportError:
    # If already inside a package where these aren't found directly, 
    # the second try-except block in some versions of this code might have handled it, 
    # but here we just ensure clean relative imports.
    from .burstiness_analyzer import DocumentBurstinessAnalyzer
    from .fact_check_client import FactCheckResult, GoogleFactCheckClient
    from .axis2_contextual_consistency import run_axis2
    from .sentence_model_singleton import get_sentence_model


@dataclass
class Axis2DocumentResult:
    score: float             # 0.0 (consistent) → 1.0 (inconsistent)
    label: str
    color: str
    confidence: float
    explanation: dict
    flags: list[str]
    breakdown: dict


@dataclass
class Axis2FusionResult:
    fused_score: float
    consistency_confidence: float
    fused_label: str
    fused_color: str
    short_explanation: str
    document_result: Axis2DocumentResult
    multimodal_penalty: float
    multimodal_results: dict
    reference_comparison: dict | None
    source_context: dict | None
    flags: list[str]


async def analyze_context_document(
    document_features: dict,
    claim_text: str,
    language_code: str = "en",
    burstiness_analyzer: DocumentBurstinessAnalyzer | None = None,
    fact_check_client: GoogleFactCheckClient | None = None,
) -> Axis2DocumentResult:
    """
    Runs all 3 context signals for document input in parallel.
    """

    context = document_features.get("context_features", {})
    clean_text = context.get("clean_text", "")
    keywords = context.get("keywords", [])
    entities = context.get("entities", [])
    document_embedding = context.get("document_embedding")

    if document_embedding is None:
        raise ValueError("document_features.context_features.document_embedding is required")

    burstiness_analyzer = burstiness_analyzer or DocumentBurstinessAnalyzer()

    # ── Run signals in parallel ──────────────────────────────────
    burstiness_task = asyncio.to_thread(
        burstiness_analyzer.analyze,
        clean_text
    )

    fact_check_task = _run_fact_check(
        fact_check_client=fact_check_client,
        claim_text=claim_text,
        keywords=keywords,
        entities=entities,
        language_code=language_code,
    )

    burstiness_result, fact_check_result = await asyncio.gather(
        burstiness_task,
        fact_check_task
    )

    # ── Headline vs body score (cosine via embeddings) ───────────
    headline_body_score = _compute_headline_body_score(
        claim_text,
        document_embedding
    )

    # ── Fuse scores ──────────────────────────────────────────────
    inconsistency_score = (
        (1 - headline_body_score)          * 0.25 +
        burstiness_result.ai_likelihood_score * 0.30 +
        fact_check_result.credibility_penalty * 0.45
    )

    # ── Label ────────────────────────────────────────────────────
    if inconsistency_score >= 0.70:
        label, color = "HIGH_RISK",   "red"
    elif inconsistency_score >= 0.40:
        label, color = "SUSPICIOUS",  "orange"
    else:
        label, color = "CONSISTENT",  "green"

    all_flags = (
        burstiness_result.flags +
        fact_check_result.flags +
        (["headline_body_mismatch"] if headline_body_score < 0.40 else [])
    )

    return Axis2DocumentResult(
        score       = round(inconsistency_score, 4),
        label       = label,
        color       = color,
        confidence  = round(inconsistency_score, 4),
        explanation = {
            "summary"    : _build_summary(label, fact_check_result, burstiness_result),
            "details"    : _build_details(headline_body_score, burstiness_result, fact_check_result),
            "user_advice": _build_advice(label)
        },
        flags       = all_flags,
        breakdown   = {
            "headline_body_alignment" : round(headline_body_score, 4),
            "ai_writing_likelihood"   : burstiness_result.ai_likelihood_score,
            "fact_check_penalty"      : fact_check_result.credibility_penalty,
            "fact_check_verdict"      : fact_check_result.verdict,
        }
    )


async def analyze_axis2_fused(
    document_features: dict,
    claim_text: str,
    language_code: str = "en",
    image_path: str | None = None,
    provided_caption: str | None = None,
    article_text: str | None = None,
    known_references: list[dict] | None = None,
    reverse_image_hits: list[dict] | None = None,
    run_multimodal: bool = True,
    burstiness_analyzer: DocumentBurstinessAnalyzer | None = None,
    fact_check_client: GoogleFactCheckClient | None = None,
) -> Axis2FusionResult:
    """
    Fuses document-context signals with multimodal Axis 2 checks.

    The fused score keeps document consistency as the primary signal (75%)
    and adds multimodal mismatch penalty (25%).
    """
    document_result = await analyze_context_document(
        document_features=document_features,
        claim_text=claim_text,
        language_code=language_code,
        burstiness_analyzer=burstiness_analyzer,
        fact_check_client=fact_check_client,
    )

    multimodal_results: dict = {}
    multimodal_penalty = 0.0
    multimodal_flags: list[str] = []
    reference_comparison: dict | None = None

    if known_references:
        reference_comparison = _compare_against_references(claim_text, article_text or "", known_references)

    if run_multimodal and (image_path or article_text):
        multimodal_results = await asyncio.to_thread(
            run_axis2,
            image_path=image_path,
            provided_caption=provided_caption,
            claim=claim_text,
            article_text=article_text,
            reverse_image_hits=reverse_image_hits,
        )
        multimodal_penalty, multimodal_flags = _compute_multimodal_penalty(multimodal_results)

    fused_score = min(1.0, (document_result.score * 0.75) + (multimodal_penalty * 0.25))
    consistency_confidence = max(0.0, 1.0 - fused_score)
    flags = list(dict.fromkeys(document_result.flags + multimodal_flags))

    fused_label, fused_color = _label_from_score(fused_score)
    if "caption_image_mismatch" in flags and fused_score >= 0.55:
        fused_label, fused_color = "HIGH_RISK", "red"

    short_explanation = _build_short_explanation(flags, multimodal_results, document_result)

    return Axis2FusionResult(
        fused_score=round(fused_score, 4),
        consistency_confidence=round(consistency_confidence, 4),
        fused_label=fused_label,
        fused_color=fused_color,
        short_explanation=short_explanation,
        document_result=document_result,
        multimodal_penalty=round(multimodal_penalty, 4),
        multimodal_results=multimodal_results,
        reference_comparison=reference_comparison,
        source_context=None,
        flags=flags,
    )


async def analyze_axis2_from_url(
    url: str,
    claim_text: str,
    language_code: str = "en",
    provided_caption: str | None = None,
    known_references: list[dict] | None = None,
    reverse_image_hits: list[dict] | None = None,
    burstiness_analyzer: DocumentBurstinessAnalyzer | None = None,
    fact_check_client: GoogleFactCheckClient | None = None,
) -> Axis2FusionResult:
    """
    Fetches article content from URL, extracts context, and runs fused Axis 2 analysis.

    This is useful for web/news links where claim-context mismatch must be detected.
    """
    source_context = _scrape_url_context(url)
    article_text = source_context.get("article_text", "")

    if not article_text:
        raise ValueError("Unable to extract article text from URL.")

    caption = provided_caption or source_context.get("description") or source_context.get("title")
    document_features = _build_document_features_from_text(article_text)

    image_path = None
    temp_image = None
    image_url = source_context.get("image_url")
    if image_url:
        temp_image = _download_image_to_temp(image_url)
        image_path = temp_image

    try:
        result = await analyze_axis2_fused(
            document_features=document_features,
            claim_text=claim_text,
            language_code=language_code,
            image_path=image_path,
            provided_caption=caption,
            article_text=article_text,
            known_references=known_references,
            reverse_image_hits=reverse_image_hits,
            run_multimodal=True,
            burstiness_analyzer=burstiness_analyzer,
            fact_check_client=fact_check_client,
        )
        result.source_context = {
            "url": url,
            "title": source_context.get("title", ""),
            "description": source_context.get("description", ""),
            "image_url": image_url,
        }
        return result
    finally:
        if temp_image and os.path.exists(temp_image):
            try:
                os.remove(temp_image)
            except OSError:
                pass


async def analyze_axis2_from_raw_inputs(
    claim_text: str,
    pdf_path: str | None = None,
    image_path: str | None = None,
    provided_caption: str | None = None,
    article_text: str | None = None,
    language_code: str = "en",
    known_references: list[dict] | None = None,
    reverse_image_hits: list[dict] | None = None,
    burstiness_analyzer: DocumentBurstinessAnalyzer | None = None,
    fact_check_client: GoogleFactCheckClient | None = None,
) -> Axis2FusionResult:
    """
    Runs Axis 2 directly from raw files (PDF and/or image) plus claim text.

    Priority for textual context:
    1) article_text argument
    2) text extracted from pdf_path
    """
    extracted_text = article_text or ""
    if not extracted_text and pdf_path:
        extracted_text = _extract_text_from_pdf(pdf_path)

    if not extracted_text:
        raise ValueError("No textual context available. Provide article_text or pdf_path.")

    doc_features = _build_document_features_from_text(extracted_text)
    result = await analyze_axis2_fused(
        document_features=doc_features,
        claim_text=claim_text,
        language_code=language_code,
        image_path=image_path,
        provided_caption=provided_caption,
        article_text=extracted_text,
        known_references=known_references,
        reverse_image_hits=reverse_image_hits,
        run_multimodal=True,
        burstiness_analyzer=burstiness_analyzer,
        fact_check_client=fact_check_client,
    )
    result.source_context = {
        "pdf_path": pdf_path,
        "image_path": image_path,
        "text_source": "article_text" if article_text else "pdf_extraction",
    }
    return result


async def _run_fact_check(
    fact_check_client: GoogleFactCheckClient | None,
    claim_text: str,
    keywords: list[str],
    entities: list[dict],
    language_code: str,
) -> FactCheckResult:
    client = fact_check_client
    if client is None:
        try:
            client = GoogleFactCheckClient()
        except Exception:
            return _default_fact_check_result(claim_text)

    try:
        return await client.check_with_fallback_queries(
            primary_query=claim_text,
            keywords=keywords,
            entities=entities,
            language_code=language_code,
        )
    except Exception as exc:
        fallback = _default_fact_check_result(claim_text)
        fallback.flags.append(f"fact_check_error: {exc}")
        return fallback


def _default_fact_check_result(claim_text: str) -> FactCheckResult:
    return FactCheckResult(
        query_used=claim_text,
        match_found=False,
        match_count=0,
        verdict="unverified",
        credibility_penalty=0.2,
        claims=[],
        reviews=[],
        flags=["fact_check_unavailable"],
    )


def _label_from_score(score: float) -> tuple[str, str]:
    if score >= 0.70:
        return "HIGH_RISK", "red"
    if score >= 0.40:
        return "SUSPICIOUS", "orange"
    return "CONSISTENT", "green"


def _compute_multimodal_penalty(results: dict) -> tuple[float, list[str]]:
    weighted_sum = 0.0
    weight_total = 0.0
    fallback_penalty = 0.0
    flags: list[str] = []

    def _apply_signal(weight: float, risk: float, flag: str | None = None) -> None:
        nonlocal weighted_sum, weight_total
        bounded_risk = max(0.0, min(1.0, float(risk)))
        weighted_sum += bounded_risk * weight
        weight_total += weight
        if flag and bounded_risk >= 0.5 and flag not in flags:
            flags.append(flag)

    clip = results.get("clip_similarity", {})
    clip_score = clip.get("similarity_score")
    clip_verdict = str(clip.get("verdict", "")).upper()
    if isinstance(clip_score, (int, float)):
        clip_risk = max(0.0, min(1.0, (0.20 - float(clip_score)) / 0.20))
        if "MISMATCH" in clip_verdict:
            clip_risk = max(clip_risk, 0.8)
        if clip_risk > 0:
            _apply_signal(0.30, clip_risk, "caption_image_mismatch")

    cap_cmp = results.get("caption_comparison", {})
    cap_score = cap_cmp.get("similarity_score")
    if isinstance(cap_score, (int, float)):
        cap_risk = max(0.0, min(1.0, (0.55 - float(cap_score)) / 0.55))
        if cap_risk > 0:
            _apply_signal(0.20, cap_risk, "caption_semantic_mismatch")

    nli = results.get("claim_nli", {})
    nli_label = str(nli.get("nli_label", "")).lower()
    if nli_label == "contradiction":
        _apply_signal(0.35, 1.0, "claim_article_contradiction")
    elif nli_label == "neutral":
        _apply_signal(0.35, 0.45, "claim_article_unconfirmed")
    elif isinstance(nli, dict) and nli.get("error"):
        fallback_penalty += 0.10
        if "nli_unavailable" not in flags:
            flags.append("nli_unavailable")

    fake_news = results.get("fake_news", {})
    fake_score = fake_news.get("fake_score")
    fake_verdict = str(fake_news.get("verdict", "")).upper()
    if isinstance(fake_score, (int, float)):
        fake_risk = max(0.0, min(1.0, float(fake_score)))
    else:
        fake_risk = 0.0
    if "FAKE" in fake_verdict:
        fake_risk = max(fake_risk, 0.8)
    if fake_risk > 0:
        _apply_signal(0.25, fake_risk, "fake_news_signal")

    reverse = results.get("reverse_image_search", {})
    if isinstance(reverse, dict) and reverse.get("reused_likely"):
        reverse_risk = reverse.get("context_mismatch_score")
        if not isinstance(reverse_risk, (int, float)):
            reverse_risk = 0.8 if reverse.get("date_conflict") else 0.65
        _apply_signal(0.30, float(reverse_risk), "image_reuse_detected")

    if weight_total == 0:
        normalized_penalty = 0.0
    else:
        normalized_penalty = weighted_sum / weight_total

    final_penalty = max(0.0, min(1.0, normalized_penalty + fallback_penalty))
    return round(final_penalty, 4), flags


def _compare_against_references(claim_text: str, article_text: str, references: list[dict]) -> dict:
    """
    Optional reference matching against known datasets/articles.
    Expected reference item format: {"id": "...", "text": "..."}
    """
    if not references:
        return {"match_found": False, "reason": "no_references_provided"}

    model = get_sentence_model()
    probe_text = f"{claim_text}\n\n{article_text[:1000]}".strip()
    probe_emb = model.encode([probe_text])[0]

    best_id = None
    best_score = -1.0

    for item in references:
        ref_id = str(item.get("id", "unknown"))
        ref_text = str(item.get("text", "")).strip()
        if not ref_text:
            continue
        ref_emb = model.encode([ref_text])[0]
        score = float(np.dot(probe_emb, ref_emb) / (np.linalg.norm(probe_emb) * np.linalg.norm(ref_emb) + 1e-9))
        if score > best_score:
            best_score = score
            best_id = ref_id

    if best_id is None:
        return {"match_found": False, "reason": "no_valid_reference_text"}

    return {
        "match_found": True,
        "top_reference_id": best_id,
        "similarity": round(best_score, 4),
        "note": "Higher similarity indicates stronger alignment with known references.",
    }


def _build_short_explanation(flags: list[str], multimodal_results: dict, document_result: Axis2DocumentResult) -> str:
    reverse = multimodal_results.get("reverse_image_search", {})
    if isinstance(reverse, dict) and reverse.get("reused_likely"):
        original_date = reverse.get("original_publication_date") or "unknown date"
        publisher = reverse.get("original_publisher") or "unknown publisher"
        similarity = reverse.get("context_similarity")
        if isinstance(similarity, (int, float)):
            return (
                f"Image originally published on {original_date} by {publisher}; "
                f"claimed-context similarity is {similarity:.2f}, indicating likely reuse."
            )
        return f"Image originally published on {original_date} by {publisher}, likely reused in a different event context."

    if "caption_image_mismatch" in flags:
        clip = multimodal_results.get("clip_similarity", {})
        similarity = clip.get("similarity_score")
        if isinstance(similarity, (int, float)):
            return f"Caption does not match the visual content (CLIP similarity {similarity:.2f})."
        return "Caption does not match the visual content."

    if "claim_article_contradiction" in flags:
        nli = multimodal_results.get("claim_nli", {})
        contradiction_count = nli.get("contradiction_chunk_count")
        if isinstance(contradiction_count, int) and contradiction_count > 0:
            return f"Claim contradicts article context in {contradiction_count} text chunk(s)."
        return "Claim contradicts article context."

    if document_result.label == "HIGH_RISK":
        return "Multiple context signals indicate misleading usage."

    if document_result.label == "SUSPICIOUS":
        return "Some contextual inconsistencies were detected."

    if multimodal_results:
        return "Content is mostly consistent with its context."

    return "Context appears consistent based on available text signals."


def _build_document_features_from_text(article_text: str) -> dict:
    clean = _normalize_text(article_text)
    model = get_sentence_model()
    emb = model.encode([clean[:4000]])[0].tolist()

    return {
        "context_features": {
            "clean_text": clean,
            "keywords": _extract_keywords(clean),
            "entities": [],
            "document_embedding": emb,
        }
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_keywords(text: str, limit: int = 8) -> list[str]:
    stop = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "by", "with",
        "is", "are", "was", "were", "be", "this", "that", "it", "as", "from", "has", "have",
    }
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:limit]]


def _extract_text_from_pdf(pdf_path: str, max_pages: int = 20) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")

    reader = None
    import_error = None

    try:
        import importlib

        pypdf2 = importlib.import_module("PyPDF2")
        reader = pypdf2.PdfReader(pdf_path)
    except Exception as exc:
        import_error = exc

    if reader is None:
        try:
            import importlib

            pypdf = importlib.import_module("pypdf")
            reader = pypdf.PdfReader(pdf_path)
        except Exception:
            raise RuntimeError(
                "Unable to read PDF. Install PyPDF2 or pypdf in the active environment."
            ) from import_error

    chunks: list[str] = []
    for page in reader.pages[:max_pages]:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            chunks.append(text)

    joined = _normalize_text("\n".join(chunks))
    if not joined:
        raise ValueError("Could not extract text from PDF (possibly scanned/image-only PDF).")
    return joined


def _scrape_url_context(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (Axis2ContextChecker/1.0)"}
    resp = requests.get(url, timeout=20, headers=headers)
    resp.raise_for_status()

    html_text = resp.text
    title = _extract_title(html_text)
    description = _extract_meta_content(html_text, "description") or _extract_meta_property(html_text, "og:description")
    image_url = _extract_meta_property(html_text, "og:image")
    article_text = _extract_visible_text(html_text)

    return {
        "title": title,
        "description": description,
        "image_url": image_url,
        "article_text": article_text[:12000],
    }


def _extract_title(html_text: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
    return html.unescape(m.group(1)).strip() if m else ""


def _extract_meta_content(html_text: str, name: str) -> str:
    pattern = rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']'
    m = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
    return html.unescape(m.group(1)).strip() if m else ""


def _extract_meta_property(html_text: str, prop: str) -> str:
    pattern = rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\'](.*?)["\']'
    m = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
    return html.unescape(m.group(1)).strip() if m else ""


def _extract_visible_text(html_text: str) -> str:
    no_script = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.IGNORECASE)
    no_style = re.sub(r"<style[\s\S]*?</style>", " ", no_script, flags=re.IGNORECASE)
    no_tags = re.sub(r"<[^>]+>", " ", no_style)
    clean = html.unescape(no_tags)
    return _normalize_text(clean)


def _download_image_to_temp(image_url: str) -> str | None:
    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
    except Exception:
        return None

    suffix = ".jpg"
    content_type = (resp.headers.get("content-type") or "").lower()
    if "png" in content_type:
        suffix = ".png"
    elif "webp" in content_type:
        suffix = ".webp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(resp.content)
        return f.name


def _compute_headline_body_score(
    claim_text: str,
    document_embedding: list[float]
) -> float:
    """Cosine similarity between claim and doc embedding"""
    model = get_sentence_model()
    claim_emb = model.encode([claim_text])[0]
    doc_emb   = np.array(document_embedding)

    cosine = np.dot(claim_emb, doc_emb) / (
        np.linalg.norm(claim_emb) * np.linalg.norm(doc_emb) + 1e-9
    )
    return float((cosine + 1) / 2)   # normalize to [0, 1]


def _build_summary(label, fc, burst) -> str:
    if label == "HIGH_RISK":
        if fc.verdict == "false":
            return "This content has been fact-checked and marked as false."
        return "Multiple signals indicate this content is misleading or fabricated."
    if label == "SUSPICIOUS":
        return "Some inconsistencies detected. Treat with caution."
    return "Content appears consistent with its claimed context."


def _build_details(headline_score, burst, fc) -> str:
    parts = []
    if headline_score < 0.40:
        parts.append(
            f"Headline-body alignment is low ({headline_score:.0%}), "
            "suggesting possible clickbait."
        )
    if burst.ai_likelihood_score > 0.60:
        parts.append(
            f"Writing patterns suggest AI generation "
            f"(burstiness: {burst.burstiness_ratio:.2f})."
        )
    if fc.match_found:
        sources = ", ".join(r.publisher_name for r in fc.reviews[:2])
        parts.append(
            f"Fact-checkers ({sources}) rated this as: {fc.verdict}."
        )
    return " ".join(parts) or "No major issues detected."


def _build_advice(label: str) -> str:
    advice_map = {
        "HIGH_RISK"  : "Do not share. Verified fact-checkers have flagged this content.",
        "SUSPICIOUS" : "Verify with additional sources before sharing.",
        "CONSISTENT" : "Content appears reliable, but always verify before sharing.",
    }
    return advice_map.get(label, "Exercise caution.")