from __future__ import annotations

"""
works/consistency.py
====================
Unified contextual-consistency evaluator.

Usage::

    from works import Consistency

    c = Consistency(hf_token="hf_xxx")

    # Image
    result = c.evaluate({
        "type": "image",
        "image_path": "photo.jpg",
        "provided_caption": "Protest in Paris 2024",
        "claim_text": "This photo shows the 2024 Paris climate march.",
    })

    # Video
    result = c.evaluate({
        "type": "video",
        "claim_text": "This protest happened in 2024.",
        "transcript_segments": [
            {"start": 0, "end": 4, "text": "The crowd met in 2019..."},
        ],
    })

    # Document - from URL
    result = c.evaluate({
        "type": "document",
        "source": "url",
        "claim_text": "Vaccines cause autism.",
        "url": "https://example.com/article",
    })

    print(result["label"])       # "CONSISTENT" | "SUSPICIOUS" | "HIGH_RISK"
    print(result["flags"])       # ["caption_image_mismatch", ...]
    print(result["timeline_issues"])  # video only, with start_fmt/end_fmt
"""

import logging

logger = logging.getLogger("Consistency")

axis2_contextual_consistency = None
check_image_caption_clip = None
generate_caption_blip = None
compare_captions = None
check_claim_consistency_nli = None
assess_reverse_image_reuse = None
classify_fake_news = None
HF_TOKEN = "your_hf_token"
_CTX_IMPORT_ERROR: Exception | None = None


_has_real_token = lambda token_value, placeholder: bool(token_value and token_value.strip() and token_value.strip() != placeholder)

try:
    from . import axis2_contextual_consistency as axis2_contextual_consistency
    from .axis2_contextual_consistency import (
        HF_TOKEN,
        _has_real_token,
        assess_reverse_image_reuse,
        check_claim_consistency_nli,
        check_image_caption_clip,
        classify_fake_news,
        compare_captions,
        generate_caption_blip,
    )
    logger.info("Successfully imported contextual consistency modules.")
except ImportError as exc:
    _CTX_IMPORT_ERROR = exc
    logger.error(f"Failed to import contextual consistency modules: {exc}")

analyze_axis2_from_url = None
analyze_axis2_from_raw_inputs = None
analyze_context_document = None
_compute_multimodal_penalty = None
Axis2FusionResult = any
Axis2DocumentResult = any
_DOC_IMPORT_ERROR: Exception | None = None

try:
    from .axis2_document_handler import (
        Axis2DocumentResult,
        Axis2FusionResult,
        _compute_multimodal_penalty,
        analyze_axis2_from_raw_inputs,
        analyze_axis2_from_url,
        analyze_context_document,
    )
    logger.info("Successfully imported document handler modules.")
except ImportError as exc:
    _DOC_IMPORT_ERROR = exc
    logger.error(f"Failed to import document handler modules: {exc}")

analyze_video_context = None
_VIDEO_IMPORT_ERROR: Exception | None = None

try:
    from .axis2_video_handler import analyze_video_context
    logger.info("Successfully imported video context handler.")
except ImportError as exc:
    _VIDEO_IMPORT_ERROR = exc
    logger.error(f"Failed to import video context handler: {exc}")

DocumentBurstinessAnalyzer = None
try:
    from .burstiness_analyzer import DocumentBurstinessAnalyzer
except ImportError as exc:
    logger.warning(f"Burstiness analyzer unavailable: {exc}")
    DocumentBurstinessAnalyzer = None

GoogleFactCheckClient = None
try:
    from .fact_check_client import GoogleFactCheckClient
except ImportError as exc:
    logger.warning(f"Google Fact Check client unavailable: {exc}")
    GoogleFactCheckClient = None


class Consistency:
    def __init__(
        self,
        burstiness_analyzer: DocumentBurstinessAnalyzer | None = None,
        fact_check_client: GoogleFactCheckClient | None = None,
        hf_token: str | None = None,
        whisper_model_size: str = "base",
    ):
        logger.info(f"Initializing Consistency with hf_token provided: {bool(hf_token)}")
        if burstiness_analyzer is None and DocumentBurstinessAnalyzer is not None:
            try:
                burstiness_analyzer = DocumentBurstinessAnalyzer()
            except Exception as e:
                logger.warning(f"Error initializing DocumentBurstinessAnalyzer: {e}")
                burstiness_analyzer = None

        self.burstiness_analyzer = burstiness_analyzer
        self.fact_check_client = fact_check_client
        self.hf_token = hf_token
        self.whisper_model_size = whisper_model_size

        if hf_token and axis2_contextual_consistency is not None:
            try:
                axis2_contextual_consistency.HF_TOKEN = hf_token
            except Exception as e:
                logger.error(f"Error setting HF_TOKEN in axis2_contextual_consistency: {e}")
                pass

    def evaluate(self, features: dict[str, Any]) -> dict[str, Any]:
        input_type = features.get("type", "unknown")
        logger.info(f"Evaluating consistency for type: {input_type}")
        handlers = {
            "image": self._image,
            "video": self._video,
            "document": self._document,
            "url": self._url,
        }
        try:
            handler = handlers.get(input_type)
            if not handler:
                logger.warning(f"No consistency handler found for type: {input_type}")
                return self._default()
            
            result = handler(features)
            logger.info(f"Consistency evaluation complete for {input_type}. Score: {result.get('score')}")
            return result
        except Exception as e:
            logger.error(f"Unexpected error in Consistency.evaluate: {e}", exc_info=True)
            return self._default()

    def _url(self, f: dict[str, Any]) -> dict[str, Any]:
        """Treat URLs as documents — analyze article text for consistency."""
        body_text = f.get("body_text") or f.get("clean_text") or ""
        claim_text = f.get("claim_text") or ""

        if not body_text or len(body_text.strip()) < 50:
            return self._result(
                score=0.3,
                label=self._label(0.3),
                explanation="Insufficient article text scraped for consistency analysis.",
                flags=["insufficient_text"],
            )

        # Build document-like features and delegate to _document handler
        doc_features = dict(f)
        doc_features["source"] = "url"
        doc_features["type"] = "document"

        # If we have the full document handler available, use it
        if analyze_axis2_from_url is not None and claim_text:
            try:
                fusion = analyze_axis2_from_url(
                    url=f.get("url", ""),
                    claim_text=claim_text,
                )
                # If it returns a coroutine, we need to run it
                if hasattr(fusion, '__await__'):
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        # We're in an async context called via to_thread, safe to use new loop
                        fusion = asyncio.run(fusion)
                    else:
                        fusion = asyncio.run(fusion)
                return self._from_fusion(fusion)
            except Exception as exc:
                # Fall through to NLI-only analysis
                pass

        # Fallback: use NLI claim checking if available
        flags: list[str] = []
        nli_result: dict[str, Any] = {}
        fake_result: dict[str, Any] = {}

        token_value = self.hf_token or HF_TOKEN
        has_hf_token = False
        try:
            has_hf_token = _has_real_token(token_value, "your_hf_token")
        except Exception:
            has_hf_token = bool(token_value)

        if has_hf_token and claim_text:
            if check_claim_consistency_nli is not None:
                try:
                    nli_result = check_claim_consistency_nli(claim_text, body_text[:2000])
                except Exception as exc:
                    nli_result = {"error": str(exc)}
                    flags.append("nli_error")

            if classify_fake_news is not None:
                try:
                    fake_result = classify_fake_news(body_text[:2000])
                except Exception as exc:
                    fake_result = {"error": str(exc)}
                    flags.append("fake_news_error")

        # Compute score from NLI result
        score = 0.3  # default: low risk
        if nli_result and not nli_result.get("error"):
            nli_label = nli_result.get("label", "").lower()
            nli_score = nli_result.get("score", 0.5)
            if "contradiction" in nli_label:
                score = max(0.6, nli_score)
                flags.append("claim_article_contradiction")
            elif "entailment" in nli_label:
                score = min(0.2, 1.0 - nli_score)
                flags.append("claim_article_entailment")

        if fake_result and not fake_result.get("error"):
            fake_label = fake_result.get("label", "").lower()
            if "fake" in fake_label:
                score = min(1.0, score + 0.2)
                flags.append("fake_news_signal")

        score = round(min(max(score, 0.0), 1.0), 4)
        label = self._label(score)

        explanation = "No significant inconsistencies detected in article text."
        if "claim_article_contradiction" in flags:
            explanation = "Claim contradicts the article content."
        elif "fake_news_signal" in flags:
            explanation = "Article text shows fake news signals."
        elif "claim_article_entailment" in flags:
            explanation = "Claim is supported by the article content."

        return self._result(
            score, label, explanation, flags,
            details={"nli": nli_result, "fake_news": fake_result},
        )

    def _image(self, f: dict[str, Any]) -> dict[str, Any]:
        flags: list[str] = []
        clip_result: dict[str, Any] = {}
        blip_result: dict[str, Any] = {}
        caption_cmp: dict[str, Any] = {}
        reverse_result: dict[str, Any] = {}
        nli_result: dict[str, Any] = {}
        fake_result: dict[str, Any] = {}

        image_path = str(f.get("image_path") or "")
        provided_caption = str(f.get("provided_caption") or "")
        claim_text = str(f.get("claim_text") or "")
        article_text = str(f.get("article_text") or "")

        if check_image_caption_clip is None:
            clip_result = {"error": str(_CTX_IMPORT_ERROR or "image caption checker unavailable")}
            flags.append("clip_error")
        else:
            try:
                clip_result = check_image_caption_clip(image_path, provided_caption)
            except Exception as exc:
                clip_result = {"error": str(exc)}
                flags.append("clip_error")

        token_value = self.hf_token or HF_TOKEN
        has_hf_token = False
        try:
            has_hf_token = _has_real_token(token_value, "your_hf_token")
        except Exception:
            has_hf_token = bool(token_value)

        if has_hf_token and generate_caption_blip is not None and compare_captions is not None:
            try:
                blip_result = generate_caption_blip(image_path)
            except Exception as exc:
                blip_result = {"error": str(exc)}
                flags.append("blip_error")

            auto_cap = str(blip_result.get("auto_generated_caption", "")).strip()
            if auto_cap:
                try:
                    caption_cmp = compare_captions(provided_caption, auto_cap)
                except Exception as exc:
                    caption_cmp = {"error": str(exc)}
                    flags.append("caption_compare_error")

        reverse_hits = f.get("reverse_image_hits")
        if reverse_hits and assess_reverse_image_reuse is not None:
            try:
                reverse_result = assess_reverse_image_reuse(
                    reverse_hits,
                    claim_text,
                    article_text,
                )
            except Exception as exc:
                reverse_result = {"error": str(exc)}
                flags.append("reverse_image_error")

        if article_text and has_hf_token:
            if check_claim_consistency_nli is not None:
                try:
                    nli_result = check_claim_consistency_nli(claim_text, article_text)
                except Exception as exc:
                    nli_result = {"error": str(exc)}
                    flags.append("nli_error")

            if classify_fake_news is not None:
                try:
                    fake_result = classify_fake_news(article_text)
                except Exception as exc:
                    fake_result = {"error": str(exc)}
                    flags.append("fake_news_error")

        multimodal_results: dict[str, Any] = {
            "clip_similarity": clip_result,
        }
        if blip_result:
            multimodal_results["blip_caption"] = blip_result
        if caption_cmp:
            multimodal_results["caption_comparison"] = caption_cmp
        if reverse_result:
            multimodal_results["reverse_image_search"] = reverse_result
        if nli_result:
            multimodal_results["claim_nli"] = nli_result
        if fake_result:
            multimodal_results["fake_news"] = fake_result

        penalty = 0.5
        penalty_flags: list[str] = []
        if _compute_multimodal_penalty is None:
            flags.append("fusion_penalty_error")
            if _DOC_IMPORT_ERROR is not None:
                multimodal_results["fusion_error"] = {"error": str(_DOC_IMPORT_ERROR)}
        else:
            try:
                penalty, penalty_flags = _compute_multimodal_penalty(multimodal_results)
            except Exception as exc:
                penalty = 0.5
                penalty_flags = []
                flags.append("fusion_penalty_error")
                multimodal_results["fusion_error"] = {"error": str(exc)}

        all_flags = list(dict.fromkeys(flags + penalty_flags))
        score = round(min(max(float(penalty), 0.0), 1.0), 4)
        label = self._label(score)

        explanation = "No significant inconsistencies detected."
        if "caption_image_mismatch" in all_flags:
            explanation = "Caption does not match the visual content."
        elif "image_reuse_detected" in all_flags:
            explanation = "Image appears reused from a different context."
        elif "fake_news_signal" in all_flags:
            explanation = "Article text shows fake news signals."
        elif "claim_nli_contradiction" in all_flags or "claim_article_contradiction" in all_flags:
            explanation = "Claim contradicts the article text."

        return self._result(
            score,
            label,
            explanation,
            all_flags,
            details=multimodal_results,
            timeline_issues=[],
        )

    def _video(self, f: dict[str, Any]) -> dict[str, Any]:
        logger.info("Starting video consistency analysis.")
        if analyze_video_context is None:
            err_msg = str(_VIDEO_IMPORT_ERROR or "video analyzer unavailable")
            logger.error(f"Video analysis dependencies missing: {err_msg}")
            return self._result(
                score=0.5,
                label=self._label(0.5),
                explanation="Video analysis dependencies are unavailable.",
                flags=["video_error"],
                details={"error": err_msg},
                timeline_issues=[],
            )

        try:
            video_path = f.get("video_path")
            claim_text = f.get("claim_text", "")
            logger.info(f"Calling analyze_video_context. Video path: {video_path}")
            
            raw = analyze_video_context(
                claim_text=claim_text,
                video_path=video_path,
                transcript_text=f.get("transcript_text"),
                transcript_segments=f.get("transcript_segments"),
                whisper_model_size=self.whisper_model_size,
                language=f.get("language"),
            )
            logger.info("analyze_video_context returned successfully.")
        except Exception as exc:
            logger.error(f"Video analysis failed with exception: {exc}", exc_info=True)
            return self._result(
                score=0.5,
                label=self._label(0.5),
                explanation=f"Video analysis failed: {exc}",
                flags=["video_error"],
                details={"error": str(exc)},
                timeline_issues=[],
            )

        score = float(raw.get("inconsistency_score", 0.5))
        label = self._label(score)
        explanation = str(raw.get("short_explanation", "Video analysis complete."))
        flags = raw.get("flags", [])
        if not isinstance(flags, list):
            flags = [str(flags)]

        timeline_issues = self._extract_video_timeline_issues(raw)

        breakdown = {
            "claim_nli": raw.get("claim_nli", {}),
            "temporal_consistency": raw.get("temporal_consistency", {}),
            "visual_signals": raw.get("visual_signals", {}),
            "year_mismatch": raw.get("year_mismatch", False),
        }

        return self._result(
            score,
            label,
            explanation,
            flags,
            details=breakdown,
            timeline_issues=timeline_issues,
        )

    def _document(self, f: dict[str, Any]) -> dict[str, Any]:
        source = str(f.get("source", "raw_text"))

        try:
            if source == "url":
                fusion = asyncio.run(
                    analyze_axis2_from_url(
                        url=f["url"],
                        claim_text=f["claim_text"],
                    )
                )
                return self._from_fusion(fusion)

            if source == "pdf":
                fusion = asyncio.run(
                    analyze_axis2_from_raw_inputs(
                        claim_text=f["claim_text"],
                        pdf_path=f["pdf_path"],
                        image_path=f.get("image_path"),
                        provided_caption=f.get("provided_caption"),
                    )
                )
                return self._from_fusion(fusion)

            doc_result = asyncio.run(
                analyze_context_document(
                    document_features=f["document_features"],
                    claim_text=f["claim_text"],
                    burstiness_analyzer=self.burstiness_analyzer,
                    fact_check_client=self.fact_check_client,
                )
            )
            return self._from_doc(doc_result)
        except Exception as exc:
            return self._result(
                score=0.5,
                label=self._label(0.5),
                explanation=f"Document analysis failed: {exc}",
                flags=["document_error"],
                details={"source": source, "error": str(exc)},
                timeline_issues=[],
            )

    def _from_fusion(self, r: Axis2FusionResult) -> dict[str, Any]:
        score = float(r.fused_score)
        return self._result(
            score=score,
            label=self._label(score),
            explanation=r.short_explanation,
            flags=r.flags,
            details={
                "document": vars(r.document_result) if r.document_result else {},
                "multimodal": r.multimodal_results,
                "penalty": r.multimodal_penalty,
            },
            timeline_issues=[],
        )

    def _from_doc(self, r: Axis2DocumentResult) -> dict[str, Any]:
        expl = r.explanation.get("summary", "") if isinstance(r.explanation, dict) else str(r.explanation)
        score = float(r.score)
        return self._result(
            score=score,
            label=self._label(score),
            explanation=expl or r.label,
            flags=r.flags,
            details=r.breakdown,
            timeline_issues=[],
        )

    def _default(self) -> dict[str, Any]:
        return self._result(
            score=0.5,
            label="unknown",
            explanation="Content type not recognised - no handler available.",
            flags=["unsupported_type"],
        )

    def _label(self, score: float) -> str:
        if score < 0.40:
            return "CONSISTENT"
        if score < 0.70:
            return "SUSPICIOUS"
        return "HIGH_RISK"

    def _result(
        self,
        score: float,
        label: str,
        explanation: str,
        flags: list[str] | None = None,
        details: dict[str, Any] | None = None,
        timeline_issues: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "score": round(float(score), 4),
            "label": label,
            "explanation": explanation,
            "flags": flags or [],
            "details": details or {},
            "timeline_issues": timeline_issues or [],
        }

    def _extract_video_timeline_issues(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        timeline_issues = raw.get("timeline_issues")
        if isinstance(timeline_issues, list) and timeline_issues:
            return [self._normalize_timeline_issue(item) for item in timeline_issues if isinstance(item, dict)]

        synthesized: list[dict[str, Any]] = []
        for item in raw.get("segment_anomalies", []) or []:
            if not isinstance(item, dict):
                continue
            synthesized.append(
                self._normalize_timeline_issue(
                    {
                        "start_sec": item.get("start_sec"),
                        "end_sec": item.get("end_sec"),
                        "start_fmt": item.get("start_fmt"),
                        "end_fmt": item.get("end_fmt"),
                        "flag": item.get("flag", "segment_contradiction"),
                        "description": "Transcript interval contradicts the claim.",
                    }
                )
            )

        for item in raw.get("drift_events", []) or []:
            if not isinstance(item, dict):
                continue
            synthesized.append(
                self._normalize_timeline_issue(
                    {
                        "start_sec": item.get("start_sec"),
                        "end_sec": item.get("end_sec"),
                        "start_fmt": item.get("start_fmt"),
                        "end_fmt": item.get("end_fmt"),
                        "flag": item.get("flag", "semantic_drift"),
                        "description": item.get("description", "Narrative drift detected in transcript window."),
                    }
                )
            )

        if bool(raw.get("year_mismatch", False)):
            transcript_segments = raw.get("transcript_segments", [])
            start_sec = 0.0
            end_sec = 0.0
            if isinstance(transcript_segments, list) and transcript_segments:
                first = transcript_segments[0]
                if isinstance(first, dict):
                    start_sec = float(first.get("start", 0.0) or 0.0)
                    end_sec = float(first.get("end", start_sec) or start_sec)

            claim_year = raw.get("claim_year")
            transcript_year = raw.get("transcript_year")
            synthesized.append(
                self._normalize_timeline_issue(
                    {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "flag": "claim_transcript_year_mismatch",
                        "description": f"Year mismatch: claim references {claim_year} but transcript suggests {transcript_year}.",
                    }
                )
            )

        synthesized.sort(key=lambda item: float(item.get("start_sec", 0.0)))
        return synthesized

    def _normalize_timeline_issue(self, item: dict[str, Any]) -> dict[str, Any]:
        start_sec = float(item.get("start_sec", 0.0) or 0.0)
        end_sec = float(item.get("end_sec", start_sec) or start_sec)
        return {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_fmt": str(item.get("start_fmt") or self._format_seconds(start_sec)),
            "end_fmt": str(item.get("end_fmt") or self._format_seconds(end_sec)),
            "flag": str(item.get("flag") or item.get("issue_type") or "timeline_issue"),
            "description": str(item.get("description") or "Potential contextual inconsistency detected."),
        }

    def _format_seconds(self, seconds: float) -> str:
        safe_seconds = int(max(0.0, float(seconds)))
        mins, secs = divmod(safe_seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"
