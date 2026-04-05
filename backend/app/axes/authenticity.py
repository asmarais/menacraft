import asyncio
import os
import io
import base64
import requests
import httpx
import json
import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import sys
from pathlib import Path

# Ensure app/utils and its subfolders are in path for modular forensic pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
UTILS_DIR = BASE_DIR / "utils"
if UTILS_DIR.exists() and str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Authenticity")

try:
    from utils.config_dataclasses import InferenceConfig
    from inference import InferenceEngine
    from utils.inference_results_dataclasses import VideoInferenceResult
    VIDEO_FORENSICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Video forensics modules not fully found: {e}")
    VIDEO_FORENSICS_AVAILABLE = False

# Load .env variables immediately
load_dotenv()

from PIL import Image, ImageChops, ImageEnhance
import piexif

# ── Optional SDKs ─────────────────────────────────────────────────────────────
try:
    from realitydefender import RealityDefender
    REALITY_DEFENDER_AVAILABLE = True
except ImportError:
    REALITY_DEFENDER_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import tldextract
    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

import re
from urllib.parse import urlparse

# ══════════════════════════════════════════════════════════════════════════════
# ENV VARS
# ══════════════════════════════════════════════════════════════════════════════
SIGHTENGINE_USER     = os.environ.get("SIGHTENGINE_USER", "")
SIGHTENGINE_SECRET   = os.environ.get("SIGHTENGINE_SECRET", "")
REALITY_DEFENDER_KEY = os.environ.get("REALITY_DEFENDER_KEY", "")
HF_TOKEN             = os.environ.get("HF_TOKEN", "")
RAPIDAPI_KEY         = os.environ.get("RAPIDAPI_KEY", "")
AIORNOT_API_KEY      = os.environ.get("AIORNOT_API_KEY", "")
GROQ_API_KEY         = os.environ.get("groq_key", "")
SERPAPI_KEY          = os.environ.get("SERPAPI_KEY", "")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

HF_INFERENCE_URL = (
    "https://router.huggingface.co/hf-inference/models/Organika/sdxl-detector"
)

HF_TEXT_DETECTOR_URL = (
    "https://router.huggingface.co/hf-inference/models/Hello-SimpleAI/chatgpt-detector-roberta"
)

RAPIDAPI_HOST       = "ai-generated-image-detection-api.p.rapidapi.com"
RAPIDAPI_DETECT_URL = f"https://{RAPIDAPI_HOST}/detect"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
async def download_image_from_url(url: str) -> str:
    """Download a remote image to a temp file; return local path."""
    import tempfile
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=15)
        response.raise_for_status()
        content = response.content

    suffix = ".jpg"
    for ext in (".png", ".webp", ".jpeg", ".jpg"):
        if ext in url.lower():
            suffix = ext
            break
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


def detect_face_present(image_path: str) -> bool:
    """Returns True if at least one face is detected. Falls back to False without OpenCV."""
    if not CV2_AVAILABLE:
        return False
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return len(faces) > 0
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════
class Authenticity:

    async def evaluate(self, features: dict) -> dict:
        """Entry point that routes to specialized media handlers."""
        handlers = {
            "image":    self._image,
            "video":    self._video,
            "document": self._document,
            "url":      self._url,
        }
        handler = handlers.get(features.get("type"))
        if handler:
            return await handler(features)
        return self._default()

    # ── Public entry-point for images ─────────────────────────────────────────
    async def _image(self, f: dict) -> dict:
        """
        Orchestrates all detectors and returns a weighted authenticity score.
        """
        image_path = f.get("path")
        image_url  = f.get("url")
        logger.info(f"=== Starting IMAGE Authenticity Evaluation for: {image_path or image_url} ===")
        start_time = time.time()

        # ── 1. Load PIL image & Prepare bytes ────────────────────────────────
        try:
            if not image_path and image_url:
                image_path = await download_image_from_url(image_url)
                logger.info(f"Downloaded remote image to: {image_path}")

            if not image_path:
                return self._result(0.5, "error", "No image path or URL provided")

            pil_image = await asyncio.to_thread(lambda: Image.open(image_path).convert("RGB"))
            with open(image_path, "rb") as fh:
                image_bytes = fh.read()

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return self._result(0.5, "error", f"Could not process image: {e}")

        # ── 2. Face detection (CPU bound) ────────────────────────────────────
        face_present = await asyncio.to_thread(detect_face_present, image_path)
        logger.info(f"Face detection complete. Face present: {face_present}")

        # ── 3. Run all detectors IN PARALLEL ─────────────────────────────────
        logger.info("Running all detectors in parallel...")

        api_tasks = [
            self.detect_ai_sightengine(image_url or image_path),
            self.detect_ai_huggingface(image_path),
            self.detect_ai_rapidapi(image_path),
            self.detect_ai_or_not(image_path),
            self.detect_deepfake_reality_defender(image_path),
        ]

        results = await asyncio.gather(*api_tasks)

        se_result       = results[0]
        hf_result       = results[1]
        rapid_result    = results[2]
        aiornot_result  = results[3]
        deepfake_result = results[4]

        logger.info(f"All parallel detectors completed in {time.time() - start_time:.2f}s")

        # 3b. Local forensics
        logger.info("Running local forensics (EXIF)...")
        exif_data = await asyncio.to_thread(self.extract_exif, image_bytes)

        # ── 4. Normalise → [0, 1]  (1 = suspicious / likely fake) ────────────

        # 4a. Weighted average of available AI-generation scores
        ai_gen_sources = {
            "sightengine": se_result.get("ai_generated_score"),
            "huggingface": hf_result.get("ai_generated_score"),
            "rapidapi":    rapid_result.get("ai_generated_score"),
            "aiornot":     aiornot_result.get("ai_generated_score"),
        }
        valid_scores = {
            k: v for k, v in ai_gen_sources.items()
            if v is not None and isinstance(v, (int, float))
        }

        source_weights = {
            "sightengine": 2.5,
            "huggingface": 2.5,
            "rapidapi":    1.0,
            "aiornot":     1.0,
        }

        weighted_sum = sum(valid_scores[k] * source_weights.get(k, 1.0) for k in valid_scores)
        total_weight = sum(source_weights.get(k, 1.0) for k in valid_scores)

        # FIX 3: Default to 0.65 (suspicious) when all APIs fail, not 0.5 (neutral)
        ai_score: float = (
            weighted_sum / total_weight
            if total_weight > 0 else 0.65
        )

        # 4b. Deepfake score (Reality Defender)
        deepfake_score: float = float(
            deepfake_result.get("manipulation_score", 0.0) or 0.0
        )



        # 4d. EXIF risk with specific justification
        exif_reason = "Metadata appears clean"
        if exif_data.get("likely_edited"):
            exif_risk = 1.0  # Explicitly edited
            exif_reason = f"Explicit editing software signatures found: {exif_data.get('software')}"
        elif not exif_data.get("has_exif"):
            exif_risk = 0.5  # Neutral/Suspicious
            exif_reason = "No metadata (EXIF) found in image file"
        elif exif_data.get("make") or exif_data.get("model"):
            exif_risk = 0.1  # Likely authentic camera signature
            exif_reason = f"Authentic camera signature found: {exif_data.get('make')} {exif_data.get('model')}"
        else:
            exif_risk = 0.0

        # ── 5. Weighted final score ───────────────────────────────────────────
        if face_present:
            final_score = (
                ai_score         * 0.55
                + deepfake_score * 0.35
                + exif_risk      * 0.10
            )
            weights_used = {
                "ai_generation": 0.60,
                "deepfake":      0.30,
                "exif_risk":     0.10,
                "face_detected": True,
            }
        else:
            # No face: redistribute deepfake weight
            final_score = (
                ai_score     * 0.85
                + exif_risk  * 0.15
            )
            weights_used = {
                "ai_generation": 0.85,
                "deepfake":      0.00,
                "exif_risk":     0.15,
                "face_detected": False,
            }

        # ── 6. Label ──────────────────────────────────────────────────────────
        # FIX 4: Tighten thresholds — 0.50+ is fake, not 0.60+
        if final_score < 0.35:
            label = "real"
        elif final_score < 0.50:
            label = "uncertain"
        else:
            label = "fake"

        # ── 7. AI Explanation ─────────────────────────────────────────────────
        logger.info(f"Generating LLM explanation for verdict: {label} (score: {final_score:.4f})")
        explanation = await self._generate_llm_explanation(
            final_score, label, {
                "content_type":  "image",
                "ai_score":       ai_score,
                "deepfake_score": deepfake_score,
                "exif_risk":      exif_risk,
                "exif_reason":    exif_reason,
                "face_detected":  face_present,
                "api_breakdown":  valid_scores,
            }
        )

        total_duration = time.time() - start_time
        logger.info(f"=== IMAGE evaluation complete in {total_duration:.2f}s ===")

        # ── 8. Flags ──────────────────────────────────────────────────────────
        flags = []
        if ai_score > 0.5:
            src_str = ", ".join(f"{k}={v:.2f}" for k, v in valid_scores.items())
            flags.append(f"AI-generation score high ({ai_score:.2f}) [{src_str}]")
        if face_present and deepfake_score > 0.5:
            flags.append(f"Deepfake score elevated ({deepfake_score:.2f})")
        if exif_data.get("likely_edited"):
            flags.append(exif_reason)
        elif not exif_data.get("has_exif"):
            flags.append(exif_reason)
        elif exif_risk <= 0.1:
            flags.append(exif_reason)

        return self._result(
            score=round(final_score, 4),
            label=label,
            explanation=explanation,
            flags=flags,
            details={
                "weights": weights_used,
                "component_scores": {
                    "ai_generation":    round(ai_score, 4),
                    "ai_gen_by_source": {k: round(v, 4) for k, v in valid_scores.items()},
                    "deepfake":         round(deepfake_score, 4),
                    "exif_risk":        round(exif_risk, 4),
                },
                "exif":      exif_data,
                "api_raw": {
                    "sightengine":      se_result,
                    "rapidapi":         rapid_result,
                    "aiornot":          aiornot_result,
                    "reality_defender": deepfake_result,
                },
            },
        )

    # ══════════════════════════════════════════════════════════════════════════
    # AI-GENERATION DETECTORS
    # ══════════════════════════════════════════════════════════════════════════

    # ── 1A. SightEngine GenAI ─────────────────────────────────────────────────
    async def detect_ai_sightengine(self, image_path: str) -> dict:
        url = "https://api.sightengine.com/1.0/check.json"
        st  = time.time()
        logger.info(f"SightEngine call starting for: {image_path[:50]}...")
        try:
            params = {
                "models":     "genai",
                "api_user":   SIGHTENGINE_USER,
                "api_secret": SIGHTENGINE_SECRET,
            }
            async with httpx.AsyncClient() as client:
                if image_path and image_path.startswith(("http://", "https://")):
                    params["url"] = image_path
                    response = await client.get(url, params=params, timeout=15)
                else:
                    with open(image_path, "rb") as fh:
                        response = await client.post(
                            url, files={"media": fh}, data=params, timeout=15
                        )
                result = response.json()

            dur = time.time() - st
            if result.get("status") == "success":
                genai_data = result.get("type", {}).get("ai_generated", 0)
                score = (
                    genai_data.get("prob", 0)
                    if isinstance(genai_data, dict)
                    else float(genai_data)
                )
                logger.info(f"SightEngine success. Score: {score:.4f} (took {dur:.2f}s)")
                return {
                    "tool":               "SightEngine (GenAI)",
                    "ai_generated_score": float(score),
                    "verdict":            "AI-GENERATED" if score > 0.5 else "LIKELY REAL",
                    "confidence":         f"{score * 100:.1f}%",
                }
            logger.warning(f"SightEngine failed. Response: {result} (took {dur:.2f}s)")
            return {
                "tool":               "SightEngine (GenAI)",
                "ai_generated_score": None,
                "error":              result.get("error", result),
            }
        except Exception as e:
            dur = time.time() - st
            logger.error(f"SightEngine exception: {e} (took {dur:.2f}s)")
            return {
                "tool":               "SightEngine (GenAI)",
                "ai_generated_score": None,
                "error":              str(e),
            }

    # ── 1B. HuggingFace — Organika/sdxl-detector ──────────────────────────────
    async def detect_ai_huggingface(self, image_path: str) -> dict:
        st = time.time()
        logger.info(f"HuggingFace GenAI call starting for: {image_path[:50]}...")
        try:
            temp_file = None
            if image_path.startswith(("http://", "https://")):
                temp_file   = await download_image_from_url(image_path)
                file_to_use = temp_file
            else:
                file_to_use = image_path

            with open(file_to_use, "rb") as fh:
                image_bytes = fh.read()

            if temp_file:
                try: os.unlink(temp_file)
                except Exception: pass

            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type":  "image/jpeg",
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    HF_INFERENCE_URL, headers=headers, content=image_bytes, timeout=20
                )

            dur = time.time() - st

            if response.status_code != 200:
                logger.warning(f"HuggingFace failed. HTTP {response.status_code} (took {dur:.2f}s)")
                return {
                    "tool":               "HuggingFace (sdxl-detector)",
                    "ai_generated_score": None,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                }

            results: list = response.json()
            ai_score = next(
                (item["score"] for item in results
                 if item.get("label", "").lower() in ("artificial", "ai", "fake")),
                None,
            )
            if ai_score is None:
                human_score = next(
                    (item["score"] for item in results
                     if item.get("label", "").lower() in ("human", "real")),
                    None,
                )
                ai_score = (1.0 - human_score) if human_score is not None else None

            logger.info(f"HuggingFace success. Score: {ai_score} (took {dur:.2f}s)")
            return {
                "tool":               "HuggingFace (sdxl-detector)",
                "ai_generated_score": float(ai_score) if ai_score is not None else None,
                "verdict":            "AI-GENERATED" if (ai_score or 0) > 0.5 else "LIKELY REAL",
                "raw":                results,
            }
        except Exception as e:
            dur = time.time() - st
            logger.error(f"HuggingFace exception: {e} (took {dur:.2f}s)")
            return {
                "tool":               "HuggingFace (sdxl-detector)",
                "ai_generated_score": None,
                "error":              str(e),
            }

    # ── 1C. RapidAPI — ai-generated-image-detection-api ───────────────────────
    async def detect_ai_rapidapi(self, image_path: str) -> dict:
        st = time.time()
        logger.info(f"RapidAPI AI Detect call starting for: {image_path[:50]}...")
        try:
            temp_file = None
            if image_path.startswith(("http://", "https://")):
                temp_file   = await download_image_from_url(image_path)
                file_to_use = temp_file
            else:
                file_to_use = image_path

            headers = {
                "X-RapidAPI-Key":  RAPIDAPI_KEY,
                "X-RapidAPI-Host": RAPIDAPI_HOST,
            }
            async with httpx.AsyncClient() as client:
                with open(file_to_use, "rb") as fh:
                    response = await client.post(
                        RAPIDAPI_DETECT_URL,
                        headers=headers,
                        files={"image": fh},
                        timeout=15,
                    )
            dur = time.time() - st

            if temp_file:
                try: os.unlink(temp_file)
                except Exception: pass

            if response.status_code != 200:
                return {
                    "tool":               "RapidAPI (ai-image-detector)",
                    "ai_generated_score": None,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                }

            result     = response.json()
            is_ai      = bool(result.get("isAI", False))
            confidence = float(result.get("confidence", 0.5))
            ai_score   = confidence if is_ai else (1.0 - confidence)

            logger.info(f"RapidAPI success. Score: {ai_score} (took {dur:.2f}s)")
            return {
                "tool":               "RapidAPI (ai-image-detector)",
                "ai_generated_score": ai_score,
                "verdict":            "AI-GENERATED" if is_ai else "LIKELY REAL",
                "confidence":         f"{confidence * 100:.1f}%",
                "raw":                result,
            }
        except Exception as e:
            dur = time.time() - st
            logger.error(f"RapidAPI exception: {e} (took {dur:.2f}s)")
            return {
                "tool":               "RapidAPI (ai-image-detector)",
                "ai_generated_score": None,
                "error":              str(e),
            }

    # ── 1D. AI or Not — aiornot.com ───────────────────────────────────────────
    async def detect_ai_or_not(self, image_path: str) -> dict:
        IMAGE_ENDPOINT = "https://api.aiornot.com/v2/image/sync"
        st = time.time()
        logger.info(f"AI or Not call starting for: {image_path[:50]}...")
        try:
            temp_file = None
            if image_path.startswith(("http://", "https://")):
                temp_file   = await download_image_from_url(image_path)
                file_to_use = temp_file
            else:
                file_to_use = image_path

            async with httpx.AsyncClient() as client:
                with open(file_to_use, "rb") as fh:
                    response = await client.post(
                        IMAGE_ENDPOINT,
                        headers={"Authorization": f"Bearer {AIORNOT_API_KEY}"},
                        files={"image": fh},
                        timeout=15,
                    )
            dur = time.time() - st

            if temp_file:
                try: os.unlink(temp_file)
                except Exception: pass

            if response.status_code != 200:
                return {
                    "tool":               "AI or Not",
                    "ai_generated_score": None,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                }

            result      = response.json()
            verdict_raw = result.get("verdict", "").lower()
            confidence  = float(result.get("confidence", 0.5))
            is_ai       = verdict_raw in ("ai", "artificial")
            ai_score    = confidence if is_ai else (1.0 - confidence)

            logger.info(f"AI or Not success. Score: {ai_score} (took {dur:.2f}s)")
            return {
                "tool":               "AI or Not",
                "ai_generated_score": ai_score,
                "verdict":            "AI-GENERATED" if is_ai else "LIKELY REAL",
                "confidence":         f"{confidence * 100:.1f}%",
                "raw_verdict":        verdict_raw,
                "raw":                result,
            }
        except Exception as e:
            dur = time.time() - st
            logger.error(f"AI or Not exception: {e} (took {dur:.2f}s)")
            return {
                "tool":               "AI or Not",
                "ai_generated_score": None,
                "error":              str(e),
            }

    # ══════════════════════════════════════════════════════════════════════════
    # DEEPFAKE DETECTOR — Reality Defender
    # ══════════════════════════════════════════════════════════════════════════
    async def detect_deepfake_reality_defender(self, image_path: str) -> dict:
        if not REALITY_DEFENDER_AVAILABLE:
            return {
                "tool":               "Reality Defender",
                "manipulation_score": 0.0,
                "error": "realitydefender SDK not installed.",
            }

        st = time.time()
        logger.info(f"Reality Defender call starting for: {image_path[:50]}...")
        try:
            client    = RealityDefender(api_key=REALITY_DEFENDER_KEY)
            temp_file = None
            if image_path.startswith(("http://", "https://")):
                temp_file   = await download_image_from_url(image_path)
                file_to_use = temp_file
            else:
                file_to_use = image_path

            result = await asyncio.to_thread(client.detect_file, file_to_use)
            dur = time.time() - st

            if temp_file:
                try: os.unlink(temp_file)
                except Exception: pass

            if isinstance(result, dict):
                score = float(result.get("score", result.get("probability", 0.0)) or 0.0)
                logger.info(f"Reality Defender success. Score: {score:.4f} (took {dur:.2f}s)")
                return {
                    "tool":               "Reality Defender",
                    "manipulation_score": score,
                    "verdict":            "MANIPULATED" if score > 0.5 else "LIKELY AUTHENTIC",
                    "details":            result,
                }

            return {
                "tool":               "Reality Defender",
                "manipulation_score": 0.0,
                "result":             str(result),
            }
        except Exception as e:
            dur = time.time() - st
            logger.error(f"Reality Defender exception: {e} (took {dur:.2f}s)")
            return {
                "tool":               "Reality Defender",
                "manipulation_score": 0.0,
                "error":              str(e),
            }

    # ══════════════════════════════════════════════════════════════════════════
    # EXIF extraction
    # ══════════════════════════════════════════════════════════════════════════
    def extract_exif(self, image_bytes: bytes) -> dict:
        try:
            exif_dict = piexif.load(image_bytes)
            ifd       = exif_dict.get("0th", {})

            make     = ifd.get(piexif.ImageIFD.Make,     b"").decode(errors="ignore").strip()
            model    = ifd.get(piexif.ImageIFD.Model,    b"").decode(errors="ignore").strip()
            software = ifd.get(piexif.ImageIFD.Software, b"").decode(errors="ignore").strip()
            dt       = ifd.get(piexif.ImageIFD.DateTime, b"").decode(errors="ignore").strip()
            gps      = bool(exif_dict.get("GPS"))

            suspicious_sw = [
                "photoshop", "gimp", "lightroom", "affinity",
                "canva", "pixlr", "snapseed", "facetune",
            ]
            edited = any(sw in software.lower() for sw in suspicious_sw)

            return {
                "has_exif":      True,
                "make":          make,
                "model":         model,
                "software":      software,
                "datetime":      dt,
                "gps_present":   gps,
                "likely_edited": edited,
            }
        except Exception:
            return {
                "has_exif":      False,
                "make":          "", "model": "", "software": "", "datetime": "",
                "gps_present":   False,
                "likely_edited": False,
            }

    # ══════════════════════════════════════════════════════════════════════════
    # DOCUMENT HANDLER
    # ══════════════════════════════════════════════════════════════════════════
    async def _document(self, f: dict) -> dict:
        logger.info("=== Starting DOCUMENT Authenticity Evaluation ===")
        start_time = time.time()

        clean_text        = f.get("clean_text", f.get("text", ""))
        word_count        = f.get("word_count", len(clean_text.split()))
        burstiness_ratio  = f.get("burstiness_ratio", 0.0)
        sent_len_var      = f.get("sentence_length_variance", 0.0)
        avg_sent_len      = f.get("avg_sentence_length", 0.0)
        meta_anomalies    = f.get("metadata_anomalies", [])
        layout_anomalies  = f.get("layout_anomalies", [])
        font_consistency  = f.get("font_consistency_score", 1.0)
        metadata          = f.get("metadata", {})

        logger.info("Running RoBERTa AI Text Detection...")
        ai_text_result = await self.detect_ai_text_roberta(clean_text)
        ai_text_score: float = float(ai_text_result.get("ai_generated_score") or 0.5)

        logger.info("Computing Burstiness and Metadata scores...")
        burstiness_result = self.compute_burstiness_score(burstiness_ratio, sent_len_var, avg_sent_len)
        burstiness_score: float = burstiness_result["ai_likelihood"]

        metadata_result = self.compute_metadata_anomaly_score(
            meta_anomalies, layout_anomalies, font_consistency, metadata
        )
        metadata_risk: float = metadata_result["risk_score"]

        final_score  = (ai_text_score * 0.55 + burstiness_score * 0.30 + metadata_risk * 0.15)
        weights_used = {"ai_text": 0.55, "burstiness": 0.30, "metadata": 0.15}

        # FIX 4: Tightened thresholds
        label = "fake" if final_score >= 0.50 else "uncertain" if final_score >= 0.35 else "real"

        explanation = await self._generate_llm_explanation(
            final_score, label, {
                "content_type":     "document",
                "ai_text_score":    ai_text_score,
                "burstiness_score": burstiness_score,
                "metadata_risk":    metadata_risk,
                "word_count":       word_count,
                "api_breakdown": {
                    "roberta":    ai_text_score,
                    "burstiness": burstiness_score,
                    "metadata":   metadata_risk,
                },
            }
        )

        flags = []
        if ai_text_score > 0.5:
            flags.append(f"AI-generated text probability high ({ai_text_score:.2f})")
        if burstiness_score > 0.5:
            flags.append(f"Low burstiness — suspiciously uniform writing ({burstiness_score:.2f})")
        if metadata_risk > 0.3:
            flags.append(
                f"Metadata anomalies detected ({metadata_risk:.2f}): "
                + ", ".join(meta_anomalies + layout_anomalies) if (meta_anomalies or layout_anomalies)
                else f"Metadata risk elevated ({metadata_risk:.2f})"
            )

        return self._result(
            score=round(final_score, 4),
            label=label,
            explanation=explanation,
            flags=flags,
            details={
                "weights": weights_used,
                "component_scores": {
                    "ai_text_detection": round(ai_text_score, 4),
                    "burstiness":        round(burstiness_score, 4),
                    "metadata_risk":     round(metadata_risk, 4),
                },
                "signal_details": {
                    "roberta":    ai_text_result,
                    "burstiness": burstiness_result,
                    "metadata":   metadata_result,
                },
                "document_stats": {
                    "word_count":       word_count,
                    "burstiness_ratio": burstiness_ratio,
                    "avg_sentence_len": avg_sent_len,
                    "font_consistency": font_consistency,
                },
            }
        )

    # ══════════════════════════════════════════════════════════════════════════
    # DOCUMENT DETECTION METHODS
    # ══════════════════════════════════════════════════════════════════════════

    async def detect_ai_text_roberta(self, text: str) -> dict:
        st = time.time()
        logger.info(f"RoBERTa Text Detect call starting (text length: {len(text)})...")
        if not text or len(text.strip()) < 50:
            return {"tool": "RoBERTa", "ai_generated_score": None, "error": "Text too short"}

        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {"inputs": text[:2000]}
            async with httpx.AsyncClient() as client:
                response = await client.post(HF_TEXT_DETECTOR_URL, headers=headers, json=payload, timeout=20)

            dur = time.time() - st
            if response.status_code != 200:
                return {"tool": "RoBERTa", "ai_generated_score": None, "error": f"HTTP {response.status_code}"}

            results = response.json()
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0]

            ai_score = next(
                (item["score"] for item in results if item.get("label", "").lower() in ("chatgpt", "ai", "fake")),
                None,
            )
            if ai_score is None:
                human_score = next(
                    (item["score"] for item in results if item.get("label", "").lower() in ("human", "real")),
                    None,
                )
                ai_score = (1.0 - human_score) if human_score is not None else None

            logger.info(f"RoBERTa success. Score: {ai_score} (took {dur:.2f}s)")
            return {
                "tool":               "RoBERTa (chatgpt-detector)",
                "ai_generated_score": float(ai_score) if ai_score is not None else None,
                "verdict":            "AI-GENERATED" if (ai_score or 0) > 0.5 else "LIKELY HUMAN",
            }
        except Exception as e:
            return {"tool": "RoBERTa", "ai_generated_score": None, "error": str(e)}

    def compute_burstiness_score(
        self,
        burstiness_ratio: float,
        sentence_length_variance: float,
        avg_sentence_length: float,
    ) -> dict:
        if burstiness_ratio >= 0.7:
            burst_ai = 0.1
        elif burstiness_ratio >= 0.5:
            burst_ai = 0.3
        elif burstiness_ratio >= 0.3:
            burst_ai = 0.6
        else:
            burst_ai = 0.9

        if sentence_length_variance < 2.0:
            var_ai = 0.8
        elif sentence_length_variance < 5.0:
            var_ai = 0.5
        elif sentence_length_variance < 10.0:
            var_ai = 0.2
        else:
            var_ai = 0.1

        if 12 <= avg_sentence_length <= 18:
            len_ai = 0.4
        else:
            len_ai = 0.1

        ai_likelihood = (burst_ai * 0.50) + (var_ai * 0.35) + (len_ai * 0.15)

        return {
            "tool":           "Burstiness Analysis (statistical)",
            "ai_likelihood":  round(min(ai_likelihood, 1.0), 4),
            "burst_ai":       round(burst_ai, 4),
            "variance_ai":    round(var_ai, 4),
            "length_ai":      round(len_ai, 4),
            "raw_burstiness": burstiness_ratio,
        }

    def compute_metadata_anomaly_score(
        self,
        metadata_anomalies: list,
        layout_anomalies: list,
        font_consistency_score: float,
        metadata: dict,
    ) -> dict:
        risk = 0.0
        triggered_rules = []

        author = metadata.get("author", metadata.get("Author", ""))
        if not author or len(str(author).strip()) < 2:
            risk += 0.20
            triggered_rules.append("missing_author")

        creation_date = metadata.get("creation_date", metadata.get("CreationDate", ""))
        mod_date      = metadata.get("modification_date", metadata.get("ModDate", ""))
        if not creation_date:
            risk += 0.15
            triggered_rules.append("missing_creation_date")
        elif mod_date and creation_date and str(mod_date) < str(creation_date):
            risk += 0.30
            triggered_rules.append("creation_date_mismatch")

        anomaly_set = set(a.lower() if isinstance(a, str) else str(a) for a in metadata_anomalies)
        if "excessive_font_variety" in anomaly_set:
            risk += 0.20
            triggered_rules.append("too_many_fonts")
        if "inconsistent_font_sizes" in anomaly_set:
            risk += 0.20
            triggered_rules.append("inconsistent_spacing")

        layout_set = set(a.lower() if isinstance(a, str) else str(a) for a in layout_anomalies)
        if layout_set:
            risk += 0.10
            triggered_rules.extend(list(layout_set))

        if font_consistency_score < 0.5:
            risk += 0.10
            triggered_rules.append("low_font_consistency")

        risk = min(risk, 1.0)

        return {
            "tool":            "Metadata Anomaly Scanner (rule-based)",
            "risk_score":      round(risk, 4),
            "triggered_rules": triggered_rules,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # URL HANDLER — Full Web-Scraping + SerpAPI Cross-Verification
    # ══════════════════════════════════════════════════════════════════════════
    async def _url(self, f: dict) -> dict:
        """
        Comprehensive URL/article authenticity analysis.
        6 Signals:
          1. AI Text Detection (RoBERTa)           — Is the text AI-written?
          2. Burstiness / Writing Style             — Statistical writing analysis
          3. SerpAPI Cross-Reference                — Do other sources corroborate?
          4. Domain Trust                           — Is the domain reputable?
          5. Content Structure Analysis             — Signs of auto-generated pages
          6. Image Authenticity (if featured image) — Is the main image AI-generated?
        """
        url = f.get("url", "")
        logger.info(f"=== Starting URL Authenticity Evaluation for: {url} ===")
        start_time = time.time()

        if not url:
            return self._result(0.5, "error", "No URL provided")

        # ── 1. Scrape the page ───────────────────────────────────────────────
        scrape_result = await self._scrape_page(url)
        if scrape_result.get("error"):
            return self._result(0.5, "error", f"Could not scrape URL: {scrape_result['error']}")

        body_text    = scrape_result.get("body_text", "")
        page_title   = scrape_result.get("title", "")
        meta_desc    = scrape_result.get("meta_description", "")
        author       = scrape_result.get("author", "")
        publish_date = scrape_result.get("publish_date", "")
        image_urls   = scrape_result.get("image_urls", [])
        link_count   = scrape_result.get("internal_link_count", 0)
        ext_link_count = scrape_result.get("external_link_count", 0)
        word_count   = len(body_text.split()) if body_text else 0

        if word_count < 20:
            return self._result(0.5, "uncertain", "Page has too little text to analyze meaningfully.")

        # ── 1.5 SOCIAL MEDIA PROFILE DETECTION ──────────────────────────────
        social_domains = ["instagram.com", "twitter.com", "x.com", "facebook.com", "tiktok.com", "youtube.com"]
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if any(sd in domain for sd in social_domains):
            logger.info(f"Social media profile detected: {domain}. Routing to profile handler.")
            return await self._social_media_profile(scrape_result, url)

        # ── 2. Run all signals IN PARALLEL ───────────────────────────────────
        burstiness_ratio = f.get("burstiness_ratio", 0.0)
        sent_len_var     = f.get("sentence_length_variance", 0.0)
        avg_sent_len     = f.get("avg_sentence_length", 0.0)

        tasks = [
            self.detect_ai_text_roberta(body_text),                          # Signal 1
            asyncio.to_thread(
                self.compute_burstiness_score,
                burstiness_ratio, sent_len_var, avg_sent_len
            ),                                                               # Signal 2
            self._serpapi_cross_reference(page_title, body_text, url),        # Signal 3
            asyncio.to_thread(self._analyze_domain_trust, url),              # Signal 4
            asyncio.to_thread(
                self._analyze_content_structure, scrape_result
            ),                                                               # Signal 5
        ]

        # Signal 6: Image authenticity (optional)
        has_image = False
        if image_urls:
            has_image = True
            tasks.append(self._image({"url": image_urls[0]}))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── 3. Unpack results with safe error handling ───────────────────────
        ai_text_result = results[0] if not isinstance(results[0], Exception) else {"ai_generated_score": None}
        burstiness_result = results[1] if not isinstance(results[1], Exception) else {"ai_likelihood": 0.5}
        serp_result = results[2] if not isinstance(results[2], Exception) else {"cross_ref_score": 0.5}
        domain_result = results[3] if not isinstance(results[3], Exception) else {"trust_score": 0.5}
        structure_result = results[4] if not isinstance(results[4], Exception) else {"risk_score": 0.5}

        image_result = None
        if has_image:
            image_result = results[5] if not isinstance(results[5], Exception) else None

        ai_text_score: float     = float(ai_text_result.get("ai_generated_score") or 0.5)
        burstiness_score: float  = float(burstiness_result.get("ai_likelihood", 0.5))
        cross_ref_score: float   = float(serp_result.get("cross_ref_score", 0.5))
        domain_trust: float      = float(domain_result.get("trust_score", 0.5))
        structure_risk: float    = float(structure_result.get("risk_score", 0.5))
        image_auth_score: float  = float(image_result.get("score", 0.5)) if image_result else 0.0

        # ── 4. Weighted Final Score ──────────────────────────────────────────
        # domain_trust is 1=trustworthy, need to invert for fake score
        domain_suspicion = 1.0 - domain_trust

        if has_image:
            final_score = (
                ai_text_score     * 0.25
                + burstiness_score * 0.10
                + cross_ref_score  * 0.25
                + domain_suspicion * 0.10
                + structure_risk   * 0.10
                + image_auth_score * 0.20
            )
            weights_used = {
                "ai_text": 0.25, "burstiness": 0.10, "cross_reference": 0.25,
                "domain_trust": 0.10, "structure": 0.10, "image_auth": 0.20,
            }
        else:
            final_score = (
                ai_text_score     * 0.30
                + burstiness_score * 0.15
                + cross_ref_score  * 0.30
                + domain_suspicion * 0.10
                + structure_risk   * 0.15
            )
            weights_used = {
                "ai_text": 0.30, "burstiness": 0.15, "cross_reference": 0.30,
                "domain_trust": 0.10, "structure": 0.15,
            }

        final_score = max(0.0, min(1.0, final_score))

        # ── 5. Label ─────────────────────────────────────────────────────────
        if final_score < 0.35:
            label = "real"
        elif final_score < 0.50:
            label = "uncertain"
        else:
            label = "fake"

        # ── 6. LLM Explanation ───────────────────────────────────────────────
        explanation = await self._generate_llm_explanation(
            final_score, label, {
                "content_type":      "url/article",
                "ai_text_score":     ai_text_score,
                "burstiness_score":  burstiness_score,
                "cross_ref_score":   cross_ref_score,
                "domain_trust":      domain_trust,
                "structure_risk":    structure_risk,
                "image_auth_score":  image_auth_score if has_image else "no image",
                "word_count":        word_count,
                "page_title":        page_title,
                "corroborating_sources": serp_result.get("corroborating_count", 0),
                "contradicting_sources": serp_result.get("contradicting_count", 0),
                "api_breakdown": {
                    "text_ai":          ai_text_score,
                    "burstiness":       burstiness_score,
                    "cross_reference":  cross_ref_score,
                    "domain":           domain_trust,
                    "structure":        structure_risk,
                    "image":            image_auth_score,
                },
            }
        )

        # ── 7. Flags ─────────────────────────────────────────────────────────
        flags = []
        if ai_text_score > 0.5:
            flags.append(f"Article text likely AI-generated ({ai_text_score:.2f})")
        if burstiness_score > 0.5:
            flags.append(f"Suspiciously uniform writing style ({burstiness_score:.2f})")
        if cross_ref_score > 0.5:
            flags.append(f"Claims not corroborated by other sources ({cross_ref_score:.2f})")
        if domain_trust < 0.3:
            flags.append(f"Domain has low trust indicators ({domain_trust:.2f})")
        if structure_risk > 0.5:
            flags.append(f"Page structure suggests auto-generated content ({structure_risk:.2f})")
        if has_image and image_auth_score > 0.5:
            flags.append(f"Featured image flagged as suspicious ({image_auth_score:.2f})")
        if not author:
            flags.append("No author attribution found")
        if serp_result.get("contradicting_count", 0) > 0:
            flags.append(f"{serp_result['contradicting_count']} sources contradict key claims")

        total_duration = time.time() - start_time
        logger.info(f"=== URL evaluation complete in {total_duration:.2f}s ===")

        component_scores = {
            "text_ai_detection": round(ai_text_score, 4),
            "burstiness":        round(burstiness_score, 4),
            "cross_reference":   round(cross_ref_score, 4),
            "domain_trust":      round(domain_trust, 4),
            "content_structure":  round(structure_risk, 4),
        }
        if has_image:
            component_scores["image_authenticity"] = round(image_auth_score, 4)

        return self._result(
            score=round(final_score, 4),
            label=label,
            explanation=explanation,
            flags=flags,
            details={
                "weights": weights_used,
                "component_scores": component_scores,
                "signal_details": {
                    "roberta":        ai_text_result,
                    "burstiness":     burstiness_result,
                    "cross_reference": serp_result,
                    "domain":          domain_result,
                    "structure":       structure_result,
                    "image":           image_result if has_image else None,
                },
                "article_stats": {
                    "url":                 url,
                    "page_title":          page_title,
                    "author":              author or "Unknown",
                    "publish_date":        publish_date or "Unknown",
                    "word_count":          word_count,
                    "burstiness_ratio":    burstiness_ratio,
                    "avg_sentence_len":    avg_sent_len,
                    "has_featured_image":  has_image,
                    "internal_links":      link_count,
                    "external_links":      ext_link_count,
                    "analysis_duration":   round(total_duration, 2),
                },
                "api_raw": {
                    "roberta":         ai_text_result,
                    "serp_api":        serp_result,
                    "domain_analysis": domain_result,
                },
            },
        )

    # ══════════════════════════════════════════════════════════════════════════
    # URL: WEB SCRAPING ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    async def _scrape_page(self, url: str) -> dict:
        """
        Deep scrapes a URL and extracts structured content:
        title, meta, author, date, body text, images, links.
        """
        st = time.time()
        logger.info(f"Scraping page: {url}")
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

            if not BS4_AVAILABLE:
                return {"error": "BeautifulSoup not installed"}

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # ── Title ────────────────────────────────────────────────────────
            title = ""
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)

            # ── Meta description ─────────────────────────────────────────────
            meta_desc = ""
            md_tag = soup.find("meta", attrs={"name": "description"})
            if md_tag and md_tag.get("content"):
                meta_desc = md_tag["content"].strip()

            # ── Author ───────────────────────────────────────────────────────
            author = ""
            for selector in [
                {"name": "author"},
                {"property": "article:author"},
                {"name": "twitter:creator"},
            ]:
                tag = soup.find("meta", attrs=selector)
                if tag and tag.get("content"):
                    author = tag["content"].strip()
                    break
            if not author:
                # Try common class patterns
                for cls in ["author", "byline", "writer", "author-name"]:
                    el = soup.find(class_=re.compile(cls, re.I))
                    if el:
                        author = el.get_text(strip=True)
                        break

            # ── Publish date ─────────────────────────────────────────────────
            publish_date = ""
            for attr in ["article:published_time", "datePublished", "date"]:
                tag = soup.find("meta", attrs={"property": attr}) or soup.find("meta", attrs={"name": attr})
                if tag and tag.get("content"):
                    publish_date = tag["content"].strip()
                    break
            if not publish_date:
                time_tag = soup.find("time")
                if time_tag:
                    publish_date = time_tag.get("datetime", time_tag.get_text(strip=True))

            # ── Body text ────────────────────────────────────────────────────
            article_el = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
                or soup.find("div", id=re.compile(r"content|article|post|entry", re.I))
                or soup
            )
            paragraphs = article_el.find_all("p")
            body_text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)

            if len(body_text) < 100:
                # Fallback: use all text
                body_text = article_el.get_text(separator=" ", strip=True)

            body_text = re.sub(r"\s+", " ", body_text).strip()[:8000]

            # ── Images ───────────────────────────────────────────────────────
            image_urls = []
            og_img = soup.find("meta", property="og:image")
            if og_img and og_img.get("content"):
                image_urls.append(og_img["content"])
            for img in article_el.find_all("img", src=True)[:5]:
                src = img["src"]
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    parsed = urlparse(url)
                    src = f"{parsed.scheme}://{parsed.netloc}{src}"
                if src.startswith("http") and src not in image_urls:
                    image_urls.append(src)

            # ── Links ────────────────────────────────────────────────────────
            parsed_url = urlparse(url)
            base_domain = parsed_url.netloc
            internal_links = 0
            external_links = 0
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    if base_domain in href:
                        internal_links += 1
                    else:
                        external_links += 1
                elif href.startswith("/"):
                    internal_links += 1

            # ── Posts/Snippets (Social Media special) ──────────────────────
            posts = []
            if any(sd in url for sd in ["instagram", "twitter", "x.com", "facebook", "tiktok"]):
                # Look for common post containers
                for p in soup.find_all(["div", "span", "p"], text=True):
                    txt = p.get_text(strip=True)
                    if 30 < len(txt) < 500 and txt not in posts:
                        posts.append(txt)
            
            dur = time.time() - st
            logger.info(f"Scrape complete. Title='{title[:50]}', words={len(body_text.split())}, images={len(image_urls)}, posts={len(posts)}, duration={dur:.2f}s")

            return {
                "body_text":           body_text,
                "posts":               posts[:10],
                "title":               title,
                "meta_description":    meta_desc,
                "author":              author,
                "publish_date":        publish_date,
                "image_urls":          image_urls,
                "internal_link_count": internal_links,
                "external_link_count": external_links,
            }

        except Exception as e:
            logger.error(f"Scrape failed for {url}: {e}")
            return {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # URL: SERPAPI CROSS-REFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    async def _serpapi_cross_reference(self, title: str, body_text: str, source_url: str) -> dict:
        """
        Uses SerpAPI to search for the article's claims and checks whether
        other reputable sources corroborate or contradict the content.
        Returns a cross_ref_score: 0 = well-corroborated, 1 = no corroboration / suspicious.
        """
        st = time.time()
        logger.info("SerpAPI cross-reference starting...")

        if not SERPAPI_AVAILABLE or not SERPAPI_KEY:
            logger.warning("SerpAPI not available or key missing, skipping cross-reference.")
            return {
                "tool": "SerpAPI Cross-Reference",
                "cross_ref_score": 0.5,
                "error": "SerpAPI not configured",
                "corroborating_count": 0,
                "contradicting_count": 0,
            }

        try:
            # Build search query from title (most specific identifier)
            search_query = title.strip() if title else body_text[:120].strip()
            if not search_query:
                return {
                    "tool": "SerpAPI Cross-Reference",
                    "cross_ref_score": 0.5,
                    "error": "No title/text to search",
                }

            # ── Run two searches: title search + key claims search ───────────
            source_domain = urlparse(source_url).netloc

            # Search 1: Exact title search to find corroborating sources
            title_search = await asyncio.to_thread(
                self._run_serp_search, f'"{search_query}"'
            )

            # Search 2: Extract key claims and verify them
            key_claims = self._extract_key_claims(body_text)
            claim_results = []
            if key_claims:
                # Search for 1-2 key claims to save API quota
                for claim in key_claims[:2]:
                    result = await asyncio.to_thread(
                        self._run_serp_search, claim
                    )
                    claim_results.append({
                        "claim": claim,
                        "results": result,
                    })

            # ── Analyze corroboration ────────────────────────────────────────
            corroborating = 0
            contradicting = 0
            total_other_sources = 0
            corroborating_sources = []

            # Analyze title search results
            for item in title_search.get("organic_results", []):
                item_domain = urlparse(item.get("link", "")).netloc
                if item_domain and item_domain != source_domain:
                    total_other_sources += 1
                    corroborating += 1
                    corroborating_sources.append({
                        "title": item.get("title", ""),
                        "source": item_domain,
                        "snippet": item.get("snippet", "")[:150],
                    })

            # Analyze claim verification results
            claims_verified = 0
            claims_unverified = 0
            for cr in claim_results:
                found_corroboration = False
                for item in cr.get("results", {}).get("organic_results", []):
                    item_domain = urlparse(item.get("link", "")).netloc
                    if item_domain and item_domain != source_domain:
                        found_corroboration = True
                        break
                if found_corroboration:
                    claims_verified += 1
                else:
                    claims_unverified += 1

            # ── Score calculation ────────────────────────────────────────────
            # No other sources reporting = highly suspicious
            if total_other_sources == 0:
                base_score = 0.85  # Very suspicious — no one else reports this
            elif total_other_sources == 1:
                base_score = 0.60
            elif total_other_sources <= 3:
                base_score = 0.35
            else:
                base_score = 0.10  # Well-corroborated

            # Adjust for claim verification
            if key_claims:
                total_claims = claims_verified + claims_unverified
                if total_claims > 0:
                    claim_penalty = (claims_unverified / total_claims) * 0.3
                    base_score = min(1.0, base_score + claim_penalty)

            # Check for fact-check articles in search results
            fact_check_found = any(
                "fact" in item.get("title", "").lower() and "check" in item.get("title", "").lower()
                for item in title_search.get("organic_results", [])
            )
            if fact_check_found:
                base_score = min(1.0, base_score + 0.15)

            dur = time.time() - st
            logger.info(
                f"SerpAPI cross-reference complete. Score: {base_score:.2f}, "
                f"corroborating: {corroborating}, claims_verified: {claims_verified}/{len(key_claims)} "
                f"(took {dur:.2f}s)"
            )

            return {
                "tool":                 "SerpAPI Cross-Reference",
                "cross_ref_score":      round(base_score, 4),
                "corroborating_count":  corroborating,
                "contradicting_count":  claims_unverified,
                "total_other_sources":  total_other_sources,
                "claims_verified":      claims_verified,
                "claims_total":         len(key_claims),
                "fact_check_found":     fact_check_found,
                "corroborating_sources": corroborating_sources[:5],
                "key_claims_searched":  [cr["claim"] for cr in claim_results],
            }

        except Exception as e:
            dur = time.time() - st
            logger.error(f"SerpAPI cross-reference failed: {e} (took {dur:.2f}s)")
            return {
                "tool": "SerpAPI Cross-Reference",
                "cross_ref_score": 0.5,
                "error": str(e),
                "corroborating_count": 0,
                "contradicting_count": 0,
            }

    def _run_serp_search(self, query: str) -> dict:
        """Execute a single SerpAPI Google search (synchronous, use in thread)."""
        try:
            params = {
                "q": query,
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "num": 10,
                "gl": "us",
                "hl": "en",
            }
            search = GoogleSearch(params)
            return search.get_dict()
        except Exception as e:
            logger.error(f"SerpAPI search failed for query '{query[:60]}': {e}")
            return {"organic_results": []}

    def _extract_key_claims(self, text: str) -> list:
        """
        Extracts the most specific, verifiable claims from article text.
        Looks for sentences containing numbers, names, dates, quotes.
        """
        sentences = re.split(r'[.!?]+', text)
        claims = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30 or len(sent) > 200:
                continue

            # Prioritise sentences with verifiable content
            has_number = bool(re.search(r'\d+', sent))
            has_quote = bool(re.search(r'["\u201c\u201d]', sent))
            has_proper_noun = bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', sent))
            has_date = bool(re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{4})\b', sent, re.I))

            score = sum([has_number, has_quote, has_proper_noun, has_date])
            if score >= 2:
                claims.append((score, sent))

        # Sort by specificity score, take top 3
        claims.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in claims[:3]]

    # ══════════════════════════════════════════════════════════════════════════
    # URL: DOMAIN TRUST ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def _analyze_domain_trust(self, url: str) -> dict:
        """
        Analyzes domain reputation based on:
        - Known trusted/untrusted domain lists
        - Domain age (via WHOIS)
        - TLD reputation
        - Domain structure patterns
        Returns trust_score: 0 = untrusted, 1 = highly trusted.
        """
        st = time.time()
        logger.info(f"Domain trust analysis starting for: {url}")

        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")

        trust_score = 0.5  # Neutral default
        triggered = []

        # ── Known reputable outlets ──────────────────────────────────────────
        trusted_domains = {
            # Major wire services
            "reuters.com", "apnews.com", "afp.com",
            # International broadcasters
            "bbc.com", "bbc.co.uk", "cnn.com", "aljazeera.com", "france24.com",
            "dw.com", "nhk.or.jp",
            # Major newspapers
            "nytimes.com", "washingtonpost.com", "theguardian.com",
            "ft.com", "wsj.com", "economist.com", "lemonde.fr",
            # MENA region trusted
            "arabnews.com", "thenationalnews.com", "gulfnews.com",
            "middleeasteye.net", "al-monitor.com",
            # Fact checkers
            "snopes.com", "factcheck.org", "politifact.com",
            "fullfact.org", "misbar.com",
            # Government / institutional
            "who.int", "un.org", "nasa.gov",
            # Science / education
            "nature.com", "science.org", "arxiv.org",
        }

        known_unreliable = {
            "infowars.com", "naturalnews.com", "beforeitsnews.com",
            "yournewswire.com", "worldnewsdailyreport.com",
            "theonion.com", "babylonbee.com",  # satire
        }

        if domain in trusted_domains:
            trust_score = 0.95
            triggered.append("known_trusted_outlet")
        elif domain in known_unreliable:
            trust_score = 0.10
            triggered.append("known_unreliable_source")
        else:
            # ── TLD analysis ──────────────────────────────────────────────
            if TLDEXTRACT_AVAILABLE:
                ext = tldextract.extract(url)
                tld = ext.suffix

                trusted_tlds = {"com", "org", "net", "edu", "gov", "int", "co.uk", "ac.uk"}
                suspicious_tlds = {"xyz", "top", "buzz", "click", "info", "site", "online", "live", "club", "work"}

                if tld in suspicious_tlds:
                    trust_score -= 0.15
                    triggered.append(f"suspicious_tld: .{tld}")
                elif tld in {"gov", "edu", "int"}:
                    trust_score += 0.20
                    triggered.append(f"authoritative_tld: .{tld}")

                # Subdomain depth
                if ext.subdomain and ext.subdomain.count(".") > 1:
                    trust_score -= 0.10
                    triggered.append("deep_subdomain")

            # ── Domain age (WHOIS) ───────────────────────────────────────
            try:
                import whois
                w = whois.whois(domain)
                creation_date = w.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                if creation_date:
                    from datetime import datetime
                    age_days = (datetime.now() - creation_date).days
                    if age_days < 90:
                        trust_score -= 0.25
                        triggered.append(f"very_new_domain: {age_days} days")
                    elif age_days < 365:
                        trust_score -= 0.10
                        triggered.append(f"new_domain: {age_days} days")
                    elif age_days > 3650:  # 10+ years
                        trust_score += 0.10
                        triggered.append(f"established_domain: {age_days // 365} years")
            except Exception as e:
                logger.debug(f"WHOIS lookup failed for {domain}: {e}")
                triggered.append("whois_unavailable")

            # ── Domain name heuristics ───────────────────────────────────
            # Excessive hyphens
            if domain.count("-") > 2:
                trust_score -= 0.10
                triggered.append("excessive_hyphens")

            # Very long domain
            if len(domain) > 40:
                trust_score -= 0.10
                triggered.append("very_long_domain")

            # Domain mimics known brand (typosquatting)
            for trusted in ["reuters", "bbc", "cnn", "nytimes", "guardian", "washingtonpost"]:
                if trusted in domain and domain not in trusted_domains:
                    trust_score -= 0.20
                    triggered.append(f"possible_typosquatting: mimics {trusted}")
                    break

        trust_score = max(0.0, min(1.0, trust_score))

        dur = time.time() - st
        logger.info(f"Domain trust analysis complete. Domain={domain}, trust={trust_score:.2f} (took {dur:.2f}s)")

        return {
            "tool":              "Domain Trust Analysis",
            "domain":            domain,
            "trust_score":       round(trust_score, 4),
            "triggered_rules":   triggered,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # URL: CONTENT STRUCTURE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def _analyze_content_structure(self, scrape_result: dict) -> dict:
        """
        Detects signs of auto-generated or low-quality content:
        - Missing metadata (author, date)
        - Thin content
        - Keyword stuffing
        - No outbound references
        - Suspicious link ratios
        Returns risk_score: 0 = legitimate, 1 = likely auto-generated.
        """
        risk = 0.0
        triggered = []

        body_text   = scrape_result.get("body_text", "")
        title       = scrape_result.get("title", "")
        author      = scrape_result.get("author", "")
        date        = scrape_result.get("publish_date", "")
        int_links   = scrape_result.get("internal_link_count", 0)
        ext_links   = scrape_result.get("external_link_count", 0)
        word_count  = len(body_text.split()) if body_text else 0

        # ── Missing metadata ─────────────────────────────────────────────────
        if not author:
            risk += 0.10
            triggered.append("no_author")
        if not date:
            risk += 0.05
            triggered.append("no_publish_date")
        if not title:
            risk += 0.10
            triggered.append("no_title")

        # ── Thin content ─────────────────────────────────────────────────────
        if word_count < 100:
            risk += 0.20
            triggered.append(f"very_thin_content: {word_count} words")
        elif word_count < 300:
            risk += 0.10
            triggered.append(f"thin_content: {word_count} words")

        # ── No external references ───────────────────────────────────────────
        if ext_links == 0 and word_count > 200:
            risk += 0.15
            triggered.append("no_external_references")

        # ── Keyword stuffing detection ───────────────────────────────────────
        if body_text and title:
            title_words = set(title.lower().split())
            stop_words = {"the", "a", "an", "is", "in", "to", "of", "and", "for", "on", "at", "by", "with", "from"}
            title_keywords = title_words - stop_words
            if title_keywords:
                text_lower = body_text.lower()
                keyword_density = sum(
                    text_lower.count(kw) for kw in title_keywords
                ) / max(word_count, 1)
                if keyword_density > 0.08:
                    risk += 0.15
                    triggered.append(f"possible_keyword_stuffing: density={keyword_density:.3f}")

        # ── Title is clickbait-style ─────────────────────────────────────────
        clickbait_patterns = [
            r"you won.?t believe",
            r"shocking",
            r"\d+ (reasons|ways|things|facts|secrets)",
            r"this (one|simple) trick",
            r"what .+ doesn.?t want you to know",
            r"breaking.*:",
        ]
        if title:
            for pattern in clickbait_patterns:
                if re.search(pattern, title, re.I):
                    risk += 0.10
                    triggered.append(f"clickbait_title_pattern: {pattern}")
                    break

        # ── ALL CAPS title ───────────────────────────────────────────────────
        if title and title == title.upper() and len(title) > 10:
            risk += 0.10
            triggered.append("all_caps_title")

        risk = max(0.0, min(1.0, risk))

        return {
            "tool":            "Content Structure Analyzer",
            "risk_score":      round(risk, 4),
            "triggered_rules": triggered,
        }
    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO HANDLER
    # ══════════════════════════════════════════════════════════════════════════
    async def _video(self, f: dict) -> dict:
        """
        Orchestrates video authenticity analysis using the modular forensic pipeline.
        Samples frames and runs deepfake detection on detected faces.
        """
        video_path = f.get("video_path")
        video_url  = f.get("video_url")
        logger.info(f"=== Starting VIDEO Authenticity Evaluation for: {video_path or video_url} ===")
        start_time = time.time()

        if not VIDEO_FORENSICS_AVAILABLE:
            return self._result(0.5, "error", "Video forensic modules not installed or missing.")

        # ── 1. Prepare video path ─────────────────────────────────────────────
        try:
            if not video_path and video_url:
                # FUTURE: implement download_video_from_url
                return self._result(0.5, "error", "Video URL download not yet implemented")

            if not video_path or not os.path.exists(video_path):
                return self._result(0.5, "error", f"Video file not found: {video_path}")

        except Exception as e:
            logger.error(f"Failed to prepare video path: {e}")
            return self._result(0.5, "error", f"Could not process video: {e}")

        # ── 2. Run InferenceEngine ────────────────────────────────────────────
        try:
            # Lazy initialization of the inference engine
            if not hasattr(self, "video_engine"):
                logger.info("Initializing Video Inference Engine (VeridisQuo)...")
                config = InferenceConfig() # Use default configuration
                self.video_engine = InferenceEngine(config=config)

            # Sampling parameters
            fps_sampling = 1 # Sample 1 frame per second by default for speed
            
            logger.info(f"Running video inference on: {video_path} at {fps_sampling} FPS...")
            video_result: VideoInferenceResult = await asyncio.to_thread(
                self.video_engine.predict_video,
                video_path=video_path,
                frames_per_second=fps_sampling,
                aggregation_method="weighted_average"
            )

            # ── 3. Normalise results → [0, 1] (1 = fake) ──────────────────────
            # veridisquo_25M.pth outputs FAKE=0, REAL=1 class indices? 
            # Wait, check InferenceEngine.CLASS_LABELS: ["FAKE", "REAL"]
            # So index 0 = FAKE, index 1 = REAL.
            
            # aggregate_prediction is label "FAKE" or "REAL"
            # aggregate_confidence is prob of THAT label.
            
            if video_result.aggregate_prediction == "FAKE":
                final_score = float(video_result.aggregate_confidence)
            else:
                final_score = 1.0 - float(video_result.aggregate_confidence)

            # Cap score for safety
            final_score = max(0.0, min(1.0, final_score))

            if final_score < 0.35:
                label = "real"
            elif final_score < 0.50:
                label = "uncertain"
            else:
                label = "fake"

            # ── 4. Generate AI Explanation ───────────────────────────────────
            explanation = await self._generate_llm_explanation(
                final_score, label, {
                    "content_type":  "video",
                    "final_score":   final_score,
                    "num_frames":    video_result.num_frames_analyzed,
                    "api_breakdown": {
                        "frame_by_frame_deepfake": final_score,
                    }
                }
            )

            # ── 5. Build Flags ────────────────────────────────────────────────
            flags = []
            if final_score > 0.5:
                flags.append(f"Deepfake detection high confidence ({final_score:.2f})")
            if video_result.num_frames_analyzed < 3:
                 flags.append("Limited frame analysis — results may be less reliable")

            total_duration = time.time() - start_time
            logger.info(f"=== VIDEO evaluation complete in {total_duration:.2f}s ===")

            return self._result(
                score=round(final_score, 4),
                label=label,
                explanation=explanation,
                flags=flags,
                details={
                    "component_scores": {
                        "deepfake_detection": round(final_score, 4),
                    },
                    "video_metadata": {
                        "num_frames_analyzed": video_result.num_frames_analyzed,
                        "aggregation_method":  video_result.aggregation_metadata.get("method"),
                        "analysis_duration":   round(total_duration, 2),
                    },
                    "api_raw": {
                        "in_house_detector": {
                            "prediction": video_result.aggregate_prediction,
                            "confidence": video_result.aggregate_confidence,
                            "raw_metadata": video_result.aggregation_metadata
                        }
                    }
                },
            )

        except Exception as e:
            logger.error(f"Video forensic engine failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._result(0.5, "error", f"Video analysis failed: {str(e)}")
    def _default(self) -> dict:
        """Fallback for unknown media types."""
        return self._result(0.5, "unknown", "No handler for this media type")

    async def _social_media_profile(self, scrape: dict, url: str) -> dict:
        """
        Special handler for social media profiles.
        Analyzes the first 3 images and first 3 posts found in the feed.
        """
        st = time.time()
        logger.info(f"=== Starting Social Media Profile Verification for: {url} ===")
        
        images = scrape.get("image_urls", [])[:3]
        posts = scrape.get("posts", [])[:3]
        
        if not images and not posts:
            return self._result(0.5, "uncertain", "Could not find any images or posts to verify on this profile.")

        # ── 1. Create Parallel Tasks ───────────────────────────
        tasks = []
        for img_url in images:
            tasks.append(self._image({"url": img_url}))
        
        for post_text in posts:
            tasks.append(self.detect_ai_text_roberta(post_text))

        # ── 2. Run Tasks with timeout ─────────────────────────
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # ── 3. Aggregate ──────────────────────────────────────
        image_scores = []
        text_scores = []
        flags = []
        
        for i, res in enumerate(results):
            if isinstance(res, Exception): continue
            
            # If it's an image result (from _image)
            if "score" in res and "verdict" in res:
                image_scores.append(res["score"])
                if res["label"] == "fake":
                    flags.append(f"AI/Manipulated image detected in feed (Score: {res['score']:.2f})")
            
            # If it's a text result (from detect_ai_text_roberta)
            elif "ai_generated_score" in res:
                score = res.get("ai_generated_score")
                if score is not None:
                    text_scores.append(score)
                    if score > 0.6:
                        flags.append(f"AI-generated text detected in posts (Score: {score:.2f})")

        # ── 4. Calculate Final Score ──────────────────────────
        total_scores = image_scores + text_scores
        if not total_scores:
            return self._result(0.5, "uncertain", "Verification failed: All content analysis tasks returned errors.")
            
        final_auth_score = sum(total_scores) / len(total_scores)
        
        if final_auth_score < 0.35:
            label = "real"
        elif final_auth_score < 0.50:
            label = "uncertain"
        else:
            label = "fake"

        # ── 5. Detailed Explanation ───────────────────────────
        explanation = await self._generate_llm_explanation(
            final_auth_score, label, {
                "content_type": f"Profile ({urlparse(url).netloc})",
                "images_checked": len(image_scores),
                "texts_checked": len(text_scores),
                "flags_found": len(flags),
                "avg_image_risk": sum(image_scores)/len(image_scores) if image_scores else 0,
                "avg_text_risk": sum(text_scores)/len(text_scores) if text_scores else 0,
            }
        )

        return self._result(
            score=round(final_auth_score, 4),
            label=label,
            explanation=explanation,
            flags=list(set(flags)), # unique flags
            details={
                "images_verdicts": image_scores,
                "text_verdicts": text_scores,
                "duration": time.time() - st,
                "profile_url": url
            }
        )

    # ══════════════════════════════════════════════════════════════════════════
    # LLM EXPLANATION
    # ══════════════════════════════════════════════════════════════════════════
    async def _generate_llm_explanation(self, score: float, label: str, signals: dict) -> str:
        if not GROQ_AVAILABLE or not GROQ_API_KEY:
            return f"Analysis complete. Score: {score:.2f} ({label})."

        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=GROQ_API_KEY)

            content_type = signals.get("content_type", "image")
            prompt = (
                f"Explain concisely why this {content_type} was labeled as '{label}' "
                f"(weighted suspiciousness score: {score:.2f}, where 0 is real and 1 is fake). "
                f"Individual detector AI-probability scores (lower is more authentic): "
                f"{json.dumps(signals.get('api_breakdown', {}))}"
            )

            chat_completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Digital Forensic Analyst. Concisely explain the verdict in 1-2 sentences. "
                            "Do NOT mention raw numerical scores or percentages (e.g., skip '0.8' or '92%'). "
                            "Instead, use descriptive terms like 'highly suspicious', 'authentic signature', "
                            "'consistent features', or 'minor anomalies' to explain based on the forensic evidence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.1-8b-instant",
                max_tokens=150,
                temperature=0.3,
                timeout=10.0,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling Groq Async: {e}")
            return f"Analysis complete. Score: {score:.2f} ({label})."

    # ══════════════════════════════════════════════════════════════════════════
    # RESULT BUILDER
    # ══════════════════════════════════════════════════════════════════════════
    def _result(
        self,
        score:       float,
        label:       str,
        explanation: str,
        flags:       list = None,
        details:     dict = None,
    ) -> dict:
        out = {
            "score":       score,
            "label":       label,
            "explanation": explanation,
            "flags":       flags or [],
        }
        if details:
            out["details"] = details
        return out