"""
AXIS 2 - Video Contextual Consistency Handler.

Three-layer video analysis:
1) transcript extraction (Whisper)
2) transcript context checks (NLI + timeline consistency)
3) frame-level visual context (CLIP + consistency heuristics)
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from sentence_transformers import util

import logging

logger = logging.getLogger("VideoHandler")

try:
    from .axis2_contextual_consistency import check_claim_consistency_nli, _extract_year
except ImportError as e:
    logger.error(f"Failed to import from axis2_contextual_consistency: {e}")
    # Fallback if needed or let it fail
    raise

try:
    from .sentence_model_singleton import get_sentence_model
except ImportError as e:
    logger.error(f"Failed to import from sentence_model_singleton: {e}")
    raise


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class FrameAnalysis:
    frame_index: int
    timestamp_sec: float
    clip_score: float
    face_count: int
    brightness: float
    dominant_colors: list[str]
    is_scene_change: bool


_WHISPER_MODEL = None
_CLIP_MODEL_VIDEO = None
_CLIP_PREPROCESS_VIDEO = None
_CLIP_DEVICE_VIDEO = None

_CLAIM_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "were", "with", "video", "shows", "about",
    "promotes", "show", "your", "you", "we", "our",
}


def _load_whisper(model_size: str = "base"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper") from exc
        _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL


def extract_audio_from_video(video_path: str) -> str:
    logger.info(f"Extracting audio from {video_path}")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_name = tmp.name
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        tmp_name,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg audio extraction failed: {result.stderr[:300]}")
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:300]}")
        logger.info(f"Successfully extracted audio to {tmp_name}")
        return tmp_name
    except FileNotFoundError:
        logger.error("ffmpeg command not found. Please install ffmpeg and add it to your PATH.")
        raise RuntimeError("ffmpeg command not found. Audio extraction required for transcription.")


def transcribe_video(
    video_path: str,
    whisper_model_size: str = "base",
    language: str | None = None,
) -> tuple[str, list[TranscriptSegment]]:
    audio_path = None
    try:
        audio_path = extract_audio_from_video(video_path)
        model = _load_whisper(whisper_model_size)
        options = {"fp16": False}
        if language:
            options["language"] = language

        result = model.transcribe(audio_path, **options)
        full_text = str(result.get("text", "")).strip()
        raw_segments = result.get("segments", [])
        segments = [
            TranscriptSegment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")).strip(),
            )
            for s in raw_segments
            if str(s.get("text", "")).strip()
        ]
        return full_text, segments
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


def _parse_timestamp_seconds(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return (int(h) * 3600) + (int(m) * 60) + float(s)
        if len(parts) == 2:
            m, s = parts
            return (int(m) * 60) + float(s)
        return float(text)
    except ValueError:
        return None


def _dicts_to_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:
    normalized: list[TranscriptSegment] = []
    for item in raw_segments:
        start = _parse_timestamp_seconds(item.get("start"))
        end = _parse_timestamp_seconds(item.get("end"))
        text = str(item.get("text", "")).strip()
        if start is None or end is None or not text:
            continue
        normalized.append(TranscriptSegment(start=start, end=end, text=text))
    return normalized


def _fmt_ts(seconds: float) -> str:
    mins, secs = divmod(int(max(0.0, seconds)), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def resolve_transcript(
    video_path: str | None,
    transcript_text: str | None,
    transcript_segments: list[dict] | None,
    whisper_model_size: str = "base",
    language: str | None = None,
) -> tuple[str, list[TranscriptSegment], str]:
    if transcript_text and transcript_text.strip():
        return transcript_text.strip(), _dicts_to_segments(transcript_segments or []), "provided"

    if transcript_segments:
        segs = _dicts_to_segments(transcript_segments)
        joined = " ".join(seg.text for seg in segs if seg.text)
        return joined, segs, "segments_joined"

    if video_path and os.path.exists(video_path):
        text, segs = transcribe_video(video_path, whisper_model_size=whisper_model_size, language=language)
        return text, segs, "whisper"

    raise ValueError(
        "No transcript source available. Provide one of: transcript_text, transcript_segments, or video_path."
    )


def check_transcript_temporal_consistency(transcript_segments) -> dict:
    """
    Validates ordering, overlap and inverted ranges.
    Accepts list[TranscriptSegment] or list[dict].
    """
    raw_segments = transcript_segments or []
    if raw_segments and isinstance(raw_segments[0], dict):
        segments = _dicts_to_segments(raw_segments)
    else:
        segments = raw_segments

    if not segments:
        return {
            "tool": "Transcript Temporal Consistency",
            "consistent": True,
            "checked_segments": 0,
            "issues": [],
            "overlap_count": 0,
            "out_of_order_count": 0,
            "inverted_range_count": 0,
        }

    prev_start = None
    prev_end = None
    overlap_count = 0
    out_of_order_count = 0
    inverted_range_count = 0
    issues: list[str] = []

    for idx, seg in enumerate(segments):
        start = float(seg.start)
        end = float(seg.end)

        if end < start:
            inverted_range_count += 1
            issues.append(f"segment_{idx}_inverted_range")

        if prev_start is not None and start < prev_start:
            out_of_order_count += 1
            issues.append(f"segment_{idx}_out_of_order")

        if prev_end is not None and start < prev_end:
            overlap_count += 1
            issues.append(f"segment_{idx}_overlap")

        prev_start = start
        prev_end = max(end, prev_end or end)

    return {
        "tool": "Transcript Temporal Consistency",
        "consistent": not issues,
        "checked_segments": len(segments),
        "issues": issues,
        "overlap_count": overlap_count,
        "out_of_order_count": out_of_order_count,
        "inverted_range_count": inverted_range_count,
    }


def detect_segment_anomalies(
    claim_text: str,
    segments: list[TranscriptSegment],
    window_size: int = 3,
    contradiction_threshold: float = 0.65,
) -> list[dict]:
    """
    Runs NLI over overlapping sliding windows and returns exact timestamps
    where claim/transcript contradictions are detected.
    """
    if not segments or not claim_text.strip():
        return []

    anomalies: list[dict] = []
    step = max(1, window_size // 2)

    for i in range(0, len(segments), step):
        window = segments[i : i + window_size]
        if not window:
            continue

        window_text = " ".join(s.text for s in window if s.text).strip()
        if not window_text:
            continue

        start_ts = float(window[0].start)
        end_ts = float(window[-1].end)

        nli = check_claim_consistency_nli(claim_text, window_text)
        label = str(nli.get("nli_label", "neutral")).lower()
        score_map = nli.get("scores", {}) if isinstance(nli.get("scores"), dict) else {}
        contra_score = float(nli.get("contradiction_score", score_map.get("contradiction", 0.0)))

        if label == "contradiction" or contra_score >= contradiction_threshold:
            anomalies.append(
                {
                    "start_sec": start_ts,
                    "end_sec": end_ts,
                    "start_fmt": _fmt_ts(start_ts),
                    "end_fmt": _fmt_ts(end_ts),
                    "nli_label": label,
                    "contradiction_score": round(contra_score, 4),
                    "window_text": window_text[:300],
                    "issue_type": "nli_contradiction",
                    "flag": "segment_contradiction",
                }
            )

    return anomalies


def detect_semantic_drift(
    segments: list[TranscriptSegment],
    window_size: int = 5,
    drift_threshold: float = 0.30,
) -> list[dict]:
    """
    Detects narrative shifts using rolling cosine distance between transcript windows.
    """
    if len(segments) <= window_size:
        return []

    texts = [s.text for s in segments]
    starts = [float(s.start) for s in segments]
    ends = [float(s.end) for s in segments]
    model = get_sentence_model()

    drift_events: list[dict] = []

    for i in range(window_size, len(texts)):
        prev_window = " ".join(texts[i - window_size : i]).strip()
        curr_window = " ".join(texts[i : i + window_size]).strip()
        if not prev_window or not curr_window:
            continue

        emb_prev = model.encode(prev_window, convert_to_tensor=True)
        emb_curr = model.encode(curr_window, convert_to_tensor=True)
        similarity = float(util.cos_sim(emb_prev, emb_curr))
        drift = 1.0 - similarity

        if drift >= drift_threshold:
            end_index = min(i + window_size - 1, len(ends) - 1)
            drift_events.append(
                {
                    "start_sec": starts[i],
                    "end_sec": ends[end_index],
                    "start_fmt": _fmt_ts(starts[i]),
                    "end_fmt": _fmt_ts(ends[end_index]),
                    "drift_score": round(drift, 4),
                    "issue_type": "semantic_drift",
                    "flag": "semantic_drift",
                    "description": f"Topic shift detected (cosine distance {drift:.2f})",
                }
            )

    return drift_events


def _extract_claim_terms(claim_text: str) -> list[str]:
    words = re.findall(r"[A-Za-z]{3,}", claim_text or "")
    return [w.lower() for w in words if w.lower() not in _CLAIM_STOPWORDS]


def _extract_claim_anchors(claim_text: str) -> list[str]:
    text = claim_text or ""
    proper_nouns = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    years = re.findall(r"\b(?:19\d{2}|20\d{2}|21\d{2})\b", text)

    anchors: list[str] = []
    for token in proper_nouns + years:
        if token.lower() not in _CLAIM_STOPWORDS and token not in anchors:
            anchors.append(token)
    return anchors


def _compute_claim_alignment(claim_text: str, transcript_text: str) -> dict:
    claim = (claim_text or "").strip()
    transcript = (transcript_text or "").strip()
    transcript_lower = transcript.lower()

    if not claim or not transcript:
        return {
            "semantic_similarity": 0.0,
            "term_coverage": 0.0,
            "claim_terms": [],
            "matched_terms": [],
            "anchors": [],
            "missing_anchors": [],
            "topic_mismatch": False,
        }

    claim_terms = _extract_claim_terms(claim)
    matched_terms = [
        term
        for term in claim_terms
        if re.search(rf"\b{re.escape(term)}\b", transcript_lower)
    ]
    term_coverage = (len(matched_terms) / len(claim_terms)) if claim_terms else 1.0

    anchors = _extract_claim_anchors(claim)
    missing_anchors = [
        anchor
        for anchor in anchors
        if not re.search(rf"\b{re.escape(anchor.lower())}\b", transcript_lower)
    ]

    model = get_sentence_model()
    emb_claim = model.encode(claim, convert_to_tensor=True)
    emb_transcript = model.encode(transcript[:4000], convert_to_tensor=True)
    semantic_similarity = float(util.cos_sim(emb_claim, emb_transcript))

    topic_mismatch = semantic_similarity < 0.50 and term_coverage < 0.30

    return {
        "semantic_similarity": round(semantic_similarity, 4),
        "term_coverage": round(term_coverage, 4),
        "claim_terms": claim_terms,
        "matched_terms": matched_terms,
        "anchors": anchors,
        "missing_anchors": missing_anchors,
        "topic_mismatch": topic_mismatch,
    }


def _load_clip_for_video():
    global _CLIP_MODEL_VIDEO, _CLIP_PREPROCESS_VIDEO, _CLIP_DEVICE_VIDEO
    if _CLIP_MODEL_VIDEO is None:
        try:
            import clip
            import torch
        except ImportError as exc:
            raise ImportError(
                "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        _CLIP_DEVICE_VIDEO = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL_VIDEO, _CLIP_PREPROCESS_VIDEO = clip.load("ViT-B/32", device=_CLIP_DEVICE_VIDEO)
    return _CLIP_MODEL_VIDEO, _CLIP_PREPROCESS_VIDEO, _CLIP_DEVICE_VIDEO


def sample_frames(video_path: str, max_frames: int = 10, scene_threshold: float = 30.0) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0.0

    sample_timestamps = np.linspace(0, max(0, duration - 0.1), max_frames).tolist()
    sample_set = set(int(t * fps) for t in sample_timestamps)

    sampled: list[tuple[float, np.ndarray]] = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ts = frame_idx / fps if fps else 0.0

        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))))
            if diff > scene_threshold and len(sampled) < max_frames:
                sampled.append((ts, frame.copy()))

        if frame_idx in sample_set and len(sampled) < max_frames:
            sampled.append((ts, frame.copy()))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    unique: list[tuple[float, np.ndarray]] = []
    seen_ts: list[float] = []
    for ts, fr in sorted(sampled, key=lambda item: item[0]):
        if all(abs(ts - prev) > 0.5 for prev in seen_ts):
            unique.append((ts, fr))
            seen_ts.append(ts)
        if len(unique) >= max_frames:
            break

    return unique


def analyze_frame_clip(frame_bgr: np.ndarray, claim_text: str) -> float:
    try:
        import clip
        import torch
    except ImportError:
        return 0.0

    model, preprocess, device = _load_clip_for_video()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([claim_text[:77]]).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(image_tensor)
        txt_feat = model.encode_text(text_tokens)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        score = float((img_feat @ txt_feat.T).item())
    return score


def detect_faces_in_frame(frame_bgr: np.ndarray) -> int:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces)


def get_frame_brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def get_dominant_colors(frame_bgr: np.ndarray, n: int = 3) -> list[str]:
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return []

    try:
        small = cv2.resize(frame_bgr, (50, 50))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        arr = rgb.reshape(-1, 3).astype(float)

        km = KMeans(n_clusters=n, n_init=3, random_state=0)
        km.fit(arr)
        centers = km.cluster_centers_.astype(int)
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]
    except Exception:
        return []


def is_scene_change_frame(frame_bgr: np.ndarray, prev_frame_bgr: np.ndarray | None, threshold: float = 30.0) -> bool:
    if prev_frame_bgr is None:
        return False
    g1 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    g2 = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    diff = float(np.mean(np.abs(g1 - g2)))
    return diff > threshold


def analyze_frames(frames: list[tuple[float, np.ndarray]], claim_text: str) -> list[FrameAnalysis]:
    results: list[FrameAnalysis] = []
    prev_frame = None

    clip_available = True
    try:
        _load_clip_for_video()
    except Exception:
        clip_available = False

    for idx, (ts, frame) in enumerate(frames):
        clip_score = 0.0
        if clip_available:
            try:
                clip_score = analyze_frame_clip(frame, claim_text)
            except Exception:
                clip_score = 0.0

        results.append(
            FrameAnalysis(
                frame_index=idx,
                timestamp_sec=round(ts, 2),
                clip_score=round(clip_score, 4),
                face_count=detect_faces_in_frame(frame),
                brightness=round(get_frame_brightness(frame), 2),
                dominant_colors=get_dominant_colors(frame),
                is_scene_change=is_scene_change_frame(frame, prev_frame),
            )
        )
        prev_frame = frame

    return results


def compute_visual_signals(frame_analyses: list[FrameAnalysis]) -> dict:
    if not frame_analyses:
        return {
            "avg_clip_score": 0.0,
            "clip_verdict": "NO_FRAMES",
            "scene_change_count": 0,
            "face_consistency_score": 1.0,
            "brightness_spike_count": 0,
            "visual_flags": ["no_frames_analyzed"],
        }

    clip_scores = [f.clip_score for f in frame_analyses]
    face_counts = [f.face_count for f in frame_analyses]
    brightnesses = [f.brightness for f in frame_analyses]
    scene_count = sum(1 for f in frame_analyses if f.is_scene_change)

    avg_clip = float(np.mean(clip_scores))
    face_std = float(np.std(face_counts))
    face_mean = float(np.mean(face_counts)) + 1e-9
    face_cv = face_std / face_mean
    face_consistency = max(0.0, 1.0 - min(face_cv, 1.0))

    brightness_spikes = 0
    for i in range(1, len(brightnesses)):
        if abs(brightnesses[i] - brightnesses[i - 1]) > 40:
            brightness_spikes += 1

    if avg_clip > 0.20:
        clip_verdict = "FRAMES_MATCH_CLAIM"
    elif avg_clip > 0.10:
        clip_verdict = "WEAK_VISUAL_MATCH"
    else:
        clip_verdict = "FRAMES_DO_NOT_MATCH_CLAIM"

    visual_flags: list[str] = []
    if avg_clip < 0.10:
        visual_flags.append("low_frame_claim_similarity")
    if scene_count >= 3:
        visual_flags.append(f"many_scene_cuts_{scene_count}")
    if face_consistency < 0.40:
        visual_flags.append("inconsistent_face_count_across_frames")
    if brightness_spikes >= 3:
        visual_flags.append(f"brightness_spikes_{brightness_spikes}")
    if scene_count >= 5:
        visual_flags.append("possible_spliced_footage")

    return {
        "avg_clip_score": round(avg_clip, 4),
        "clip_verdict": clip_verdict,
        "scene_change_count": scene_count,
        "face_consistency_score": round(face_consistency, 4),
        "brightness_spike_count": brightness_spikes,
        "visual_flags": visual_flags,
    }


def _fuse_video_scores(
    nli_label: str,
    temporal_ok: bool,
    year_mismatch: bool,
    avg_clip_score: float,
    visual_flags: list[str],
    nli_has_error: bool,
    segment_anomalies: list[dict],
    drift_events: list[dict],
    claim_alignment: dict,
) -> float:
    score = 0.0

    if nli_label == "contradiction":
        score += 1.0 * 0.40
    elif nli_label == "neutral":
        score += 0.45 * 0.40
    elif nli_has_error:
        score += 0.10 * 0.40

    if avg_clip_score < 0.10:
        score += 1.0 * 0.25
    elif avg_clip_score < 0.20:
        score += 0.50 * 0.25

    if not temporal_ok:
        score += 1.0 * 0.20

    if year_mismatch:
        score += 1.0 * 0.20

    if segment_anomalies:
        max_contra = max(float(item.get("contradiction_score", 0.0)) for item in segment_anomalies)
        if max_contra > 0.75:
            score = max(score, 0.75)
        elif max_contra > 0.50:
            score = max(score, 0.45)

    if drift_events:
        max_drift = max(float(item.get("drift_score", 0.0)) for item in drift_events)
        if max_drift > 0.40:
            score = min(1.0, score + 0.15)
        else:
            score = min(1.0, score + 0.05)

    term_coverage = float(claim_alignment.get("term_coverage", 1.0))
    semantic_similarity = float(claim_alignment.get("semantic_similarity", 1.0))
    missing_anchor_count = len(claim_alignment.get("missing_anchors", []))

    if term_coverage < 0.15:
        score = min(1.0, score + 0.20)
    elif term_coverage < 0.30:
        score = min(1.0, score + 0.10)

    if semantic_similarity < 0.40:
        score = min(1.0, score + 0.25)
    elif semantic_similarity < 0.50:
        score = min(1.0, score + 0.10)

    if missing_anchor_count >= 2:
        score = min(1.0, score + 0.15)
    elif missing_anchor_count == 1 and semantic_similarity < 0.50:
        score = min(1.0, score + 0.10)

    if "possible_spliced_footage" in visual_flags:
        score = min(1.0, score + 0.05)
    if "inconsistent_face_count_across_frames" in visual_flags:
        score = min(1.0, score + 0.05)

    return round(min(score, 1.0), 4)


def _build_video_short_explanation(
    nli_result: dict,
    temporal_result: dict,
    visual_signals: dict,
    flags: list[str],
    claim_year: int | None,
    transcript_year: int | None,
) -> str:
    parts: list[str] = []

    if "claim_transcript_contradiction" in flags:
        contradiction_chunks = nli_result.get("contradiction_chunk_count")
        suffix = f" in {contradiction_chunks} chunk(s)" if isinstance(contradiction_chunks, int) and contradiction_chunks > 0 else ""
        parts.append(f"Claim contradicts transcript{suffix}.")

    if "low_frame_claim_similarity" in flags:
        avg_clip = float(visual_signals.get("avg_clip_score", 0.0))
        parts.append(f"Video frames do not visually match the claim (avg CLIP score {avg_clip:.2f}).")

    if "transcript_temporal_inconsistency" in flags:
        issue_count = len(temporal_result.get("issues", []))
        parts.append(f"Transcript timeline has {issue_count} ordering issue(s).")

    if "possible_spliced_footage" in flags:
        scene_count = int(visual_signals.get("scene_change_count", 0))
        parts.append(f"Possible spliced footage detected ({scene_count} scene cuts).")

    if "claim_transcript_year_mismatch" in flags and claim_year and transcript_year:
        parts.append(f"Year mismatch: claim references {claim_year} but transcript mentions {transcript_year}.")

    if "segment_contradiction" in flags:
        parts.append("One or more transcript intervals contradict the claim.")

    if "semantic_drift" in flags:
        parts.append("Narrative drift was detected across transcript windows.")

    if "claim_transcript_topic_mismatch" in flags:
        parts.append("Claim topic does not align with transcript content.")

    if "claim_term_mismatch" in flags:
        parts.append("Key claim terms are weakly supported in transcript.")

    if not parts:
        if "claim_transcript_unconfirmed" in flags:
            return "Transcript does not clearly confirm or deny the claim."
        return "Transcript and visual frames appear consistent with the claim."

    return " ".join(parts)


def _intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def _build_timeline_report(
    timeline_issues: list[dict],
    claim_year: int | None,
    transcript_year: int | None,
) -> list[dict]:
    report: list[dict] = []

    for issue in timeline_issues:
        issue_type = str(issue.get("issue_type") or issue.get("flag") or "unknown")
        start_sec = float(issue.get("start_sec", 0.0))
        end_sec = float(issue.get("end_sec", start_sec))
        overlap_count = sum(
            1
            for other in timeline_issues
            if _intervals_overlap(
                start_sec,
                end_sec,
                float(other.get("start_sec", 0.0)),
                float(other.get("end_sec", 0.0)),
            )
        )

        severity = "low"
        if issue_type == "nli_contradiction":
            contradiction_score = float(issue.get("contradiction_score", 0.0))
            if contradiction_score > 0.75:
                severity = "high"
            elif contradiction_score > 0.50:
                severity = "medium"
        elif issue_type == "semantic_drift":
            drift_score = float(issue.get("drift_score", 0.0))
            if drift_score > 0.40:
                severity = "high"
            elif drift_score > 0.30:
                severity = "medium"
        elif issue_type == "year_mismatch":
            severity = "high"
        elif issue_type == "claim_topic_mismatch":
            severity = "high"

        if overlap_count > 1:
            severity = "high"

        if issue_type == "nli_contradiction":
            description = (
                "This interval directly contradicts the claim, indicating the narrative conflicts with "
                "the asserted context."
            )
        elif issue_type == "semantic_drift":
            description = issue.get("description") or "Narrative topic shifts significantly in this interval."
        elif issue_type == "year_mismatch":
            description = (
                f"Claim references {claim_year} while transcript context points to {transcript_year}, "
                "which changes the event timeframe."
            )
        elif issue_type == "claim_topic_mismatch":
            description = "Claim topic is not supported by the spoken narrative in this interval."
        else:
            description = "Potential contextual inconsistency detected in this interval."

        report.append(
            {
                "interval": f"{issue.get('start_fmt', _fmt_ts(start_sec))}-{issue.get('end_fmt', _fmt_ts(end_sec))}",
                "issue_type": issue_type,
                "severity": severity,
                "description": str(description),
            }
        )

    return report


def analyze_video_context(
    claim_text: str,
    video_path: str | None = None,
    transcript_text: str | None = None,
    transcript_segments: list[dict] | None = None,
    whisper_model_size: str = "base",
    language: str | None = None,
    max_frames: int = 10,
    run_frame_analysis: bool = True,
) -> dict:
    logger.info("Resolving transcript...")
    full_text, segments, transcript_source = resolve_transcript(
        video_path=video_path,
        transcript_text=transcript_text,
        transcript_segments=transcript_segments,
        whisper_model_size=whisper_model_size,
        language=language,
    )
    logger.info(f"Transcript resolved via: {transcript_source}. Length: {len(full_text)}")

    logger.info("Running NLI claim consistency check...")
    nli_result = check_claim_consistency_nli(claim_text, full_text)
    
    logger.info("Running temporal consistency check...")
    temporal_result = check_transcript_temporal_consistency(segments)
    
    logger.info("Computing claim alignment...")
    claim_alignment = _compute_claim_alignment(claim_text=claim_text, transcript_text=full_text)

    logger.info("Detecting segment anomalies...")
    segment_anomalies = detect_segment_anomalies(claim_text=claim_text, segments=segments)
    
    logger.info("Detecting semantic drift...")
    drift_events = detect_semantic_drift(segments=segments)
    timeline_issues = sorted(segment_anomalies + drift_events, key=lambda item: float(item.get("start_sec", 0.0)))

    nli_label = str(nli_result.get("nli_label", "")).lower()
    nli_has_error = "error" in nli_result

    claim_year = _extract_year(claim_text)
    transcript_year = _extract_year(full_text)
    year_mismatch = bool(claim_year and transcript_year and claim_year != transcript_year)

    frame_analyses: list[FrameAnalysis] = []
    visual_signals = {
        "avg_clip_score": 0.0,
        "clip_verdict": "SKIPPED",
        "scene_change_count": 0,
        "face_consistency_score": 1.0,
        "brightness_spike_count": 0,
        "visual_flags": [],
    }

    if run_frame_analysis and video_path and os.path.exists(video_path):
        try:
            frames = sample_frames(video_path, max_frames=max_frames)
            frame_analyses = analyze_frames(frames, claim_text)
            visual_signals = compute_visual_signals(frame_analyses)
        except Exception as exc:
            visual_signals["visual_flags"].append(f"frame_analysis_error: {str(exc)[:100]}")

    flags: list[str] = []
    if nli_label == "contradiction":
        flags.append("claim_transcript_contradiction")
    elif nli_label == "neutral":
        flags.append("claim_transcript_unconfirmed")
    elif nli_has_error:
        flags.append("nli_unavailable")

    if not temporal_result.get("consistent", True):
        flags.append("transcript_temporal_inconsistency")

    if year_mismatch:
        flags.append("claim_transcript_year_mismatch")

    if segment_anomalies:
        flags.append("segment_contradiction")

    if drift_events:
        flags.append("semantic_drift")
        if any(float(item.get("drift_score", 0.0)) > 0.40 for item in drift_events):
            flags.append("narrative_shift")

    topic_mismatch = bool(claim_alignment.get("topic_mismatch", False))
    if topic_mismatch:
        flags.append("claim_transcript_topic_mismatch")

    term_coverage = float(claim_alignment.get("term_coverage", 1.0))
    if term_coverage < 0.30:
        flags.append("claim_term_mismatch")

    if len(claim_alignment.get("missing_anchors", [])) >= 2:
        flags.append("claim_entity_mismatch")

    if topic_mismatch:
        timestamp_ref = float(segments[0].start) if segments else 0.0
        timeline_issues.append(
            {
                "start_sec": timestamp_ref,
                "end_sec": timestamp_ref,
                "start_fmt": _fmt_ts(timestamp_ref),
                "end_fmt": _fmt_ts(timestamp_ref),
                "issue_type": "claim_topic_mismatch",
                "flag": "claim_transcript_topic_mismatch",
            }
        )
        timeline_issues.sort(key=lambda item: float(item.get("start_sec", 0.0)))

    if year_mismatch:
        timestamp_ref = float(segments[0].start) if segments else 0.0
        timeline_issues.append(
            {
                "start_sec": timestamp_ref,
                "end_sec": timestamp_ref,
                "start_fmt": _fmt_ts(timestamp_ref),
                "end_fmt": _fmt_ts(timestamp_ref),
                "issue_type": "year_mismatch",
                "flag": "claim_transcript_year_mismatch",
                "claim_year": claim_year,
                "transcript_year": transcript_year,
            }
        )
        timeline_issues.sort(key=lambda item: float(item.get("start_sec", 0.0)))

    flags.extend(visual_signals.get("visual_flags", []))
    flags = list(dict.fromkeys(flags))

    inconsistency_score = _fuse_video_scores(
        nli_label=nli_label,
        temporal_ok=temporal_result.get("consistent", True),
        year_mismatch=year_mismatch,
        avg_clip_score=float(visual_signals.get("avg_clip_score", 0.0)),
        visual_flags=visual_signals.get("visual_flags", []),
        nli_has_error=nli_has_error,
        segment_anomalies=segment_anomalies,
        drift_events=drift_events,
        claim_alignment=claim_alignment,
    )
    consistency_confidence = round(max(0.0, 1.0 - inconsistency_score), 4)

    if inconsistency_score >= 0.70:
        label, color = "HIGH_RISK", "red"
    elif inconsistency_score >= 0.40:
        label, color = "SUSPICIOUS", "orange"
    else:
        label, color = "CONSISTENT", "green"

    short_explanation = _build_video_short_explanation(
        nli_result=nli_result,
        temporal_result=temporal_result,
        visual_signals=visual_signals,
        flags=flags,
        claim_year=claim_year,
        transcript_year=transcript_year,
    )

    return {
        "tool": "Axis2 Video Handler",
        "label": label,
        "color": color,
        "inconsistency_score": inconsistency_score,
        "confidence": consistency_confidence,
        "consistency_confidence": consistency_confidence,
        "short_explanation": short_explanation,
        "flags": flags,
        "transcript_text": full_text,
        "transcript_source": transcript_source,
        "transcript_segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments
        ],
        "claim_nli": nli_result,
        "temporal_consistency": temporal_result,
        "segment_anomalies": segment_anomalies,
        "drift_events": drift_events,
        "emotion_signals": [],
        "claim_alignment": claim_alignment,
        "year_mismatch": year_mismatch,
        "claim_year": claim_year,
        "transcript_year": transcript_year,
        "visual_signals": visual_signals,
        "frame_analyses": [
            {
                "frame_index": f.frame_index,
                "timestamp_sec": f.timestamp_sec,
                "clip_score": f.clip_score,
                "face_count": f.face_count,
                "brightness": f.brightness,
                "dominant_colors": f.dominant_colors,
                "is_scene_change": f.is_scene_change,
            }
            for f in frame_analyses
        ],
    }
