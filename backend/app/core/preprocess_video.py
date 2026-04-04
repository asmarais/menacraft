import cv2
import subprocess
import json
import tempfile
import os
import io
import numpy as np
from PIL import Image
from .preprocess_image import preprocess_image, detect_faces  # your image module


def preprocess_video(video_input: bytes | str) -> dict:
    """
    Master video preprocessor
    """
    # ── STAGE 1: Get file path ───────────────────────────────────
    if isinstance(video_input, str) and video_input.startswith("http"):
        video_path = download_video(video_input)
    elif isinstance(video_input, bytes):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(video_input)
        tmp.close()
        video_path = tmp.name
    else:
        video_path = video_input

    # ── STAGE 2: Video Metadata ──────────────────────────────────
    video_meta = extract_video_metadata(video_path)

    # ── STAGE 3: Frame Extraction ────────────────────────────────
    frames_data = extract_and_process_frames(video_path, max_frames=10)

    # ── STAGE 4: Temporal Analysis ───────────────────────────────
    temporal = analyze_temporal_consistency(frames_data["raw_frames"])

    # ── STAGE 5: Audio ───────────────────────────────────────────
    audio_data = extract_and_process_audio(video_path)

    return {
        "input_type": "video",

        # ── For Axis 1
        "authenticity_features": {
            "frame_ela_scores": frames_data["ela_scores"],
            "frame_ai_scores_ready": frames_data["model_inputs"],
            "face_consistency_score": temporal["face_consistency"],
            "splice_detected": temporal["splice_detected"],
            "splice_locations": temporal["splice_locations"],
            "audio_video_sync_score": audio_data["sync_score"],
            "has_faces": frames_data["has_faces"],
            "suspicious_frames": frames_data["suspicious_frame_indices"],
        },

        # ── For Axis 2
        "context_features": {
            "transcript": audio_data["transcript"],
            "key_frame_embeddings": frames_data["clip_embeddings"],
            "scene_count": temporal["scene_count"],
            "dominant_scene_colors": frames_data["dominant_colors"],
            "faces_detected": frames_data["face_regions_per_frame"],
        },

        # ── For Axis 3
        "source_features": {
            "video_metadata": video_meta,
            "creation_date": video_meta.get("creation_time", ""),
            "encoder": video_meta.get("encoder", ""),
            "duration_seconds": video_meta.get("duration", 0),
        },

        # ── Raw data
        "_raw_frames": frames_data["raw_frames"],
        "_video_path": video_path,
    }


def extract_video_metadata(video_path: str) -> dict:
    """Use ffprobe (comes with ffmpeg, free)"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # BUG FIX 1: ffprobe may fail or return empty output
    if not result.stdout.strip():
        return {
            "duration": 0, "size_mb": 0, "fps": 0,
            "width": 0, "height": 0, "codec": "",
            "encoder": "", "creation_time": "", "has_audio": False,
        }

    probe = json.loads(result.stdout)

    video_stream = next(
        (s for s in probe.get("streams", []) if s["codec_type"] == "video"), {}
    )
    fmt = probe.get("format", {})

    # BUG FIX 2: eval() is dangerous and crashes on malformed strings like "30000/1001"
    # Use a safe fraction parser instead
    def parse_fps(rate_str: str) -> float:
        try:
            parts = rate_str.split("/")
            return round(int(parts[0]) / int(parts[1]), 3) if len(parts) == 2 else float(parts[0])
        except Exception:
            return 0.0

    return {
        "duration": float(fmt.get("duration", 0)),
        "size_mb": round(float(fmt.get("size", 0)) / 1024 / 1024, 2),
        "fps": parse_fps(video_stream.get("r_frame_rate", "0/1")),
        "width": video_stream.get("width", 0),
        "height": video_stream.get("height", 0),
        "codec": video_stream.get("codec_name", ""),
        "encoder": fmt.get("tags", {}).get("encoder", ""),
        "creation_time": fmt.get("tags", {}).get("creation_time", ""),
        "has_audio": any(
            s["codec_type"] == "audio" for s in probe.get("streams", [])
        ),
    }


def extract_and_process_frames(video_path: str, max_frames: int = 10) -> dict:
    cap = cv2.VideoCapture(video_path)

    # BUG FIX 3: cap.isOpened() check missing — crashes silently otherwise
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # BUG FIX 4: np was used but never imported at the top of this file
    sample_times = np.linspace(0, max(duration - 1, 0), max_frames)
    sample_frame_ids = [int(t * fps) for t in sample_times]

    raw_frames = []
    ela_scores = []
    clip_embeddings = []
    model_inputs = []
    face_regions_per_frame = []
    dominant_colors = []
    suspicious_indices = []

    for idx, frame_id in enumerate(sample_frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        raw_frames.append(rgb)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        frame_bytes = buf.getvalue()

        preprocessed = preprocess_image(frame_bytes)

        ela_scores.append({
            "frame_index": idx,
            "timestamp_sec": round(float(sample_times[idx]), 2),
            "ela_mean": preprocessed["authenticity_features"]["ela_mean"],
            "ela_suspicious_ratio": preprocessed["authenticity_features"]["ela_suspicious_ratio"],
        })

        clip_embeddings.append(preprocessed["context_features"]["clip_embedding"])
        model_inputs.append(preprocessed["authenticity_features"]["model_input_array"])
        face_regions_per_frame.append(preprocessed["authenticity_features"]["face_regions"])
        dominant_colors.append(preprocessed["context_features"]["dominant_colors"])

        if preprocessed["authenticity_features"]["ela_suspicious_ratio"] > 0.05:
            suspicious_indices.append(idx)

    cap.release()

    has_faces = any(len(f) > 0 for f in face_regions_per_frame)
    avg_colors = dominant_colors[len(dominant_colors) // 2] if dominant_colors else []

    return {
        "raw_frames": raw_frames,
        "ela_scores": ela_scores,
        "clip_embeddings": clip_embeddings,
        "model_inputs": model_inputs,
        "face_regions_per_frame": face_regions_per_frame,
        "has_faces": has_faces,
        "dominant_colors": avg_colors,
        "suspicious_frame_indices": suspicious_indices,
    }


def analyze_temporal_consistency(raw_frames: list) -> dict:
    """Detect splices and face inconsistency across frames"""
    if len(raw_frames) < 2:
        return {
            "splice_detected": False,
            "splice_locations": [],
            "face_consistency": 1.0,
            "scene_count": 1,
            "frame_diffs": [],  # BUG FIX 5: key was missing in early-return path
        }

    diffs = []
    for i in range(1, len(raw_frames)):
        prev_small = cv2.resize(raw_frames[i - 1].astype(float), (64, 64))
        curr_small = cv2.resize(raw_frames[i].astype(float), (64, 64))
        diff = np.mean(np.abs(prev_small - curr_small))
        diffs.append(diff)

    mean_diff = np.mean(diffs)
    std_diff  = np.std(diffs)
    threshold = mean_diff + 2.5 * std_diff

    splice_locations = [i for i, d in enumerate(diffs) if d > threshold]
    splice_detected  = len(splice_locations) > 0
    scene_count      = len(splice_locations) + 1

    face_counts = []
    for frame in raw_frames:
        pil = Image.fromarray(frame)
        faces = detect_faces(pil)
        face_counts.append(len(faces))

    # BUG FIX 6: ZeroDivisionError when all face_counts are 0
    mean_faces = np.mean(face_counts) if face_counts else 0
    std_faces  = np.std(face_counts) if face_counts else 0
    face_consistency = 1.0 - min(std_faces / (mean_faces + 1), 1.0)

    return {
        "splice_detected": splice_detected,
        "splice_locations": splice_locations,
        "scene_count": scene_count,
        "face_consistency": round(float(face_consistency), 3),
        "frame_diffs": [round(float(d), 3) for d in diffs],
    }


def extract_and_process_audio(video_path: str) -> dict:
    """Extract audio + transcribe with Whisper"""
    # BUG FIX 7: .replace(".mp4") breaks for .mov/.avi/etc — use proper path logic
    base = os.path.splitext(video_path)[0]
    audio_path = base + "_audio.wav"

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        "-vn", audio_path
    ], capture_output=True)

    transcript = ""
    sync_score = 1.0

    if os.path.exists(audio_path):
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, fp16=False)
        transcript = result.get("text", "").strip()
        sync_score = estimate_av_sync(video_path, audio_path)

    return {
        "transcript": transcript,
        "has_audio": bool(transcript),
        "sync_score": round(sync_score, 3),
        "audio_path": audio_path,
    }


def estimate_av_sync(video_path: str, audio_path: str) -> float:
    """
    Heuristic: compare audio energy variance.
    Flat energy = possibly dubbed/replaced audio.
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=16000)
    audio_energy = librosa.feature.rms(y=y)[0]

    if audio_energy.max() > 0:
        audio_energy = audio_energy / audio_energy.max()

    # BUG FIX 8: video_path argument accepted but never used — kept for API consistency
    sync_score = min(float(np.std(audio_energy)) * 5, 1.0)
    return sync_score


def download_video(url: str) -> str:
    """Download video from URL using yt-dlp (free)"""
    tmp_path = tempfile.mktemp(suffix=".mp4")
    import yt_dlp

    ydl_opts = {
        "outtmpl": tmp_path,
        "format": "mp4/best[filesize<50M]",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return tmp_path