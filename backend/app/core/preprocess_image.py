from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import io, base64, hashlib
import piexif
import cv2
from imagehash import phash, average_hash

def preprocess_image(image_input: bytes | str) -> dict:
    """
    Master image preprocessor.
    input: raw bytes OR url string
    output: Unified Feature Object for image
    """
    
    # ── STAGE 1: Decode ──────────────────────────────────────────
    if isinstance(image_input, str):  # URL
        import httpx
        resp = httpx.get(image_input, timeout=10)
        image_bytes = resp.content
    else:
        image_bytes = image_input

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    file_size_kb = len(image_bytes) / 1024

    # ── STAGE 2: EXIF Metadata ───────────────────────────────────
    exif_features = extract_exif(image_bytes)

    # ── STAGE 3: Visual Features ─────────────────────────────────
    
    # ELA
    ela_image, ela_stats = compute_ela(img)
    
    # Model-ready array
    model_img = np.array(img.resize((224, 224))) / 255.0
    
    # CLIP embedding (for context axis)
    clip_embedding = compute_clip_embedding(img)
    
    # Color histogram
    color_hist = compute_color_histogram(img)
    
    # Edge map
    edge_map = compute_edge_map(img)
    
    # Face detection
    face_regions = detect_faces(img)
    
    # Perceptual hashes (for reverse search)
    p_hash = str(phash(img))
    a_hash = str(average_hash(img))

    # ── STAGE 4: Statistical signals ────────────────────────────
    noise_level = estimate_noise(img)
    compression_score = estimate_compression_artifacts(image_bytes, img)

    # ── STAGE 5: Base64 for API transport ───────────────────────
    thumbnail = img.copy()
    thumbnail.thumbnail((100, 100))
    thumb_buffer = io.BytesIO()
    thumbnail.save(thumb_buffer, format="JPEG", quality=85)
    thumbnail_b64 = base64.b64encode(thumb_buffer.getvalue()).decode()

    ela_buffer = io.BytesIO()
    ela_image.save(ela_buffer, format="PNG")
    ela_b64 = base64.b64encode(ela_buffer.getvalue()).decode()

    return {
        # ── Identity
        "input_type": "image",
        "width": width,
        "height": height,
        "file_size_kb": round(file_size_kb, 2),
        "format": img.format or "unknown",
        
        # ── For Axis 1 (Authenticity)
        "authenticity_features": {
            "model_input_array": model_img.tolist(),     # 224x224x3
            "ela_image_b64": ela_b64,                    # ELA visualization
            "ela_mean": ela_stats["mean"],               # float
            "ela_std": ela_stats["std"],                 # float
            "ela_max": ela_stats["max"],                 # float
            "ela_suspicious_ratio": ela_stats["suspicious_ratio"],
            "noise_level": noise_level,                  # float
            "compression_artifact_score": compression_score,
            "has_faces": len(face_regions) > 0,
            "face_count": len(face_regions),
            "face_regions": face_regions,                # [{x,y,w,h}, ...]
            "exif": exif_features,
        },
        
        # ── For Axis 2 (Contextual Consistency)
        "context_features": {
            "clip_embedding": clip_embedding,            # list[float] 512-dim
            "color_histogram": color_hist,               # dict
            "perceptual_hash_p": p_hash,                 # string
            "perceptual_hash_a": a_hash,                 # string
            "thumbnail_b64": thumbnail_b64,              # for reverse search
            "dominant_colors": extract_dominant_colors(img),
            "scene_description": None,                   # filled later by BLIP/LLaVA
        },
        
        # ── For Axis 3 (Source Credibility)
        "source_features": {
            "exif_software": exif_features.get("software", ""),
            "exif_datetime": exif_features.get("datetime", ""),
            "exif_has_gps": exif_features.get("gps_present", False),
            "file_hash_md5": hashlib.md5(image_bytes).hexdigest(),
        }
    }


def compute_ela(img: Image, quality: int = 90) -> tuple[Image, dict]:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    ela = ImageChops.difference(img, resaved)
    ela_array = np.array(ela.convert("L"))

    mean_val = float(np.mean(ela_array))
    std_val  = float(np.std(ela_array))
    max_val  = float(np.max(ela_array))
    threshold = mean_val + 2 * std_val
    suspicious_ratio = float(np.sum(ela_array > threshold) / ela_array.size)

    # Amplify for visualization
    scale = 255.0 / max_val if max_val > 0 else 1.0
    ela_visual = ImageEnhance.Brightness(ela).enhance(scale)

    stats = {
        "mean": round(mean_val, 3),
        "std": round(std_val, 3),
        "max": round(max_val, 3),
        "suspicious_ratio": round(suspicious_ratio, 4)
    }
    return ela_visual, stats


def extract_exif(image_bytes: bytes) -> dict:
    try:
        exif_dict = piexif.load(image_bytes)
        ifd = exif_dict.get("0th", {})
        exif_ifd = exif_dict.get("Exif", {})

        make    = ifd.get(piexif.ImageIFD.Make, b"").decode(errors="ignore").strip()
        model   = ifd.get(piexif.ImageIFD.Model, b"").decode(errors="ignore").strip()
        software= ifd.get(piexif.ImageIFD.Software, b"").decode(errors="ignore").strip()
        dt      = ifd.get(piexif.ImageIFD.DateTime, b"").decode(errors="ignore").strip()
        gps     = bool(exif_dict.get("GPS"))

        suspicious_sw = ["photoshop","gimp","lightroom","affinity",
                         "canva","pixlr","snapseed","facetune"]
        edited = any(sw in software.lower() for sw in suspicious_sw)

        return {
            "has_exif": True,
            "make": make,
            "model": model,
            "software": software,
            "datetime": dt,
            "gps_present": gps,
            "likely_edited": edited,
        }
    except Exception:
        return {
            "has_exif": False,
            "make": "", "model": "",
            "software": "", "datetime": "",
            "gps_present": False,
            "likely_edited": False,
        }


def compute_clip_embedding(img: Image) -> list[float]:
    """
    Uses CLIP via HuggingFace transformers (local, free)
    Returns 512-dim embedding vector
    """
    from transformers import CLIPProcessor, CLIPModel
    import torch

    # Cache model (load once)
    if not hasattr(compute_clip_embedding, "_model"):
        compute_clip_embedding._model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        compute_clip_embedding._processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    model = compute_clip_embedding._model
    processor = compute_clip_embedding._processor

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    
    return embedding[0].tolist()  # 512 floats


def detect_faces(img: Image) -> list[dict]:
    """OpenCV face detection (free, no API)"""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for (x, y, w, h) in faces]


def compute_color_histogram(img: Image) -> dict:
    arr = np.array(img)
    return {
        "r_mean": float(np.mean(arr[:,:,0])),
        "g_mean": float(np.mean(arr[:,:,1])),
        "b_mean": float(np.mean(arr[:,:,2])),
        "r_std":  float(np.std(arr[:,:,0])),
        "g_std":  float(np.std(arr[:,:,1])),
        "b_std":  float(np.std(arr[:,:,2])),
    }


def compute_edge_map(img: Image) -> str:
    """Returns base64 Canny edge map"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    _, buffer = cv2.imencode(".png", edges)
    return base64.b64encode(buffer).decode()


def estimate_noise(img: Image) -> float:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY).astype(float)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return round(float(laplacian.var()), 4)


def estimate_compression_artifacts(image_bytes: bytes, img: Image) -> float:
    """
    Compare file size vs image dimensions
    Unusually small file for its resolution = high compression = possible re-save
    """
    pixels = img.width * img.height
    bytes_per_pixel = len(image_bytes) / pixels
    # Natural photos: ~0.3–3.0 bytes/pixel
    # Heavily re-saved: < 0.1
    score = max(0.0, 1.0 - bytes_per_pixel / 1.5)
    return round(score, 4)


def extract_dominant_colors(img: Image, n: int = 5) -> list[str]:
    """Returns top N dominant colors as hex strings"""
    small = img.copy().resize((50, 50))
    arr = np.array(small).reshape(-1, 3)
    
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=3, random_state=0)
    km.fit(arr)
    
    colors = km.cluster_centers_.astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]