#!/usr/bin/env python3
"""
SOURCE CREDIBILITY ANALYZER
Menacraft 2.0 — Advanced Threat Detection + LLM Explanation Layer
"""

import re
import json
import urllib.parse
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Credibility")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# =========================================================
# API KEYS CONFIGURATION
# =========================================================

CONFIG_FILE = os.path.join(os.path.dirname(__file__), ".api_config.json")

GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
VIRUSTOTAL_KEY   = os.environ.get("VIRUSTOTAL_API_KEY") or os.environ.get("VIRUSTOTAL", "")
APIVOID_KEY      = os.environ.get("APIVOID_API_KEY") or os.environ.get("APIVOID", "")
WHOIS_API_KEY    = os.environ.get("WHOISXML_API_KEY") or os.environ.get("WHOISXML", "")
URL_SCAN_KEY     = os.environ.get("URLSCAN_API_KEY") or os.environ.get("URLSCAN", "")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed. LLM explanations will use fallback mode.")


def load_api_keys() -> dict:
    """Load optional API keys from .api_config.json (merges with env vars)."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


_file_keys = load_api_keys()

def _key(name: str) -> str:
    """Resolve API key: env var wins, then config file."""
    env_map = {
        "VIRUSTOTAL":  VIRUSTOTAL_KEY,
        "APIVOID":     APIVOID_KEY,
        "WHOISXML":    WHOIS_API_KEY,
        "URLSCAN":     URL_SCAN_KEY,
    }
    return env_map.get(name) or _file_keys.get(name, "")


# =========================================================
# KNOWN ENTITIES (for spoofing detection)
# =========================================================

KNOWN_ENTITIES: Dict[str, List[str]] = {
    "bbc":        ["bbc.com", "bbc.co.uk"],
    "reuters":    ["reuters.com"],
    "cnn":        ["cnn.com"],
    "nytimes":    ["nytimes.com"],
    "guardian":   ["theguardian.com"],
    "aljazeera":  ["aljazeera.com", "aljazeera.net"],
    "apnews":     ["apnews.com"],
    "lemonde":    ["lemonde.fr"],
    "sputnik":    ["sputniknews.com"],
    "foxnews":    ["foxnews.com"],
    "bfmtv":      ["bfmtv.com"],
    "franceinfo": ["francetvinfo.fr"],
}

SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".ml", ".buzz", ".click", ".top",
    ".gq", ".cf", ".ga", ".work", ".loan", ".racing",
    ".download", ".stream", ".science", ".win",
}

SENSATIONAL_KEYWORDS = [
    "breaking", "urgent", "leaked", "secret", "shocking",
    "conspiracy", "anonymous source", "they don't want you",
    "wake up", "share before deleted", "banned information",
    "exclusive", "bombshell", "cover-up", "deep state",
]


# =========================================================
# CREDIBILITY ANALYZER CLASS
# =========================================================

class Credibility:
    """Advanced source credibility analyzer."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def evaluate(self, features: dict) -> dict:
        """
        Main entry point.

        features must contain a 'type' key with one of:
            url | account | text | image | video | document

        Returns a standardized result dict:
            { score, label, explanation, flags }
        """
        handlers = {
            "url":      self._url,
            "account":  self._account,
            "text":     self._text,
            "image":    self._image,
            "video":    self._video,
            "document": self._document,
        }
        handler = handlers.get(features.get("type", "unknown"))
        return handler(features) if handler else self._default()

    # ------------------------------------------------------------------
    # URL ANALYSIS
    # ------------------------------------------------------------------

    def _url(self, features: dict) -> dict:
        """Full URL credibility pipeline with account-level pivoting."""
        url = features.get("url", "").strip()
        if not url:
            return self._result(0.5, "unknown", "No URL provided.")

        # Detect social media early
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        domain = parsed.netloc.lower().replace("www.", "")

        social_platforms = {
            "instagram.com":  "Instagram",
            "twitter.com":    "Twitter/X",
            "x.com":          "Twitter/X",
            "facebook.com":   "Facebook",
            "tiktok.com":     "TikTok",
            "youtube.com":    "YouTube",
            "linkedin.com":   "LinkedIn",
            "t.me":           "Telegram",
            "reddit.com":     "Reddit",
            "threads.net":    "Threads",
        }
        is_social, platform_name = False, "Website"
        for p_domain, p_name in social_platforms.items():
            if domain == p_domain or domain.endswith("." + p_domain):
                is_social, platform_name = True, p_name
                break

        # Run all sub-checks
        heuristic  = self._check_url_heuristic(url)
        whois      = self._check_domain_whois(url)
        virustotal = self._check_url_virustotal(url)
        apivoid    = self._check_domain_apivoid(url)
        urlscan    = self._check_urlscan(url)
        
        # ── Pivot to account analysis if available ──
        account_risk = 0
        account_flags = []
        if features.get("account"):
            acc_res = self._account(features)
            account_risk = acc_res["score"] * 100
            account_flags = acc_res["flags"]

        # Weighted calculation
        if is_social and features.get("account"):
            # Stronger weight on account signals for social media
            total_score = (
                heuristic["score"]  * 0.15 +
                account_risk       * 0.60 + 
                virustotal["score"] * 0.15 +
                apivoid["score"]    * 0.10
            )
        else:
            total_score = (
                heuristic["score"]  * 0.30 +
                whois["score"]      * 0.15 +
                virustotal["score"] * 0.25 +
                apivoid["score"]    * 0.20 +
                urlscan["score"]    * 0.10
            )

        flags = (
            heuristic["flags"]  +
            whois["flags"]      +
            virustotal["flags"] +
            apivoid["flags"]    +
            urlscan["flags"]    +
            account_flags
        )

        if is_social:
            flags.insert(0, f"Source is a {platform_name} link")
            if total_score < 25: label = "trustworthy"
            elif total_score < 60: label = "suspicious"
            else: label = "dangerous"
        else:
            label = "safe" if total_score < 20 else ("suspicious" if total_score < 60 else "dangerous")

        explanation = self._explain(
            score=total_score / 100,
            label=label,
            analysis_type=f"{platform_name} link",
            signals={
                "url": url,
                "is_social_media": is_social,
                "platform": platform_name,
                "domain_risk": round(total_score, 1),
                "account_risk": account_risk if is_social else None,
                "flags": flags,
            },
        )
        return self._result(total_score / 100, label, explanation, flags)

    def _check_url_heuristic(self, url: str) -> dict:
        """Local heuristic checks — no external API needed."""
        risk, flags = 0, []
        try:
            parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
            domain = parsed.netloc.lower()
            path   = parsed.path.lower()
            full   = url.lower()

            # Protocol
            if parsed.scheme == "http":
                risk += 15
                flags.append("Non-secure connection (HTTP)")
            else:
                flags.append("Secure connection (HTTPS)")

            # TLD
            tld = "." + domain.split(".")[-1]
            if tld in SUSPICIOUS_TLDS:
                risk += 30
                flags.append(f"Suspicious TLD ({tld})")

            # Typosquatting / spoofing
            base = domain.split(":")[0]  # strip port
            for entity, official_domains in KNOWN_ENTITIES.items():
                if entity in base and base not in official_domains:
                    risk += 55
                    flags.append(f"Possible spoofing of {entity.upper()}")

            # Sensational path
            if any(k in path or k in full for k in SENSATIONAL_KEYWORDS[:6]):
                risk += 20
                flags.append("Sensationalist URL keywords")

            # Subdomain depth
            parts = base.split(".")
            if len(parts) > 3:
                risk += 15
                flags.append(f"Excessive subdomains ({len(parts)} levels)")

            # IP-based URL
            if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(":")[0]):
                risk += 35
                flags.append("IP address used instead of domain name")

            # Credential phishing indicator
            if "@" in url:
                risk += 25
                flags.append("@ symbol in URL (credential phishing risk)")

            # Excessive hyphens
            if domain.count("-") > 3:
                risk += 10
                flags.append(f"Excessive hyphens in domain ({domain.count('-')})")

            # URL length
            if len(url) > 200:
                risk += 10
                flags.append(f"Unusually long URL ({len(url)} chars)")

            # Punycode / IDN homograph
            if "xn--" in domain:
                risk += 20
                flags.append("Punycode / IDN domain (possible homograph attack)")

            if not [f for f in flags if any(w in f.lower() for w in ["suspicious","spoofing","phishing","excessive","ip address","http","punycode"])]:
                flags.append("Clean domain structure")

        except Exception as e:
            risk = 40
            flags.append(f"URL parse error: {str(e)[:60]}")

        return {"score": min(100, risk), "flags": flags}

    def _check_domain_whois(self, url: str) -> dict:
        """
        Check domain registration age via WHOISXML API.
        Newer domains (< 1 year) are riskier.
        """
        if not _key("WHOISXML"):
            return {"score": 0, "flags": ["WHOIS check skipped (no API key)"]}
        try:
            parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
            domain = parsed.netloc.lower().replace("www.", "").split(":")[0]

            resp = requests.get(
                "https://www.whoisxmlapi.com/whoisserver/WhoisService",
                params={
                    "apiKey":       _key("WHOISXML"),
                    "domainName":   domain,
                    "outputFormat": "JSON",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                return {"score": 0, "flags": [f"WHOIS API error: {resp.status_code}"]}

            data = resp.json()
            record = data.get("WhoisRecord", {})
            created_raw = record.get("createdDate", "")

            flags, risk = [], 0

            if created_raw:
                # Parse ISO date
                try:
                    created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                    age_days = (datetime.now(created.tzinfo) - created).days
                    if age_days < 30:
                        risk += 50
                        flags.append(f"Very new domain (registered {age_days} days ago)")
                    elif age_days < 180:
                        risk += 30
                        flags.append(f"Young domain (registered {age_days} days ago)")
                    elif age_days < 365:
                        risk += 15
                        flags.append(f"Domain registered < 1 year ago ({age_days} days)")
                    else:
                        years = age_days // 365
                        flags.append(f"Established domain (registered ~{years} year(s) ago)")
                except Exception:
                    flags.append("Could not parse domain registration date")
            else:
                risk += 20
                flags.append("Domain registration date unavailable")

            # Registrant privacy
            registrant = record.get("registrant", {})
            if not registrant or registrant.get("name", "").lower() in ["redacted", "privacy", ""]:
                risk += 10
                flags.append("Registrant identity hidden (privacy protection)")
            else:
                flags.append(f"Registrant: {registrant.get('name', 'unknown')}")

            return {"score": min(100, risk), "flags": flags}

        except Exception as e:
            return {"score": 0, "flags": [f"WHOIS error: {str(e)[:60]}"]}

    def _check_url_virustotal(self, url: str) -> dict:
        """Submit URL to VirusTotal and retrieve scan results."""
        if not _key("VIRUSTOTAL"):
            return {"score": 0, "flags": ["VirusTotal check skipped (no API key)"]}
        try:
            headers = {"x-apikey": _key("VIRUSTOTAL")}

            # Step 1: Submit URL
            submit = requests.post(
                "https://www.virustotal.com/api/v3/urls",
                headers=headers,
                data={"url": url},
                timeout=10,
            )
            if submit.status_code != 200:
                return {"score": 0, "flags": [f"VirusTotal submit error: {submit.status_code}"]}

            analysis_id = submit.json().get("data", {}).get("id", "")

            # Step 2: Fetch analysis
            if analysis_id:
                result_resp = requests.get(
                    f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                    headers=headers,
                    timeout=10,
                )
                if result_resp.status_code == 200:
                    attrs = result_resp.json().get("data", {}).get("attributes", {})
                    stats = attrs.get("stats", {})
                else:
                    # Fallback: try URL lookup directly
                    import base64
                    url_id = base64.urlsafe_b64encode(url.encode()).rstrip(b"=").decode()
                    r2 = requests.get(
                        f"https://www.virustotal.com/api/v3/urls/{url_id}",
                        headers=headers,
                        timeout=10,
                    )
                    stats = r2.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            else:
                stats = {}

            malicious  = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            harmless   = stats.get("harmless", 0)
            undetected = stats.get("undetected", 0)
            total      = malicious + suspicious + harmless + undetected

            risk  = min(100, (malicious * 10) + (suspicious * 3))
            flags = []
            if malicious > 0:
                flags.append(f"MALICIOUS: flagged by {malicious} security engines")
            if suspicious > 0:
                flags.append(f"SUSPICIOUS: flagged by {suspicious} security engines")
            if harmless > 0 and malicious == 0:
                flags.append(f"Clean on VirusTotal ({harmless}/{total} engines)")
            if not flags:
                flags.append("VirusTotal: no data available")

            return {"score": risk, "flags": flags}

        except Exception as e:
            return {"score": 0, "flags": [f"VirusTotal error: {str(e)[:60]}"]}

    def _check_domain_apivoid(self, url: str) -> dict:
        """Check domain against APIVoid blacklists."""
        if not _key("APIVOID"):
            return {"score": 0, "flags": ["APIVoid check skipped (no API key)"]}
        try:
            parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
            domain = parsed.netloc.lower().split(":")[0]

            resp = requests.get(
                "https://endpoint.apivoid.com/domainbl/v1/pay-as-you-go/",
                params={"key": _key("APIVOID"), "host": domain},
                timeout=10,
            )
            if resp.status_code != 200:
                return {"score": 0, "flags": [f"APIVoid error: {resp.status_code}"]}

            data   = resp.json().get("data", {})
            report = data.get("report", {})
            bl     = report.get("blacklists", {})
            detected    = bl.get("engines_detected", 0)
            total_bl    = bl.get("engines_count", 40)
            risk        = min(100, (detected / max(total_bl, 1)) * 100)

            flags = []
            if detected > 0:
                flags.append(f"Listed in {detected}/{total_bl} domain blacklists")
            else:
                flags.append(f"Clean — not in any of {total_bl} blacklists")

            # Domain score from APIVoid (0-100, higher = safer)
            domain_score = report.get("domain_score", None)
            if domain_score is not None:
                if domain_score < 40:
                    flags.append(f"Low domain trust score ({domain_score}/100)")
                elif domain_score > 70:
                    flags.append(f"Good domain trust score ({domain_score}/100)")

            return {"score": risk, "flags": flags}

        except Exception as e:
            return {"score": 0, "flags": [f"APIVoid error: {str(e)[:60]}"]}

    def _check_urlscan(self, url: str) -> dict:
        """
        Submit URL to urlscan.io and retrieve verdict.
        NOTE: scan takes ~10s; we poll once with a short delay.
        """
        if not _key("URLSCAN"):
            return {"score": 0, "flags": []}
        try:
            import time

            headers = {
                "API-Key":      _key("URLSCAN"),
                "Content-Type": "application/json",
            }
            submit = requests.post(
                "https://urlscan.io/api/v1/scan/",
                headers=headers,
                json={"url": url, "visibility": "public"},
                timeout=10,
            )
            if submit.status_code not in (200, 201):
                # Fail silently if submission fails due to API limits or policy
                return {"score": 0, "flags": []}

            result_url = submit.json().get("api", "")
            if not result_url:
                return {"score": 0, "flags": ["urlscan.io: no result URL returned"]}

            time.sleep(12)  # wait for scan to complete

            result = requests.get(result_url, timeout=15)
            if result.status_code != 200:
                return {"score": 0, "flags": ["urlscan.io: scan result not ready"]}

            data    = result.json()
            verdict = data.get("verdicts", {}).get("overall", {})
            score   = verdict.get("score", 0)      # 0-100 (higher = more malicious)
            malicious = verdict.get("malicious", False)
            tags    = verdict.get("tags", [])

            risk  = min(100, score)
            flags = []
            if malicious:
                flags.append("urlscan.io: classified as MALICIOUS")
            if tags:
                flags.append(f"urlscan.io tags: {', '.join(tags[:3])}")
            if not malicious and score < 30:
                flags.append(f"urlscan.io: low risk (score {score}/100)")

            return {"score": risk, "flags": flags}

        except Exception as e:
            return {"score": 0, "flags": [f"urlscan.io error: {str(e)[:60]}"]}

    # ------------------------------------------------------------------
    # ACCOUNT ANALYSIS
    # ------------------------------------------------------------------

    def _account(self, features: dict) -> dict:
        """Analyze social media account credibility signals."""
        account = features.get("account", {})

        risk, flags = 0, []
        
        # --- Basic Profile Details ---
        followers = account.get("followers", 0)
        following = account.get("following", 0)
        total_posts = account.get("total_posts", 0)
        verified = account.get("verified", False)
        age = account.get("account_age_days", 730)
        
        # --- Age Logic ---
        if age < 7:
            risk += 45; flags.append(f"CRITICAL: Brand new account created {age} days ago")
        elif age < 30:
            risk += 30; flags.append(f"SUSPICIOUS: Very recent account ({age} days old)")
        elif age < 365:
            risk += 10; flags.append(f"Relatively new account ({age} days old)")
        else:
            flags.append(f"✓ Established account ({age // 365}+ years old)")

        # --- Followers Logic ---
        if followers < 20:
            risk += 30; flags.append(f"CRITICAL: Isolated account ({followers:,} followers)")
        elif followers < 100:
            risk += 15; flags.append(f"SUSPICIOUS: Extremely limited social proof ({followers:,} followers)")
        elif followers < 1000:
            flags.append(f"✓ Organic personal account ({followers:,} followers)")
        elif followers > 1_000_000:
            flags.append(f"✓ Mega-influencer status ({followers:,} followers)")
        elif followers > 100_000:
            flags.append(f"✓ Highly influential ({followers:,} followers)")
        else:
            flags.append(f"✓ Strong community presence ({followers:,} followers)")

        # --- Follower/Following Ratio (Bot detection) ---
        if following > 2000 and followers < 50:
            risk += 40; flags.append("CRITICAL: Extreme mass-follower behavior (Bot pattern)")
        elif followers > 0 and following > 0:
            ratio = followers / following
            if ratio < 0.05:
                risk += 25; flags.append(f"SUSPICIOUS: Abnormal followers/following ratio ({ratio:.2f})")
            elif ratio > 2.0:
                flags.append(f"✓ Natural growth pattern (Ratio {ratio:.1f}:1)")

        # --- Activity ---
        posts_24h = account.get("posts_last_24h", 0)
        if posts_24h > 50:
            risk += 35; flags.append(f"WARNING: Abnormal burst activity ({posts_24h} posts in 24h)")
        
        if total_posts < 5:
            risk += 20; flags.append(f"SUSPICIOUS: Minimal history ({total_posts} total posts)")
        elif total_posts > 5000:
            flags.append(f"✓ Deep posting history ({total_posts:,} total posts)")
        else:
            flags.append(f"✓ Active posting history ({total_posts} posts)")

        # --- Profile Verification ---
        if verified:
            risk = max(0, risk - 50)
            flags.append("✓ Platform Verified (Strong Trust Signal)")
        else:
            flags.append("Standard (Not Verified)")

        # --- Profile Completeness ---
        if not account.get("bio") or len(account.get("bio", "")) < 5:
            risk += 10; flags.append("Minimal bio / incomplete profile information")
        if not account.get("profile_pic"):
            risk += 15; flags.append("Missing profile picture")

        risk = max(0, min(100, risk))

        if risk < 30:    label = "trustworthy"
        elif risk < 50:  label = "uncertain"
        elif risk < 75:  label = "suspicious"
        else:            label = "highly_suspicious"

        # Construct specific metadata for transparency
        details = {
            "exact_followers": followers,
            "exact_following": following,
            "total_posts": total_posts,
            "account_age_days": age,
            "verified": verified,
            "bio": account.get("bio", "No bio available")
        }

        explanation = self._explain(
            score=risk / 100,
            label=label,
            analysis_type="social media account",
            signals={
                **details,
                "flags": flags
            }
        )
        return self._result(risk / 100, label, explanation, flags, details=details)

    # ------------------------------------------------------------------
    # TEXT ANALYSIS
    # ------------------------------------------------------------------

    def _text(self, features: dict) -> dict:
        """Analyze text credibility: writing style, sensationalism, consistency."""
        texts = features.get("texts", [])
        if not texts:
            single = features.get("clean_text") or features.get("body_text", "")
            if not single:
                return self._result(0.5, "unknown", "No text provided for analysis.")
            texts = [single]

        # 1. LLM-based writing style consistency (uses Groq)
        consistency = self._check_writing_consistency_llm(texts)

        # 2. Heuristic signals
        combined_text = " ".join(texts)
        flags = list(consistency["flags"])

        # Alarmism score (exclamation mark density)
        alarm_count  = combined_text.count("!")
        alarm_avg    = alarm_count / len(texts)
        alarm_risk   = min(30, alarm_avg * 5)

        # Sensationalism score
        sens_risk, found_keywords = 0, []
        for kw in SENSATIONAL_KEYWORDS:
            if kw.lower() in combined_text.lower():
                sens_risk += 12
                found_keywords.append(f'"{kw}"')
        sens_risk = min(40, sens_risk)

        if alarm_avg > 5:
            flags.append(f"Heavy alarmist punctuation ({alarm_count} exclamation marks)")
        elif alarm_avg > 2:
            flags.append(f"Moderate alarmist punctuation ({alarm_count} exclamation marks)")

        if found_keywords:
            flags.append(f"Sensationalist keywords detected: {', '.join(found_keywords[:4])}")

        # Capitalization abuse
        cap_ratio = len(re.findall(r"[A-Z]{2,}", combined_text)) / max(len(combined_text.split()), 1)
        cap_risk  = 0
        if cap_ratio > 0.2:
            cap_risk = 20
            flags.append("Excessive ALL-CAPS words detected")
        elif cap_ratio > 0.1:
            cap_risk = 10
            flags.append("Some excessive capitalization")

        # Source citation presence (positive signal)
        citation_patterns = [
            r"\baccording to\b", r"\bcited by\b", r"\bsource:\b",
            r"\bstudy by\b", r"\breports by\b", r"\bpublished in\b",
        ]
        has_citations = any(re.search(p, combined_text, re.IGNORECASE) for p in citation_patterns)
        if has_citations:
            flags.append("✓ Contains source citations or attributions")

        # Combine risks
        total_risk = (
            consistency["score"] * 0.50 +
            alarm_risk            * 0.20 +
            sens_risk             * 0.20 +
            cap_risk              * 0.10
        )
        total_risk = min(100, total_risk)
        # Citations reduce risk slightly
        if has_citations:
            total_risk = max(0, total_risk - 10)

        if total_risk < 30:    label = "credible"
        elif total_risk < 60:  label = "questionable"
        else:                  label = "misleading"

        explanation = self._explain(
            score=total_risk / 100,
            label=label,
            analysis_type="text content and writing style",
            signals={
                "style_consistency":   consistency["score"],
                "alarmism_score":      round(alarm_risk, 1),
                "sensationalism_score": sens_risk,
                "capitalization_risk": cap_risk,
                "has_citations":       has_citations,
                "analytical_findings": consistency.get("findings", ""),
                "flags":               flags,
            },
        )
        return self._result(total_risk / 100, label, explanation, flags)

    def _image(self, features: dict) -> dict:
        """Analyze image credibility via file metadata and provenance signals."""
        image_path = features.get("path") or features.get("image_path") or features.get("file_path")
        flags = []
        risk = 0

        if not image_path or not os.path.exists(str(image_path)):
            return self._result(0.5, "unknown", "No image file available for credibility analysis.", flags)

        try:
            from PIL import Image
            import piexif

            img = Image.open(image_path)
            width, height = img.size

            # Resolution check
            if width < 200 or height < 200:
                risk += 20
                flags.append(f"SUSPICIOUS: Very low resolution ({width}x{height})")
            elif width >= 1920 or height >= 1920:
                flags.append(f"✓ High resolution original ({width}x{height})")
            else:
                flags.append(f"Standard resolution ({width}x{height})")

            # EXIF metadata analysis
            exif_data = {}
            try:
                exif_bytes = img.info.get("exif", b"")
                if exif_bytes:
                    exif_data = piexif.load(exif_bytes)
                    flags.append("✓ EXIF metadata present")

                    # Check for editing software
                    ifd = exif_data.get("0th", {})
                    software = ifd.get(piexif.ImageIFD.Software, b"").decode("utf-8", errors="ignore").strip()
                    if software:
                        editing_tools = ["photoshop", "gimp", "canva", "pixlr", "lightroom", "affinity"]
                        if any(tool in software.lower() for tool in editing_tools):
                            risk += 25
                            flags.append(f"SUSPICIOUS: Edited with {software}")
                        else:
                            flags.append(f"✓ Camera/Software: {software}")

                    # Check for original date
                    exif_ifd = exif_data.get("Exif", {})
                    date_original = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal, b"").decode("utf-8", errors="ignore").strip()
                    if date_original:
                        flags.append(f"✓ Original capture date: {date_original}")
                    else:
                        risk += 10
                        flags.append("No original capture date in EXIF")
                else:
                    risk += 15
                    flags.append("No EXIF metadata (possibly stripped or screenshot)")
            except Exception:
                risk += 10
                flags.append("Could not parse EXIF metadata")

            # Format analysis
            fmt = img.format or "unknown"
            if fmt.upper() in ("JPEG", "JPG", "TIFF", "CR2", "NEF"):
                flags.append(f"✓ Camera-native format ({fmt})")
            elif fmt.upper() in ("PNG", "WEBP"):
                risk += 5
                flags.append(f"Web/screenshot format ({fmt})")

            img.close()
        except ImportError:
            return self._result(0.5, "unknown", "Image analysis libraries not available.", flags)
        except Exception as e:
            return self._result(0.5, "unknown", f"Image metadata analysis failed: {e}", flags)

        risk = max(0, min(100, risk))
        score = risk / 100

        if risk < 25:    label = "trustworthy"
        elif risk < 50:  label = "uncertain"
        elif risk < 75:  label = "suspicious"
        else:            label = "highly_suspicious"

        explanation = self._explain(
            score=score, label=label,
            analysis_type="image metadata and provenance",
            signals={"risk": risk, "flags": flags},
        )
        return self._result(score, label, explanation, flags)

    def _video(self, features: dict) -> dict:
        """Analyze video credibility via file metadata and provenance signals."""
        video_path = features.get("video_path") or features.get("path") or features.get("file_path")
        flags = []
        risk = 0

        if not video_path or not os.path.exists(str(video_path)):
            return self._result(0.5, "unknown", "No video file available for credibility analysis.", flags)

        try:
            file_size = os.path.getsize(video_path)
            file_ext = os.path.splitext(video_path)[1].lower()

            # File size check
            if file_size < 50_000:  # < 50KB
                risk += 25
                flags.append(f"SUSPICIOUS: Extremely small file ({file_size // 1024}KB)")
            elif file_size > 100_000_000:  # > 100MB
                flags.append(f"✓ Large original file ({file_size // (1024*1024)}MB)")
            else:
                flags.append(f"File size: {file_size // (1024*1024)}MB")

            # Format analysis
            native_formats = [".mp4", ".mov", ".avi", ".mkv", ".m4v"]
            compressed_formats = [".webm", ".gif", ".3gp"]
            if file_ext in native_formats:
                flags.append(f"✓ Standard video format ({file_ext})")
            elif file_ext in compressed_formats:
                risk += 10
                flags.append(f"Compressed/web format ({file_ext})")
            else:
                risk += 5
                flags.append(f"Uncommon format ({file_ext})")

            # Try to get video metadata via cv2
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0

                    if width < 320 or height < 240:
                        risk += 20
                        flags.append(f"SUSPICIOUS: Very low resolution ({width}x{height})")
                    elif width >= 1920:
                        flags.append(f"✓ HD resolution ({width}x{height})")
                    else:
                        flags.append(f"Resolution: {width}x{height}")

                    flags.append(f"Duration: {duration:.1f}s, {fps:.0f}fps")

                    if duration < 2:
                        risk += 15
                        flags.append("SUSPICIOUS: Very short clip (< 2s)")
                    cap.release()
                else:
                    risk += 10
                    flags.append("Could not open video for metadata extraction")
            except ImportError:
                flags.append("Video metadata extraction not available (OpenCV missing)")

        except Exception as e:
            return self._result(0.5, "unknown", f"Video metadata analysis failed: {e}", flags)

        risk = max(0, min(100, risk))
        score = risk / 100

        if risk < 25:    label = "trustworthy"
        elif risk < 50:  label = "uncertain"
        elif risk < 75:  label = "suspicious"
        else:            label = "highly_suspicious"

        explanation = self._explain(
            score=score, label=label,
            analysis_type="video file metadata and provenance",
            signals={"risk": risk, "flags": flags},
        )
        return self._result(score, label, explanation, flags)

    def _document(self, features: dict) -> dict:
        """Analyze document credibility — delegates to text analysis if text is available."""
        # If we have extracted text, run the full text credibility analysis
        text = features.get("clean_text") or features.get("body_text", "")
        if text and len(text.strip()) > 50:
            # Temporarily set type to "text" for the _text handler, then restore
            original_type = features.get("type")
            features["type"] = "text"
            result = self._text(features)
            features["type"] = original_type
            return result

        # Fallback: analyze document metadata
        doc_path = features.get("document_path") or features.get("path") or features.get("file_path")
        flags = []
        risk = 0

        if not doc_path or not os.path.exists(str(doc_path)):
            return self._result(0.5, "unknown", "No document file available for credibility analysis.", flags)

        try:
            file_size = os.path.getsize(doc_path)
            file_ext = os.path.splitext(doc_path)[1].lower()

            if file_size < 1000:
                risk += 20
                flags.append(f"SUSPICIOUS: Very small document ({file_size} bytes)")
            else:
                flags.append(f"Document size: {file_size // 1024}KB")

            if file_ext == ".pdf":
                flags.append("✓ Standard PDF format")
            elif file_ext in (".docx", ".doc"):
                flags.append("✓ Standard document format")
            else:
                risk += 5
                flags.append(f"Non-standard document format ({file_ext})")

        except Exception as e:
            return self._result(0.5, "unknown", f"Document analysis failed: {e}", flags)

        risk = max(0, min(100, risk))
        score = risk / 100

        if risk < 25:    label = "trustworthy"
        elif risk < 50:  label = "uncertain"
        elif risk < 75:  label = "suspicious"
        else:            label = "highly_suspicious"

        explanation = self._explain(
            score=score, label=label,
            analysis_type="document metadata",
            signals={"risk": risk, "flags": flags},
        )
        return self._result(score, label, explanation, flags)

    def _default(self) -> dict:
        return self._result(0.5, "unknown", "No specific handler available for this analysis type.")

    # ------------------------------------------------------------------
    # RESULT HELPER
    # ------------------------------------------------------------------

    def _result(self, score: float, label: str, explanation: str, flags: list = None, details: dict = None) -> dict:
        out = {
            "score":       round(score, 3),
            "label":       label,
            "explanation": explanation,
            "flags":       flags or [],
            "timestamp":   datetime.utcnow().isoformat() + "Z",
        }
        if details:
            out["details"] = details
        return out

    # ------------------------------------------------------------------
    # LLM EXPLANATION LAYER
    # ------------------------------------------------------------------

    def _explain(self, score: float, label: str, analysis_type: str, signals: dict) -> str:
        """Generate a human-readable forensic explanation via Groq LLM (or fallback)."""
        if not GROQ_AVAILABLE or not GROQ_API_KEY:
            return self._fallback_explanation(score, label, analysis_type, signals)

        try:
            client = Groq(api_key=GROQ_API_KEY)
            flags_text    = ", ".join(signals.get("flags", [])) or "none"
            signals_clean = {k: v for k, v in signals.items() if k != "flags"}

            prompt = (
                f"You are analyzing the credibility of a {analysis_type}. "
                f"Risk score: {score:.2f} (0 = safe, 1 = dangerous). Verdict: '{label}'. "
                f"Forensic signals: {json.dumps(signals_clean)}. "
                f"Flags raised: {flags_text}. "
                f"Explain concisely why this {analysis_type} received this credibility verdict."
            )

            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Source Credibility Analyst. "
                            "Explain the verdict in 2-3 sentences maximum. "
                            "Focus on the most impactful signals. "
                            "Do NOT mention raw numerical scores or percentages. "
                            "Use descriptive terms: 'highly suspicious', 'well-established', "
                            "'recently created', 'abnormal behavior', 'known threat', etc. "
                            "Be direct, professional, and factual."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.1-8b-instant",
                max_tokens=150,
                temperature=0.3,
                timeout=10.0,
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq explanation failed: {e}")
            return self._fallback_explanation(score, label, analysis_type, signals)

    def _fallback_explanation(self, score: float, label: str, analysis_type: str, signals: dict) -> str:
        """Rule-based explanation when LLM is unavailable."""
        flags = signals.get("flags", [])
        if not flags:
            return f"Credibility analysis of this {analysis_type} complete. Verdict: {label}."

        key_flags = flags[:3]
        summary   = "; ".join(key_flags)

        if score < 0.25:    tone = "appears trustworthy"
        elif score < 0.50:  tone = "shows some concerning indicators"
        elif score < 0.75:  tone = "raises significant credibility concerns"
        else:               tone = "is highly suspicious and should not be trusted"

        return f"This {analysis_type} {tone}. Key findings: {summary}."

    # ------------------------------------------------------------------
    # WRITING CONSISTENCY (LLM + statistical fallback)
    # ------------------------------------------------------------------

    def _check_writing_consistency_llm(self, texts: list) -> dict:
        """Use Groq LLM to assess writing style consistency across samples."""
        if not texts:
            return {"score": 50, "flags": ["No text provided"]}

        if not GROQ_AVAILABLE or not GROQ_API_KEY:
            return self._check_writing_consistency_statistical(texts)

        try:
            client  = Groq(api_key=GROQ_API_KEY)
            samples = "\n---\n".join(t[:2000] for t in texts[:3])

            prompt = (
                "Analyze the following text samples from the same claimed source. "
                "Assess writing style, tone, vocabulary level, and linguistic patterns. "
                "Look for signs of: ghost-writing, AI generation, multiple authors, "
                "inconsistent register, or deliberate manipulation. "
                "Respond ONLY with a JSON object: "
                "{\"consistency_score\": 0-100, \"findings\": \"string\", \"flags\": [\"string\"]}\n\n"
                f"SAMPLES:\n{samples}"
            )

            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            res  = json.loads(completion.choices[0].message.content)
            risk = 100 - res.get("consistency_score", 50)

            return {
                "score":    risk,
                "findings": res.get("findings", ""),
                "flags":    res.get("flags", []) or (
                    ["Variable writing style — possible mixed authorship"] if risk > 40
                    else ["✓ Consistent authorial tone and style"]
                ),
            }

        except Exception as e:
            logger.error(f"LLM consistency analysis failed: {e}")
            return self._check_writing_consistency_statistical(texts)

    def _check_writing_consistency_statistical(self, texts: list) -> dict:
        """Statistical fallback for writing consistency (TF-IDF or sentence-transformers)."""
        if len(texts) < 2:
            return {"score": 0, "flags": ["Single text — consistency check skipped"]}

        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer, util
            model       = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings  = model.encode(texts, convert_to_tensor=True)
            sims        = [
                util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                for i in range(len(embeddings))
                for j in range(i + 1, len(embeddings))
            ]
            avg_sim    = sum(sims) / len(sims)
            consistency = avg_sim * 100
            risk        = 100 - consistency
            flags       = (
                ["✓ High writing consistency (semantic embeddings)"] if consistency > 70
                else ["Moderate style consistency"] if consistency > 40
                else ["Low consistency — possibly ghost-written or multi-author"]
            )
            return {"score": risk, "flags": flags}

        except ImportError:
            pass  # fall through to TF-IDF

        # TF-IDF fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf      = vectorizer.fit_transform(texts)
            matrix     = cosine_similarity(tfidf)
            sims       = [
                matrix[i, j]
                for i in range(matrix.shape[0])
                for j in range(i + 1, matrix.shape[1])
            ]
            avg_sim     = sum(sims) / len(sims)
            consistency = avg_sim * 100
            risk        = 100 - consistency
            flags       = (
                ["✓ High writing consistency (TF-IDF)"] if consistency > 70
                else ["Moderate consistency (TF-IDF)"] if consistency > 40
                else ["Low consistency (TF-IDF) — possibly ghost-written"]
            )
            return {"score": risk, "flags": flags}

        except Exception as e:
            return {"score": 50, "flags": [f"Style analysis unavailable: {str(e)[:50]}"]}


# =========================================================
# WEIGHTS (configurable)
# =========================================================

WEIGHTS = {
    "url_heuristic": 0.10,
    "virustotal":    0.15,
    "apivoid":       0.15,
    "whois":         0.10,
    "urlscan":       0.05,
    "account":       0.30,   # Account is most critical for social media sources
    "consistency":   0.15,
}


# =========================================================
# PIPELINE FUNCTION (combines all modules)
# =========================================================

def analyze_source(
    url: Optional[str] = None,
    account: Optional[dict] = None,
    texts: Optional[List[str]] = None,
) -> dict:
    """
    Run the full credibility pipeline and return a combined trust score.

    Parameters
    ----------
    url      : str   — URL to analyze
    account  : dict  — Account signals dict (see _account for expected keys)
    texts    : list  — List of text samples (>=2 enables consistency check)

    Returns
    -------
    dict with keys: score (0-100 trust), verdict, status, modules
    """
    analyzer   = Credibility()
    total      = 0.0
    weight_sum = 0.0
    modules    = {}

    if url:
        result = analyzer.evaluate({"type": "url", "url": url})
        modules["url"] = {
            "score":       int(result["score"] * 100),
            "label":       result["label"],
            "flags":       result["flags"],
            "explanation": result["explanation"],
        }
        w = WEIGHTS["url_heuristic"] + WEIGHTS["virustotal"] + WEIGHTS["apivoid"] + WEIGHTS["whois"] + WEIGHTS["urlscan"]
        total      += modules["url"]["score"] * w
        weight_sum += w

    if account:
        result = analyzer.evaluate({"type": "account", "account": account})
        modules["account"] = {
            "score":       int(result["score"] * 100),
            "label":       result["label"],
            "flags":       result["flags"],
            "explanation": result["explanation"],
        }
        total      += modules["account"]["score"] * WEIGHTS["account"]
        weight_sum += WEIGHTS["account"]

    if texts and len(texts) >= 1:
        result = analyzer.evaluate({"type": "text", "texts": texts})
        modules["text"] = {
            "score":       int(result["score"] * 100),
            "label":       result["label"],
            "flags":       result["flags"],
            "explanation": result["explanation"],
        }
        total      += modules["text"]["score"] * WEIGHTS["consistency"]
        weight_sum += WEIGHTS["consistency"]

    final_risk  = (total / weight_sum) if weight_sum else 50.0
    final_trust = round(100 - final_risk, 1)

    if final_trust > 80:
        verdict, status = "FIABLE",              "GREEN"
    elif final_trust > 60:
        verdict, status = "MODÉRÉMENT FIABLE",   "YELLOW"
    elif final_trust > 40:
        verdict, status = "QUESTIONNABLE",        "ORANGE"
    else:
        verdict, status = "NON FIABLE",           "RED"

    return {
        "trust_score": final_trust,
        "risk_score":  round(final_risk, 1),
        "verdict":     verdict,
        "status":      status,
        "modules":     modules,
        "timestamp":   datetime.utcnow().isoformat() + "Z",
    }
