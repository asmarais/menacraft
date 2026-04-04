import httpx
from urllib.parse import urlparse
import whois
import ssl
import socket
from bs4 import BeautifulSoup
from datetime import datetime

SUSPICIOUS_PATTERNS = [
    r'(bbc|cnn|reuters|aljazeera)[^.]*\.(net|org|info|co)',  # spoofed domains
    r'\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}',                     # IP address URLs
    r'(breaking|news|today)[^.]*\.(tk|ml|ga|cf|gq)',         # free TLDs
    r'[а-яёА-ЯЁ]',                                           # Cyrillic homoglyphs
]

KNOWN_LEGIT_DOMAINS = {
    "reuters.com", "bbc.com", "aljazeera.com", "apnews.com",
    "lemonde.fr", "nytimes.com", "theguardian.com", "france24.com"
}

async def preprocess_url(url: str) -> dict:
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    root_domain = ".".join(domain.split(".")[-2:])

    # ── STAGE 1: URL-level analysis ──────────────────────────────
    url_features = analyze_url_structure(url, domain, root_domain)

    # ── STAGE 2: Scrape content ───────────────────────────────────
    scraped = await scrape_page(url)

    # ── STAGE 3: Domain intelligence ─────────────────────────────
    domain_intel = await get_domain_intelligence(domain, root_domain)

    # ── STAGE 4: Process scraped content ─────────────────────────
    text_features = {}
    image_features = {}

    if scraped.get("body_text"):
        text_features = extract_text_features(scraped["body_text"])

    if scraped.get("first_image_url"):
        try:
            resp = await httpx.AsyncClient().get(
                scraped["first_image_url"], timeout=5
            )
            image_features = preprocess_image(resp.content)
        except:
            image_features = {}

    return {
        "input_type": "url",
        "url": url,
        "domain": domain,
        "root_domain": root_domain,

        # ── For Axis 1 (Authenticity)
        "authenticity_features": {
            "body_text_for_ai_detection": text_features.get("clean_text", "")[:2000],
            "burstiness_ratio": text_features.get("burstiness_ratio", 0),
            "image_authenticity_ready": bool(image_features),
            "image_features": image_features.get("authenticity_features", {}),
        },

        # ── For Axis 2 (Contextual Consistency)
        "context_features": {
            "title": scraped.get("title", ""),
            "og_description": scraped.get("og_description", ""),
            "body_text": text_features.get("clean_text", ""),
            "document_embedding": text_features.get("document_embedding", []),
            "entities": text_features.get("entities", []),
            "keywords": text_features.get("keywords", []),
            "publish_date": scraped.get("publish_date", ""),
            "author": scraped.get("author", ""),
            "image_url": scraped.get("first_image_url", ""),
            "image_context_features": image_features.get("context_features", {}),
        },

        # ── For Axis 3 (Source Credibility)
        "source_features": {
            "url_risk_flags": url_features["flags"],
            "is_known_legitimate": root_domain in KNOWN_LEGIT_DOMAINS,
            "domain_age_days": domain_intel.get("domain_age_days", -1),
            "registrar": domain_intel.get("registrar", ""),
            "ssl_valid": domain_intel.get("ssl_valid", False),
            "is_blacklisted": domain_intel.get("is_blacklisted", False),
            "whois_available": domain_intel.get("whois_available", False),
            "has_author": bool(scraped.get("author")),
            "has_publish_date": bool(scraped.get("publish_date")),
            "outbound_link_count": len(scraped.get("outbound_links", [])),
            "writing_style_signals": text_features.get("style_signals", {}),
            "sensationalism_score": text_features.get("sentiment", {}).get("sensationalism_score", 0),
        }
    }


def analyze_url_structure(url: str, domain: str, root_domain: str) -> dict:
    flags = []

    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, domain):
            flags.append(f"suspicious_domain_pattern: {pattern}")

    if len(domain.split(".")) > 4:
        flags.append("excessive_subdomains")

    if any(brand in domain for brand in ["bbc","cnn","reuters","aljazeera"]):
        if root_domain not in KNOWN_LEGIT_DOMAINS:
            flags.append("brand_name_spoofing")

    free_tlds = {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz"}
    if any(domain.endswith(t) for t in free_tlds):
        flags.append("free_suspicious_tld")

    return {"flags": flags, "domain": domain, "root_domain": root_domain}


async def scrape_page(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; Veridact/1.0)"}
            resp = await client.get(url, headers=headers)
            html = resp.text
    except Exception as e:
        return {"error": str(e)}

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = soup.find("title")
    title_text = title.text.strip() if title else ""

    # Open Graph
    og_title = (soup.find("meta", property="og:title") or {}).get("content", "")
    og_desc  = (soup.find("meta", property="og:description") or {}).get("content", "")
    og_image = (soup.find("meta", property="og:image") or {}).get("content", "")

    # Author
    author_tag = (
        soup.find("meta", {"name": "author"}) or
        soup.find("a", {"rel": "author"}) or
        soup.find(class_=re.compile("author|byline", re.I))
    )
    author = ""
    if author_tag:
        author = author_tag.get("content") or author_tag.text.strip()

    # Date
    date_tag = (
        soup.find("time") or
        soup.find("meta", {"name": "date"}) or
        soup.find("meta", {"property": "article:published_time"})
    )
    publish_date = ""
    if date_tag:
        publish_date = date_tag.get("datetime") or date_tag.get("content") or date_tag.text

    # Body text
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    body_text = soup.get_text(separator=" ", strip=True)
    body_text = re.sub(r'\s+', ' ', body_text)[:10000]

    # Links
    outbound_links = [
        a["href"] for a in soup.find_all("a", href=True)
        if a["href"].startswith("http")
    ][:20]

    # First image
    first_img = og_image
    if not first_img:
        img_tag = soup.find("img", src=True)
        if img_tag:
            first_img = img_tag["src"]
            if first_img.startswith("/"):
                first_img = f"{urlparse(url).scheme}://{urlparse(url).netloc}{first_img}"

    return {
        "title": og_title or title_text,
        "og_description": og_desc,
        "author": author,
        "publish_date": publish_date,
        "body_text": body_text,
        "first_image_url": first_img,
        "outbound_links": outbound_links,
    }


async def get_domain_intelligence(domain: str, root_domain: str) -> dict:
    result = {
        "domain_age_days": -1,
        "registrar": "",
        "ssl_valid": False,
        "whois_available": False,
        "is_blacklisted": False,
    }

    # WHOIS lookup
    try:
        w = whois.whois(root_domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            age = (datetime.now() - creation).days
            result["domain_age_days"] = age
            result["registrar"] = str(w.registrar or "")
            result["whois_available"] = True
    except:
        pass

    # SSL check
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(3)
            s.connect((domain, 443))
            result["ssl_valid"] = True
    except:
        result["ssl_valid"] = False

    return result