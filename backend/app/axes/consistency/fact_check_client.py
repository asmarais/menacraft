# fact_check_client.py
import httpx
import os
from dataclasses import dataclass, field
from urllib.parse import urlencode


@dataclass
class FactCheckClaim:
    text: str               # the original claim text
    claimant: str           # who made the claim
    claim_date: str         # when the claim was made


@dataclass
class FactCheckReview:
    publisher_name: str
    publisher_site: str
    url: str
    title: str
    rating: str             # "False", "True", "Misleading", etc.
    rating_normalized: str  # our normalized: "false"|"true"|"mixed"|"unverified"
    review_date: str


@dataclass
class FactCheckResult:
    query_used: str
    match_found: bool
    match_count: int
    verdict: str            # "false"|"true"|"mixed"|"unverified"|"not_found"
    credibility_penalty: float  # 0.0 (no penalty) → 1.0 (proven false)
    claims: list[FactCheckClaim]
    reviews: list[FactCheckReview]
    flags: list[str]
    raw_response: dict = field(default_factory=dict)


class GoogleFactCheckClient:
    """
    Uses Google Fact Check Tools API (free, requires API key).
    Searches ClaimReview markup from:
      - Snopes, PolitiFact, AFP Fact Check,
        FullFact, Reuters Fact Check, etc.

    Free tier: 1000 requests/day
    Docs: https://developers.google.com/fact-check/tools/api
    """

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    # Map publisher ratings → our normalized scale
    RATING_MAP = {
        # FALSE variants
        "false": "false",
        "pants on fire": "false",
        "incorrect": "false",
        "fabricated": "false",
        "fake": "false",
        "not true": "false",
        "debunked": "false",
        "inaccurate": "false",
        "wrong": "false",
        "خاطئ": "false",        # Arabic
        "فبركة": "false",

        # TRUE variants
        "true": "true",
        "correct": "true",
        "accurate": "true",
        "verified": "true",
        "confirmed": "true",
        "صحيح": "true",          # Arabic

        # MIXED variants
        "mixed": "mixed",
        "half true": "mixed",
        "mostly true": "mixed",
        "mostly false": "mixed",
        "partially true": "mixed",
        "misleading": "mixed",
        "missing context": "mixed",
        "needs context": "mixed",
        "partly false": "mixed",
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google Fact Check API key required. "
                "Set GOOGLE_FACT_CHECK_API_KEY env variable."
            )

    async def check(
        self,
        query: str,
        language_code: str = "en",
        max_results: int = 5
    ) -> FactCheckResult:
        """
        query         : the claim text or keywords to search
        language_code : "en", "ar", "fr", etc.
        max_results   : max claims to return (1-10)
        """

        # ── Step 1: Call API ─────────────────────────────────────
        params = {
            "key":          self.api_key,
            "query":        query[:200],     # API limit
            "languageCode": language_code,
            "pageSize":     max_results,
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    self.BASE_URL,
                    params=params
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            return self._api_error_result(query, str(e))
        except Exception as e:
            return self._api_error_result(query, str(e))

        # ── Step 2: Parse response ───────────────────────────────
        raw_claims = data.get("claims", [])

        if not raw_claims:
            return FactCheckResult(
                query_used        = query,
                match_found       = False,
                match_count       = 0,
                verdict           = "not_found",
                credibility_penalty = 0.0,
                claims            = [],
                reviews           = [],
                flags             = ["no_fact_check_found"],
                raw_response      = data
            )

        # ── Step 3: Extract claims + reviews ────────────────────
        parsed_claims  = []
        parsed_reviews = []
        all_ratings    = []

        for raw_claim in raw_claims:
            # Claim object
            claim = FactCheckClaim(
                text      = raw_claim.get("text", ""),
                claimant  = raw_claim.get("claimant", "unknown"),
                claim_date= raw_claim.get("claimDate", ""),
            )
            parsed_claims.append(claim)

            # Reviews for this claim
            for review in raw_claim.get("claimReview", []):
                publisher = review.get("publisher", {})
                rating_raw = (
                    review.get("textualRating", "")
                    or review.get("rating", {}).get("ratingValue", "")
                )
                rating_norm = self._normalize_rating(str(rating_raw))
                all_ratings.append(rating_norm)

                parsed_reviews.append(FactCheckReview(
                    publisher_name = publisher.get("name", ""),
                    publisher_site = publisher.get("site", ""),
                    url            = review.get("url", ""),
                    title          = review.get("title", ""),
                    rating         = str(rating_raw),
                    rating_normalized = rating_norm,
                    review_date    = review.get("reviewDate", ""),
                ))

        # ── Step 4: Aggregate verdict ────────────────────────────
        verdict, penalty, flags = self._aggregate_verdict(all_ratings)

        return FactCheckResult(
            query_used          = query,
            match_found         = True,
            match_count         = len(parsed_claims),
            verdict             = verdict,
            credibility_penalty = penalty,
            claims              = parsed_claims,
            reviews             = parsed_reviews,
            flags               = flags,
            raw_response        = data
        )

    async def check_with_fallback_queries(
        self,
        primary_query: str,
        keywords: list[str],
        entities: list[dict],
        language_code: str = "en"
    ) -> FactCheckResult:
        """
        Try multiple query strategies before giving up.
        Strategy 1: Full claim text
        Strategy 2: Top 3 keywords joined
        Strategy 3: Main entity name
        """

        # Strategy 1: Full claim
        result = await self.check(primary_query, language_code)
        if result.match_found:
            return result

        # Strategy 2: Keywords
        if keywords:
            kw_query = " ".join(keywords[:4])
            result = await self.check(kw_query, language_code)
            if result.match_found:
                result.query_used = f"keywords: {kw_query}"
                return result

        # Strategy 3: Main entity (first ORG or GPE)
        priority_types = {"ORG", "GPE", "PERSON"}
        for ent in entities:
            if ent.get("label") in priority_types:
                result = await self.check(ent["text"], language_code)
                if result.match_found:
                    result.query_used = f"entity: {ent['text']}"
                    return result

        # Nothing found
        return FactCheckResult(
            query_used          = primary_query,
            match_found         = False,
            match_count         = 0,
            verdict             = "not_found",
            credibility_penalty = 0.0,
            claims              = [],
            reviews             = [],
            flags               = ["no_fact_check_found_all_strategies"],
        )

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _normalize_rating(self, raw: str) -> str:
        normalized = self.RATING_MAP.get(raw.lower().strip(), "unverified")
        return normalized

    def _aggregate_verdict(
        self,
        ratings: list[str]
    ) -> tuple[str, float, list[str]]:
        """
        Multiple reviews may exist for same claim.
        Aggregate into one verdict + penalty score.
        """
        if not ratings:
            return "unverified", 0.0, ["no_ratings_found"]

        from collections import Counter
        counts = Counter(ratings)
        flags  = []

        # Penalty mapping
        penalty_map = {
            "false":      0.90,   # proven false → high penalty
            "mixed":      0.45,   # mixed → moderate penalty
            "true":       0.00,   # verified true → no penalty
            "unverified": 0.20,   # unknown → small penalty
        }

        # Determine dominant verdict
        dominant = counts.most_common(1)[0][0]

        # If any source says "false" → always flag it
        if counts.get("false", 0) > 0:
            flags.append("flagged_as_false_by_fact_checker")
            if counts.get("false", 0) >= counts.get("true", 0):
                dominant = "false"

        # If conflicting verdicts
        unique_verdicts = set(ratings)
        if "true" in unique_verdicts and "false" in unique_verdicts:
            dominant = "mixed"
            flags.append("conflicting_fact_check_verdicts")

        penalty = penalty_map.get(dominant, 0.20)

        # Extra flag for multiple false verdicts
        if counts.get("false", 0) >= 2:
            flags.append("multiple_sources_confirmed_false")
            penalty = min(penalty + 0.05, 1.0)

        return dominant, round(penalty, 3), flags

    def _api_error_result(self, query: str, error: str) -> FactCheckResult:
        return FactCheckResult(
            query_used          = query,
            match_found         = False,
            match_count         = 0,
            verdict             = "api_error",
            credibility_penalty = 0.0,
            claims              = [],
            reviews             = [],
            flags               = [f"api_error: {error}"],
        )