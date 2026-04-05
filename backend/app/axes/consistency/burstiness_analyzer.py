# burstiness_analyzer.py
import re
import numpy as np
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass
class BurstinessResult:
    ai_likelihood_score: float      # 0.0 (human) → 1.0 (AI)
    burstiness_ratio: float         # low = AI-like
    sentence_length_variance: float
    avg_sentence_length: float
    sentence_count: int
    short_sentence_ratio: float
    long_sentence_ratio: float
    repetition_score: float         # AI repeats phrases
    transition_uniformity: float    # AI uses same transitions
    flags: list[str]
    breakdown: dict


class DocumentBurstinessAnalyzer:
    """
    Human writing = HIGH variance in sentence lengths (bursty)
    AI writing   = LOW variance, uniform medium-length sentences

    Additional signals:
    - Repetitive transition words (AI loves "Furthermore", "Moreover")
    - Uniform paragraph structure
    - Low lexical diversity
    """

    # AI models overuse these transitions
    AI_TRANSITIONS = {
        "furthermore", "moreover", "additionally", "consequently",
        "therefore", "thus", "hence", "nonetheless", "nevertheless",
        "in conclusion", "in summary", "to summarize", "it is worth noting",
        "it is important to note", "it should be noted", "notably",
        "significantly", "importantly", "ultimately", "overall"
    }

    # Sentence boundary patterns (handles English + Arabic-heavy punctuation)
    SENTENCE_SPLIT_PATTERN = re.compile(
        r'(?<=[.!?؟])\s+'
        r'|(?<=[.!?؟])"?\s+'
        r'|\n{2,}'
    )

    def __init__(self):
        pass

    def analyze(self, text: str) -> BurstinessResult:
        """
        Main entry point.
        text: cleaned plain text (no HTML)
        """
        text = self._normalize(text)

        if not text or len(text.split()) < 20:
            return self._insufficient_text_result()

        sentences   = self._split_sentences(text)
        lengths     = [len(s.split()) for s in sentences if s.strip()]

        if len(lengths) < 3:
            return self._insufficient_text_result()

        # ── Core burstiness ─────────────────────────────────────
        mean_len    = float(np.mean(lengths))
        std_len     = float(np.std(lengths))
        # Key ratio: low std relative to mean = AI-like uniformity
        burstiness  = std_len / (mean_len + 1e-9)

        # ── Sentence distribution ────────────────────────────────
        short_ratio = sum(1 for l in lengths if l <= 8)  / len(lengths)
        long_ratio  = sum(1 for l in lengths if l >= 30) / len(lengths)
        # AI rarely writes very short or very long sentences

        # ── Transition word repetition ───────────────────────────
        transition_score = self._analyze_transitions(text)

        # ── Lexical diversity (Type-Token Ratio) ─────────────────
        words    = re.findall(r'\b\w+\b', text.lower())
        ttr      = len(set(words)) / (len(words) + 1e-9)
        # Low TTR = repetitive vocabulary = AI signal

        # ── Repetitive n-gram detection ──────────────────────────
        repetition_score = self._detect_repetition(sentences)

        # ── Paragraph uniformity ─────────────────────────────────
        paragraph_uniformity = self._analyze_paragraph_uniformity(text)

        # ── Compute final AI likelihood ──────────────────────────
        ai_score, flags = self._compute_ai_score(
            burstiness        = burstiness,
            short_ratio       = short_ratio,
            long_ratio        = long_ratio,
            ttr               = ttr,
            transition_score  = transition_score,
            repetition_score  = repetition_score,
            paragraph_uni     = paragraph_uniformity,
            sentence_count    = len(lengths)
        )

        return BurstinessResult(
            ai_likelihood_score      = round(ai_score, 4),
            burstiness_ratio         = round(burstiness, 4),
            sentence_length_variance = round(std_len, 3),
            avg_sentence_length      = round(mean_len, 3),
            sentence_count           = len(lengths),
            short_sentence_ratio     = round(short_ratio, 3),
            long_sentence_ratio      = round(long_ratio, 3),
            repetition_score         = round(repetition_score, 3),
            transition_uniformity    = round(transition_score, 3),
            flags                    = flags,
            breakdown = {
                "burstiness_contribution"   : round(self._burstiness_to_score(burstiness), 3),
                "transition_contribution"   : round(transition_score * 0.20, 3),
                "repetition_contribution"   : round(repetition_score * 0.15, 3),
                "lexical_diversity_ttr"     : round(ttr, 4),
                "paragraph_uniformity"      : round(paragraph_uniformity, 3),
            }
        )

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'<[^>]+>', ' ', text)       # strip HTML
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_sentences(self, text: str) -> list[str]:
        parts = self.SENTENCE_SPLIT_PATTERN.split(text)
        # Filter noise: keep only parts with >= 3 words
        return [p.strip() for p in parts if len(p.split()) >= 3]

    def _burstiness_to_score(self, burstiness: float) -> float:
        """
        Convert burstiness ratio to AI likelihood contribution.
        burstiness < 0.3  → very AI-like  → contribution ~0.50
        burstiness > 0.8  → very human    → contribution ~0.05
        """
        # Sigmoid-like mapping (inverted)
        if burstiness < 0.20:
            return 0.55
        elif burstiness < 0.35:
            return 0.40
        elif burstiness < 0.50:
            return 0.25
        elif burstiness < 0.70:
            return 0.12
        else:
            return 0.04

    def _analyze_transitions(self, text: str) -> float:
        """
        Count AI-typical transition words.
        Returns 0.0-1.0 (higher = more AI-like)
        """
        lower = text.lower()
        total_words = len(text.split())
        if total_words == 0:
            return 0.0

        found = []
        for t in self.AI_TRANSITIONS:
            count = lower.count(t)
            if count > 0:
                found.append((t, count))

        total_transition_words = sum(c for _, c in found)
        # Normalize: more than 2% transition density is suspicious
        density = total_transition_words / total_words
        score   = min(density / 0.02, 1.0)

        return score

    def _detect_repetition(self, sentences: list[str]) -> float:
        """
        Detect repeated n-grams across sentences.
        AI often reuses the same 3-4 word phrases.
        Returns 0.0-1.0
        """
        if len(sentences) < 4:
            return 0.0

        # Extract 3-grams from all sentences
        all_trigrams = []
        for s in sentences:
            words = re.findall(r'\b\w+\b', s.lower())
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            all_trigrams.extend(trigrams)

        if not all_trigrams:
            return 0.0

        from collections import Counter
        counts  = Counter(all_trigrams)
        # Filter out common stop-word trigrams
        repeated = {k: v for k, v in counts.items()
                    if v >= 3 and not self._is_stopword_trigram(k)}

        # Score based on ratio of repeated trigrams
        score = min(len(repeated) / max(len(set(all_trigrams)), 1) * 5, 1.0)
        return score

    def _is_stopword_trigram(self, trigram: str) -> bool:
        stopwords = {"the", "a", "an", "is", "in", "of", "and",
                     "to", "it", "was", "for", "on", "that", "this"}
        words = trigram.split()
        return all(w in stopwords for w in words)

    def _analyze_paragraph_uniformity(self, text: str) -> float:
        """
        AI generates paragraphs of suspiciously similar length.
        Returns 0.0-1.0 (high = uniform = AI-like)
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            return 0.0

        para_lengths = [len(p.split()) for p in paragraphs]
        mean_p = np.mean(para_lengths)
        std_p  = np.std(para_lengths)
        cv     = std_p / (mean_p + 1e-9)  # coefficient of variation

        # Low CV = uniform paragraphs = AI signal
        uniformity = max(0.0, 1.0 - cv)
        return uniformity

    def _compute_ai_score(
        self,
        burstiness: float,
        short_ratio: float,
        long_ratio: float,
        ttr: float,
        transition_score: float,
        repetition_score: float,
        paragraph_uni: float,
        sentence_count: int
    ) -> tuple[float, list[str]]:

        flags = []

        # Component scores
        burst_score      = self._burstiness_to_score(burstiness)
        variety_score    = max(0.0, 0.5 - short_ratio - long_ratio)
        # AI avoids extremes → low short+long ratio is suspicious
        lexical_score    = max(0.0, (0.72 - ttr) / 0.72)
        # TTR below 0.72 is suspicious for long texts

        # Weighted sum
        ai_score = (
            burst_score       * 0.35 +
            transition_score  * 0.20 +
            repetition_score  * 0.15 +
            lexical_score     * 0.15 +
            paragraph_uni     * 0.10 +
            variety_score     * 0.05
        )

        # Flag generation
        if burstiness < 0.30:
            flags.append("low_sentence_length_variation")
        if transition_score > 0.5:
            flags.append("excessive_ai_transition_words")
        if repetition_score > 0.4:
            flags.append("high_phrase_repetition")
        if ttr < 0.55:
            flags.append("low_lexical_diversity")
        if paragraph_uni > 0.7:
            flags.append("suspiciously_uniform_paragraphs")
        if short_ratio < 0.05:
            flags.append("no_short_sentences_unusual")
        if long_ratio < 0.05 and sentence_count > 10:
            flags.append("no_long_sentences_unusual")

        return min(ai_score, 1.0), flags

    def _insufficient_text_result(self) -> BurstinessResult:
        return BurstinessResult(
            ai_likelihood_score      = 0.5,
            burstiness_ratio         = 0.0,
            sentence_length_variance = 0.0,
            avg_sentence_length      = 0.0,
            sentence_count           = 0,
            short_sentence_ratio     = 0.0,
            long_sentence_ratio      = 0.0,
            repetition_score         = 0.0,
            transition_uniformity    = 0.0,
            flags                    = ["insufficient_text"],
            breakdown                = {}
        )