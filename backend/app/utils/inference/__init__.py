"""
Inference module for VeridisQuo deepfake detection.

This module provides the InferenceEngine class and related utilities
for performing deepfake detection inference on images and videos.
"""

from .inference_engine import InferenceEngine
from utils.inference_results_dataclasses import InferenceResult, VideoInferenceResult
from .score_aggregator import ScoreAggregator, AggregatedScore

__version__ = "0.1.0"
__author__ = "Clément BARRIÈRE (@clembarr), Théo CASTILLO (@theosorus)"

__all__ = [
    "InferenceEngine",
    "InferenceResult",
    "VideoInferenceResult",
    "ScoreAggregator",
    "AggregatedScore"
]
