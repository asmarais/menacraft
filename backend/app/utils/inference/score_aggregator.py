from typing import List, Tuple, Dict, Any, Callable, Union
from dataclasses import dataclass
import logging

from utils.inference_results_dataclasses import InferenceResult


@dataclass
class AggregatedScore:
    """Result of score aggregation across multiple frames.
    
    Attributes:
        prediction (str): Aggregated prediction label ('REAL' or 'FAKE')
        confidence (float): Aggregated confidence score (0-1)
        method (str): Aggregation method used
        frame_count (int): Number of frames aggregated
        metadata (Dict[str, Any]): Additional aggregation metadata
    """
    prediction: Union[str, List[str]]
    confidence: Union[float, List[float]]
    method: str
    frame_count: int
    metadata: Dict[str, Any]

class ScoreAggregator:
    """Aggregates frame-level predictions into video-level predictions.
    Supports multiple aggregation strategies:
    - majority: Majority vote of frame predictions
    - average: Average of confidence scores
    - weighted_average: Weighted average based on confidence
    - max_confidence: Prediction with highest confidence
    - threshold: Count frames above confidence threshold
    """

    AVAILABLE_METHODS: List[str] = [
        "majority",
        "average",
        "weighted_average",
        "max_confidence",
        "threshold"
    ]
    """Available aggregation methods"""

    def __init__(self, method: str ="majority"):
        """Initialize ScoreAggregator.
        Parameters:
            method (str): Aggregation method to use
                - "majority": majority vote of predictions
                - "average": average of confidence scores
                - "weighted_average": weighted average by confidence
                - "max_confidence": prediction with max confidence
                - "threshold": count frames above threshold
        Raises:
            ValueError: if method is not in AVAILABLE_METHODS
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Invalid aggregation method: {method}. "
                f"Available methods: {self.AVAILABLE_METHODS}"
            )

        self.method: str = method
        """Aggregation method to use"""
        self.logger: logging.Logger = logging.getLogger("/".join(__file__.split("/")[-2:]))
        """Logger for the ScoreAggregator class"""

        self._aggregation_functions: Dict[str, Callable] = {
            "majority": self._aggregate_majority,
            "average": self._aggregate_average,
            "weighted_average": self._aggregate_weighted_average,
            "max_confidence": self._aggregate_max_confidence,
            "threshold": self._aggregate_threshold
        }
        """Mapping of method names to aggregation functions"""

    def aggregate(self, frame_results: List[InferenceResult]) -> AggregatedScore:
        """Aggregate frame-level predictions into video-level prediction.
        Parameters:
            frame_results (List[InferenceResult]): List of frame predictions
        Returns:
            AggregatedScore: Aggregated prediction result
        Raises:
            ValueError: if frame_results is empty
        """
        if not frame_results:
            self.logger.error("No frame results provided for aggregation")
            raise ValueError("Cannot aggregate empty frame_results")

        self.logger.debug(f"Aggregating {len(frame_results)} frames using method '{self.method}'")

        aggregation_fn: callable = self._aggregation_functions[self.method]
        prediction, confidence, metadata = aggregation_fn(frame_results)

        return AggregatedScore(
            prediction=prediction,
            confidence=confidence,
            method=self.method,
            frame_count=len(frame_results),
            metadata=metadata
        )
        
    def check_labels(self, frame_results: List[InferenceResult]) -> set[str]:
        """Identify unique labels present in frame results. And assert that 
        all frames contain the same set of labels.
        Doing so allows dynamic handling of multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
        Returns:
            List[str]: Unique labels found in predictions
        Raises:
            ValueError: if inconsistent labels are found across frames
        """
        unique_labels = set([label for label in frame_results[0].probabilities.keys()])
        
        for i, frame in enumerate(frame_results):
            frame_labels = frame.probabilities.keys()
            
            if frame_labels != unique_labels:
                err_msg = (
                    f"Inconsistent labels in frame results. "
                    f"Some frames have different labels: cannot aggregate. "
                    f"- Frame 0 labels: {unique_labels},\n- Frame {i} labels: {frame_labels}"
                )
                self.logger.error(err_msg)
                raise ValueError(err_msg)
            
        return unique_labels

    def _aggregate_majority(self, frame_results: List[InferenceResult]) -> Tuple[str, float, Dict[str, Any]]:
        """Majority vote aggregation. The most frequent prediction across frames becomes the final prediction.
        Confidence is the proportion of frames voting for that prediction. Supports multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
        """
        predictions = self.check_labels(frame_results)
        
        counts: Dict[str, int] = {label: 0 for label in predictions}
        for result in frame_results:
            counts[result.prediction] += 1
        
        max_label = max(counts, key=counts.get)
        total = len(frame_results)
        confidence = counts[max_label] / total

        metadata = {
            **{f"count_{label}": count for label, count in counts.items()},
            **{f"ratio_{label}": count / total for label, count in counts.items()}
        }

        return max_label, confidence, metadata

    def _aggregate_average(self, 
        frame_results: List[InferenceResult], 
        threshold: float =0.5
    ) -> Union[Tuple[str, float, Dict[str, Any]], List[Tuple[str, float, Dict[str, Any]]]]:
        """Average confidence aggregation. Supports multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
            threshold (float): Threshold to retain the prediction. 
        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
            List[Tuple[str, float, Dict]]: (prediction, confidence, metadata) for each label if multi-class
        """
        _ = self.check_labels(frame_results) #ensure consistent labels

        avg_confidence_scores = {
            label: sum(frame.probabilities[label] for frame in frame_results) / len(frame_results)
            for label in frame_results[0].probabilities.keys()
        }

        above_threshold = {label: score for label, score in avg_confidence_scores.items() if score > threshold}
        predictions_above_threshold = list(above_threshold.keys())
        confidences_above_threshold = list(above_threshold.values())

        metadata = {
            **{f"avg_confidence_{label}": avg_score for label, avg_score in avg_confidence_scores.items()},
            "min_confidence": min(avg_confidence_scores.values()),
            "max_confidence": max(avg_confidence_scores.values()),
            "threshold_used": threshold
        }

        if len(predictions_above_threshold) > 1:
            #multi-class scenario
            results = []
            for label, avg_score in zip(predictions_above_threshold, confidences_above_threshold):
                results.append((label, avg_score, metadata))
            return results
        
        elif len(predictions_above_threshold) == 1:
            #binary scenario
            return predictions_above_threshold[0], confidences_above_threshold[0], metadata
        
        else:
            self.logger.warning("No predictions above the specified threshold")
            return "UNKNOWN", 0.0, metadata

    def _aggregate_weighted_average(self, 
        frame_results: List[InferenceResult],
        threshold: float =0.5
    ) -> Union[Tuple[str, float, Dict[str, Any]], List[Tuple[str, float, Dict[str, Any]]]]:
        """Weighted average aggregation. Supports multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
            threshold (float): Threshold to retain the prediction.
        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
        """
        labels = self.check_labels(frame_results) #ensure consistent labels
        
        weights: Dict[str, float] = {label: 0.0 for label in labels}
        weighted_scores: Dict[str, float] = {label: 0.0 for label in labels}
        
        for frame in frame_results:
            for label in labels:
                prob = frame.probabilities[label]
                weight = frame.confidence
                weighted_scores[label] += prob * weight
                weights[label] += weight

        weighted_avg_scores: Dict[str, float] = {}
        for label in labels:
            if weights[label] > 0:
                weighted_avg_scores[label] = weighted_scores[label] / weights[label]
            else:
                weighted_avg_scores[label] = 0.0
        
        prediction = max(weighted_avg_scores, key=weighted_avg_scores.get)
        confidence = weighted_avg_scores[prediction]

        metadata = {
            **{f"weighted_avg_confidence_{label}": score for label, score in weighted_avg_scores.items()},
            **{f"total_weight_{label}": weights[label] for label in labels},
            "min_weighted_avg_confidence": min(weighted_avg_scores.values()),
            "max_weighted_avg_confidence": max(weighted_avg_scores.values()),
            "threshold_used": threshold
        }

        avg_above_threshold = {k: v for k, v in weighted_avg_scores.items() if v > threshold}
        
        if len(avg_above_threshold) == 0:
            self.logger.warning("No predictions above the specified threshold")
            return "UNKNOWN", 0.0, metadata

        elif len(avg_above_threshold) > 1:
            #multi-class scenario
            results = []
            for label, score in avg_above_threshold.items():
                results.append((label, score, metadata))
            return results
        
        else:
            #binary scenario
            return prediction, confidence, metadata

    def _aggregate_max_confidence(self, frame_results: List[InferenceResult]) -> Tuple[str, float, Dict[str, Any]]:
        """Maximum confidence aggregation. Supports multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
        """
        _ = self.check_labels(frame_results) #ensure consistent labels
        max_result = max(frame_results, key=lambda r: r.confidence)

        prediction = max_result.prediction
        confidence = max_result.confidence

        metadata = {
            "max_confidence_frame": frame_results.index(max_result),
            "max_confidence_value": confidence,
            "max_frame_prediction": prediction
        }

        return prediction, confidence, metadata

    def _aggregate_threshold(self, frame_results: List[InferenceResult], threshold: float = 0.5) -> Tuple[str, float, Dict[str, Any]]:
        """Threshold-based aggregation. Supports multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
            threshold (float): Confidence threshold
        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
        """
        labels = self.check_labels(frame_results) #ensure consistent labels
        total = len(frame_results)
        counts_above_threshold: Dict[str, int] = {label: 0 for label in labels}
        
        for result in frame_results:
            for label in labels:
                if result.probabilities[label] > threshold:
                    counts_above_threshold[label] += 1
                    
        ratio_above_threshold: Dict[str, float] = {label: counts_above_threshold[label] / total for label in labels}

        if any(ratio > 0.5 for ratio in ratio_above_threshold.values()):
            prediction = max(ratio_above_threshold, key=ratio_above_threshold.get)
            confidence = ratio_above_threshold[prediction]
        
        else:
            prediction = "NONE"
            confidence = 0.0

        metadata = {
            **{f"count_above_threshold_{label}": count for label, count in counts_above_threshold.items()},
            **{f"ratio_above_threshold_{label}": ratio_above_threshold[label] for label in labels},
            "threshold_used": threshold,
            "total_frames": total,
            "frames_below_threshold": total - sum(counts_above_threshold.values())
        }

        return prediction, confidence, metadata

    @staticmethod
    def compare_methods(cls,
        frame_results: List[InferenceResult],
        methods: List[str] = None
    ) -> Dict[str, AggregatedScore]:
        """Compare multiple aggregation methods on the same frame results.
        Useful for analyzing which aggregation strategy works best.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
            methods (List[str], optional): Methods to compare.
                If None, compares all available methods.
        Returns:
            Dict[str, AggregatedScore]: Results for each method
        """
        if methods is None:
            methods = ScoreAggregator.AVAILABLE_METHODS

        comparison = {
            method: ScoreAggregator(method=method).aggregate(frame_results)
            for method in methods
        }

        return comparison
