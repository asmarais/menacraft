from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class InferenceResult:
    """Result of a single inference prediction.

    Attributes:
        prediction (str): Predicted class label ('REAL' or 'FAKE')
        confidence (float): Confidence score for the prediction (0-1)
        probabilities (Dict[str, float]): Probability for each class
        raw_logits (List[float]): Raw model output logits
    """
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    raw_logits: List[float]

@dataclass
class VideoInferenceResult:
    """Result of inference on a video.

    Attributes:
        video_path (str): Path to the analyzed video
        frame_results (List[InferenceResult]): Results for each analyzed frame
        aggregate_prediction (str): Overall prediction for the video
        aggregate_confidence (float): Overall confidence score
        num_frames_analyzed (int): Number of frames analyzed
        aggregation_metadata (Dict[str, Any], optional): Additional metadata from aggregation
        gradcam_output_path (Optional[str]): Path to GradCAM visualization video (if generated)
    """
    video_path: str
    frame_results: List[InferenceResult]
    aggregate_prediction: str
    aggregate_confidence: float
    num_frames_analyzed: int
    aggregation_metadata: Dict = None
    gradcam_output_path: Optional[str] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.aggregation_metadata is None:
            self.aggregation_metadata = {}
