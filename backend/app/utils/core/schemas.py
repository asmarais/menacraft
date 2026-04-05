"""
Data schemas for CSV files.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CSVItem:
    File_Path : str
    Label : str
    Frame_Count : int
    Width : int
    Height : int
    Codec : str
    File_Size_MB : float
     
@dataclass
class FrameCSVItem:
    Frame_Path: str
    Video_Path: str
    Label: str
    Width: int
    Height: int
    Frame_Number: int
    Timestamp: float
    Dataset: str
      
@dataclass
class FaceCSVItem:
    Face_Path: str
    Video_Path: str
    Label: str
    Frame_Number: int
    Dataset: str
    Confidence: float
    BBox_X: int
    BBox_Y: int
    BBox_Width: int
    BBox_Height: int
    Face_Width: int
    Face_Height: int
    
    
@dataclass
class PreprocessCSVItem:
    Face_Path: str
    Label: str
    Frame_Number: int
    Dataset: str
    Confidence: float
    BBox_X: int
    BBox_Y: int
    BBox_Width: int
    BBox_Height: int
    Face_Width: int
    Face_Height: int


@dataclass
class DatasetSplit:
    """Represents a train/test/eval split of the dataset."""
    train: List[FaceCSVItem]
    test: List[FaceCSVItem]
    eval: List[FaceCSVItem]

    @property
    def train_labels(self) -> List[str]:
        """Get labels for training set."""
        return [item.Label for item in self.train]

    @property
    def test_labels(self) -> List[str]:
        """Get labels for test set."""
        return [item.Label for item in self.test]

    @property
    def eval_labels(self) -> List[str]:
        """Get labels for eval set."""
        return [item.Label for item in self.eval]

    def get_statistics(self) -> dict:
        """Get statistics about the split."""
        def count_labels(items):
            fake = sum(1 for item in items if item.Label == "FAKE")
            real = sum(1 for item in items if item.Label == "REAL")
            return {"total": len(items), "fake": fake, "real": real}

        return {
            "train": count_labels(self.train),
            "test": count_labels(self.test),
            "eval": count_labels(self.eval)
        }

