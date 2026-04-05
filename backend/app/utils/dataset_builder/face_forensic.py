"""
FaceForensic++ Dataset loader.
"""

from typing import Dict, List
from utils.csv_services import CSVService
from core.schemas import CSVItem


class FaceForensicDataset:
    """
    Dataset loader for FaceForensics++ videos.

    Loads video metadata from CSV files in the specified folder.
    """

    def __init__(self, csv_folder: str):
        """
        Initialize FaceForensicDataset.

        Args:
            csv_folder: Path to folder containing CSV files with video metadata
        """
        self.csv_folder = csv_folder
        self.items: Dict[str, List[CSVItem]] = {}

    def load_all_csvs(self) -> Dict[str, List[CSVItem]]:
        """
        Load all CSV files from the folder.

        Returns:
            Dictionary mapping dataset name to list of CSVItem objects
        """
        self.items = CSVService.load_all_csvs(self.csv_folder, CSVItem)
        return self.items