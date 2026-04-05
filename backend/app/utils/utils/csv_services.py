"""
CSV Service for loading and parsing CSV files into dataclass objects.
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Type, TypeVar
from dataclasses import fields

T = TypeVar('T')


class CSVService:
    """Service for loading CSV files and converting rows to dataclass objects."""

    @staticmethod
    def _normalize_field_name(name: str) -> str:
        """
        Normalize field name by replacing spaces with underscores.

        Args:
            name: Field name to normalize

        Returns:
            Normalized field name
        """
        # Replace spaces with underscores
        # Handle parentheses by replacing them with underscores too
        normalized = name.replace(' ', '_')
        normalized = normalized.replace('(', '_').replace(')', '')
        # Remove double underscores
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
        # Remove trailing underscores
        normalized = normalized.rstrip('_')
        return normalized

    @staticmethod
    def row_csv_into_items(row: dict, schema: Type[T]) -> T:
        """
        Convert a CSV row dictionary into a dataclass instance.

        Args:
            row: Dictionary representing a CSV row
            schema: Dataclass type to instantiate

        Returns:
            Instance of the schema dataclass populated with row data
        """
        # Normalize CSV column names (replace spaces with underscores)
        normalized_row = {
            CSVService._normalize_field_name(k): v
            for k, v in row.items()
        }

        # Get field names and types from dataclass
        field_map = {f.name: f.type for f in fields(schema)}

        # Convert row values to appropriate types
        kwargs = {}
        for field_name, field_type in field_map.items():
            # Get value from row, or use empty string if missing
            value = normalized_row.get(field_name, "")

            # Handle type conversion with defaults for missing/empty values
            if field_type == int:
                kwargs[field_name] = int(value) if value else 0
            elif field_type == float:
                kwargs[field_name] = float(value) if value else 0.0
            elif field_type == bool:
                kwargs[field_name] = value.lower() in ('true', '1', 'yes') if value else False
            else:  # str or other
                kwargs[field_name] = value if value else ""

        return schema(**kwargs)

    @staticmethod
    def load_csv(csv_path: str, schema: Type[T]) -> List[T]:
        """
        Load a single CSV file and convert rows to dataclass instances.

        Args:
            csv_path: Path to the CSV file
            schema: Dataclass type to instantiate for each row

        Returns:
            List of dataclass instances

        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        items = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = CSVService.row_csv_into_items(row, schema)
                items.append(item)

        return items

    @staticmethod
    def load_all_csvs(csv_folder: str, schema: Type[T]) -> Dict[str, List[T]]:
        """
        Load all CSV files from a folder and convert to dataclass instances.

        Args:
            csv_folder: Path to folder containing CSV files
            schema: Dataclass type to instantiate for each row

        Returns:
            Dictionary mapping CSV filename (without extension) to list of items

        Raises:
            FileNotFoundError: If folder doesn't exist
        """
        if not os.path.exists(csv_folder):
            raise FileNotFoundError(f"CSV folder not found: {csv_folder}")

        all_csvs = CSVService.get_all_csvs(csv_folder)
        result = {}

        for csv_file in all_csvs:
            csv_name = Path(csv_file).stem  # Get filename without extension
            csv_path = os.path.join(csv_folder, csv_file)
            result[csv_name] = CSVService.load_csv(csv_path, schema)

        return result

    @staticmethod
    def get_all_csvs(csv_folder: str) -> List[str]:
        """
        Get list of all CSV files in a folder.

        Args:
            csv_folder: Path to folder to search

        Returns:
            List of CSV filenames (not full paths)
        """
        if not os.path.exists(csv_folder):
            return []

        csv_files = [
            f for f in os.listdir(csv_folder)
            if f.endswith('.csv') and os.path.isfile(os.path.join(csv_folder, f))
        ]

        return sorted(csv_files)

    @staticmethod
    def save_csv(csv_path: str, items: List[T], schema: Type[T]) -> None:
        """
        Save list of dataclass instances to a CSV file.

        Args:
            csv_path: Path where to save the CSV file
            items: List of dataclass instances to save
            schema: Dataclass type (used to get field names)

        Raises:
            ValueError: If items list is empty or items don't match schema
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if not items:
            # Create empty CSV with headers
            field_names = [f.name for f in fields(schema)]
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=field_names)
                writer.writeheader()
            return

        # Get field names from schema
        field_names = [f.name for f in fields(schema)]

        # Convert dataclass instances to dictionaries
        rows = []
        for item in items:
            if not isinstance(item, schema):
                raise ValueError(f"Item is not an instance of {schema.__name__}")
            row = {field: getattr(item, field) for field in field_names}
            rows.append(row)

        # Write to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def items_to_dicts(items: List[T]) -> List[Dict]:
        """
        Convert list of dataclass instances to list of dictionaries.

        Args:
            items: List of dataclass instances

        Returns:
            List of dictionaries
        """
        if not items:
            return []

        field_names = [f.name for f in fields(items[0])]
        return [
            {field: getattr(item, field) for field in field_names}
            for item in items
        ]
