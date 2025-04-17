"""
Metadata handling utilities for vascular simulation data.

This module provides functions for querying, filtering, and managing metadata
associated with vascular simulation datasets.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def query_metadata(
    metadata_files: Union[List[str], List[Path]], query: Dict[str, Any]
) -> List[Dict]:
    """
    Query metadata files based on a set of criteria.

    Args:
        metadata_files: List of paths to metadata files.
        query: Dictionary of key-value pairs to match.

    Returns:
        List of metadata dictionaries that match the query.
    """
    results = []

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Check if all query criteria match
            matches = all(
                key in metadata and metadata[key] == value for key, value in query.items()
            )

            if matches:
                # Add file path to metadata for reference
                metadata["_file_path"] = str(metadata_file)
                results.append(metadata)

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading metadata file {metadata_file}: {e}")

    return results


def filter_by_attribute(
    metadata_list: List[Dict], attribute: str, filter_fn: Callable[[Any], bool]
) -> List[Dict]:
    """
    Filter a list of metadata dictionaries based on an attribute.

    Args:
        metadata_list: List of metadata dictionaries.
        attribute: Attribute name to filter on.
        filter_fn: Function that takes the attribute value and returns a boolean.

    Returns:
        Filtered list of metadata dictionaries.
    """
    return [
        metadata
        for metadata in metadata_list
        if attribute in metadata and filter_fn(metadata[attribute])
    ]


def get_unique_values(metadata_list: List[Dict], attribute: str) -> List[Any]:
    """
    Get unique values for a specific attribute across metadata.

    Args:
        metadata_list: List of metadata dictionaries.
        attribute: Attribute name to extract values from.

    Returns:
        List of unique values for the attribute.
    """
    values = set()

    for metadata in metadata_list:
        if attribute in metadata:
            # Handle list values by adding each item
            if isinstance(metadata[attribute], list):
                values.update(metadata[attribute])
            else:
                values.add(metadata[attribute])

    # Convert to list for easier handling
    return list(values)


def merge_metadata(metadata_list: List[Dict], allow_conflicts: bool = False) -> Dict:
    """
    Merge multiple metadata dictionaries into one.

    Args:
        metadata_list: List of metadata dictionaries to merge.
        allow_conflicts: Whether to allow conflicting values (later ones override earlier).

    Returns:
        Merged metadata dictionary.

    Raises:
        ValueError: If there are conflicting values and allow_conflicts is False.
    """
    if not metadata_list:
        return {}

    result = {}

    for metadata in metadata_list:
        for key, value in metadata.items():
            if key in result and result[key] != value and not allow_conflicts:
                raise ValueError(f"Conflicting values for key '{key}': {result[key]} vs {value}")
            result[key] = value

    return result


def find_metadata_files(
    directory: Union[str, Path], recursive: bool = True, pattern: str = "*.json"
) -> List[Path]:
    """
    Find all metadata files in a directory.

    Args:
        directory: Directory to search.
        recursive: Whether to search recursively.
        pattern: File pattern to match.

    Returns:
        List of paths to metadata files.
    """
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        logger.warning(f"Directory not found: {directory}")
        return []

    metadata_files = []

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    metadata_files.append(Path(root) / file)
    else:
        metadata_files = list(directory.glob(pattern))

    return metadata_files


def index_metadata(
    directories: Union[List[str], List[Path]],
    output_file: Optional[Union[str, Path]] = None,
    recursive: bool = True,
) -> Dict[str, Dict]:
    """
    Create an index of all metadata files in the specified directories.

    Args:
        directories: List of directories to search.
        output_file: Optional path to save the index.
        recursive: Whether to search recursively.

    Returns:
        Dictionary mapping file paths to metadata.
    """
    index = {}

    for directory in directories:
        metadata_files = find_metadata_files(directory, recursive)

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                # Use relative path as key
                rel_path = str(metadata_file.relative_to(Path(directory)))
                index[rel_path] = metadata

            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}")

    # Save index if output file is specified
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, "w") as f:
                json.dump(index, f, indent=4)
        except IOError as e:
            logger.error(f"Error writing index to {output_file}: {e}")

    return index


def group_by_attribute(metadata_list: List[Dict], attribute: str) -> Dict[Any, List[Dict]]:
    """
    Group metadata dictionaries by a specific attribute.

    Args:
        metadata_list: List of metadata dictionaries.
        attribute: Attribute name to group by.

    Returns:
        Dictionary mapping attribute values to lists of metadata dictionaries.
    """
    groups = {}

    for metadata in metadata_list:
        if attribute in metadata:
            value = metadata[attribute]

            # Convert value to hashable type if needed
            if isinstance(value, list):
                value = tuple(value)
            elif isinstance(value, dict):
                value = frozenset(value.items())

            if value not in groups:
                groups[value] = []

            groups[value].append(metadata)

    return groups


def summarize_metadata(
    metadata_list: List[Dict], attributes: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Summarize metadata across a list of dictionaries.

    Args:
        metadata_list: List of metadata dictionaries.
        attributes: Optional list of attributes to include in the summary.
                   If None, all attributes are included.

    Returns:
        Dictionary with summary statistics for each attribute.
    """
    if not metadata_list:
        return {}

    # If attributes not specified, use all keys from the first metadata
    if attributes is None:
        attributes = list(metadata_list[0].keys())

    # Count frequency of each value for each attribute
    summary = {}

    for attribute in attributes:
        # Get all values for this attribute
        values = [metadata.get(attribute) for metadata in metadata_list if attribute in metadata]

        if not values:
            continue

        # Count frequencies
        freq = {}
        for value in values:
            # Convert non-hashable types
            if isinstance(value, list):
                value = tuple(value)
            elif isinstance(value, dict):
                value = frozenset(value.items())

            freq[value] = freq.get(value, 0) + 1

        # Compute statistics
        summary[attribute] = {
            "count": len(values),
            "unique_values": len(freq),
            "frequencies": freq,
        }

        # Add type-specific statistics
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            non_none_values = [v for v in values if v is not None]
            if non_none_values:
                summary[attribute].update(
                    {
                        "min": min(non_none_values),
                        "max": max(non_none_values),
                        "mean": sum(non_none_values) / len(non_none_values),
                    }
                )

    return summary


def update_metadata(
    metadata_file: Union[str, Path], updates: Dict[str, Any], create_if_missing: bool = False
) -> bool:
    """
    Update a metadata file with new values.

    Args:
        metadata_file: Path to the metadata file.
        updates: Dictionary of key-value pairs to update.
        create_if_missing: Whether to create the file if it doesn't exist.

    Returns:
        Whether the update was successful.

    Raises:
        FileNotFoundError: If the file doesn't exist and create_if_missing is False.
        IOError: If the file cannot be read or written.
    """
    metadata_file = Path(metadata_file)

    # Check if file exists
    if not metadata_file.exists():
        if create_if_missing:
            metadata = {}
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    else:
        # Read existing metadata
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading metadata file {metadata_file}: {e}")
            raise

    # Update metadata
    metadata.update(updates)

    # Write back to file
    try:
        # Create parent directory if it doesn't exist
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        return True
    except IOError as e:
        logger.error(f"Error writing metadata file {metadata_file}: {e}")
        return False
