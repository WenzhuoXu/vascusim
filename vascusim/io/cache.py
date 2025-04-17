"""
Cache management for vascular simulation data.

This module provides a cache management system for efficient handling of
downloaded data files, with support for size limits and eviction policies.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages cache storage for downloaded files.

    This class handles tracking of cache usage, access patterns, and implements
    policies for cache eviction when space limits are reached.

    Attributes:
        cache_dir (Path): Directory where cached files are stored.
        max_size (Optional[int]): Maximum cache size in bytes.
        index_file (Path): Path to the cache index file.
        files (Dict): Dictionary of cached files with access information.
    """

    def __init__(self, cache_dir: Path, max_size: Optional[int] = None):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cached files.
            max_size: Maximum cache size in bytes. If None, no limit is applied.
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index file to store cache metadata
        self.index_file = self.cache_dir / ".cache_index.json"

        # Dictionary of cached files with access information
        # {file_path: {'size': size_in_bytes, 'last_access': timestamp}}
        self.files: Dict[str, Dict] = {}

        # Load existing cache index if available
        self._load_index()

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    self.files = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load cache index: {e}")
                self.files = {}
                # Create a new index by scanning the directory
                self._rebuild_index()
        else:
            # Create a new index by scanning the directory
            self._rebuild_index()

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.files, f)
        except IOError as e:
            logger.error(f"Failed to save cache index: {e}")

    def _rebuild_index(self) -> None:
        """Rebuild the cache index by scanning the cache directory."""
        self.files = {}

        for root, _, files in os.walk(self.cache_dir):
            for file in files:
                # Skip the index file itself
                if file == self.index_file.name:
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.cache_dir)

                try:
                    file_stat = os.stat(file_path)
                    self.files[rel_path] = {
                        "size": file_stat.st_size,
                        "last_access": file_stat.st_atime,
                    }
                except OSError as e:
                    logger.error(f"Failed to stat file {file_path}: {e}")

        self._save_index()

    def add_file(self, file_path: Path) -> None:
        """
        Add a file to the cache index.

        Args:
            file_path: Path to the file to add.
        """
        rel_path = os.path.relpath(file_path, self.cache_dir)

        try:
            file_stat = os.stat(file_path)
            self.files[rel_path] = {"size": file_stat.st_size, "last_access": time.time()}

            # Check cache size and cleanup if necessary
            if self.max_size is not None:
                self._check_size()

            self._save_index()
        except OSError as e:
            logger.error(f"Failed to add file {file_path} to cache: {e}")

    def mark_accessed(self, file_path: Path) -> None:
        """
        Update the last access time for a cached file.

        Args:
            file_path: Path to the accessed file.
        """
        rel_path = os.path.relpath(file_path, self.cache_dir)

        if rel_path in self.files:
            self.files[rel_path]["last_access"] = time.time()
            self._save_index()
        else:
            # File exists but not in our index, add it
            self.add_file(file_path)

    def _check_size(self) -> None:
        """
        Check if the cache size exceeds the limit and cleanup if necessary.
        """
        if self.max_size is None:
            return

        # Calculate current cache size
        current_size = sum(file_info["size"] for file_info in self.files.values())

        # If we're over the limit, remove least recently used files
        if current_size > self.max_size:
            logger.info(
                f"Cache size ({current_size} bytes) exceeds limit ({self.max_size} bytes). Cleaning up..."
            )
            self._cleanup_lru(current_size - self.max_size)

    def _cleanup_lru(self, bytes_to_free: int) -> None:
        """
        Remove least recently used files to free up space.

        Args:
            bytes_to_free: Amount of space to free in bytes.
        """
        # Sort files by last access time (oldest first)
        sorted_files = sorted(self.files.items(), key=lambda x: x[1]["last_access"])

        freed = 0
        for rel_path, file_info in sorted_files:
            if freed >= bytes_to_free:
                break

            file_path = self.cache_dir / rel_path
            file_size = file_info["size"]

            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed cached file: {rel_path}")

                del self.files[rel_path]
                freed += file_size
            except OSError as e:
                logger.error(f"Failed to remove cached file {file_path}: {e}")

        self._save_index()
        logger.info(f"Freed {freed} bytes from cache")

    def cleanup(self) -> None:
        """
        Perform cache cleanup based on size limits.
        """
        if self.max_size is not None:
            self._check_size()

    def clear_all(self) -> None:
        """
        Remove all files from the cache.
        """
        # Get list of all files to remove
        files_to_remove = list(self.files.keys())

        for rel_path in files_to_remove:
            file_path = self.cache_dir / rel_path

            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed cached file: {rel_path}")

                del self.files[rel_path]
            except OSError as e:
                logger.error(f"Failed to remove cached file {file_path}: {e}")

        self._save_index()
        logger.info("Cleared all files from cache")

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the current cache usage.

        Returns:
            Dictionary with cache statistics.
        """
        total_size = sum(file_info["size"] for file_info in self.files.values())

        return {
            "file_count": len(self.files),
            "total_size": total_size,
            "max_size": self.max_size,
            "usage_percent": (total_size / self.max_size * 100) if self.max_size else None,
        }
