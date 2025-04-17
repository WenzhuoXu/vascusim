"""
Streaming module for efficiently accessing vascular simulation data.

This module provides functionality for streaming data from various sources,
including local files, network attached storage, and Hugging Face datasets.
"""

import os
import json
import shutil
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download

from .cache import CacheManager


logger = logging.getLogger(__name__)


class DataStreamer(ABC):
    """
    Abstract base class for data streaming functionality.
    
    This class provides the interface for all data streaming implementations,
    including methods for retrieving files, managing cache, and cleanup.
    
    Attributes:
        source_url (str): The URL or path to the data source.
        cache_dir (Path): Directory to store cached files.
        cache_manager (CacheManager): Manager for handling cache operations.
        max_cache_size (Optional[int]): Maximum cache size in bytes.
    """
    
    def __init__(
        self, 
        source_url: str, 
        cache_dir: Optional[str] = None,
        max_cache_size: Optional[int] = None
    ):
        """
        Initialize the data streamer.
        
        Args:
            source_url: URL or path to the data source.
            cache_dir: Directory to store cached files. If None, a default is used.
            max_cache_size: Maximum cache size in bytes. If None, no limit is applied.
        """
        self.source_url = source_url
        
        if cache_dir is None:
            home_dir = os.path.expanduser('~')
            cache_dir = os.path.join(home_dir, '.vascusim', 'cache')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_manager = CacheManager(self.cache_dir, max_size=max_cache_size)
        self.max_cache_size = max_cache_size
    
    @abstractmethod
    def get_file(self, file_id: str, download_if_missing: bool = True) -> Path:
        """
        Get the path to a file, downloading it if necessary.
        
        Args:
            file_id: Identifier for the file to retrieve.
            download_if_missing: Whether to download the file if not in cache.
            
        Returns:
            Path to the requested file.
        
        Raises:
            FileNotFoundError: If the file is not in cache and download_if_missing is False.
            ConnectionError: If there's an issue downloading the file.
        """
        pass
    
    @abstractmethod
    def get_metadata(self, file_id: str) -> Dict:
        """
        Get metadata for a specific file.
        
        Args:
            file_id: Identifier for the file whose metadata to retrieve.
            
        Returns:
            Dictionary containing the metadata.
            
        Raises:
            FileNotFoundError: If the metadata file doesn't exist.
        """
        pass
    
    def cleanup(self, strategy: str = "lru") -> None:
        """
        Clean up cached files based on the specified strategy.
        
        Args:
            strategy: Strategy to use for cleanup.
                     "lru" - Least Recently Used
                     "all" - Remove all cached files
        """
        if strategy == "all":
            self.cache_manager.clear_all()
        else:  # Default to LRU
            self.cache_manager.cleanup()
    
    def __del__(self):
        """Clean up resources when the object is deleted."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class NASStreamer(DataStreamer):
    """
    Streamer for accessing data from a Network Attached Storage (NAS) drive.
    
    This class implements the DataStreamer interface for NAS drives, handling
    authentication and network file access.
    """
    
    def __init__(
        self, 
        source_url: str, 
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_cache_size: Optional[int] = None
    ):
        """
        Initialize the NAS streamer.
        
        Args:
            source_url: URL to the NAS share.
            username: Username for NAS authentication.
            password: Password for NAS authentication.
            cache_dir: Directory to store cached files.
            max_cache_size: Maximum cache size in bytes.
        """
        super().__init__(source_url, cache_dir, max_cache_size)
        self.username = username
        self.password = password
        # Additional setup for NAS connection would go here
    
    def get_file(self, file_id: str, download_if_missing: bool = True) -> Path:
        """
        Get the path to a file on the NAS, downloading it to cache if necessary.
        
        Args:
            file_id: Identifier (relative path) for the file to retrieve.
            download_if_missing: Whether to download the file if not in cache.
            
        Returns:
            Path to the requested file.
        """
        cache_path = self.cache_dir / file_id
        
        # Check if file is already in cache
        if cache_path.exists():
            # Update access time
            self.cache_manager.mark_accessed(cache_path)
            return cache_path
        
        if not download_if_missing:
            raise FileNotFoundError(f"File {file_id} not in cache")
        
        # Ensure directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Construct source path
        source_path = os.path.join(self.source_url, file_id)
        
        try:
            # Implementation would depend on the NAS protocol (SMB, NFS, etc.)
            # For simplicity, assuming a mounted drive or accessible network path
            shutil.copy2(source_path, cache_path)
            
            # Register file with cache manager
            self.cache_manager.add_file(cache_path)
            return cache_path
            
        except Exception as e:
            raise ConnectionError(f"Failed to download {file_id}: {e}")
    
    def get_metadata(self, file_id: str) -> Dict:
        """
        Get metadata for a specific file from the NAS.
        
        Args:
            file_id: Identifier for the file whose metadata to retrieve.
            
        Returns:
            Dictionary containing the metadata.
        """
        # Metadata files should have the same name as the data file but with .json extension
        metadata_id = file_id.rsplit('.', 1)[0] + '.json'
        metadata_path = self.get_file(metadata_id)
        
        with open(metadata_path, 'r') as f:
            return json.load(f)


class HuggingFaceStreamer(DataStreamer):
    """
    Streamer for accessing data from Hugging Face datasets.
    
    This class implements the DataStreamer interface for Hugging Face,
    handling authentication and file retrieval from Hugging Face repositories.
    """
    
    def __init__(
        self, 
        repo_id: str, 
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_cache_size: Optional[int] = None,
        revision: str = "main"
    ):
        """
        Initialize the Hugging Face streamer.
        
        Args:
            repo_id: Hugging Face repository ID.
            token: Hugging Face API token for private repositories.
            cache_dir: Directory to store cached files.
            max_cache_size: Maximum cache size in bytes.
            revision: Repository revision to use.
        """
        super().__init__(repo_id, cache_dir, max_cache_size)
        self.repo_id = repo_id
        self.token = token
        self.revision = revision
        self._file_listing = None
    
    def _ensure_file_listing(self) -> List[str]:
        """
        Ensure we have a listing of available files in the repository.
        
        Returns:
            List of available file IDs.
        """
        if self._file_listing is None:
            try:
                # Get repository info to build file listing
                # This approach avoids downloading the entire repository
                snapshot_info = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                    revision=self.revision,
                    local_dir=None,  # Don't actually download, just get info
                    local_dir_use_symlinks=False,
                    max_workers=1,
                    tqdm_class=None
                )
                self._file_listing = list(snapshot_info.keys())
            except Exception as e:
                logger.error(f"Failed to get file listing from Hugging Face: {e}")
                self._file_listing = []
                
        return self._file_listing
    
    def get_file(self, file_id: str, download_if_missing: bool = True) -> Path:
        """
        Get the path to a file from Hugging Face, downloading it to cache if necessary.
        
        Args:
            file_id: Identifier (relative path) for the file to retrieve.
            download_if_missing: Whether to download the file if not in cache.
            
        Returns:
            Path to the requested file.
        """
        cache_path = self.cache_dir / file_id
        
        # Check if file is already in cache
        if cache_path.exists():
            # Update access time
            self.cache_manager.mark_accessed(cache_path)
            return cache_path
        
        if not download_if_missing:
            raise FileNotFoundError(f"File {file_id} not in cache")
        
        # Ensure directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download file from Hugging Face
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=file_id,
                repo_type="dataset",
                token=self.token,
                revision=self.revision,
                cache_dir=str(self.cache_dir),
                local_dir_use_symlinks=False
            )
            
            # If the downloaded path is different from our expected cache path
            # (which can happen with the HF cache structure), copy or link it
            if Path(downloaded_path) != cache_path:
                # Create a symlink or copy the file
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    
                # Try to create a symlink first
                try:
                    os.symlink(downloaded_path, cache_path)
                except (OSError, NotImplementedError):
                    # If symlink fails, copy the file
                    shutil.copy2(downloaded_path, cache_path)
            
            # Register file with cache manager
            self.cache_manager.add_file(cache_path)
            return cache_path
            
        except Exception as e:
            raise ConnectionError(f"Failed to download {file_id} from Hugging Face: {e}")
    
    def get_metadata(self, file_id: str) -> Dict:
        """
        Get metadata for a specific file from Hugging Face.
        
        Args:
            file_id: Identifier for the file whose metadata to retrieve.
            
        Returns:
            Dictionary containing the metadata.
        """
        # Metadata files should have the same name as the data file but with .json extension
        metadata_id = file_id.rsplit('.', 1)[0] + '.json'
        metadata_path = self.get_file(metadata_id)
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_files(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all available files in the repository.
        
        Args:
            pattern: Optional glob pattern to filter files.
            
        Returns:
            List of file IDs.
        """
        files = self._ensure_file_listing()
        
        if pattern:
            import fnmatch
            return [f for f in files if fnmatch.fnmatch(f, pattern)]
        return files