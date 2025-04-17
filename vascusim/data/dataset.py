"""
Dataset implementations for vascular simulation data.

This module provides PyTorch Dataset implementations for loading and processing
vascular simulation data, with support for streaming from remote sources.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Tuple, Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..io.streaming import DataStreamer, HuggingFaceStreamer, NASStreamer
from .conversion import vtu_to_pyg, vtp_to_pyg


logger = logging.getLogger(__name__)


class VascuDataset(Dataset):
    """
    Dataset for vascular simulation data.
    
    This class provides a PyTorch Dataset implementation for loading and
    processing vascular simulation data from VTU/VTP files.
    
    Attributes:
        source_url (str): URL or path to the data source.
        cache_dir (Path): Directory to store cached files.
        file_list (List[str]): List of files to load.
        transform (Optional[Callable]): Transform to apply to the data.
        pre_transform (Optional[Callable]): Transform to apply during loading.
        filter_fn (Optional[Callable]): Function to filter files by metadata.
        load_vtu (bool): Whether to load VTU files.
        load_vtp (bool): Whether to load VTP files.
        streamer (DataStreamer): Data streamer for retrieving files.
    """
    
    def __init__(
        self,
        source_url: str,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        load_vtu: bool = True,
        load_vtp: bool = True,
        max_cache_size: Optional[int] = None,
        include_attributes: Optional[List[str]] = None,
        normalize: bool = False,
        streaming_type: str = "auto",
    ):
        """
        Initialize the dataset.
        
        Args:
            source_url: URL or path to the data source.
            cache_dir: Directory to store cached files.
            transform: Transform to apply to the data.
            pre_transform: Transform to apply during loading.
            filter_fn: Function to filter files by metadata.
            load_vtu: Whether to load VTU files.
            load_vtp: Whether to load VTP files.
            max_cache_size: Maximum cache size in bytes.
            include_attributes: List of specific attributes to include.
            normalize: Whether to normalize node positions.
            streaming_type: Type of streaming to use ("auto", "hf", "nas").
        """
        self.source_url = source_url
        
        # Set up cache directory
        if cache_dir is None:
            home_dir = os.path.expanduser('~')
            cache_dir = os.path.join(home_dir, '.vascusim', 'cache')
        
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.pre_transform = pre_transform
        self.filter_fn = filter_fn
        self.load_vtu = load_vtu
        self.load_vtp = load_vtp
        self.include_attributes = include_attributes
        self.normalize = normalize
        
        # Create streamer based on source URL
        if streaming_type == "auto":
            if "huggingface.co" in source_url or "hf.co" in source_url:
                streaming_type = "hf"
            else:
                streaming_type = "nas"
        
        if streaming_type == "hf":
            self.streamer = HuggingFaceStreamer(
                repo_id=source_url,
                cache_dir=str(self.cache_dir),
                max_cache_size=max_cache_size
            )
        else:  # Default to NAS
            self.streamer = NASStreamer(
                source_url=source_url,
                cache_dir=str(self.cache_dir),
                max_cache_size=max_cache_size
            )
        
        # Initialize file list
        self.file_list = self._initialize_file_list()
        
    def _initialize_file_list(self) -> List[str]:
        """
        Initialize the list of files to load.
        
        Returns:
            List of file paths.
        """
        try:
            # If available, use the streamer's list_files method
            if hasattr(self.streamer, 'list_files'):
                all_files = self.streamer.list_files()
            else:
                # Otherwise, try to find an index file
                index_path = self.streamer.get_file('index.json', download_if_missing=True)
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                all_files = index_data.get('files', [])
            
            # Filter files by type
            vtu_files = [f for f in all_files if f.endswith('.vtu')] if self.load_vtu else []
            vtp_files = [f for f in all_files if f.endswith('.vtp')] if self.load_vtp else []
            
            # Combine and sort
            file_list = sorted(vtu_files + vtp_files)
            
            # Apply metadata filter if provided
            if self.filter_fn is not None:
                filtered_files = []
                for file_path in file_list:
                    try:
                        # Get metadata for the file
                        metadata_id = file_path.rsplit('.', 1)[0] + '.json'
                        metadata = self.streamer.get_metadata(metadata_id)
                        
                        # Apply filter
                        if self.filter_fn(metadata):
                            filtered_files.append(file_path)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logger.warning(f"Error loading metadata for {file_path}: {e}")
                
                file_list = filtered_files
            
            return file_list
            
        except Exception as e:
            logger.error(f"Failed to initialize file list: {e}")
            return []
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get.
            
        Returns:
            PyTorch Geometric Data object.
            
        Raises:
            IndexError: If the index is out of range.
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is invalid.
        """
        if idx >= len(self.file_list) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.file_list)} samples")
        
        file_path = self.file_list[idx]
        
        # Download file if necessary
        local_path = self.streamer.get_file(file_path, download_if_missing=True)
        
        # Load data based on file type
        if file_path.endswith('.vtu'):
            data = vtu_to_pyg(
                str(local_path),
                attributes=self.include_attributes,
                normalize=self.normalize
            )
        elif file_path.endswith('.vtp'):
            data = vtp_to_pyg(
                str(local_path),
                attributes=self.include_attributes,
                normalize=self.normalize
            )
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Load metadata if available
        try:
            metadata_id = file_path.rsplit('.', 1)[0] + '.json'
            metadata_path = self.streamer.get_file(metadata_id, download_if_missing=True)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Add metadata as global attributes
            for key, value in metadata.items():
                if isinstance(value, (int, float, bool, str)):
                    data[key] = value
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    data[key] = torch.tensor(value)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading metadata for {file_path}: {e}")
        
        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)
        
        return data


class StreamingVascuDataset(VascuDataset):
    """
    Streaming dataset for vascular simulation data.
    
    This class extends VascuDataset with additional streaming functionality,
    including background downloading and dynamic cache management.
    
    Attributes:
        prefetch (bool): Whether to prefetch data in the background.
        prefetch_size (int): Number of samples to prefetch.
        delete_after_use (bool): Whether to delete files after use.
    """
    
    def __init__(
        self,
        source_url: str,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        load_vtu: bool = True,
        load_vtp: bool = True,
        max_cache_size: Optional[int] = None,
        include_attributes: Optional[List[str]] = None,
        normalize: bool = False,
        streaming_type: str = "auto",
        prefetch: bool = True,
        prefetch_size: int = 5,
        delete_after_use: bool = False,
    ):
        """
        Initialize the streaming dataset.
        
        Args:
            source_url: URL or path to the data source.
            cache_dir: Directory to store cached files.
            transform: Transform to apply to the data.
            pre_transform: Transform to apply during loading.
            filter_fn: Function to filter files by metadata.
            load_vtu: Whether to load VTU files.
            load_vtp: Whether to load VTP files.
            max_cache_size: Maximum cache size in bytes.
            include_attributes: List of specific attributes to include.
            normalize: Whether to normalize node positions.
            streaming_type: Type of streaming to use ("auto", "hf", "nas").
            prefetch: Whether to prefetch data in the background.
            prefetch_size: Number of samples to prefetch.
            delete_after_use: Whether to delete files after use.
        """
        super().__init__(
            source_url=source_url,
            cache_dir=cache_dir,
            transform=transform,
            pre_transform=pre_transform,
            filter_fn=filter_fn,
            load_vtu=load_vtu,
            load_vtp=load_vtp,
            max_cache_size=max_cache_size,
            include_attributes=include_attributes,
            normalize=normalize,
            streaming_type=streaming_type,
        )
        
        self.prefetch = prefetch
        self.prefetch_size = prefetch_size
        self.delete_after_use = delete_after_use
        self._last_index = None
        
        # Start prefetching if enabled
        if self.prefetch:
            self._prefetch(0)
    
    def _prefetch(self, start_idx: int) -> None:
        """
        Prefetch data starting from the given index.
        
        Args:
            start_idx: Index to start prefetching from.
        """
        if not self.prefetch or start_idx >= len(self.file_list):
            return
        
        import threading
        
        def prefetch_worker(idx_list):
            for idx in idx_list:
                if idx >= len(self.file_list):
                    break
                
                try:
                    file_path = self.file_list[idx]
                    # Just download the file, don't process it
                    self.streamer.get_file(file_path, download_if_missing=True)
                    
                    # Also prefetch metadata
                    metadata_id = file_path.rsplit('.', 1)[0] + '.json'
                    self.streamer.get_file(metadata_id, download_if_missing=True)
                    
                except Exception as e:
                    logger.warning(f"Error prefetching file at index {idx}: {e}")
        
        # Create prefetch indices
        prefetch_indices = list(range(start_idx, start_idx + self.prefetch_size))
        
        # Start prefetch thread
        thread = threading.Thread(
            target=prefetch_worker,
            args=(prefetch_indices,),
            daemon=True
        )
        thread.start()
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a sample from the dataset with streaming functionality.
        
        Args:
            idx: Index of the sample to get.
            
        Returns:
            PyTorch Geometric Data object.
        """
        # Get data from parent class
        data = super().__getitem__(idx)
        
        # Start prefetching next batch if needed
        if self.prefetch and (self._last_index is None or idx >= self._last_index):
            self._prefetch(idx + 1)
            self._last_index = idx + self.prefetch_size
        
        # Delete file after use if requested
        if self.delete_after_use:
            try:
                file_path = self.file_list[idx]
                local_path = self.streamer.get_file(file_path, download_if_missing=False)
                
                if local_path.exists():
                    os.remove(local_path)
                    
                # Also remove metadata
                metadata_id = file_path.rsplit('.', 1)[0] + '.json'
                metadata_path = self.streamer.get_file(metadata_id, download_if_missing=False)
                
                if metadata_path.exists():
                    os.remove(metadata_path)
                    
            except Exception as e:
                logger.warning(f"Error deleting file at index {idx}: {e}")
        
        return data