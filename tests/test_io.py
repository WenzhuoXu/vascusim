"""
Tests for the I/O module of vascusim.

This module tests the functionality for reading, streaming, and caching
VTU/VTP files and associated metadata.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pytest

from vascusim.io import formats, streaming, cache, vtk_utils


class TestFileReading:
    """Tests for reading VTU/VTP files and metadata."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Skip tests if VTK is not available
        if not vtk_utils.check_vtk_availability():
            pytest.skip("VTK is not available, skipping VTK tests.")
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_metadata_reading(self):
        """Test reading metadata JSON files."""
        # Create a test metadata file
        metadata_file = Path(self.temp_dir) / "test_metadata.json"
        test_metadata = {
            "case_id": "test_case",
            "timestep": 0,
            "resolution": 1.0,
            "is_healthy": True,
            "parameters": {
                "viscosity": 0.0035,
                "density": 1060.0
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(test_metadata, f)
        
        # Test reading the metadata
        read_metadata = formats.read_metadata(metadata_file)
        
        # Check that metadata was read correctly
        assert read_metadata == test_metadata
        assert read_metadata["case_id"] == "test_case"
        assert read_metadata["is_healthy"] is True
        assert read_metadata["parameters"]["viscosity"] == 0.0035
    
    def test_metadata_reading_nonexistent_file(self):
        """Test reading a nonexistent metadata file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.json"
        
        # Reading a nonexistent file should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            formats.read_metadata(nonexistent_file)
    
    @pytest.mark.skipif(not vtk_utils.check_vtk_availability(), 
                     reason="VTK is not available")
    def test_vtu_reading_mock(self):
        """Test reading VTU files with mock VTK objects."""
        # This is a mock test since actual VTU file creation is complex
        # In a real test, you would create a small VTU file programmatically
        
        with mock.patch('vascusim.io.vtk_utils.extract_mesh_from_vtu') as mock_extract:
            # Setup mock return values
            import vtk
            mock_mesh = mock.MagicMock(spec=vtk.vtkUnstructuredGrid)
            mock_cell_data = {"pressure": np.array([1.0, 2.0, 3.0])}
            mock_point_data = {"velocity": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}
            
            mock_extract.return_value = (mock_mesh, mock_cell_data, mock_point_data)
            
            # Test reading a VTU file (which doesn't actually exist)
            # The mock will intercept the call and return our mock data
            test_file = Path(self.temp_dir) / "test.vtu"
            
            # Create an empty file
            with open(test_file, 'w') as f:
                f.write("dummy vtu content")
            
            # Read the mock VTU file
            mesh, cell_data, point_data = formats.read_vtu(test_file)
            
            # Check that the function returned our mock objects
            assert mesh == mock_mesh
            assert cell_data == mock_cell_data
            assert point_data == mock_point_data


class TestCacheManager:
    """Tests for the cache management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for cache
        self.cache_dir = Path(tempfile.mkdtemp())
        
        # Create cache manager with a small maximum size
        self.cache_manager = cache.CacheManager(self.cache_dir, max_size=1024*10)  # 10 KB
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.cache_dir)
    
    def test_add_file(self):
        """Test adding a file to the cache."""
        # Create a test file
        test_file = self.cache_dir / "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("x" * 100)  # 100 bytes file
        
        # Add file to cache
        self.cache_manager.add_file(test_file)
        
        # Check that the file is in the index
        rel_path = test_file.relative_to(self.cache_dir)
        assert str(rel_path) in self.cache_manager.files
        
        # Check that the file size was recorded
        assert self.cache_manager.files[str(rel_path)]['size'] == 100
    
    def test_mark_accessed(self):
        """Test marking a file as accessed."""
        # Create a test file
        test_file = self.cache_dir / "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("x" * 100)  # 100 bytes file
        
        # Add file to cache
        self.cache_manager.add_file(test_file)
        
        # Get original access time
        rel_path = test_file.relative_to(self.cache_dir)
        original_time = self.cache_manager.files[str(rel_path)]['last_access']
        
        # Wait a bit to ensure the timestamp changes
        import time
        time.sleep(0.1)
        
        # Mark file as accessed
        self.cache_manager.mark_accessed(test_file)
        
        # Check that the access time was updated
        new_time = self.cache_manager.files[str(rel_path)]['last_access']
        assert new_time > original_time
    
    def test_cache_eviction(self):
        """Test that files are evicted when cache size is exceeded."""
        # Create multiple test files to exceed cache size
        files = []
        for i in range(5):
            test_file = self.cache_dir / f"test_file_{i}.txt"
            with open(test_file, 'w') as f:
                f.write("x" * 4000)  # Each file is 4 KB
            files.append(test_file)
            self.cache_manager.add_file(test_file)
            
            # Ensure different access times
            import time
            time.sleep(0.1)
        
        # Access files in reverse order to change their access time
        for file in reversed(files[1:]):
            self.cache_manager.mark_accessed(file)
        
        # Now the oldest file should be the first one
        # Check that it was removed (it's the least recently used)
        rel_path = files[0].relative_to(self.cache_dir)
        assert str(rel_path) not in self.cache_manager.files


class TestDataStreamer:
    """Tests for the data streaming functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for source and cache
        self.source_dir = Path(tempfile.mkdtemp())
        self.cache_dir = Path(tempfile.mkdtemp())
        
        # Create a custom streamer for testing
        self.streamer = streaming.DataStreamer(
            source_url=str(self.source_dir),
            cache_dir=str(self.cache_dir)
        )
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove temporary directories
        shutil.rmtree(self.source_dir)
        shutil.rmtree(self.cache_dir)
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.streamer.get_file("dummy.txt")
        
        with pytest.raises(NotImplementedError):
            self.streamer.get_metadata("dummy.json")


class TestNASStreamer:
    """Tests for the NAS streamer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for source and cache
        self.source_dir = Path(tempfile.mkdtemp())
        self.cache_dir = Path(tempfile.mkdtemp())
        
        # Create test files in the source directory
        self.test_file = self.source_dir / "test_file.vtu"
        with open(self.test_file, 'w') as f:
            f.write("dummy vtu content")
        
        self.test_metadata = self.source_dir / "test_file.json"
        with open(self.test_metadata, 'w') as f:
            json.dump({"case_id": "test_case"}, f)
        
        # Create NAS streamer
        self.streamer = streaming.NASStreamer(
            source_url=str(self.source_dir),
            cache_dir=str(self.cache_dir)
        )
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove temporary directories
        shutil.rmtree(self.source_dir)
        shutil.rmtree(self.cache_dir)
    
    def test_get_file(self):
        """Test getting a file from the NAS streamer."""
        # Get the test file
        cache_path = self.streamer.get_file("test_file.vtu")
        
        # Check that the file was cached
        assert cache_path.exists()
        
        # Check file content
        with open(cache_path, 'r') as f:
            content = f.read()
        assert content == "dummy vtu content"
    
    def test_get_nonexistent_file(self):
        """Test getting a nonexistent file."""
        # Getting a nonexistent file should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            self.streamer.get_file("nonexistent.vtu", download_if_missing=False)
    
    def test_get_metadata(self):
        """Test getting metadata from the NAS streamer."""
        # Get the test metadata
        metadata = self.streamer.get_metadata("test_file.vtu")
        
        # Check that metadata was read correctly
        assert metadata["case_id"] == "test_case"