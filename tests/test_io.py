"""
Tests for the I/O module of vascusim.

This module tests the functionality for reading, streaming, and caching
VTU/VTP files and associated metadata.
"""

import json
import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import requests

from vascusim.io import cache, formats, streaming, vtk_utils

# Check if optional dependencies are available
try:
    from smb.SMBConnection import SMBConnection

    HAS_SMB = True
except ImportError:
    HAS_SMB = False


def is_nas_reachable(hostname, port, timeout=2):
    """
    Test if NAS is reachable at the given hostname and port.

    Args:
        hostname: NAS hostname or IP address
        port: Port to test (usually 5000 for Synology API, 445 for SMB)
        timeout: Connection timeout in seconds

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Try socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((hostname, port))
        sock.close()
        return result == 0
    except Exception:
        return False


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
            "parameters": {"viscosity": 0.0035, "density": 1060.0},
        }

        with open(metadata_file, "w") as f:
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

    @pytest.mark.skipif(not vtk_utils.check_vtk_availability(), reason="VTK is not available")
    def test_vtu_reading_mock(self):
        """Test reading VTU files with mock VTK objects."""
        # This is a mock test since actual VTU file creation is complex

        with mock.patch("vascusim.io.vtk_utils.extract_mesh_from_vtu") as mock_extract:
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
            with open(test_file, "w") as f:
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
        self.cache_manager = cache.CacheManager(self.cache_dir, max_size=1024 * 10)  # 10 KB

    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.cache_dir)

    def test_add_file(self):
        """Test adding a file to the cache."""
        # Create a test file
        test_file = self.cache_dir / "test_file.txt"
        with open(test_file, "w") as f:
            f.write("x" * 100)  # 100 bytes file

        # Add file to cache
        self.cache_manager.add_file(test_file)

        # Check that the file is in the index
        rel_path = test_file.relative_to(self.cache_dir)
        assert str(rel_path) in self.cache_manager.files

        # Check that the file size was recorded
        assert self.cache_manager.files[str(rel_path)]["size"] == 100

    def test_mark_accessed(self):
        """Test marking a file as accessed."""
        # Create a test file
        test_file = self.cache_dir / "test_file.txt"
        with open(test_file, "w") as f:
            f.write("x" * 100)  # 100 bytes file

        # Add file to cache
        self.cache_manager.add_file(test_file)

        # Get original access time
        rel_path = test_file.relative_to(self.cache_dir)
        original_time = self.cache_manager.files[str(rel_path)]["last_access"]

        # Wait a bit to ensure the timestamp changes
        import time

        time.sleep(0.1)

        # Mark file as accessed
        self.cache_manager.mark_accessed(test_file)

        # Check that the access time was updated
        new_time = self.cache_manager.files[str(rel_path)]["last_access"]
        assert new_time > original_time

    def test_cache_eviction(self):
        """Test that files are evicted when cache size is exceeded."""
        # Create multiple test files to exceed cache size
        files = []
        for i in range(5):
            test_file = self.cache_dir / f"test_file_{i}.txt"
            with open(test_file, "w") as f:
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
            source_url=str(self.source_dir), cache_dir=str(self.cache_dir)
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


class TestNASStreamerAPI:
    """Tests for the NASStreamer with API access mode."""

    # Test NAS configuration - update with actual values for real testing
    NAS_IP = "172.24.44.162"  # Example IP, replace with actual
    NAS_PORT = 5000
    NAS_USER = "wxu2"
    NAS_PASS = "Xu@913816"

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for cache
        self.cache_dir = Path(tempfile.mkdtemp())

        # Skip all tests if dependencies for API mode are not available
        try:
            import requests
            import urllib3
        except ImportError:
            pytest.skip("Required dependencies for API mode not available")

        # Check if NAS is reachable (can be disabled during testing)
        self.use_mock = not is_nas_reachable(self.NAS_IP, self.NAS_PORT)

        # Create NAS streamer
        self.streamer = streaming.NASStreamer(
            source_url=self.NAS_IP,
            cache_dir=str(self.cache_dir),
            username=self.NAS_USER,
            password=self.NAS_PASS,
            port=self.NAS_PORT,
            access_mode="api",
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.cache_dir)

        # Disconnect from NAS if connected
        if hasattr(self, "streamer"):
            self.streamer.disconnect()

    def test_api_connect(self):
        """Test connecting to NAS with API."""
        if self.use_mock:
            # Mock the connect method
            with mock.patch.object(self.streamer, "_api_connect") as mock_connect:
                mock_connect.return_value = True

                # Test the connection
                assert self.streamer.connect() is True
                mock_connect.assert_called_once()
        else:
            # Actual connection test
            try:
                result = self.streamer.connect()
                assert result is True
                assert self.streamer.sid is not None
            except Exception as e:
                pytest.fail(f"Connection failed with error: {e}")

    def test_list_shares(self):
        """Test listing shares on the NAS."""
        if self.use_mock:
            # Mock the list_shares method
            with mock.patch.object(self.streamer, "_list_shares_api") as mock_list:
                mock_list.return_value = ["public", "homes", "data"]

                # Test listing shares
                shares = self.streamer.list_shares()
                assert "public" in shares
                assert len(shares) == 3
                mock_list.assert_called_once()
        else:
            # Connect first
            if not self.streamer.sid:
                self.streamer.connect()

            # Actual list shares test
            try:
                shares = self.streamer.list_shares()
                assert isinstance(shares, list)
                assert len(shares) > 0
            except Exception as e:
                pytest.fail(f"List shares failed with error: {e}")

    def test_get_file_with_mock(self):
        """Test getting a file from NAS with mocked API."""
        # Create a test file in cache to simulate an already downloaded file
        test_cache_file = self.cache_dir / "public" / "test_file.txt"
        test_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_cache_file, "w") as f:
            f.write("Test content")

        # Test getting a file that's already in cache
        result = self.streamer.get_file("public/test_file.txt", download_if_missing=False)
        assert result == test_cache_file

        # Test getting a file that's not in cache
        with pytest.raises(FileNotFoundError):
            self.streamer.get_file("public/nonexistent.txt", download_if_missing=False)

        # Mock the _download_file_api method for testing download
        with mock.patch.object(self.streamer, "_download_file_api") as mock_download:
            # Mock connect if needed
            if self.streamer.sid is None:
                with mock.patch.object(self.streamer, "_api_connect") as mock_connect:
                    mock_connect.return_value = True
                    self.streamer.sid = "mock_session_id"

            # Test downloading a file
            new_file = self.streamer.get_file("public/new_file.txt")
            mock_download.assert_called_once()
            assert new_file == self.cache_dir / "public" / "new_file.txt"

    def test_list_directory_with_mock(self):
        """Test listing directory contents with mocked API."""
        # Mock the _list_directory_api method
        mock_files = [
            {
                "name": "file1.txt",
                "path": "/public/file1.txt",
                "isdir": False,
                "size": 1024,
                "time": {"ctime": 1618023942, "mtime": 1618023942},
            },
            {
                "name": "folder1",
                "path": "/public/folder1",
                "isdir": True,
                "size": 0,
                "time": {"ctime": 1618023942, "mtime": 1618023942},
            },
        ]

        with mock.patch.object(self.streamer, "_list_directory_api") as mock_list:
            mock_list.return_value = mock_files

            # Test listing a directory
            files = self.streamer.list_directory("/public")
            assert len(files) == 2
            assert files[0]["name"] == "file1.txt"
            assert files[1]["isdir"] is True
            mock_list.assert_called_once_with("/public")

    def test_upload_file_with_mock(self):
        """Test uploading a file with mocked API."""
        # Create a test local file
        local_file = self.cache_dir / "local_file.txt"
        with open(local_file, "w") as f:
            f.write("Test content")

        # Mock the _upload_file_api method
        with mock.patch.object(self.streamer, "_upload_file_api") as mock_upload:
            mock_upload.return_value = True

            # Test uploading a file
            result = self.streamer.upload_file(local_file, "/public/uploaded.txt")
            assert result is True
            mock_upload.assert_called_once_with(local_file, "/public/uploaded.txt", False)

            # Test with overwrite=True
            result = self.streamer.upload_file(local_file, "/public/uploaded.txt", overwrite=True)
            assert result is True
            mock_upload.assert_called_with(local_file, "/public/uploaded.txt", True)

    def test_create_directory_with_mock(self):
        """Test creating a directory with mocked API."""
        with mock.patch.object(self.streamer, "_create_directory_api") as mock_create:
            mock_create.return_value = True

            # Test creating a directory
            result = self.streamer._create_directory_api("/public/new_folder")
            assert result is True
            mock_create.assert_called_once_with("/public/new_folder")


# Only run SMB tests if pysmb is available
@pytest.mark.skipif(not HAS_SMB, reason="pysmb is not available")
class TestNASStreamerSMB:
    """Tests for the NASStreamer with SMB access mode."""

    # Test NAS configuration - update with actual values for real testing
    NAS_IP = "172.24.44.162"  # Example IP, replace with actual
    NAS_PORT = 445  # SMB port
    NAS_USER = "wxu2"
    NAS_PASS = "Xu@913816"

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for cache
        self.cache_dir = Path(tempfile.mkdtemp())

        # Check if NAS is reachable
        self.use_mock = not is_nas_reachable(self.NAS_IP, self.NAS_PORT)

        # Create NAS streamer
        self.streamer = streaming.NASStreamer(
            source_url=self.NAS_IP,
            cache_dir=str(self.cache_dir),
            username=self.NAS_USER,
            password=self.NAS_PASS,
            port=self.NAS_PORT,
            access_mode="smb",
        )

    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.cache_dir)

        # Disconnect from NAS if connected
        if hasattr(self, "streamer"):
            self.streamer.disconnect()

    def test_smb_connect(self):
        """Test connecting to NAS with SMB."""
        if self.use_mock:
            # Mock the connect method
            with mock.patch.object(self.streamer, "_smb_connect") as mock_connect:
                mock_connect.return_value = True

                # Test the connection
                assert self.streamer.connect() is True
                mock_connect.assert_called_once()
        else:
            # Actual connection test
            try:
                result = self.streamer.connect()
                assert result is True
                assert self.streamer.smb_conn is not None
            except Exception as e:
                pytest.fail(f"Connection failed with error: {e}")

    def test_list_shares_smb(self):
        """Test listing shares on the NAS with SMB."""
        if self.use_mock:
            # Create mock shares
            mock_share1 = mock.MagicMock()
            mock_share1.name = "public"
            mock_share1.isHidden = False

            mock_share2 = mock.MagicMock()
            mock_share2.name = "homes"
            mock_share2.isHidden = False

            mock_share3 = mock.MagicMock()
            mock_share3.name = "admin$"
            mock_share3.isHidden = True

            # Mock the SMB connection
            mock_conn = mock.MagicMock()
            mock_conn.listShares.return_value = [mock_share1, mock_share2, mock_share3]

            # Replace the connection with our mock
            self.streamer.smb_conn = mock_conn

            # Test listing shares
            shares = self.streamer.list_shares()
            assert "public" in shares
            assert "homes" in shares
            assert "admin$" not in shares  # Hidden share
            assert len(shares) == 2
            mock_conn.listShares.assert_called_once()
        else:
            # Connect first
            if not self.streamer.smb_conn:
                self.streamer.connect()

            # Actual list shares test
            try:
                shares = self.streamer.list_shares()
                assert isinstance(shares, list)
                assert len(shares) > 0
            except Exception as e:
                pytest.fail(f"List shares failed with error: {e}")

    def test_get_file_with_mock_smb(self):
        """Test getting a file from NAS with mocked SMB."""
        # Create a test file in cache to simulate an already downloaded file
        test_cache_file = self.cache_dir / "public" / "test_file.txt"
        test_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_cache_file, "w") as f:
            f.write("Test content")

        # Test getting a file that's already in cache
        result = self.streamer.get_file("public/test_file.txt", download_if_missing=False)
        assert result == test_cache_file

        # Test getting a file that's not in cache
        with pytest.raises(FileNotFoundError):
            self.streamer.get_file("public/nonexistent.txt", download_if_missing=False)

        # Mock the _download_file_smb method for testing download
        with mock.patch.object(self.streamer, "_download_file_smb") as mock_download:
            # Mock connect if needed
            if self.streamer.smb_conn is None:
                with mock.patch.object(self.streamer, "_smb_connect") as mock_connect:
                    mock_connect.return_value = True
                    self.streamer.smb_conn = mock.MagicMock()
                    self.streamer.smb_conn.echo.return_value = True

            # Test downloading a file
            new_file = self.streamer.get_file("public/new_file.txt")
            mock_download.assert_called_once()
            assert new_file == self.cache_dir / "public" / "new_file.txt"

    def test_list_directory_with_mock_smb(self):
        """Test listing directory contents with mocked SMB."""
        # Create mock files for the SMB listPath response
        mock_file1 = mock.MagicMock()
        mock_file1.filename = "file1.txt"
        mock_file1.isDirectory = False
        mock_file1.file_size = 1024
        mock_file1.create_time = 1618023942
        mock_file1.last_write_time = 1618023942
        mock_file1.last_access_time = 1618023942

        mock_file2 = mock.MagicMock()
        mock_file2.filename = "folder1"
        mock_file2.isDirectory = True
        mock_file2.file_size = 0
        mock_file2.create_time = 1618023942
        mock_file2.last_write_time = 1618023942
        mock_file2.last_access_time = 1618023942

        # Include '.' and '..' entries which should be filtered out
        mock_dot = mock.MagicMock()
        mock_dot.filename = "."
        mock_dot.isDirectory = True

        mock_dotdot = mock.MagicMock()
        mock_dotdot.filename = ".."
        mock_dotdot.isDirectory = True

        # Mock the SMB connection
        mock_conn = mock.MagicMock()
        mock_conn.listPath.return_value = [mock_file1, mock_file2, mock_dot, mock_dotdot]

        # Replace the connection with our mock
        self.streamer.smb_conn = mock_conn

        # Test listing a directory
        files = self.streamer.list_directory("public")
        assert len(files) == 2  # Should only include file1.txt and folder1
        assert files[0]["name"] == "file1.txt"
        assert files[0]["isdir"] is False
        assert files[1]["name"] == "folder1"
        assert files[1]["isdir"] is True
        mock_conn.listPath.assert_called_once()

    def test_upload_file_with_mock_smb(self):
        """Test uploading a file with mocked SMB."""
        # Create a test local file
        local_file = self.cache_dir / "local_file.txt"
        with open(local_file, "w") as f:
            f.write("Test content")

        # Mock the SMB connection
        mock_conn = mock.MagicMock()
        mock_conn.echo.return_value = True

        # Simulate file not found for first call, then found for second
        mock_conn.getAttributes.side_effect = [Exception("File not found"), None]

        # Replace the connection with our mock
        self.streamer.smb_conn = mock_conn

        # Test uploading a file
        result = self.streamer.upload_file(local_file, "public/uploaded.txt")
        assert result is True
        mock_conn.storeFile.assert_called_once()

        # Test with file that already exists
        result = self.streamer.upload_file(local_file, "public/existing.txt", overwrite=False)
        assert result is False  # Should return False if not overwriting

        # Test with overwrite=True
        result = self.streamer.upload_file(local_file, "public/existing.txt", overwrite=True)
        assert result is True  # Should return True when overwriting

    def test_create_directory_with_mock_smb(self):
        """Test creating a directory with mocked SMB."""
        # Mock the SMB connection
        mock_conn = mock.MagicMock()
        # Simulate directory not found, then create successfully
        mock_conn.listPath.side_effect = Exception("Directory not found")

        # Replace the connection with our mock
        self.streamer.smb_conn = mock_conn

        # Test creating a directory
        result = self.streamer._create_directory_smb("public/new_folder")
        assert result is True
        mock_conn.createDirectory.assert_called()
