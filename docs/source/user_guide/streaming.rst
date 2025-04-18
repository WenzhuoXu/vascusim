===================
Streaming Data
===================

VascuSim provides robust capabilities for streaming vascular simulation data from various sources,
allowing efficient access to large datasets without having to download all files at once.

This guide covers how to use VascuSim's streaming functionality to access data from different sources.

Overview of Streaming Functionality
----------------------------------

VascuSim's streaming functionality allows you to:

- Access data from remote sources (NAS, Hugging Face datasets)
- Cache downloaded data for faster repeated access
- Prefetch data in the background while processing
- Efficiently manage memory usage for large datasets
- Filter data based on metadata criteria

The streaming functionality is primarily provided through the ``StreamingVascuDataset`` class,
which extends the base ``VascuDataset`` with additional streaming capabilities.

Available Streaming Sources
-------------------------

VascuSim currently supports streaming from the following sources:

NAS Storage
~~~~~~~~~~

Network Attached Storage (NAS) streaming allows you to access data stored on a Synology or
similar NAS device. VascuSim supports two access modes:

1. **API Mode**: Uses Synology's REST API for access (more feature-rich)
2. **SMB Mode**: Uses the SMB/CIFS protocol for more universal compatibility

Hugging Face Datasets
~~~~~~~~~~~~~~~~~~~

Stream data directly from Hugging Face datasets repositories. This is useful for
accessing publicly available datasets or sharing your own datasets with the community.

Local Storage with Streaming Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use the streaming functionality with local storage, which provides
consistent caching, prefetching, and memory management behavior across different
storage types.

Setting Up NAS Streaming
-----------------------

Basic NAS Connection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vascusim.data import StreamingVascuDataset
    
    # Connect to NAS using API mode
    dataset = StreamingVascuDataset(
        source_url='192.168.1.100',  # NAS IP address
        username='your_username',
        password='your_password',
        streaming_type='nas',
        access_mode='api'
    )
    
    # Access data
    data = dataset[0]

Advanced NAS Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vascusim.data import StreamingVascuDataset
    
    # Connect to NAS with advanced options
    dataset = StreamingVascuDataset(
        source_url='192.168.1.100',     # NAS IP address
        username='your_username',
        password='your_password',
        streaming_type='nas',
        access_mode='smb',              # Use SMB protocol
        port=445,                       # SMB port
        cache_dir='/path/to/cache',     # Custom cache location
        max_cache_size=1024*1024*1024,  # 1GB cache limit
        prefetch=True,                  # Enable prefetching
        prefetch_size=5                 # Prefetch 5 files ahead
    )

Direct NAS Access with NASStreamer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control over NAS access, you can use the ``NASStreamer`` class directly:

.. code-block:: python

    from vascusim.io import NASStreamer
    
    # Create NAS streamer
    nas = NASStreamer(
        source_url='192.168.1.100',
        username='your_username',
        password='your_password',
        port=5000,
        access_mode='api',
        secure=True  # Use HTTPS
    )
    
    # Connect to NAS
    nas.connect()
    
    # List available shares
    shares = nas.list_shares()
    print(f"Available shares: {shares}")
    
    # List directory contents
    files = nas.list_directory('share_name/path/to/dir')
    
    # Get a file (downloads to cache if needed)
    file_path = nas.get_file('share_name/path/to/file.vtu')
    
    # Get file metadata
    metadata = nas.get_metadata('share_name/path/to/file.json')
    
    # Disconnect when done
    nas.disconnect()

Setting Up Hugging Face Streaming
--------------------------------

Basic Hugging Face Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vascusim.data import StreamingVascuDataset
    
    # Connect to Hugging Face dataset
    dataset = StreamingVascuDataset(
        source_url='username/dataset-name',  # HF repo ID
        streaming_type='hf'
    )
    
    # Access data
    data = dataset[0]

Accessing Private Hugging Face Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vascusim.io import HuggingFaceStreamer
    
    # Create streamer with token for private repos
    hf = HuggingFaceStreamer(
        repo_id='username/private-dataset',
        token='your_hf_token',  # HF API token with read access
        revision='main'         # Branch or tag to use
    )
    
    # List files in the repository
    files = hf.list_files(pattern='*.vtu')  # Optional glob pattern
    
    # Get a file
    file_path = hf.get_file('path/to/file.vtu')

Advanced Streaming Features
-------------------------

Prefetching
~~~~~~~~~~

Prefetching allows VascuSim to download files in the background while you're processing
other data, which can significantly improve performance:

.. code-block:: python

    # Enable prefetching with custom settings
    dataset = StreamingVascuDataset(
        source_url='source',
        prefetch=True,           # Enable prefetching
        prefetch_size=10,        # Number of samples to prefetch
        delete_after_use=True,   # Free up space after using a file
    )

Filtering by Metadata
~~~~~~~~~~~~~~~~~~~

You can filter datasets based on metadata criteria:

.. code-block:: python

    # Define a filter function
    def filter_healthy_cases(metadata):
        return (
            metadata.get('is_healthy', False) and 
            metadata.get('patient_age', 0) > 50
        )
    
    # Create filtered dataset
    dataset = StreamingVascuDataset(
        source_url='source',
        filter_fn=filter_healthy_cases  # Apply filter function
    )

Custom Cache Management
~~~~~~~~~~~~~~~~~~~~~

You can customize how VascuSim manages its cache:

.. code-block:: python

    from vascusim.io import CacheManager
    
    # Create a custom cache manager
    cache_dir = '/path/to/cache'
    cache_manager = CacheManager(
        cache_dir=cache_dir,
        max_size=5 * 1024**3  # 5GB cache limit
    )
    
    # Get cache statistics
    stats = cache_manager.get_cache_stats()
    print(f"Cache usage: {stats['total_size']/1024**2:.2f} MB")
    print(f"File count: {stats['file_count']}")
    
    # Clear cache
    cache_manager.clear_all()
    
    # Or run LRU cleanup
    cache_manager.cleanup()

Handling Large Datasets
---------------------

For very large datasets, consider these best practices:

1. **Set a reasonable cache size**: Balance between memory usage and performance
2. **Enable prefetching**: But keep prefetch_size moderate (5-10 files)
3. **Use delete_after_use**: For extremely large datasets
4. **Filter early**: Use metadata filtering to reduce the dataset size
5. **Process in batches**: Use PyTorch Geometric's DataLoader for batch processing

Example large dataset workflow:

.. code-block:: python

    import torch
    from torch_geometric.loader import DataLoader
    from vascusim.data import StreamingVascuDataset
    
    # Create filtered streaming dataset
    dataset = StreamingVascuDataset(
        source_url='source',
        max_cache_size=2 * 1024**3,  # 2GB cache
        prefetch=True,
        prefetch_size=5,
        filter_fn=lambda meta: meta.get('resolution', 0) > 0.5  # Filter by resolution
    )
    
    # Create DataLoader for batch processing
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2  # Parallel loading
    )
    
    # Process in batches
    for batch in loader:
        # Process batch
        output = model(batch)
        
        # Free up memory
        del batch
        torch.cuda.empty_cache()  # If using GPU

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~

1. **Connection failures**: Check network settings, firewall rules, and credentials
2. **Cache exhaustion**: Increase max_cache_size or use delete_after_use=True
3. **Slow performance**: Enable prefetching or adjust prefetch_size
4. **Memory errors**: Reduce batch size or use delete_after_use=True

Checking Connection Status
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from vascusim.io import NASStreamer
    
    nas = NASStreamer(source_url='192.168.1.100', username='user', password='pass')
    
    # Check if connection is working
    if nas.connect():
        print("Connection successful!")
        
        # Test basic operations
        try:
            shares = nas.list_shares()
            print(f"Shares: {shares}")
            
            # List a directory
            files = nas.list_directory(f"{shares[0]}")
            print(f"Files in {shares[0]}: {len(files)}")
            
        except Exception as e:
            print(f"Error during operations: {e}")
        
        # Disconnect
        nas.disconnect()
    else:
        print("Connection failed.")

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import time
    
    # Measure data access performance
    start_time = time.time()
    
    for i in range(10):
        data = dataset[i]
        print(f"Sample {i}: {len(data.pos)} nodes, {len(data.edge_index[0])} edges")
    
    elapsed = time.time() - start_time
    print(f"Accessed 10 samples in {elapsed:.2f} seconds ({elapsed/10:.2f}s per sample)")