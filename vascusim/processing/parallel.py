"""
Parallel processing utilities for vascular simulation data.

This module provides functions for efficient parallel processing of vascular 
simulation data, including worker pool management and batch processing.
"""

import functools
import logging
import multiprocessing as mp
from multiprocessing.pool import Pool
import os
from typing import Any, Callable, List, Optional, TypeVar, Union

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Define type variables for generic typing
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

logger = logging.getLogger(__name__)


def worker_pool(n_workers: Optional[int] = None) -> Pool:
    """
    Create a worker pool for parallel processing.
    
    Args:
        n_workers: Number of worker processes to create. If None, uses CPU count.
        
    Returns:
        Multiprocessing Pool object.
    """
    if n_workers is None:
        # Use CPU count with a sensible default maximum
        n_workers = min(mp.cpu_count(), 16)
    
    # Ensure at least one worker
    n_workers = max(1, n_workers)
    
    logger.debug(f"Creating worker pool with {n_workers} workers")
    return mp.Pool(processes=n_workers)


def process_batch(
    func: Callable[[T], R],
    items: List[T],
    n_workers: Optional[int] = None,
    chunksize: Optional[int] = None,
    show_progress: bool = False
) -> List[R]:
    """
    Process a batch of items in parallel.
    
    Args:
        func: Function to apply to each item.
        items: List of items to process.
        n_workers: Number of worker processes to use.
        chunksize: Size of chunks to send to each worker.
        show_progress: Whether to show a progress bar.
        
    Returns:
        List of results.
    """
    if not items:
        return []
    
    # Determine appropriate chunk size if not specified
    if chunksize is None:
        chunksize = max(1, len(items) // (n_workers or mp.cpu_count()) // 4)
    
    # Create worker pool
    pool = worker_pool(n_workers)
    
    try:
        # Apply function to each item
        if show_progress and HAS_TQDM:
            # Use tqdm for progress bar
            results = list(tqdm(
                pool.imap(func, items, chunksize=chunksize),
                total=len(items),
                desc="Processing"
            ))
        else:
            # No progress bar
            results = pool.map(func, items, chunksize=chunksize)
    finally:
        # Always close the pool
        pool.close()
        pool.join()
    
    return results


def parallelize(n_workers: Optional[int] = None, **kwargs) -> Callable:
    """
    Decorator for parallelizing functions that operate on batches.
    
    Args:
        n_workers: Number of worker processes to use.
        **kwargs: Additional keyword arguments for process_batch.
        
    Returns:
        Decorated function.
    """
    def decorator(func: Callable[[List[T]], List[R]]) -> Callable[[List[T]], List[R]]:
        @functools.wraps(func)
        def wrapper(items: List[T], **inner_kwargs) -> List[R]:
            # Extract single item case
            if not isinstance(items, list):
                items = [items]
                single_item = True
            else:
                single_item = False
            
            # Check if parallelization should be bypassed
            inner_n_workers = inner_kwargs.pop('n_workers', n_workers)
            
            # If only 1 worker or small batch, run sequentially
            if inner_n_workers == 1 or len(items) <= 1:
                results = [func(item) for item in items]
            else:
                # Merge decorator kwargs with inner kwargs
                process_kwargs = kwargs.copy()
                process_kwargs.update(inner_kwargs)
                
                # Process in parallel
                results = process_batch(
                    func=lambda item: func(item),
                    items=items,
                    n_workers=inner_n_workers,
                    **process_kwargs
                )
            
            # Return single item if input was single item
            if single_item and results:
                return results[0]
            return results
        
        return wrapper
    
    return decorator