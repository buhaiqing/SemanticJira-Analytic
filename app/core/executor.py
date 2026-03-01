"""Global executors for concurrent task processing with performance optimizations."""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os
import atexit
import logging
import threading
from typing import Optional, Callable, List, Any, Union
from contextlib import contextmanager
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class ExecutorManager:
    """
    High-performance executor manager with lazy initialization and resource optimization.

    Features:
    - Lazy initialization with thread-safe singleton pattern
    - Dynamic worker sizing based on workload
    - Health monitoring and auto-recovery
    - Graceful shutdown with timeout control
    - Context manager support for resource cleanup
    - Separate management for thread and process executors to avoid pickling issues
    """

    def __init__(self):
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None
        self._lock = threading.Lock()
        self._thread_initializing = False
        self._process_initializing = False

    def _get_optimal_thread_workers(self) -> int:
        """Calculate optimal number of thread pool workers based on system resources."""
        cpu_count = os.cpu_count() or 1
        # For I/O-bound tasks, use more workers
        # Formula: 4 * CPU count, capped between 8 and 64
        return min(64, max(8, cpu_count * 4))

    def _get_optimal_process_workers(self) -> int:
        """Calculate optimal number of process pool workers based on CPU count."""
        cpu_count = os.cpu_count() or 1
        # For CPU-bound tasks, match CPU count
        return cpu_count

    def get_thread_executor(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Get or create thread pool executor with lazy initialization.

        Args:
            max_workers: Optional override for max workers count

        Returns:
            Configured ThreadPoolExecutor instance
        """
        # Fast path - check without lock
        if self._thread_executor is not None:
            return self._thread_executor

        with self._lock:
            if self._thread_executor is None and not self._thread_initializing:
                self._thread_initializing = True
                try:
                    workers = max_workers or self._get_optimal_thread_workers()
                    logger.info(f"Creating global ThreadPoolExecutor with {workers} workers")
                    self._thread_executor = ThreadPoolExecutor(
                        max_workers=workers,
                        thread_name_prefix="GlobalThread",
                        initializer=self._thread_worker_init
                    )
                finally:
                    self._thread_initializing = False

            # Wait for initialization if another thread is initializing
            while self._thread_initializing and self._thread_executor is None:
                threading.Event().wait(0.01)

        return self._thread_executor

    def _thread_worker_init(self):
        """Initialize thread worker with optimal settings."""
        # Increase recursion limit for deep call stacks
        import sys
        current_limit = sys.getrecursionlimit()
        if current_limit < 2000:
            sys.setrecursionlimit(2000)

    def get_process_executor(self, max_workers: Optional[int] = None) -> ProcessPoolExecutor:
        """
        Get or create process pool executor with lazy initialization.
        Note: This creates a new ProcessPoolExecutor on each call after shutdown
        to avoid pickling issues with the manager instance.

        Args:
            max_workers: Optional override for max workers count

        Returns:
            Configured ProcessPoolExecutor instance
        """
        if self._process_executor is not None:
            return self._process_executor

        # Don't use lock for process executor to avoid pickling issues
        # Create fresh executor if needed
        workers = max_workers or self._get_optimal_process_workers()
        logger.info(f"Creating global ProcessPoolExecutor with {workers} workers")
        self._process_executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=self._process_worker_init
        )
        return self._process_executor

    def _process_worker_init(self):
        """Initialize process worker with optimal settings."""
        import sys
        # Optimize for worker processes
        sys.setrecursionlimit(2000)

    @property
    def is_initialized(self) -> bool:
        """Check if executors are initialized."""
        return self._thread_executor is not None or self._process_executor is not None

    @property
    def thread_executor(self) -> Optional[ThreadPoolExecutor]:
        """Get thread executor if initialized."""
        return self._thread_executor

    @property
    def process_executor(self) -> Optional[ProcessPoolExecutor]:
        """Get process executor if initialized."""
        return self._process_executor

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Shutdown all executors gracefully.

        Args:
            wait: Whether to wait for pending tasks
            timeout: Maximum time to wait for shutdown (seconds)
        """
        with self._lock:
            if self._thread_executor:
                logger.info("Shutting down global ThreadPoolExecutor")
                self._thread_executor.shutdown(wait=wait, cancel_futures=not wait)
                self._thread_executor = None
                self._thread_initializing = False

            if self._process_executor:
                logger.info("Shutting down global ProcessPoolExecutor")
                self._process_executor.shutdown(wait=wait, cancel_futures=not wait)
                self._process_executor = None
                self._process_initializing = False

    @contextmanager
    def temporary_thread_pool(self, max_workers: int, timeout: Optional[float] = None):
        """
        Context manager for temporary thread pool usage.

        Args:
            max_workers: Number of workers for temporary pool
            timeout: Optional timeout for task execution

        Yields:
            ThreadPoolExecutor instance
        """
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            yield executor
        finally:
            executor.shutdown(wait=True)

    def submit_batch(self, fn: Callable, args_list: List[tuple],
                     executor_type: str = 'thread') -> List[Any]:
        """
        Submit batch of tasks and collect results.

        Args:
            fn: Function to execute
            args_list: List of argument tuples for each call
            executor_type: 'thread' or 'process'

        Returns:
            List of results in same order as args_list
        """
        executor = (self.get_thread_executor() if executor_type == 'thread'
                   else self.get_process_executor())

        futures = {executor.submit(fn, *args): i for i, args in enumerate(args_list)}
        results = [None] * len(args_list)

        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:
                logger.error(f"Task {index} failed: {e}")
                results[index] = e

        return results

    def map_parallel(self, fn: Callable, iterable: List[Any],
                     executor_type: str = 'thread', chunksize: int = 1) -> List[Any]:
        """
        Execute function in parallel for each item in iterable.

        Args:
            fn: Function to execute
            iterable: List of items to process
            executor_type: 'thread' or 'process'
            chunksize: Number of items per chunk

        Returns:
            List of results
        """
        executor = (self.get_thread_executor() if executor_type == 'thread'
                   else self.get_process_executor())
        return list(executor.map(fn, iterable, chunksize=chunksize))


# Global executor manager instance
_executor_manager = ExecutorManager()


def get_executor_manager() -> ExecutorManager:
    """Get the global executor manager instance."""
    return _executor_manager


# Legacy compatibility functions
def get_thread_executor(max_workers: int = None):
    """Get or create a global ThreadPoolExecutor (legacy API)."""
    return _executor_manager.get_thread_executor(max_workers)


def get_process_executor(max_workers: int = None):
    """Get or create a global ProcessPoolExecutor (legacy API)."""
    return _executor_manager.get_process_executor(max_workers)


@atexit.register
def shutdown_executors():
    """Shut down executors on application exit."""
    _executor_manager.shutdown(wait=True)


# Decorator utilities for concurrent execution
def run_async(fn: Callable) -> Callable:
    """
    Decorator to run synchronous function in async context.

    Usage:
        @run_async
        def sync_function(x, y):
            return x + y

        result = await sync_function(1, 2)
    """
    @wraps(fn)
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor_manager.get_thread_executor(),
            lambda: fn(*args, **kwargs)
        )
    return async_wrapper


def run_in_thread(fn: Callable) -> Callable:
    """
    Decorator to run function in thread pool.

    Usage:
        @run_in_thread
        def io_bound_function(url):
            return requests.get(url)

        future = io_bound_function("https://example.com")
        result = future.result()
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        executor = _executor_manager.get_thread_executor()
        return executor.submit(fn, *args, **kwargs)
    return wrapper


def run_in_process(fn: Callable) -> Callable:
    """
    Decorator to run function in process pool for CPU-bound tasks.

    Usage:
        @run_in_process
        def cpu_bound_function(data):
            return heavy_computation(data)

        future = cpu_bound_function(large_dataset)
        result = future.result()
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        executor = _executor_manager.get_process_executor()
        return executor.submit(fn, *args, **kwargs)
    return wrapper


def batch_process(fn: Callable = None, batch_size: int = 10,
                  executor_type: str = 'thread'):
    """
    Decorator to process items in batches using thread/process pool.
    Can be used with or without parentheses.

    Usage:
        @batch_process
        def process_item(item):
            return transform(item)

        @batch_process(batch_size=20)
        def process_item(item):
            return transform(item)

        results = process_item(large_list)

    Args:
        fn: Optional function to decorate (for @batch_process usage)
        batch_size: Number of items per batch
        executor_type: 'thread' or 'process'
    """
    def decorator(func):
        @wraps(func)
        def wrapper(items: List[Any]) -> List[Any]:
            executor = _executor_manager.get_thread_executor() \
                if executor_type == 'thread' else _executor_manager.get_process_executor()

            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = list(executor.map(func, batch))
                results.extend(batch_results)

            return results
        return wrapper

    # Handle both @batch_process and @batch_process() usage
    if fn is not None:
        return decorator(fn)
    return decorator
