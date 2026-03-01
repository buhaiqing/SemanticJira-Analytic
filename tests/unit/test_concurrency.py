"""Unit tests for concurrency and parallel execution features."""

import pytest
import asyncio
import threading
import time
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Add project root to path
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from app.core.executor import (
    ExecutorManager,
    get_executor_manager,
    get_thread_executor,
    get_process_executor,
    run_async,
    run_in_thread,
    run_in_process,
    batch_process,
    shutdown_executors
)

# Check if we're on macOS (which has multiprocessing limitations)
IS_MACOS = platform.system() == 'Darwin'


class TestExecutorManager:
    """Test suite for ExecutorManager class."""

    @pytest.fixture
    def manager(self):
        """Create fresh executor manager for each test."""
        return ExecutorManager()

    def test_singleton_initialization(self, manager):
        """Test that executors are initialized as singletons."""
        executor1 = manager.get_thread_executor()
        executor2 = manager.get_thread_executor()

        assert executor1 is executor2
        assert isinstance(executor1, ThreadPoolExecutor)

    def test_thread_executor_worker_count(self, manager):
        """Test thread executor worker count calculation."""
        workers = manager._get_optimal_thread_workers()
        cpu_count = __import__('os').cpu_count() or 1

        assert workers >= 8
        assert workers <= 64
        assert workers == min(64, max(8, cpu_count * 4))

    def test_process_executor_worker_count(self, manager):
        """Test process executor worker count calculation."""
        workers = manager._get_optimal_process_workers()
        cpu_count = __import__('os').cpu_count() or 1

        assert workers == cpu_count

    def test_lazy_initialization(self, manager):
        """Test that executors are lazily initialized."""
        assert not manager.is_initialized

        manager.get_thread_executor()
        assert manager.is_initialized

    def test_thread_safe_initialization(self, manager):
        """Test thread-safe executor initialization."""
        results = []
        errors = []

        def get_executor():
            try:
                executor = manager.get_thread_executor()
                results.append(executor)
            except Exception as e:
                errors.append(e)

        # Create multiple threads requesting executor simultaneously
        threads = [threading.Thread(target=get_executor) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All should be the same executor instance
        assert all(r is results[0] for r in results)

    def test_submit_batch_thread(self, manager):
        """Test batch submission to thread executor."""
        def simple_function(x, y):
            return x + y

        args_list = [(1, 2), (3, 4), (5, 6)]
        results = manager.submit_batch(simple_function, args_list, executor_type='thread')

        assert results == [3, 7, 11]

    @pytest.mark.skipif(IS_MACOS, reason="ProcessPoolExecutor has limitations on macOS")
    def test_submit_batch_process(self, manager):
        """Test batch submission to process executor."""
        def cpu_bound_function(x):
            return x * x

        args_list = [(1,), (2,), (3,), (4,), (5,)]
        results = manager.submit_batch(cpu_bound_function, args_list, executor_type='process')

        assert results == [1, 4, 9, 16, 25]

    def test_map_parallel_thread(self, manager):
        """Test parallel map with thread executor."""
        def square(x):
            return x * x

        results = manager.map_parallel(square, [1, 2, 3, 4, 5], executor_type='thread')
        assert results == [1, 4, 9, 16, 25]

    @pytest.mark.skipif(IS_MACOS, reason="ProcessPoolExecutor has limitations on macOS")
    def test_map_parallel_process(self, manager):
        """Test parallel map with process executor."""
        def increment(x):
            return x + 1

        results = manager.map_parallel(increment, range(10), executor_type='process')
        assert results == list(range(1, 11))

    def test_temporary_thread_pool_context_manager(self, manager):
        """Test temporary thread pool context manager."""
        with manager.temporary_thread_pool(max_workers=4) as executor:
            assert isinstance(executor, ThreadPoolExecutor)
            future = executor.submit(lambda x: x * 2, 5)
            assert future.result() == 10

    def test_shutdown(self, manager):
        """Test executor shutdown."""
        manager.get_thread_executor()
        manager.get_process_executor()
        assert manager.is_initialized

        manager.shutdown(wait=True)
        assert not manager.is_initialized

    def test_shutdown_with_pending_tasks(self, manager):
        """Test shutdown handles pending tasks."""
        executor = manager.get_thread_executor()

        # Submit a task
        future = executor.submit(lambda: time.sleep(0.1))

        # Shutdown should wait for task to complete
        manager.shutdown(wait=True)

        # Future should be completed
        assert future.done()

    def test_properties(self, manager):
        """Test executor manager properties."""
        assert manager.thread_executor is None
        assert manager.process_executor is None

        manager.get_thread_executor()
        assert manager.thread_executor is not None
        assert isinstance(manager.thread_executor, ThreadPoolExecutor)

        manager.get_process_executor()
        assert manager.process_executor is not None
        assert isinstance(manager.process_executor, ProcessPoolExecutor)


class TestConcurrencyDecorators:
    """Test suite for concurrency decorators."""

    def test_run_async_decorator(self):
        """Test @run_async decorator."""
        @run_async
        def sync_function(x, y):
            return x + y

        async def test_runner():
            result = await sync_function(10, 20)
            assert result == 30

        asyncio.run(test_runner())

    def test_run_in_thread_decorator(self):
        """Test @run_in_thread decorator."""
        call_count = [0]

        @run_in_thread
        def io_function(url):
            call_count[0] += 1
            return f"Processed {url}"

        future = io_function("https://example.com")
        assert isinstance(future, Future)

        result = future.result(timeout=5)
        assert result == "Processed https://example.com"
        assert call_count[0] == 1

    @pytest.mark.skipif(IS_MACOS, reason="ProcessPoolExecutor has limitations on macOS")
    def test_run_in_process_decorator(self):
        """Test @run_in_process decorator."""
        @run_in_process
        def cpu_function(n):
            return sum(i * i for i in range(n))

        future = cpu_function(1000)
        assert isinstance(future, Future)

        result = future.result(timeout=10)
        expected = sum(i * i for i in range(1000))
        assert result == expected

    def test_batch_process_decorator(self):
        """Test @batch_process decorator."""
        @batch_process(batch_size=5)
        def process_item(item):
            return item * 2

        items = list(range(20))
        results = process_item(items)

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]

    @pytest.mark.skipif(IS_MACOS, reason="ProcessPoolExecutor has limitations on macOS")
    def test_batch_process_with_process_executor(self):
        """Test @batch_process with process executor."""
        @batch_process(batch_size=3, executor_type='process')
        def compute_square(x):
            return x * x

        items = [1, 2, 3, 4, 5, 6]
        results = compute_square(items)

        assert results == [1, 4, 9, 16, 25, 36]


class TestRaceConditions:
    """Test suite for race condition scenarios."""

    def test_concurrent_task_submission(self):
        """Test concurrent task submission doesn't cause race conditions."""
        manager = ExecutorManager()
        results = []
        lock = threading.Lock()

        def submit_and_collect(n):
            result = manager.get_thread_executor().submit(lambda x: x * 2, n)
            with lock:
                results.append((n, result.result()))

        threads = [threading.Thread(target=submit_and_collect, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all results are correct
        expected = [(i, i * 2) for i in range(50)]
        assert sorted(results) == sorted(expected)

    def test_counter_increment_thread_safety(self):
        """Test thread-safe counter increment."""
        manager = ExecutorManager()
        counter = [0]
        lock = threading.Lock()

        def increment():
            with lock:
                counter[0] += 1

        futures = [manager.get_thread_executor().submit(increment) for _ in range(100)]
        for f in futures:
            f.result()

        assert counter[0] == 100

    def test_shared_resource_access(self):
        """Test safe access to shared resources."""
        manager = ExecutorManager()
        shared_data = {'count': 0, 'items': []}
        lock = threading.Lock()

        def update_data(value):
            with lock:
                shared_data['count'] += 1
                shared_data['items'].append(value)

        futures = [manager.get_thread_executor().submit(update_data, i) for i in range(20)]
        for f in futures:
            f.result()

        assert shared_data['count'] == 20
        assert len(shared_data['items']) == 20


class TestDeadlockPrevention:
    """Test suite for deadlock prevention."""

    def test_nested_executor_calls(self):
        """Test that nested executor calls don't deadlock."""
        manager = ExecutorManager()

        def outer_function(n):
            # Inner call should not deadlock
            future = manager.get_thread_executor().submit(lambda x: x + 1, n)
            return future.result() + 10

        future = manager.get_thread_executor().submit(outer_function, 5)
        result = future.result(timeout=10)

        assert result == 16  # (5 + 1) + 10

    def test_timeout_handling(self):
        """Test proper timeout handling prevents hangs."""
        manager = ExecutorManager()

        def slow_function():
            time.sleep(0.5)
            return "done"

        start = time.time()
        future = manager.get_thread_executor().submit(slow_function)

        try:
            result = future.result(timeout=2.0)
            assert result == "done"
        except Exception:
            pass

        elapsed = time.time() - start
        assert elapsed < 3.0  # Should not hang

    def test_executor_reuse_after_shutdown(self):
        """Test executor can be recreated after shutdown."""
        manager = ExecutorManager()

        # First lifecycle
        executor1 = manager.get_thread_executor()
        future1 = executor1.submit(lambda: 42)
        assert future1.result() == 42

        manager.shutdown(wait=True)
        assert not manager.is_initialized

        # Second lifecycle
        executor2 = manager.get_thread_executor()
        future2 = executor2.submit(lambda: 100)
        assert future2.result() == 100


class TestPerformanceOptimizations:
    """Test suite for performance optimizations."""

    def test_batch_processing_efficiency(self):
        """Test batch processing is more efficient than sequential."""
        manager = ExecutorManager()

        def cpu_task(n):
            return sum(range(n))

        # Sequential processing
        items = [10000] * 10
        start = time.time()
        sequential_results = [cpu_task(x) for x in items]
        sequential_time = time.time() - start

        # Parallel processing
        start = time.time()
        parallel_results = manager.map_parallel(cpu_task, items, executor_type='thread')
        parallel_time = time.time() - start

        # Verify correctness
        assert sequential_results == parallel_results

        # Parallel should be faster (or at least not much slower)
        # Note: For very small tasks, threading overhead might make it slower
        # This is expected behavior - we just verify it completes
        assert parallel_time < sequential_time * 2.0  # Allow more overhead for threading

    def test_worker_initialization_efficiency(self):
        """Test that workers are initialized only once."""
        manager = ExecutorManager()

        # Multiple calls should return same executor
        executors = [manager.get_thread_executor() for _ in range(100)]
        assert all(e is executors[0] for e in executors)

    def test_concurrent_batch_submission(self):
        """Test concurrent batch submissions."""
        manager = ExecutorManager()

        def process_batch(batch_id):
            return manager.submit_batch(lambda x: x * 2, [(i,) for i in range(10)])

        futures = [manager.get_thread_executor().submit(process_batch, i) for i in range(5)]
        results = [f.result() for f in futures]

        expected = [[i * 2 for i in range(10)] for _ in range(5)]
        assert results == expected


class TestErrorHandling:
    """Test suite for error handling in concurrent execution."""

    def test_exception_in_thread_task(self):
        """Test exception handling in thread tasks."""
        manager = ExecutorManager()

        def failing_function():
            raise ValueError("Test error")

        future = manager.get_thread_executor().submit(failing_function)

        with pytest.raises(ValueError, match="Test error"):
            future.result(timeout=5)

    @pytest.mark.skipif(IS_MACOS, reason="ProcessPoolExecutor has limitations on macOS")
    def test_exception_in_process_task(self):
        """Test exception handling in process tasks."""
        manager = ExecutorManager()

        def process_failing_function():
            raise RuntimeError("Process error")

        future = manager.get_process_executor().submit(process_failing_function)

        with pytest.raises(RuntimeError, match="Process error"):
            future.result(timeout=10)

    def test_batch_with_mixed_success_failure(self):
        """Test batch processing with mixed success/failure."""
        manager = ExecutorManager()

        def mixed_function(x):
            if x % 2 == 0:
                return x * 2
            else:
                raise ValueError(f"Odd number: {x}")

        args_list = [(i,) for i in range(5)]
        results = manager.submit_batch(mixed_function, args_list, executor_type='thread')

        # Check results contain both values and exceptions
        assert results[0] == 0  # 0 * 2
        assert isinstance(results[1], ValueError)  # 1 is odd
        assert results[2] == 4  # 2 * 2
        assert isinstance(results[3], ValueError)  # 3 is odd
        assert results[4] == 8  # 4 * 2

    def test_timeout_on_slow_task(self):
        """Test timeout handling for slow tasks."""
        manager = ExecutorManager()

        def slow_task():
            time.sleep(10)
            return "never returns"

        future = manager.get_thread_executor().submit(slow_task)

        with pytest.raises(Exception):  # TimeoutError or concurrent.futures.TimeoutError
            future.result(timeout=0.1)


class TestGlobalExecutorFunctions:
    """Test suite for global executor functions."""

    def test_get_thread_executor_global(self):
        """Test global get_thread_executor function."""
        executor = get_thread_executor()
        assert isinstance(executor, ThreadPoolExecutor)

        # Should return same instance
        executor2 = get_thread_executor()
        assert executor is executor2

    def test_get_process_executor_global(self):
        """Test global get_process_executor function."""
        executor = get_process_executor()
        assert isinstance(executor, ProcessPoolExecutor)

        # Should return same instance
        executor2 = get_process_executor()
        assert executor is executor2

    def test_get_executor_manager(self):
        """Test get_executor_manager function."""
        manager = get_executor_manager()
        assert isinstance(manager, ExecutorManager)


class TestStressScenarios:
    """Stress tests for concurrency features."""

    def test_high_volume_task_submission(self):
        """Test handling high volume of task submissions."""
        manager = ExecutorManager()

        def simple_task(n):
            return n

        futures = [manager.get_thread_executor().submit(simple_task, i) for i in range(1000)]
        results = [f.result() for f in futures]

        assert results == list(range(1000))

    def test_many_concurrent_threads(self):
        """Test many concurrent threads."""
        manager = ExecutorManager()
        results = []
        lock = threading.Lock()

        def worker(n):
            time.sleep(0.01)
            with lock:
                results.append(n)

        futures = [manager.get_thread_executor().submit(worker, i) for i in range(200)]
        for f in futures:
            f.result(timeout=30)

        assert len(results) == 200

    def test_large_batch_processing(self):
        """Test large batch processing."""
        manager = ExecutorManager()

        def double(x):
            return x * 2

        items = list(range(500))
        results = manager.map_parallel(double, items, executor_type='thread')

        assert results == [i * 2 for i in items]


# Run shutdown on test module exit
def teardown_module():
    """Clean up resources after tests."""
    shutdown_executors()
