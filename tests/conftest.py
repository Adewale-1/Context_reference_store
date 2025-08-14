#!/usr/bin/env python3
"""
Pytest configuration and fixtures for Context Reference Store tests.

This module provides common fixtures, configuration, and utilities
for all test modules in the Context Reference Store test suite.
"""

import pytest
import tempfile
import os
import shutil
import gc
import sys
from typing import Generator, Any


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_store import ContextReferenceStore, CacheEvictionPolicy


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def basic_store() -> Generator[ContextReferenceStore, None, None]:
    """Provide a basic context store for testing."""
    store = ContextReferenceStore(
        cache_size=50, eviction_policy=CacheEvictionPolicy.LRU
    )
    try:
        yield store
    finally:
        del store
        gc.collect()


@pytest.fixture
def advanced_store(temp_dir: str) -> Generator[ContextReferenceStore, None, None]:
    """Provide an advanced context store with all features enabled."""
    store = ContextReferenceStore(
        cache_size=100,
        eviction_policy=CacheEvictionPolicy.LRU,
        enable_compression=True,
        use_disk_storage=True,
        binary_cache_dir=temp_dir,
        large_binary_threshold=1024,
        memory_threshold=0.8,
    )
    try:
        yield store
    finally:
        del store
        gc.collect()


@pytest.fixture
def memory_constrained_store() -> Generator[ContextReferenceStore, None, None]:
    """Provide a memory-constrained store for testing edge cases."""
    store = ContextReferenceStore(
        cache_size=5,  # Very small cache
        eviction_policy=CacheEvictionPolicy.MEMORY_PRESSURE,
        memory_threshold=0.1,  # Very low threshold
    )
    try:
        yield store
    finally:
        del store
        gc.collect()


@pytest.fixture
def sample_text_content() -> str:
    """Provide sample text content for testing."""
    return "This is sample text content for testing the Context Reference Store functionality."


@pytest.fixture
def sample_json_content() -> dict:
    """Provide sample JSON content for testing."""
    return {
        "test_data": True,
        "nested": {"level1": {"level2": "deep_value"}},
        "array": [1, 2, 3, 4, 5],
        "metadata": {"created": "2024-01-01T00:00:00Z", "version": "1.0.0"},
    }


@pytest.fixture
def sample_binary_content() -> bytes:
    """Provide sample binary content for testing."""
    return b"Binary test data: \x00\x01\x02\x03\xff\xfe\xfd"


@pytest.fixture
def large_text_content() -> str:
    """Provide large text content for testing performance."""
    base_text = "Large content for performance testing with repeated patterns. "
    return base_text * 1000  # ~60KB of text


@pytest.fixture
def multimodal_content():
    """Provide sample multimodal content for testing."""
    from context_store import MultimodalContent, MultimodalPart

    parts = [
        MultimodalPart(text="Text part of multimodal content"),
        MultimodalPart(binary_data=b"Binary part data"),
        MultimodalPart(json_data={"part": "json", "type": "metadata"}),
    ]

    return MultimodalContent(role="user", parts=parts)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically cleanup after each test."""
    yield
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup for the entire test session."""
    print("\n🚀 Starting Context Reference Store test suite")
    yield
    print("\n✅ Context Reference Store test suite completed")


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "stress: marks tests as stress/performance tests"
    )
    config.addinivalue_line("markers", "edge_case: marks tests as edge case tests")
    config.addinivalue_line("markers", "adapter: marks tests for framework adapters")
    config.addinivalue_line(
        "markers", "optimization: marks tests for optimization features"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file names."""
    for item in items:
        if "stress" in item.module.__name__:
            item.add_marker(pytest.mark.stress)
            item.add_marker(pytest.mark.slow)

        if "integration" in item.module.__name__:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)

        if "edge" in item.module.__name__:
            item.add_marker(pytest.mark.edge_case)

        if "adapter" in item.module.__name__:
            item.add_marker(pytest.mark.adapter)

        if "optimization" in item.module.__name__:
            item.add_marker(pytest.mark.optimization)


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    if "slow" in item.keywords and item.config.getoption("--quick"):
        pytest.skip("Skipping slow test in quick mode")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run only quick tests, skip slow tests",
    )
    parser.addoption(
        "--stress", action="store_true", default=False, help="Run stress tests"
    )
    parser.addoption(
        "--adapters-only",
        action="store_true",
        default=False,
        help="Run only adapter tests",
    )


@pytest.fixture
def quick_mode(request):
    """Provide quick mode flag to tests."""
    return request.config.getoption("--quick")


@pytest.fixture
def stress_mode(request):
    """Provide stress mode flag to tests."""
    return request.config.getoption("--stress")


class TestHelpers:
    """Helper functions for tests."""

    @staticmethod
    def create_test_contexts(store: ContextReferenceStore, count: int = 10) -> list:
        """Create multiple test contexts in the store."""
        context_ids = []
        for i in range(count):
            content = f"Test content {i}: " + ("data " * 50)
            context_id = store.store(content)
            context_ids.append(context_id)
        return context_ids

    @staticmethod
    def verify_context_integrity(
        store: ContextReferenceStore, context_id: str, expected_content: Any
    ) -> bool:
        """Verify that a context contains the expected content."""
        try:
            retrieved = store.retrieve(context_id)
            return retrieved == expected_content
        except Exception:
            return False

    @staticmethod
    def measure_operation_time(operation_func, *args, **kwargs) -> tuple:
        """Measure the time taken by an operation."""
        import time

        start_time = time.time()
        result = operation_func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @staticmethod
    def create_large_content(size_kb: int) -> str:
        """Create large text content of specified size in KB."""
        base_text = "Large content for testing purposes. "
        target_size = size_kb * 1024
        repetitions = target_size // len(base_text.encode()) + 1
        return base_text * repetitions

    @staticmethod
    def skip_if_no_dependency(module_name: str, reason: str = None):
        """Skip test if a dependency is not available."""
        try:
            __import__(module_name)
        except ImportError:
            pytest.skip(reason or f"{module_name} not available")


@pytest.fixture
def test_helpers():
    """Provide test helper functions."""
    return TestHelpers


class PerformanceTracker:
    """Track performance metrics across tests."""

    def __init__(self):
        self.metrics = {}

    def record_metric(self, test_name: str, metric_name: str, value: float):
        """Record a performance metric."""
        if test_name not in self.metrics:
            self.metrics[test_name] = {}
        self.metrics[test_name][metric_name] = value

    def get_summary(self) -> dict:
        """Get summary of all recorded metrics."""
        return self.metrics.copy()


@pytest.fixture(scope="session")
def performance_tracker():
    """Provide performance tracking across the test session."""
    tracker = PerformanceTracker()
    yield tracker

    summary = tracker.get_summary()
    if summary:
        print("\n📊 Performance Summary:")
        for test_name, metrics in summary.items():
            print(f"  {test_name}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value}")


class TestExceptions:
    """Custom exceptions for testing."""

    class MockDependencyError(ImportError):
        """Exception for missing test dependencies."""

        pass

    class PerformanceThresholdError(AssertionError):
        """Exception for performance threshold violations."""

        pass


@pytest.fixture
def test_exceptions():
    """Provide test exception classes."""
    return TestExceptions


@pytest.fixture
def debug_mode():
    """Check if tests are running in debug mode."""
    return os.getenv("PYTEST_DEBUG", "false").lower() == "true"


def pytest_runtest_logreport(report):
    """Log test results for debugging."""
    if os.getenv("PYTEST_DEBUG", "false").lower() == "true":
        if report.when == "call":
            print(f"\n{'Passed' if report.passed else 'Failed'} {report.nodeid}")
            if report.failed:
                print(f"   Failure: {report.longrepr}")


@pytest.fixture
def resource_monitor():
    """Monitor resource usage during tests."""
    try:
        import psutil

        process = psutil.Process()

        initial_memory = process.memory_info().rss
        initial_cpu = process.cpu_percent()

        yield {
            "initial_memory": initial_memory,
            "initial_cpu": initial_cpu,
            "process": process,
        }

        final_memory = process.memory_info().rss
        memory_diff = final_memory - initial_memory

        if memory_diff > 100 * 1024 * 1024:  # 100MB increase
            print(f"High memory usage increase: {memory_diff / 1024 / 1024:.1f}MB")

    except ImportError:
        yield {"psutil_available": False}
