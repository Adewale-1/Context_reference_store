#!/usr/bin/env python3
"""
Performance tests for Context Reference Store.

This module contains performance benchmarks and regression tests.
"""

import pytest
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from context_store import (
    ContextReferenceStore,
    CacheEvictionPolicy,
    MultimodalContent,
    MultimodalPart,
)


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.fixture
    def store(self):
        """Create a context store for testing."""
        return ContextReferenceStore(
            cache_size=100,
            eviction_policy=CacheEvictionPolicy.LRU,
        )

    def create_large_content(self, size_kb: int = 100) -> str:
        """Create large text content for testing."""
        base_text = "This is sample content for performance testing. " * 100
        multiplier = (size_kb * 1024) // len(base_text.encode('utf-8'))
        return base_text * multiplier

    def create_structured_content(self, complexity: int = 1000) -> dict:
        """Create complex structured content."""
        return {
            "metadata": {
                "title": f"Test Document {i}",
                "created_at": time.time(),
                "tags": [f"tag_{j}" for j in range(10)],
            },
            "sections": [
                {
                    "id": f"section_{j}",
                    "content": f"Section content {j} " * 50,
                    "subsections": [
                        {"id": f"sub_{k}", "data": f"Data {k}" * 20}
                        for k in range(5)
                    ],
                }
                for j in range(complexity // 100)
            ],
            "references": [f"ref_{i}" for i in range(complexity // 10)],
        }

    @pytest.mark.benchmark
    def test_storage_performance(self, store, benchmark):
        """Benchmark context storage performance."""
        content = self.create_large_content(100)  # 100KB
        
        def store_content():
            return store.store(content)
        
        result = benchmark(store_content)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.benchmark
    def test_retrieval_performance(self, store, benchmark):
        """Benchmark context retrieval performance."""
        content = self.create_large_content(100)
        context_id = store.store(content)
        
        def retrieve_content():
            return store.retrieve(context_id)
        
        result = benchmark(retrieve_content)
        assert result == content

    @pytest.mark.benchmark
    def test_serialization_performance(self, store, benchmark):
        """Benchmark serialization performance vs traditional approach."""
        # Store multiple large contexts
        context_ids = []
        for i in range(10):
            content = self.create_structured_content(500)
            context_id = store.store(content)
            context_ids.append(context_id)
        
        def serialize_references():
            # Simulate serializing just references (Context Store approach)
            return json.dumps({"context_refs": context_ids})
        
        result = benchmark(serialize_references)
        serialized_size = len(result.encode('utf-8'))
        assert serialized_size < 1024  # Should be less than 1KB for references

    @pytest.mark.benchmark
    def test_deduplication_performance(self, store, benchmark):
        """Benchmark deduplication performance."""
        content = self.create_large_content(50)
        
        def store_duplicate_content():
            # Store the same content multiple times
            ids = []
            for _ in range(10):
                context_id = store.store(content)
                ids.append(context_id)
            return ids
        
        result = benchmark(store_duplicate_content)
        
        # Verify all IDs are the same (deduplication worked)
        assert len(set(result)) == 1, "Deduplication should result in same ID"

    @pytest.mark.benchmark
    def test_multimodal_storage_performance(self, store, benchmark):
        """Benchmark multimodal content storage."""
        # Create multimodal content
        text_part = MultimodalPart.from_text("This is a text part")
        binary_data = b"BINARY_DATA" * 1000  # 11KB binary
        binary_part = MultimodalPart.from_binary(binary_data, "application/octet-stream")
        
        content = MultimodalContent(parts=[text_part, binary_part])
        
        def store_multimodal():
            return store.store_multimodal_content(content)
        
        result = benchmark(store_multimodal)
        assert isinstance(result, str)

    @pytest.mark.slow
    def test_concurrent_access_performance(self, store):
        """Test performance under concurrent access."""
        content = self.create_large_content(50)
        context_id = store.store(content)
        
        results = []
        errors = []
        
        def worker():
            try:
                start_time = time.time()
                retrieved = store.retrieve(context_id)
                end_time = time.time()
                
                assert retrieved == content
                results.append(end_time - start_time)
            except Exception as e:
                errors.append(e)
        
        # Test with 20 concurrent threads
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(100)]
            for future in futures:
                future.result()
        
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 100
        max_time = max(results)
        avg_time = sum(results) / len(results)
        
        print(f"Concurrent access - Max: {max_time*1000:.2f}ms, Avg: {avg_time*1000:.2f}ms")
        assert max_time < 0.1, f"Max retrieval time too high: {max_time*1000:.2f}ms"

    @pytest.mark.slow
    def test_memory_pressure_performance(self, store):
        """Test performance under memory pressure."""
        # Fill cache beyond capacity
        context_ids = []
        large_content = self.create_large_content(200)  # 200KB each
        
        start_time = time.time()
        
        # Store 150 contexts (cache size is 100)
        for i in range(150):
            content = f"{large_content}_version_{i}"
            context_id = store.store(content)
            context_ids.append((context_id, content))
        
        storage_time = time.time() - start_time
        
        # Test retrieval performance after evictions
        start_time = time.time()
        successful_retrievals = 0
        
        for context_id, original_content in context_ids[-50:]:  # Test last 50
            try:
                retrieved = store.retrieve(context_id)
                assert retrieved == original_content
                successful_retrievals += 1
            except KeyError:
                # Expected for evicted contexts
                pass
        
        retrieval_time = time.time() - start_time
        
        print(f"Memory pressure test - Storage: {storage_time:.2f}s, Retrieval: {retrieval_time:.2f}s")
        print(f"Successful retrievals: {successful_retrievals}/50")
        
        # Verify cache is working (some contexts should be evicted)
        assert successful_retrievals < 50, "Some contexts should have been evicted"
        assert successful_retrievals > 0, "Some contexts should still be available"

    def test_cache_hit_rate_performance(self, store):
        """Test cache hit rate optimization."""
        # Store some content
        contents = []
        context_ids = []
        
        for i in range(20):
            content = self.create_structured_content(100)
            content["id"] = i
            context_id = store.store(content)
            contents.append(content)
            context_ids.append(context_id)
        
        # Access patterns that should improve hit rate
        access_pattern = [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]  # Repeated access
        
        start_time = time.time()
        
        for idx in access_pattern:
            retrieved = store.retrieve(context_ids[idx])
            assert retrieved == contents[idx]
        
        access_time = time.time() - start_time
        # Get cache statistics
        stats = store.get_cache_stats()
        hit_rate = stats["hit_rate"]
        
        print(f"Cache hit rate: {hit_rate:.2%}, Access time: {access_time*1000:.2f}ms")
        
        # Hit rate should be reasonable due to repeated access
        assert hit_rate > 0.3, f"Hit rate too low: {hit_rate:.2%}"


class TestRegressionTests:
    """Regression tests to prevent performance degradation."""

    def test_storage_time_regression(self):
        """Ensure storage time doesn't regress."""
        store = ContextReferenceStore()
        content = "x" * 10000  # 10KB content
        
        times = []
        for _ in range(10):
            start_time = time.time()
            store.store(content)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Storage should be very fast (under 10ms for 10KB)
        assert avg_time < 0.01, f"Storage time regression: {avg_time*1000:.2f}ms"

    def test_retrieval_time_regression(self):
        """Ensure retrieval time doesn't regress."""
        store = ContextReferenceStore()
        content = "x" * 10000  # 10KB content
        context_id = store.store(content)
        
        times = []
        for _ in range(10):
            start_time = time.time()
            retrieved = store.retrieve(context_id)
            end_time = time.time()
            times.append(end_time - start_time)
            assert retrieved == content
        
        avg_time = sum(times) / len(times)
        
        # Retrieval should be very fast (under 5ms for 10KB)
        assert avg_time < 0.005, f"Retrieval time regression: {avg_time*1000:.2f}ms"

    def test_memory_usage_regression(self):
        """Ensure memory usage doesn't regress."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        store = ContextReferenceStore(cache_size=50)
        
        # Store 100 contexts (should trigger evictions)
        for i in range(100):
            content = f"Content {i} " * 1000  # ~10KB each
            store.store(content)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (under 50MB for this test)
        assert memory_increase < 50 * 1024 * 1024, f"Memory usage regression: {memory_increase / 1024 / 1024:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
