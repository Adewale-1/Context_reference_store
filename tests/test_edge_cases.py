#!/usr/bin/env python3
"""
Comprehensive Edge Case Tests for Context Reference Store

This module contains extensive edge case testing to ensure robustness
and reliability of the Context Reference Store under various stress conditions.
"""

import pytest
import json
import time
import threading
import tempfile
import os
import gc
import sys
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from context_store import (
    ContextReferenceStore,
    LargeContextState,
    ContextMetadata,
    CacheEvictionPolicy,
    MultimodalContent,
    MultimodalPart,
)


class TestEdgeCases:
    """Comprehensive edge case testing."""

    def test_extreme_memory_pressure(self):
        """Test behavior under extreme memory pressure."""
        store = ContextReferenceStore(
            cache_size=2,  # Very small cache
            eviction_policy=CacheEvictionPolicy.MEMORY_PRESSURE,
            memory_threshold=0.01,  # Very low threshold
        )
        large_content = "x" * (1024 * 1024)  # 1MB string
        context_ids = []

        for i in range(10):
            context_id = store.store(f"{large_content}_{i}")
            context_ids.append(context_id)
        # Verify store handles memory pressure gracefully
        assert len(context_ids) == 10

        # Some contexts should be evicted due to memory pressure
        cache_stats = store.get_cache_stats()
        assert cache_stats["total_evictions"] > 0

    def test_concurrent_operations_stress(self):
        """Test concurrent operations under stress conditions."""
        store = ContextReferenceStore(cache_size=100)

        def worker(worker_id):
            """Worker function for concurrent testing."""
            results = []
            for i in range(50):
                # Mix of store and retrieve operations
                if i % 2 == 0:
                    content = f"Worker {worker_id} content {i}" * 100
                    context_id = store.store(content)
                    results.append(("store", context_id))
                else:
                    # Try to retrieve a previously stored context
                    if results:
                        _, context_id = results[-1]
                        try:
                            retrieved = store.retrieve(context_id)
                            results.append(
                                ("retrieve", len(retrieved) if retrieved else 0)
                            )
                        except KeyError:
                            results.append(("retrieve", "not_found"))
            return results

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())

        # Verify no corruption or crashes
        assert len(all_results) > 0
        store_operations = [r for r in all_results if r[0] == "store"]
        assert len(store_operations) > 0

    def test_malformed_data_handling(self):
        """Test handling of malformed or invalid data."""
        store = ContextReferenceStore()
        invalid_inputs = [
            None,
            "",
            b"",  
            float("nan"),
            float("inf"),
            {"circular": None},  # Will make circular
            "\x00\x01\x02",  # Binary data in string
            "�" * 1000,  # Invalid unicode
        ]
        # Make circular reference
        invalid_inputs[5]["circular"] = invalid_inputs[5]

        for i, invalid_input in enumerate(invalid_inputs):
            try:
                context_id = store.store(invalid_input)
                retrieved = store.retrieve(context_id)
                if invalid_input == "" or invalid_input == b"":
                    assert retrieved == invalid_input
            except (ValueError, TypeError, RecursionError) as e:
                # Expected for some invalid inputs
                assert isinstance(e, (ValueError, TypeError, RecursionError))

    def test_unicode_edge_cases(self):
        """Test edge cases with Unicode handling."""
        store = ContextReferenceStore()

        unicode_tests = [
            "🚀🎯💡🔧⚡📊🛠️✅❌",  # Emojis
            "こんにちは世界",  # Japanese
            "مرحبا بالعالم",  # Arabic
            "Здравствуй мир",  # Russian
            "🇺🇸🇬🇧🇫🇷🇩🇪🇯🇵",  # Flag emojis
            "\u0001\u0002\u0003",  # Control characters
            "a" * 100000,  # Very long string
            "ñáéíóúüç" * 10000,  # Accented characters
        ]

        for test_string in unicode_tests:
            context_id = store.store(test_string)
            retrieved = store.retrieve(context_id)
            assert retrieved == test_string

    def test_large_context_limits(self):
        """Test behavior with extremely large contexts."""
        store = ContextReferenceStore(
            use_disk_storage=True, large_binary_threshold=1024  # 1KB threshold
        )

        # Test progressively larger contexts
        sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB

        for size in sizes:
            large_content = "x" * size
            context_id = store.store(large_content)
            retrieved = store.retrieve(context_id)
            assert len(retrieved) == size
            assert retrieved == large_content

    def test_multimodal_edge_cases(self):
        """Test edge cases with multimodal content."""
        store = ContextReferenceStore()

        # Test empty multimodal content
        empty_multimodal = MultimodalContent(role="user", parts=[])
        context_id = store.store(empty_multimodal)
        retrieved = store.retrieve(context_id)
        assert isinstance(retrieved, MultimodalContent)
        assert len(retrieved.parts) == 0

        # Test multimodal with mixed content types
        mixed_parts = [
            MultimodalPart(text="Text part"),
            MultimodalPart(binary_data=b"Binary data"),
            MultimodalPart(json_data={"key": "value"}),
            MultimodalPart(text=""),  # Empty text
            MultimodalPart(binary_data=b""),  # Empty binary
        ]

        mixed_multimodal = MultimodalContent(role="assistant", parts=mixed_parts)
        context_id = store.store(mixed_multimodal)
        retrieved = store.retrieve(context_id)

        assert isinstance(retrieved, MultimodalContent)
        assert len(retrieved.parts) == 5
        assert retrieved.parts[0].text == "Text part"
        assert retrieved.parts[1].binary_data == b"Binary data"
        assert retrieved.parts[2].json_data == {"key": "value"}

    def test_cache_eviction_edge_cases(self):
        """Test edge cases in cache eviction policies."""
        # Test LRU with frequent access patterns
        lru_store = ContextReferenceStore(
            cache_size=3, eviction_policy=CacheEvictionPolicy.LRU
        )
        ids = []
        for i in range(5):
            context_id = lru_store.store(f"content_{i}")
            ids.append(context_id)
        for _ in range(10):
            lru_store.retrieve(ids[0])
        cache_stats = lru_store.get_cache_stats()
        assert cache_stats["total_evictions"] > 0

        # Test TTL with rapid expiration
        ttl_store = ContextReferenceStore(
            cache_size=10,
            eviction_policy=CacheEvictionPolicy.TTL,
            ttl_check_interval=1,  # 1 second
        )
        # Store with short TTL
        context_id = ttl_store.store("ttl_content", ttl_seconds=1)
        # Should be available immediately
        retrieved = ttl_store.retrieve(context_id)
        assert retrieved == "ttl_content"
        time.sleep(2)
        try:
            ttl_store.retrieve(context_id)
            # May or may not be evicted yet, depends on background cleanup
        except KeyError:
            pass  # Expected if TTL cleanup occurred

    def test_serialization_edge_cases(self):
        """Test edge cases in serialization/deserialization."""
        store = ContextReferenceStore()
        # Test deeply nested structures
        deep_dict = {"level": 0}
        current = deep_dict
        for i in range(100):  # Create deep nesting
            current["next"] = {"level": i + 1}
            current = current["next"]

        context_id = store.store(deep_dict)
        retrieved = store.retrieve(context_id)

        # Verify deep structure is preserved
        current_retrieved = retrieved
        for i in range(100):
            assert current_retrieved["level"] == i
            if "next" in current_retrieved:
                current_retrieved = current_retrieved["next"]

    def test_compression_edge_cases(self):
        """Test edge cases in compression functionality."""
        store = ContextReferenceStore(enable_compression=True)

        # Test content that doesn't compress well
        random_data = os.urandom(1024)  # Random bytes
        context_id = store.store(random_data)
        retrieved = store.retrieve(context_id)
        assert retrieved == random_data
        # Test highly compressible content
        repetitive_data = "a" * 10000
        context_id = store.store(repetitive_data)
        retrieved = store.retrieve(context_id)
        assert retrieved == repetitive_data

        # Test content at compression threshold
        threshold_data = "x" * 1023  # Just under default threshold
        context_id = store.store(threshold_data)
        retrieved = store.retrieve(context_id)
        assert retrieved == threshold_data

    def test_state_corruption_recovery(self):
        """Test recovery from state corruption scenarios."""
        store = ContextReferenceStore()
        state = LargeContextState(context_store=store)

        valid_ref = state.store_context("valid_data")
        # Simulate corruption by directly modifying internal state
        if hasattr(state, "_context_references"):
            # Add invalid reference
            state._context_references["corrupted"] = "invalid_context_id"

        # Should handle corruption gracefully
        try:
            state.get_context("corrupted")
        except KeyError:
            pass  # Expected for corrupted reference

        # Valid data should still be accessible
        retrieved = state.get_context(valid_ref)
        assert retrieved == "valid_data"

    def test_metadata_edge_cases(self):
        """Test edge cases with metadata handling."""
        store = ContextReferenceStore()
        # Test metadata with edge case values
        edge_metadata = ContextMetadata(
            content_type="",  # Empty string
            size_bytes=0,  # Zero size
            created_at=0.0,  # Epoch time
            last_accessed=float("inf"),  # Infinity
            access_count=-1,  # Negative count
            ttl_seconds=0,  # Zero TTL
            compression_algorithm="nonexistent", 
            is_multimodal=None,  # None value
        )

        context_id = store.store("test_content", metadata=edge_metadata)
        retrieved_metadata = store.get_metadata(context_id)
        # Verify metadata is preserved or sanitized appropriately
        assert retrieved_metadata is not None

    def test_disk_storage_edge_cases(self):
        """Test edge cases with disk storage functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ContextReferenceStore(
                use_disk_storage=True,
                binary_cache_dir=temp_dir,
                large_binary_threshold=100, 
            )
            # Test with read-only directory
            readonly_dir = os.path.join(temp_dir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # Read-only
            store_readonly = ContextReferenceStore(
                use_disk_storage=True,
                binary_cache_dir=readonly_dir,
                large_binary_threshold=100,
            )
            try:
                large_content = b"x" * 1000
                context_id = store_readonly.store(large_content)

            except (PermissionError, OSError):
                pass  # Expected for read-only directory

            # Test disk space exhaustion simulation
            # (Note: This is hard to test reliably without actually filling disk)
            # We can test with very large content that might trigger disk issues
            huge_content = b"x" * (10 * 1024 * 1024)  # 10MB
            try:
                context_id = store.store(huge_content)
                retrieved = store.retrieve(context_id)
                assert len(retrieved) == len(huge_content)
            except (OSError, MemoryError):
                pass  # Expected if system can't handle the size

    def test_threading_safety_edge_cases(self):
        """Test threading safety under edge conditions."""
        store = ContextReferenceStore(cache_size=50)

        # Test rapid concurrent access to same context
        def rapid_access_worker():
            context_id = store.store("shared_content")
            results = []
            for _ in range(100):
                try:
                    retrieved = store.retrieve(context_id)
                    results.append(len(retrieved))
                except Exception as e:
                    results.append(str(e))
            return results

        # Run many threads with rapid access
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(rapid_access_worker) for _ in range(20)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())

        # Verify no threading issues
        successful_retrievals = [r for r in all_results if isinstance(r, int)]
        assert len(successful_retrievals) > 0

    def test_memory_leak_prevention(self):
        """Test that operations don't cause memory leaks."""
        store = ContextReferenceStore(cache_size=10)
        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many operations
        for i in range(1000):
            content = f"content_{i}" * 100
            context_id = store.store(content)
            if i % 2 == 0:
                store.retrieve(context_id)
        gc.collect()

        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects

        assert (
            object_growth < 10000
        ), f"Potential memory leak: {object_growth} new objects"

    def test_error_propagation(self):
        """Test proper error propagation and handling."""
        store = ContextReferenceStore()

        # Test invalid context ID
        with pytest.raises(KeyError):
            store.retrieve("nonexistent_id")
        # Test invalid eviction policy
        with pytest.raises((ValueError, TypeError)):
            ContextReferenceStore(eviction_policy="invalid_policy")
        # Test invalid cache size
        with pytest.raises((ValueError, TypeError)):
            ContextReferenceStore(cache_size=-1)

        # Test invalid memory threshold
        with pytest.raises((ValueError, TypeError)):
            ContextReferenceStore(memory_threshold=2.0)  # > 1.0

    def test_context_lifecycle_edge_cases(self):
        """Test edge cases in context lifecycle management."""
        store = ContextReferenceStore(cache_size=3)

        # Test rapid store/delete cycles
        for cycle in range(10):
            context_ids = []
            for i in range(5):
                context_id = store.store(f"cycle_{cycle}_content_{i}")
                context_ids.append(context_id)
            for i, context_id in enumerate(context_ids):
                if i % 2 == 0:
                    store.delete(context_id)
            for i, context_id in enumerate(context_ids):
                if i % 2 == 0:
                    with pytest.raises(KeyError):
                        store.retrieve(context_id)
                else:
                    # Should still be accessible
                    retrieved = store.retrieve(context_id)
                    assert retrieved is not None

    @pytest.mark.parametrize(
        "policy",
        [
            CacheEvictionPolicy.LRU,
            CacheEvictionPolicy.LFU,
            CacheEvictionPolicy.TTL,
            CacheEvictionPolicy.MEMORY_PRESSURE,
        ],
    )
    def test_eviction_policy_consistency(self, policy):
        """Test consistency across different eviction policies."""
        store = ContextReferenceStore(
            cache_size=5, eviction_policy=policy, memory_threshold=0.8
        )

        # Store more contexts than cache size
        context_ids = []
        for i in range(10):
            context_id = store.store(f"content_{i}")
            context_ids.append(context_id)

        # Verify evictions occurred
        cache_stats = store.get_cache_stats()
        assert cache_stats["total_evictions"] > 0

        # Verify some contexts are still accessible
        accessible_count = 0
        for context_id in context_ids:
            try:
                store.retrieve(context_id)
                accessible_count += 1
            except KeyError:
                pass

        # Should have some contexts accessible based on cache size
        assert accessible_count > 0
        assert accessible_count <= 5  # Cache size limit

    def test_stress_test_combined(self):
        """Combined stress test with multiple edge conditions."""
        store = ContextReferenceStore(
            cache_size=20,
            eviction_policy=CacheEvictionPolicy.LRU,
            use_disk_storage=True,
            enable_compression=True,
        )

        def stress_worker(worker_id):
            """Worker that combines multiple stress conditions."""
            results = {"stored": 0, "retrieved": 0, "errors": 0}

            for i in range(100):
                try:
                    # Mix different content types and sizes
                    if i % 4 == 0:
                        content = "x" * (1024 * (i % 10 + 1))
                    elif i % 4 == 1:
                        content = os.urandom(1024) 
                    elif i % 4 == 2:
                        content = {
                            "worker": worker_id,
                            "iteration": i,
                            "data": ["a"] * 100,
                        }
                    else:
                        parts = [
                            MultimodalPart(text=f"Worker {worker_id} text {i}"),
                            MultimodalPart(binary_data=os.urandom(256)),
                            MultimodalPart(json_data={"iteration": i}),
                        ]
                        content = MultimodalContent(role="user", parts=parts)

                    context_id = store.store(content)
                    results["stored"] += 1

                    # Randomly retrieve some content
                    if i % 3 == 0:
                        retrieved = store.retrieve(context_id)
                        results["retrieved"] += 1

                except Exception as e:
                    results["errors"] += 1

            return results

        # Run stress test with multiple workers
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(8)]
            total_results = {"stored": 0, "retrieved": 0, "errors": 0}

            for future in as_completed(futures):
                worker_results = future.result()
                for key in total_results:
                    total_results[key] += worker_results[key]

        # Verify reasonable success rate
        total_operations = total_results["stored"] + total_results["retrieved"]
        error_rate = total_results["errors"] / (
            total_operations + total_results["errors"]
        )

        assert error_rate < 0.1, f"Error rate too high: {error_rate:.2%}"
        assert total_results["stored"] > 0
        assert total_results["retrieved"] > 0


class TestAdapterEdgeCases:
    """Edge case tests for framework adapters."""

    def test_adapter_import_failures(self):
        """Test adapter behavior when dependencies are missing."""
        try:
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

        except ImportError:
            pass 

        try:
            from context_store.adapters.llamaindex_adapter import (
                LlamaIndexContextAdapter,
            )
        except ImportError:
            pass

        try:
            from context_store.adapters.composio_adapter import ComposioContextAdapter
        except ImportError:
            pass

    def test_adapter_mock_integration(self):
        """Test adapter functionality with mocked dependencies."""
        with patch(
            "context_store.adapters.langchain_adapter.LANGCHAIN_AVAILABLE", False
        ):
            try:
                from context_store.adapters.langchain_adapter import (
                    LangChainContextAdapter,
                )

                adapter = LangChainContextAdapter()
                assert adapter is not None
            except ImportError:
                pass 


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
