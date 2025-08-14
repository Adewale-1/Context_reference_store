#!/usr/bin/env python3
"""
Async Context Reference Store tests.

This module tests the async functionality of the Context Reference Store.
"""

import pytest
import asyncio
import tempfile
from context_store.core.async_context_store import (
    AsyncContextReferenceStore,
    create_async_store,
)
from context_store import MultimodalContent, MultimodalPart, CacheEvictionPolicy


class TestAsyncContextReferenceStore:
    """Test async context store functionality."""

    @pytest.fixture
    def async_store(self):
        """Create an async context store for testing."""
        return create_async_store(cache_size=20)

    @pytest.mark.asyncio
    async def test_async_store_retrieve(self, async_store):
        """Test basic async store and retrieve operations."""
        content = "This is a test context for async operations"

        # Store content asynchronously
        context_id = await async_store.store_async(content)
        assert isinstance(context_id, str)

        # Retrieve content asynchronously
        retrieved = await async_store.retrieve_async(context_id)
        assert retrieved == content

    @pytest.mark.asyncio
    async def test_async_structured_content(self, async_store):
        """Test async operations with structured content."""
        structured_content = {
            "type": "conversation",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
        }
        context_id = await async_store.store_async(structured_content)
        retrieved = await async_store.retrieve_async(context_id)

        assert retrieved == structured_content

    @pytest.mark.asyncio
    async def test_batch_operations(self, async_store):
        """Test batch store and retrieve operations."""
        contents = [
            "First context content",
            "Second context content",
            {"structured": "content", "value": 123},
            "Fourth context content",
        ]

        # Batch store
        context_ids = await async_store.batch_store_async(contents)
        assert len(context_ids) == len(contents)
        assert all(isinstance(cid, str) for cid in context_ids)
        # Batch retrieve
        retrieved_contents = await async_store.batch_retrieve_async(context_ids)
        assert len(retrieved_contents) == len(contents)

        # Verify content matches
        for original, retrieved in zip(contents, retrieved_contents):
            assert original == retrieved

    @pytest.mark.asyncio
    async def test_async_multimodal_content(self, async_store):
        """Test async multimodal content operations."""
        # Create multimodal content with text and binary
        text_part = MultimodalPart.from_text("Description of the binary data")
        binary_data = b"BINARY_TEST_DATA" * 100  # ~1.6KB
        binary_part = MultimodalPart.from_binary(binary_data, "application/test")

        content = MultimodalContent(parts=[text_part, binary_part])
        # Store asynchronously
        context_id = await async_store.store_multimodal_content_async(content)
        # Retrieve asynchronously
        retrieved = await async_store.retrieve_multimodal_content_async(context_id)

        assert isinstance(retrieved, MultimodalContent)
        assert len(retrieved.parts) == 2
        assert retrieved.parts[0].text == "Description of the binary data"
        assert retrieved.parts[1].binary_data == binary_data

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, async_store):
        """Test concurrent async operations."""

        # Create multiple tasks that store content concurrently
        async def store_content(index):
            content = f"Concurrent content {index}"
            context_id = await async_store.store_async(content)
            retrieved = await async_store.retrieve_async(context_id)
            return content, retrieved

        # Run 10 concurrent operations
        tasks = [store_content(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        # Verify all operations completed successfully
        assert len(results) == 10
        for original, retrieved in results:
            assert original == retrieved

    @pytest.mark.asyncio
    async def test_async_stats(self, async_store):
        """Test async statistics retrieval."""
        # Store some content
        await async_store.store_async("Test content 1")
        await async_store.store_async("Test content 2")
        # Get stats asynchronously
        stats = await async_store.get_stats_async()
        assert isinstance(stats, dict)
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "total_contexts" in stats

    @pytest.mark.asyncio
    async def test_async_cleanup(self, async_store):
        """Test async cleanup operations."""
        # Store content with short TTL
        metadata = {"ttl": 1}  # 1 second TTL
        context_id = await async_store.store_async("Short-lived content", metadata)
        # Wait for expiration
        await asyncio.sleep(1.1)
        # Run async cleanup
        cleaned_count = await async_store.cleanup_expired_async()
        # Content should be expired and cleaned
        with pytest.raises(KeyError):
            await async_store.retrieve_async(context_id)

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        async with create_async_store(cache_size=10) as store:
            context_id = await store.store_async("Context manager test")
            retrieved = await store.retrieve_async(context_id)
            assert retrieved == "Context manager test"

        # Store should be properly cleaned up after context exit

    @pytest.mark.asyncio
    async def test_async_warm_contexts(self, async_store):
        """Test async context warming."""
        # Store some contexts
        context_ids = []
        for i in range(5):
            context_id = await async_store.store_async(f"Context {i}")
            context_ids.append(context_id)
        # Warm contexts asynchronously
        await async_store.warm_contexts_async(context_ids[:3])
        # All contexts should still be retrievable
        for context_id in context_ids:
            content = await async_store.retrieve_async(context_id)
            assert content.startswith("Context")

    @pytest.mark.asyncio
    async def test_async_binary_storage(self, async_store):
        """Test async binary data storage and retrieval."""
        # Create large binary data
        large_binary = b"LARGE_BINARY_DATA" * 1000  # ~17KB
        binary_part = MultimodalPart.from_binary(large_binary, "video/mp4")
        content = MultimodalContent(parts=[binary_part])
        # Store with async I/O
        context_id = await async_store.store_multimodal_content_async(content)
        # Retrieve with async I/O
        retrieved = await async_store.retrieve_multimodal_content_async(context_id)

        assert retrieved.parts[0].binary_data == large_binary
        assert retrieved.parts[0].mime_type == "video/mp4"

    @pytest.mark.asyncio
    async def test_concurrent_multimodal_storage(self, async_store):
        """Test concurrent multimodal content storage."""

        async def store_multimodal(index):
            binary_data = f"BINARY_DATA_{index}".encode() * 50
            part = MultimodalPart.from_binary(binary_data, "application/test")
            content = MultimodalContent(parts=[part])
            context_id = await async_store.store_multimodal_content_async(content)
            retrieved = await async_store.retrieve_multimodal_content_async(context_id)
            return binary_data, retrieved.parts[0].binary_data

        # Run concurrent multimodal operations
        tasks = [store_multimodal(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        # Verify all operations completed correctly
        for original, retrieved in results:
            assert original == retrieved

    @pytest.mark.asyncio
    async def test_async_error_handling(self, async_store):
        """Test error handling in async operations."""
        # Test retrieving non-existent context
        with pytest.raises(KeyError):
            await async_store.retrieve_async("non-existent-id")
        # Test retrieing non-existent multimodal content
        with pytest.raises(KeyError):
            await async_store.retrieve_multimodal_content_async(
                "non-existent-multimodal-id"
            )

    @pytest.mark.asyncio
    async def test_async_performance_comparison(self):
        """Test that async operations don't significantly degrade performance."""
        import time

        # Test sync vs async performance for batch operations
        contents = [f"Performance test content {i}" for i in range(20)]
        # Create both sync and async stores
        sync_store = AsyncContextReferenceStore(cache_size=50)
        async with create_async_store(cache_size=50) as async_store:
            # Time sync batch operations
            start_time = time.time()
            sync_ids = []
            for content in contents:
                sync_ids.append(sync_store.store(content))
            sync_time = time.time() - start_time

            # Time async batch operations
            start_time = time.time()
            async_ids = await async_store.batch_store_async(contents)
            async_time = time.time() - start_time
            # Async should be comparable or better for batch operations
            # Allow some overhead for async machinery
            assert (
                async_time < sync_time * 2
            ), f"Async too slow: {async_time:.3f}s vs sync {sync_time:.3f}s"

            print(
                f"Sync batch: {sync_time*1000:.1f}ms, Async batch: {async_time*1000:.1f}ms"
            )


class TestAsyncStoreCreation:
    """Test async store creation utilities."""

    @pytest.mark.asyncio
    async def test_create_async_store_defaults(self):
        """Test creating async store with default parameters."""
        async with create_async_store() as store:
            assert isinstance(store, AsyncContextReferenceStore)

            # Test basic functionality
            context_id = await store.store_async("Default store test")
            retrieved = await store.retrieve_async(context_id)
            assert retrieved == "Default store test"

    @pytest.mark.asyncio
    async def test_create_async_store_custom_params(self):
        """Test creating async store with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            async with create_async_store(
                cache_size=100,
                eviction_policy=CacheEvictionPolicy.LFU,
                use_disk_storage=True,
                binary_cache_dir=temp_dir,
            ) as store:
                # Store multimodal content to test custom parameters
                binary_data = b"CUSTOM_PARAMS_TEST" * 100
                part = MultimodalPart.from_binary(binary_data, "application/test")
                content = MultimodalContent(parts=[part])

                context_id = await store.store_multimodal_content_async(content)
                retrieved = await store.retrieve_multimodal_content_async(context_id)

                assert retrieved.parts[0].binary_data == binary_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
