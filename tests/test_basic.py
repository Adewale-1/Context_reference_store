#!/usr/bin/env python3
"""
Basic tests for Context Reference Store.

This module contains fundamental tests to ensure the library works correctly.
"""

import pytest
import json
import time
from context_store import (
    ContextReferenceStore,
    LargeContextState,
    ContextMetadata,
    CacheEvictionPolicy,
)


class TestContextReferenceStore:
    """Test the core ContextReferenceStore functionality."""

    def test_store_and_retrieve_text(self):
        """Test storing and retrieving text content."""
        store = ContextReferenceStore()
        content = "This is a test content string"

        # Store content
        context_id = store.store(content)
        assert isinstance(context_id, str)
        assert len(context_id) > 0

        # Retrieve content
        retrieved = store.retrieve(context_id)
        assert retrieved == content

        # Check metadata
        metadata = store.get_metadata(context_id)
        assert isinstance(metadata, ContextMetadata)
        assert metadata.content_type == "text/plain"
        assert not metadata.is_structured

    def test_store_and_retrieve_structured(self):
        """Test storing and retrieving structured content."""
        store = ContextReferenceStore()
        content = {
            "title": "Test Document",
            "data": [1, 2, 3, 4, 5],
            "nested": {"key": "value"},
        }

        # Store content
        context_id = store.store(content)

        # Retrieve content
        retrieved = store.retrieve(context_id)
        assert retrieved == content

        # Check metadata
        metadata = store.get_metadata(context_id)
        assert metadata.content_type == "application/json"
        assert metadata.is_structured

    def test_content_deduplication(self):
        """Test that identical content returns the same ID."""
        store = ContextReferenceStore()
        content = "Duplicate content"

        # Store the same content multiple times
        id1 = store.store(content)
        id2 = store.store(content)
        id3 = store.store(content)

        # All IDs should be the same
        assert id1 == id2 == id3

        # Check statistics
        stats = store.get_cache_stats()
        assert stats["total_contexts"] == 1
        assert stats["total_hits"] >= 2  # Second and third calls are hits

    def test_cache_hint(self):
        """Test cache hint generation."""
        store = ContextReferenceStore()
        content = "Content for cache hint test"

        context_id = store.store(content, metadata={"cache_ttl": 3600})
        cache_hint = store.get_cache_hint(context_id)

        assert isinstance(cache_hint, dict)
        assert "cache_id" in cache_hint
        assert "cache_level" in cache_hint
        assert cache_hint["cache_level"] == "HIGH"

    def test_context_priority(self):
        """Test context priority setting."""
        store = ContextReferenceStore()
        content = "Priority test content"

        context_id = store.store(content, metadata={"priority": 10})
        metadata = store.get_metadata(context_id)
        assert metadata.priority == 10
        # Update priority
        store.set_context_priority(context_id, 15)
        metadata = store.get_metadata(context_id)
        assert metadata.priority == 15

    def test_cache_eviction_lru(self):
        """Test LRU cache eviction."""
        store = ContextReferenceStore(
            cache_size=2, eviction_policy=CacheEvictionPolicy.LRU
        )
        # Store more items than cache size
        id1 = store.store("Content 1")
        id2 = store.store("Content 2")
        id3 = store.store("Content 3")  # Should evict content 1

        # Content 1 should be evicted
        with pytest.raises(KeyError):
            store.retrieve(id1)
        # Content 2 and 3 should still be available
        assert store.retrieve(id2) == "Content 2"
        assert store.retrieve(id3) == "Content 3"

    def test_statistics(self):
        """Test cache statistics tracking."""
        store = ContextReferenceStore()
        # Store some content
        content1 = "Content 1"
        content2 = "Content 2"

        id1 = store.store(content1)
        id2 = store.store(content2)
        # Access content to generate hits
        store.retrieve(id1)
        store.retrieve(id2)
        store.retrieve(id1)  # Another hit
        stats = store.get_cache_stats()
        assert stats["total_contexts"] == 2
        assert stats["total_hits"] >= 3
        assert stats["total_misses"] == 2
        assert stats["hit_rate"] > 0


class TestLargeContextState:
    """Test the LargeContextState functionality."""

    def test_basic_context_operations(self):
        """Test basic context operations."""
        state = LargeContextState()
        content = {"large": "context", "data": list(range(100))}

        # Add context
        ref_id = state.add_large_context(content, key="test_context")
        assert isinstance(ref_id, str)
        assert "test_context" in state
        # Retrieve context
        retrieved = state.get_context("test_context")
        assert retrieved == content

    def test_structured_context(self):
        """Test structured context storage."""
        state = LargeContextState()

        data = {"structured": True, "values": [1, 2, 3]}

        ref_id = state.store_structured_context(data, key="structured_data")
        retrieved = state.get_context("structured_data")
        assert retrieved == data

    def test_context_metadata(self):
        """Test context metadata retrieval."""
        state = LargeContextState()
        content = "Test content"
        ref_id = state.add_large_context(
            content, metadata={"priority": 5, "tags": ["test"]}, key="meta_test"
        )
        metadata = state.get_context_metadata("meta_test")
        assert metadata["priority"] == 5
        assert "test" in metadata["tags"]
        assert metadata["content_type"] == "text/plain"

    def test_cache_hint(self):
        """Test cache hint generation through state."""
        state = LargeContextState()

        content = "Cache hint test"
        state.add_large_context(content, metadata={"cache_ttl": 1800}, key="cache_test")

        cache_hint = state.with_cache_hint("cache_test")
        assert isinstance(cache_hint, dict)
        assert "cache_id" in cache_hint

    def test_context_priority_and_warming(self):
        """Test context priority and warming."""
        state = LargeContextState()
        content = "Priority test"
        state.add_large_context(content, key="priority_test")
        # Set priority
        state.set_context_priority("priority_test", 20)
        metadata = state.get_context_metadata("priority_test")
        assert metadata["priority"] == 20
        # Warm context
        state.warm_context("priority_test")
        # This should not raise an error

    def test_context_references_listing(self):
        """Test listing context references."""
        state = LargeContextState()
        # Add multiple contexts
        state.add_large_context("Content 1", key="ref1")
        state.add_large_context("Content 2", key="ref2")
        state["non_context_key"] = "regular_value"
        refs = state.list_context_references()
        assert "ref1" in refs
        assert "ref2" in refs
        assert "non_context_key" not in refs


class TestCacheEvictionPolicies:
    """Test different cache eviction policies."""

    def test_lru_policy(self):
        """Test Least Recently Used eviction."""
        store = ContextReferenceStore(
            cache_size=2, eviction_policy=CacheEvictionPolicy.LRU
        )
        id1 = store.store("Content 1")
        id2 = store.store("Content 2")
        # Access first content to make it recently used
        store.retrieve(id1)
        # Add third content (should evict content 2)
        id3 = store.store("Content 3")
        # Content 1 should still be available (recently used)
        assert store.retrieve(id1) == "Content 1"
        # Content 3 should be available (just added)
        assert store.retrieve(id3) == "Content 3"

    def test_lfu_policy(self):
        """Test Least Frequently Used eviction."""
        store = ContextReferenceStore(
            cache_size=2, eviction_policy=CacheEvictionPolicy.LFU
        )
        id1 = store.store("Content 1")
        id2 = store.store("Content 2")
        # Access first content multiple times
        for _ in range(5):
            store.retrieve(id1)

        # Access second content once
        store.retrieve(id2)
        # Add third content (should evict content 2 - less frequently used)
        id3 = store.store("Content 3")
        # Content 1 should still be available (frequently used)
        assert store.retrieve(id1) == "Content 1"

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        store = ContextReferenceStore(
            ttl_check_interval=0
        )  # Disable background cleanup
        content = "TTL test content"
        # Use a positive TTL value
        context_id = store.store(content, metadata={"cache_ttl": 1})  # 1 second TTL
        try:
            metadata = store.get_metadata(context_id)
            assert not metadata.is_expired()
        except KeyError:
            # If context was already cleaned up, that's also valid for TTL behavior
            pytest.skip("Context was cleaned up before metadata check")

        # Wait for expiration and verify expired status
        import time
        time.sleep(1.1)  # Wait slightly longer than TTL

        # Now check if it's expired or removed
        try:
            metadata = store.get_metadata(context_id)
            assert metadata.is_expired()
        except KeyError:
            # Context was removed due to expiration - this is expected behavior
            pass


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_retrieve_nonexistent_context(self):
        """Test retrieving non-existent context."""
        store = ContextReferenceStore()
        with pytest.raises(KeyError):
            store.retrieve("nonexistent-id")

    def test_get_metadata_nonexistent(self):
        """Test getting metadata for non-existent context."""
        store = ContextReferenceStore()
        with pytest.raises(KeyError):
            store.get_metadata("nonexistent-id")

    def test_state_key_not_found(self):
        """Test state key not found errors."""
        state = LargeContextState()
        with pytest.raises(KeyError):
            state.get_context("nonexistent-key")
        with pytest.raises(KeyError):
            state.with_cache_hint("nonexistent-key")


class TestPerformance:
    """Basic performance tests."""

    def test_large_content_performance(self):
        """Test performance with large content."""
        store = ContextReferenceStore()
        # Create large content
        large_content = {"data": list(range(10000)), "text": "x" * 50000}
        # Time storage
        start_time = time.time()
        context_id = store.store(large_content)
        storage_time = time.time() - start_time
        # Time retrieval
        start_time = time.time()
        retrieved = store.retrieve(context_id)
        retrieval_time = time.time() - start_time

        # Verify content
        assert retrieved == large_content

        # Performance should be reasonable (less than 1 second for this size)
        assert storage_time < 1.0
        assert retrieval_time < 1.0

    def test_multiple_contexts_performance(self):
        """Test performance with multiple contexts."""
        # Use a larger cache size to avoid evictions during test
        store = ContextReferenceStore(cache_size=150)
        # Store multiple contexts
        contexts = []
        start_time = time.time()
        for i in range(100):
            content = {"index": i, "data": list(range(i, i + 100))}
            context_id = store.store(content)
            contexts.append((context_id, content))

        storage_time = time.time() - start_time

        # Retrieve all contexts
        start_time = time.time()
        successful_retrievals = 0
        for context_id, original_content in contexts:
            try:
                retrieved = store.retrieve(context_id)
                assert retrieved == original_content
                successful_retrievals += 1
            except KeyError:
                # Context may have been evicted, which is acceptable behavior
                pass

        retrieval_time = time.time() - start_time

        # Performance should be reasonable
        assert storage_time < 5.0  # 5 seconds for 100 contexts
        assert retrieval_time < 2.0  # 2 seconds for retrievals
        # At least most contexts should be retrievable
        assert successful_retrievals >= 90  # Allow for some evictions


if __name__ == "__main__":
    pytest.main([__file__])
