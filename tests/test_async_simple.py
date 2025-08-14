#!/usr/bin/env python3
"""
Simple async tests to verify basic functionality.
"""

import pytest
import asyncio
from context_store.core.async_context_store import AsyncContextReferenceStore


class TestAsyncBasic:
    """Basic async functionality tests."""

    @pytest.mark.asyncio
    async def test_basic_async_operations(self):
        """Test basic async store and retrieve."""
        store = AsyncContextReferenceStore(cache_size=10)
        # Test async store
        content = "Test async content"
        context_id = await store.store_async(content)
        
        # Test async retrieve
        retrieved = await store.retrieve_async(context_id)
        assert retrieved == content

    @pytest.mark.asyncio
    async def test_batch_async_operations(self):
        """Test batch async operations."""
        store = AsyncContextReferenceStore(cache_size=20)
        
        contents = ["Content 1", "Content 2", "Content 3"]
        
        # Batch store
        context_ids = await store.batch_store_async(contents)
        assert len(context_ids) == 3
        
        # Batch retrieve
        retrieved = await store.batch_retrieve_async(context_ids)
        assert retrieved == contents

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent async operations."""
        store = AsyncContextReferenceStore(cache_size=30)
        
        async def store_and_retrieve(index):
            content = f"Concurrent content {index}"
            context_id = await store.store_async(content)
            retrieved = await store.retrieve_async(context_id)
            return content == retrieved
        
        # Run 5 concurrent operations
        tasks = [store_and_retrieve(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
