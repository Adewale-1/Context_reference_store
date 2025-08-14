"""
Async Context Reference Store for Efficient Management of Large Context Windows

This module implements async variants of the Context Reference Store operations
for better integration with async frameworks and improved I/O performance.
"""

import asyncio
import aiofiles
import time
import json
import uuid
import hashlib
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .context_reference_store import (
    ContextReferenceStore,
    ContextMetadata,
    CacheEvictionPolicy,
    MultimodalContent,
    MultimodalPart,
)


class AsyncContextReferenceStore(ContextReferenceStore):
    """
    Async version of ContextReferenceStore with non-blocking I/O operations.

    This class extends the synchronous ContextReferenceStore to provide
    async/await support for better integration with async frameworks.
    """

    def __init__(self, *args, **kwargs):
        """Initialize async context store with same parameters as sync version."""
        super().__init__(*args, **kwargs)
        self._async_locks: Dict[str, asyncio.Lock] = {}
        self._write_semaphore = asyncio.Semaphore(10)

    async def store_async(
        self, content: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Async version of store method.

        Args:
            content: The context content to store
            metadata: Optional metadata about the context

        Returns:
            A reference ID for the stored context
        """
        loop = asyncio.get_event_loop()

        # Run the synchronous store operation in a thread pool
        return await loop.run_in_executor(None, self.store, content, metadata)

    async def retrieve_async(self, context_id: str) -> Any:
        """
        Async version of retrieve method.

        Args:
            context_id: The reference ID for the context

        Returns:
            The context content
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, context_id)

    async def store_multimodal_content_async(
        self,
        content: Union[str, Dict, MultimodalContent, MultimodalPart, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Async version of store_multimodal_content method.

        Args:
            content: Multimodal content to store
            metadata: Optional metadata

        Returns:
            A reference ID for the stored content
        """
        if isinstance(content, (MultimodalContent, MultimodalPart)):
            # Handle binary data asynchronously
            return await self._store_multimodal_async(content, metadata)
        else:
            # Fall back
            return await self.store_async(content, metadata)

    async def _store_multimodal_async(
        self,
        content: Union[MultimodalContent, MultimodalPart],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """Handle multimodal content storage asynchronously."""
        if isinstance(content, MultimodalPart):
            content = MultimodalContent(parts=[content])

        # Process binary data asynchronously
        async with self._write_semaphore:
            return await self._store_content_with_parts_async(content, metadata)

    async def _store_content_with_parts_async(
        self, content: MultimodalContent, metadata
    ):
        """Async version of _store_content_with_parts."""
        content_data = {"role": content.role, "parts": []}

        # Process only binary parts concurrently
        binary_tasks = []
        binary_parts = []
        for part in content.parts:
            if part.binary_data:
                task = self._store_binary_data_async(part.binary_data, part.mime_type)
                binary_tasks.append(task)
                binary_parts.append(part)

        # Get results for binary parts only
        binary_results = await asyncio.gather(*binary_tasks) if binary_tasks else []

        # Build content data structure
        binary_index = 0
        for part in content.parts:
            if part.text:
                content_data["parts"].append({"type": "text", "data": part.text})
            elif part.binary_data:
                binary_hash = binary_results[binary_index]
                content_data["parts"].append(
                    {
                        "type": "binary_ref",
                        "binary_hash": binary_hash,
                        "mime_type": part.mime_type,
                    }
                )
                binary_index += 1
            elif part.file_uri:
                content_data["parts"].append(
                    {
                        "type": "file_uri",
                        "uri": part.file_uri,
                        "mime_type": part.mime_type,
                    }
                )
        # Store the structured content
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._store_structured_multimodal, content_data, metadata, True
        )

    async def _store_binary_data_async(self, data: bytes, mime_type: str) -> str:
        """Store binary data asynchronously."""
        binary_hash = hashlib.sha256(data).hexdigest()
        # Check if already exists
        if binary_hash in self._binary_store:
            self._binary_metadata[binary_hash]["ref_count"] += 1
            return binary_hash
        # Determine storage location
        if len(data) > self._large_binary_threshold and self._use_disk_storage:

            file_path = os.path.join(self._binary_cache_dir, f"{binary_hash}.bin")

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(data)

            self._binary_store[binary_hash] = file_path
            is_disk_stored = True
        else:
            # Store in memory
            self._binary_store[binary_hash] = data
            is_disk_stored = False

        # Update metadata
        self._binary_metadata[binary_hash] = {
            "mime_type": mime_type,
            "size": len(data),
            "ref_count": 1,
            "created_at": time.time(),
            "is_disk_stored": is_disk_stored,
        }

        return binary_hash

    async def retrieve_multimodal_content_async(self, context_id: str):
        """Async version of retrieve_multimodal_content."""
        if context_id not in self._contexts:
            raise KeyError(f"Context ID {context_id} not found")

        if self._metadata[context_id].is_expired():
            self._evict_context(context_id)
            raise KeyError(f"Context ID {context_id} has expired")

        self._metadata[context_id].update_access_stats()
        self._track_access_pattern(context_id)
        self._stats["hits"] += 1

        content_str = self._contexts[context_id]
        metadata = self._metadata[context_id]

        if metadata.content_type == "application/json+multimodal":
            content_data = json.loads(content_str)

            if "role" in content_data:
                return await self._reconstruct_content_async(content_data)
            else:
                return await self._reconstruct_part_async(content_data)
        else:
            return await self.retrieve_async(context_id)

    async def _reconstruct_content_async(self, content_data: Dict):
        """Async version of _reconstruct_content."""
        content = MultimodalContent(role=content_data.get("role", "user"))

        # Process parts concurrently
        tasks = []
        for part_data in content_data.get("parts", []):
            tasks.append(self._reconstruct_part_async(part_data))

        content.parts = await asyncio.gather(*tasks)
        return content

    async def _reconstruct_part_async(self, part_data: Dict):
        """Async version of _reconstruct_part."""
        if part_data["type"] == "text":
            return MultimodalPart.from_text(part_data["data"])
        elif part_data["type"] == "binary_ref":
            binary_data = await self._load_binary_data_async(part_data["binary_hash"])
            return MultimodalPart.from_binary(binary_data, part_data["mime_type"])
        elif part_data["type"] == "file_uri":
            return MultimodalPart.from_file(
                part_data["uri"], part_data.get("mime_type")
            )
        else:
            raise ValueError(f"Unknown part type: {part_data['type']}")

    async def _load_binary_data_async(self, binary_hash: str) -> bytes:
        """Load binary data asynchronously."""
        if binary_hash is None:
            raise KeyError("Binary hash is None")
        if binary_hash not in self._binary_store:
            raise KeyError(f"Binary hash {binary_hash} not found")

        stored_data = self._binary_store[binary_hash]

        if isinstance(stored_data, str):
            # Data is stored on disk
            async with aiofiles.open(stored_data, "rb") as f:
                return await f.read()
        else:
            # Data is in memory
            return stored_data

    async def batch_store_async(
        self, contents: List[Any], metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Store multiple contexts asynchronously in batch.

        Args:
            contents: List of contents to store
            metadata_list: Optional list of metadata for each content

        Returns:
            List of reference IDs
        """
        if metadata_list is None:
            metadata_list = [None] * len(contents)

        # Create tasks for concurrent storage
        tasks = []
        for content, metadata in zip(contents, metadata_list):
            task = self.store_async(content, metadata)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def batch_retrieve_async(self, context_ids: List[str]) -> List[Any]:
        """
        Retrieve multiple contexts asynchronously in batch.

        Args:
            context_ids: List of context IDs to retrieve

        Returns:
            List of retrieved contents
        """
        tasks = [self.retrieve_async(context_id) for context_id in context_ids]
        return await asyncio.gather(*tasks)

    async def cleanup_expired_async(self) -> int:
        """
        Asynchronously clean up expired contexts.

        Returns:
            Number of contexts cleaned up
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._evict_expired_contexts)

    async def get_stats_async(self) -> Dict[str, Any]:
        """Get cache statistics asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_cache_stats)

    async def warm_contexts_async(self, context_ids: List[str]):
        """Mark contexts as warm asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.warm_contexts, context_ids)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        # Cleanup any pending async operations
        if hasattr(self, "_cleanup_thread") and self._cleanup_thread:
            self._stop_cleanup = True
            if self._cleanup_thread.is_alive():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._cleanup_thread.join, 1)


def create_async_store(
    cache_size: int = 50,
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
    **kwargs,
) -> AsyncContextReferenceStore:
    """
    Create an async context store with default settings.

    Args:
        cache_size: Maximum number of contexts to keep in memory
        eviction_policy: Cache eviction policy to use
        **kwargs: Additional arguments for ContextReferenceStore

    Returns:
        Configured AsyncContextReferenceStore instance
    """
    return AsyncContextReferenceStore(
        cache_size=cache_size, eviction_policy=eviction_policy, **kwargs
    )


# Example usage
async def example_async_usage():
    """Example of how to use AsyncContextReferenceStore."""
    async with create_async_store(cache_size=100) as store:
        # Store content asynchronously
        content_id = await store.store_async("Large context content here...")
        retrieved = await store.retrieve_async(content_id)
        # Batch operations
        contents = ["Content 1", "Content 2", "Content 3"]
        ids = await store.batch_store_async(contents)
        retrieved_contents = await store.batch_retrieve_async(ids)
        # Multimodal content
        binary_data = b"Binary content here"
        part = MultimodalPart.from_binary(binary_data, "application/octet-stream")
        content = MultimodalContent(parts=[part])

        multimodal_id = await store.store_multimodal_content_async(content)
        retrieved_multimodal = await store.retrieve_multimodal_content_async(
            multimodal_id
        )

        print(f"Stored {len(ids)} contexts and 1 multimodal content")


if __name__ == "__main__":
    asyncio.run(example_async_usage())
