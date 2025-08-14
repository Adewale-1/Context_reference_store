

"""
Enhanced State class for handling large context windows efficiently.

This module extends the BaseState class to provide efficient handling of large context
windows (1M-2M tokens) using a reference-based approach.
"""

import json
from typing import Dict, Any, Optional, List

from .base_state import BaseState
from .context_reference_store import ContextReferenceStore


class LargeContextState(BaseState):
    """
    Enhanced State class for efficient handling of large contexts.

    This class extends BaseState to handle large contexts efficiently by:
    - Storing references to contexts instead of the contexts themselves
    - Providing methods to resolve references when needed
    - Supporting external context caching features for cost optimization
    - Handling both text and structured contexts
    """

    def __init__(
        self,
        value: Optional[Dict[str, Any]] = None,
        delta: Optional[Dict[str, Any]] = None,
        context_store: Optional[ContextReferenceStore] = None,
    ):
        """
        Initialize the Large Context State.

        Args:
            value: The current value of the state dict
            delta: The delta change to the current value that hasn't been committed
            context_store: Context reference store to use
        """
        super().__init__(value=value or {}, delta=delta or {})
        self._context_store = context_store or ContextReferenceStore()

    def add_large_context(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        key: str = "context_ref",
    ) -> str:
        """
        Add large context to the state using reference-based storage.

        Args:
            content: The context content to store (string or structured data)
            metadata: Optional metadata about the context
            key: The key to store the reference under in the state

        Returns:
            The reference ID for the stored context
        """
        context_id = self._context_store.store(content, metadata)
        self[key] = context_id
        return context_id

    def get_context(self, ref_key: str = "context_ref") -> Any:
        """
        Retrieve context from a reference stored in the state.

        Args:
            ref_key: The key where the context reference is stored

        Returns:
            The context content

        Raises:
            KeyError: If the reference key is not found in state
        """
        if ref_key not in self:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = self[ref_key]
        return self._context_store.retrieve(context_id)

    def with_cache_hint(self, ref_key: str = "context_ref") -> Dict[str, Any]:
        """
        Get a cache hint object for external caching systems (e.g., Gemini API).

        This allows external systems to cache the context for reuse, which can
        significantly reduce costs when reusing the same context multiple times.

        Args:
            ref_key: The key where the context reference is stored

        Returns:
            A cache hint object suitable for passing to external APIs

        Raises:
            KeyError: If the reference key is not found in state
        """
        if ref_key not in self:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = self[ref_key]
        return self._context_store.get_cache_hint(context_id)

    def store_structured_context(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        key: str = "structured_context_ref",
    ) -> str:
        """
        Store structured data (JSON/dict) in the context store.

        Args:
            data: The structured data to store
            metadata: Optional metadata about the context
            key: The key to store the reference under in the state

        Returns:
            The reference ID for the stored context
        """
        if metadata is None:
            metadata = {}

        # Mark this as structured data if not already specified
        if "content_type" not in metadata:
            metadata["content_type"] = "application/json"

        return self.add_large_context(data, metadata, key)

    def store_multimodal_context(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        key: str = "multimodal_context_ref",
    ) -> str:
        """
        Store multimodal content (images, audio, video) in the context store.
        Args:
            content: The multimodal content to store
            metadata: Optional metadata about the context
            key: The key to store the reference under in the state

        Returns:
            The reference ID for the stored context
        """
        context_id = self._context_store.store_multimodal_content(content, metadata)
        self[key] = context_id
        return context_id

    def get_multimodal_context(self, ref_key: str = "multimodal_context_ref") -> Any:
        """
        Retrieve multimodal context from a reference stored in the state.

        Args:
            ref_key: The key where the multimodal context reference is stored

        Returns:
            The multimodal context content

        Raises:
            KeyError: If the reference key is not found in state
        """
        if ref_key not in self:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = self[ref_key]
        return self._context_store.retrieve_multimodal_content(context_id)

    def get_context_metadata(self, ref_key: str = "context_ref") -> Dict[str, Any]:
        """
        Get metadata for a context reference stored in the state.

        Args:
            ref_key: The key where the context reference is stored

        Returns:
            The context metadata as a dictionary

        Raises:
            KeyError: If the reference key is not found in state
        """
        if ref_key not in self:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = self[ref_key]
        metadata = self._context_store.get_metadata(context_id)

        # Convert to dict for easier use
        return {
            "content_type": metadata.content_type,
            "token_count": metadata.token_count,
            "created_at": metadata.created_at,
            "last_accessed": metadata.last_accessed,
            "access_count": metadata.access_count,
            "tags": metadata.tags,
            "cache_id": metadata.cache_id,
            "cached_until": metadata.cached_until,
            "is_structured": metadata.is_structured,
            "priority": metadata.priority,
            "frequency_score": metadata.frequency_score,
        }

    def set_context_priority(self, ref_key: str, priority: int):
        """
        Set priority for a context reference (higher priority = kept longer).

        Args:
            ref_key: The key where the context reference is stored
            priority: Priority level (higher values = higher priority)

        Raises:
            KeyError: If the reference key is not found in state
        """
        if ref_key not in self:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = self[ref_key]
        self._context_store.set_context_priority(context_id, priority)

    def warm_context(self, ref_key: str):
        """
        Mark a context as warm (should be kept in cache).

        Args:
            ref_key: The key where the context reference is stored

        Raises:
            KeyError: If the reference key is not found in state
        """
        if ref_key not in self:
            raise KeyError(f"Context reference key '{ref_key}' not found in state")

        context_id = self[ref_key]
        self._context_store.warm_contexts([context_id])

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics from the context store.

        Returns:
            Dictionary containing cache performance metrics
        """
        return self._context_store.get_cache_stats()

    def get_multimodal_stats(self) -> Dict[str, Any]:
        """
        Get detailed multimodal storage statistics.

        Returns:
            Dictionary containing multimodal storage metrics
        """
        return self._context_store.get_multimodal_stats()

    def list_context_references(self) -> List[str]:
        """
        List all context reference keys stored in this state.

        Returns:
            List of keys that contain context references
        """
        context_refs = []
        for key in self.keys():
            try:
                value = self[key]
                if isinstance(value, str):
                    self._context_store.get_metadata(value)
                    context_refs.append(key)
            except (KeyError, ValueError):
                continue
        return context_refs

    def cleanup_unused_contexts(self):
        """
        Clean up any unused contexts in the context store.

        This can help free up memory from contexts that are no longer referenced.
        """
        self._context_store.cleanup_unused_binaries()

    @property
    def context_store(self) -> ContextReferenceStore:
        """
        Get the underlying context reference store.

        This provides access to advanced features and direct control over the store.

        Returns:
            The ContextReferenceStore instance
        """
        return self._context_store
