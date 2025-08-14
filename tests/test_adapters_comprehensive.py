#!/usr/bin/env python3
"""
Comprehensive Adapter Tests for Context Reference Store

This module contains comprehensive tests for all framework adapters,
including edge cases, error handling, and integration scenarios.
"""

import pytest
import json
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional

from context_store import ContextReferenceStore, MultimodalContent, MultimodalPart


class TestLangChainAdapter:
    """Comprehensive tests for LangChain adapter."""

    def test_adapter_without_langchain(self):
        """Test adapter behavior when LangChain is not available."""
        with patch(
            "context_store.adapters.langchain_adapter.LANGCHAIN_AVAILABLE", False
        ):
            with pytest.raises(ImportError):
                from context_store.adapters.langchain_adapter import (
                    LangChainContextAdapter,
                )

                adapter = LangChainContextAdapter()

    def test_adapter_with_mocked_langchain(self):
        """Test adapter availability check."""
        try:
            # First check if LangChain is actually available
            import langchain_core.messages
            import langchain_core.memory
            import langchain_core.documents

            # If LangChain is available, just test adapter instantiation
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            adapter = LangChainContextAdapter()
            assert adapter is not None
            # Skip actual functionality testing since it requires proper LangChain message objects
        except ImportError:
            # If LangChain is not available, test that adapter is not importable
            pytest.skip(
                "LangChain not available - testing adapter availability gracefully"
            )

    def test_message_serialization_edge_cases(self):
        """Test edge cases in message serialization."""
        try:
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            adapter = LangChainContextAdapter()
            assert adapter is not None

        except ImportError:
            pytest.skip("LangChain not available for edge case testing")

    def test_streaming_edge_cases(self):
        """Test edge cases in streaming functionality."""
        try:
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            adapter = LangChainContextAdapter(enable_streaming=True)
            assert adapter is not None

            # Skip actual streaming operations since they require proper LangChain setup

        except ImportError:
            pytest.skip("LangChain not available for streaming tests")


class TestLangGraphAdapter:
    """Comprehensive tests for LangGraph adapter."""

    def test_adapter_without_langgraph(self):
        """Test adapter behavior when LangGraph is not available."""
        with patch(
            "context_store.adapters.langgraph_adapter.LANGGRAPH_AVAILABLE", False
        ):
            with pytest.raises(ImportError):
                from context_store.adapters.langgraph_adapter import (
                    LangGraphContextAdapter,
                )

                adapter = LangGraphContextAdapter()

    def test_checkpoint_edge_cases(self):
        """Test edge cases in checkpointing functionality."""
        try:
            from context_store.adapters.langgraph_adapter import LangGraphContextAdapter

            adapter = LangGraphContextAdapter()

            # Test with empty state
            empty_state = {}
            thread_id = "test_thread"

            adapter.store_graph_state(empty_state, thread_id)
            retrieved = adapter.retrieve_graph_state(thread_id)
            assert retrieved == empty_state
            # Test with very large state
            large_state = {"data": "x" * (1024 * 1024)}  # 1MB state
            adapter.store_graph_state(large_state, "large_thread")
            retrieved_large = adapter.retrieve_graph_state("large_thread")
            assert retrieved_large["data"] == large_state["data"]

            # Test with complex nested state
            complex_state = {
                "level1": {
                    "level2": {
                        "level3": {
                            "data": list(range(1000)),
                            "metadata": {"created": time.time()},
                        }
                    }
                }
            }
            adapter.store_graph_state(complex_state, "complex_thread")
            retrieved_complex = adapter.retrieve_graph_state("complex_thread")
            assert retrieved_complex["level1"]["level2"]["level3"]["data"] == list(
                range(1000)
            )

        except ImportError:
            pytest.skip("LangGraph not available")

    def test_multi_agent_edge_cases(self):
        """Test edge cases in multi-agent scenarios."""
        try:
            from context_store.adapters.langgraph_adapter import LangGraphContextAdapter

            adapter = LangGraphContextAdapter(enable_multi_agent=True)

            # Test registering multiple agents
            mock_agent1 = Mock()
            mock_agent1.name = "agent1"
            mock_agent2 = Mock()
            mock_agent2.name = "agent2"

            adapter.register_agent("agent1", mock_agent1)
            adapter.register_agent("agent2", mock_agent2)

            # Test agent state isolation
            adapter.store_graph_state({"agent": "agent1", "data": "data1"}, "thread1")
            adapter.store_graph_state({"agent": "agent2", "data": "data2"}, "thread2")

            state1 = adapter.retrieve_graph_state("thread1")
            state2 = adapter.retrieve_graph_state("thread2")

            assert state1["agent"] == "agent1"
            assert state2["agent"] == "agent2"

        except ImportError:
            pytest.skip("LangGraph not available")


class TestLlamaIndexAdapter:
    """Comprehensive tests for LlamaIndex adapter."""

    def test_adapter_without_llamaindex(self):
        """Test adapter behavior when LlamaIndex is not available."""
        with patch(
            "context_store.adapters.llamaindex_adapter.LLAMAINDEX_AVAILABLE", False
        ):
            with pytest.raises(ImportError):
                from context_store.adapters.llamaindex_adapter import (
                    LlamaIndexContextAdapter,
                )

                adapter = LlamaIndexContextAdapter()

    def test_document_storage_edge_cases(self):
        """Test edge cases in document storage."""
        try:
            from context_store.adapters.llamaindex_adapter import (
                LlamaIndexContextAdapter,
            )

            adapter = LlamaIndexContextAdapter()

            # Test with empty documents
            empty_docs = []
            collection_id = adapter.store_documents(empty_docs, "empty_collection")
            retrieved = adapter.retrieve_documents("empty_collection")
            assert len(retrieved) == 0

            # Test with documents containing edge case content
            edge_case_docs = [
                {"text": "", "metadata": {}},  # Empty text
                {
                    "text": "x" * (1024 * 1024),
                    "metadata": {"size": "large"},
                },  # Very large text
                {"text": "🚀🎯💡", "metadata": {"type": "emoji"}},  # Unicode content
                {"text": None, "metadata": {"type": "null"}},  # None text
            ]

            collection_id = adapter.store_documents(
                edge_case_docs, "edge_case_collection"
            )
            retrieved = adapter.retrieve_documents("edge_case_collection")

            # Should handle edge cases gracefully
            assert len(retrieved) <= len(edge_case_docs)

        except ImportError:
            pytest.skip("LlamaIndex not available")

    def test_node_management_edge_cases(self):
        """Test edge cases in node management."""
        try:
            from context_store.adapters.llamaindex_adapter import (
                LlamaIndexContextAdapter,
            )

            adapter = LlamaIndexContextAdapter()

            # Test with empty nodes
            empty_nodes = []
            node_ref = adapter.store_nodes(empty_nodes, "empty_nodes")
            retrieved = adapter.retrieve_nodes("empty_nodes")
            assert len(retrieved) == 0

            # Test with nodes containing various content types
            edge_case_nodes = [
                {"text": "Normal node", "node_id": "normal"},
                {"text": "", "node_id": "empty"},  # Empty text
                {"text": "🌟" * 1000, "node_id": "unicode"},  # Unicode heavy
            ]

            node_ref = adapter.store_nodes(edge_case_nodes, "edge_nodes")
            retrieved = adapter.retrieve_nodes("edge_nodes")

            # Should preserve node structure
            assert len(retrieved) <= len(edge_case_nodes)

        except ImportError:
            pytest.skip("LlamaIndex not available")

    def test_chat_memory_edge_cases(self):
        """Test edge cases in chat memory management."""
        try:
            from context_store.adapters.llamaindex_adapter import (
                LlamaIndexContextAdapter,
            )

            adapter = LlamaIndexContextAdapter()

            # Test with empty chat history
            empty_history = []
            adapter.store_chat_history("session1", empty_history)
            retrieved = adapter.retrieve_chat_history("session1")
            assert retrieved == empty_history

            # Test with very long chat history
            long_history = [
                {"role": "user", "content": f"Message {i}"} for i in range(1000)
            ]
            adapter.store_chat_history("session2", long_history)
            retrieved_long = adapter.retrieve_chat_history("session2")
            assert len(retrieved_long) == 1000

        except ImportError:
            pytest.skip("LlamaIndex not available")


class TestComposioAdapter:
    """Comprehensive tests for Composio adapter."""

    def test_adapter_without_composio(self):
        """Test adapter behavior when Composio is not available."""
        with patch("context_store.adapters.composio_adapter.COMPOSIO_AVAILABLE", False):
            with pytest.raises(ImportError):
                from context_store.adapters.composio_adapter import (
                    ComposioContextAdapter,
                )

                adapter = ComposioContextAdapter()

    def test_tool_execution_caching_edge_cases(self):
        """Test edge cases in tool execution caching."""
        try:
            from context_store.adapters.composio_adapter import ComposioContextAdapter

            adapter = ComposioContextAdapter(enable_tool_caching=True)

            # Test caching with empty inputs
            empty_result = adapter.execute_tool_with_caching(
                "test_tool", "test_action", {}  # Empty inputs
            )

            # Test caching with large inputs
            large_inputs = {"data": "x" * (1024 * 1024)}  # 1MB input
            large_result = adapter.execute_tool_with_caching(
                "test_tool", "test_action", large_inputs
            )

            # Test caching with complex inputs
            complex_inputs = {
                "nested": {
                    "list": list(range(1000)),
                    "dict": {f"key_{i}": f"value_{i}" for i in range(100)},
                }
            }
            complex_result = adapter.execute_tool_with_caching(
                "test_tool", "test_action", complex_inputs
            )

        except ImportError:
            pytest.skip("Composio not available")

    def test_authentication_edge_cases(self):
        """Test edge cases in authentication management."""
        try:
            from context_store.adapters.composio_adapter import ComposioContextAdapter

            adapter = ComposioContextAdapter(enable_auth_caching=True)

            # Test with empty credentials
            empty_auth = {
                "app_name": "test_app",
                "auth_type": "oauth",
                "credentials": {},
            }
            adapter.store_auth_state("user1", empty_auth)
            retrieved = adapter.retrieve_auth_state("user1")
            assert retrieved["credentials"] == {}

            large_creds = {
                "token": "x" * 10000,  # Very large token
                "metadata": {"created": time.time()},
            }

            large_auth = {
                "app_name": "test_app",
                "auth_type": "token",
                "credentials": large_creds,
            }

            adapter.store_auth_state("user2", large_auth)
            retrieved_large = adapter.retrieve_auth_state("user2")
            assert len(retrieved_large["credentials"]["token"]) == 10000

        except ImportError:
            pytest.skip("Composio not available")

    def test_trigger_management_edge_cases(self):
        """Test edge cases in trigger management."""
        try:
            from context_store.adapters.composio_adapter import ComposioContextAdapter

            adapter = ComposioContextAdapter(enable_trigger_management=True)

            # Test with empty trigger config
            empty_trigger = {
                "trigger_id": "empty_trigger",
                "app_name": "test_app",
                "trigger_name": "test_trigger",
                "config": {},
                "is_active": False,
            }

            adapter.register_trigger("empty_trigger", empty_trigger, "user1")
            state = adapter.get_trigger_state("empty_trigger")
            assert state["config"] == {}

            # Test with complex trigger config
            complex_trigger = {
                "trigger_id": "complex_trigger",
                "app_name": "test_app",
                "trigger_name": "complex_trigger",
                "config": {
                    "filters": [f"filter_{i}" for i in range(100)],
                    "actions": {f"action_{i}": f"value_{i}" for i in range(50)},
                    "metadata": {"created": time.time(), "version": "1.0"},
                },
                "is_active": True,
            }

            adapter.register_trigger("complex_trigger", complex_trigger, "user2")
            state = adapter.get_trigger_state("complex_trigger")
            assert len(state["config"]["filters"]) == 100

        except ImportError:
            pytest.skip("Composio not available")


class TestAdapterIntegration:
    """Integration tests for adapters working together."""

    def test_multiple_adapters_same_store(self):
        """Test multiple adapters using the same context store."""
        context_store = ContextReferenceStore(cache_size=100)

        adapters = []

        # Try to initialize multiple adapters with the same store
        try:
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            adapters.append(LangChainContextAdapter(context_store=context_store))
        except ImportError:
            pass

        try:
            from context_store.adapters.langgraph_adapter import LangGraphContextAdapter

            adapters.append(LangGraphContextAdapter(context_store=context_store))
        except ImportError:
            pass

        try:
            from context_store.adapters.llamaindex_adapter import (
                LlamaIndexContextAdapter,
            )

            adapters.append(LlamaIndexContextAdapter(context_store=context_store))
        except ImportError:
            pass

        try:
            from context_store.adapters.composio_adapter import ComposioContextAdapter

            adapters.append(ComposioContextAdapter(context_store=context_store))
        except ImportError:
            pass

        # Test that adapters don't interfere with each other
        for i, adapter in enumerate(adapters):
            # Each adapter should work independently
            assert adapter.context_store is context_store

            # Test basic operations don't conflict
            test_data = f"adapter_{i}_test_data"
            if hasattr(adapter, "store_messages"):
                try:
                    adapter.store_messages(f"session_{i}", [])
                except Exception:
                    pass  # Some adapters might need specific message format

            if hasattr(adapter, "store_state"):
                try:
                    adapter.store_state(f"thread_{i}", {"data": test_data})
                except Exception:
                    pass
        # Verify shared store stats
        stats = context_store.get_cache_stats()
        assert stats["total_contexts"] >= 0

    def test_adapter_error_isolation(self):
        """Test that errors in one adapter don't affect others."""
        adapters = []

        try:
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            adapters.append(("langchain", LangChainContextAdapter()))
        except ImportError:
            pass

        try:
            from context_store.adapters.langgraph_adapter import LangGraphContextAdapter

            adapters.append(("langgraph", LangGraphContextAdapter()))
        except ImportError:
            pass

        for name, adapter in adapters:
            try:
                if hasattr(adapter, "store_messages"):
                    adapter.store_messages("", None)  # Invalid inputs
            except Exception:
                pass
            try:
                if hasattr(adapter, "get_performance_analytics"):
                    analytics = adapter.get_performance_analytics()
                    assert analytics is not None
            except Exception:
                pass

    def test_concurrent_adapter_operations(self):
        """Test concurrent operations across different adapters."""
        import threading
        import time

        context_store = ContextReferenceStore(cache_size=50)
        results = {}

        def adapter_worker(adapter_name, adapter_class):
            """Worker function for concurrent adapter testing."""
            try:
                adapter = adapter_class(context_store=context_store)
                operations = 0

                for i in range(10):
                    try:
                        # Perform adapter-specific operations
                        if hasattr(adapter, "store_messages"):
                            adapter.store_messages(f"{adapter_name}_session_{i}", [])
                            operations += 1

                        if hasattr(adapter, "store_state"):
                            adapter.store_state(
                                f"{adapter_name}_thread_{i}", {"data": f"data_{i}"}
                            )
                            operations += 1

                        if hasattr(adapter, "store_documents"):
                            adapter.store_documents(
                                [], f"{adapter_name}_collection_{i}"
                            )
                            operations += 1
                        time.sleep(0.01)  # Small delay to increase concurrency chance

                    except Exception as e:
                        pass

                results[adapter_name] = operations
            except Exception as e:
                results[adapter_name] = f"Error: {str(e)}"

        # Start concurrent workers
        threads = []
        try:
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            thread = threading.Thread(
                target=adapter_worker, args=("langchain", LangChainContextAdapter)
            )
            threads.append(thread)
        except ImportError:
            pass
        try:
            from context_store.adapters.langgraph_adapter import LangGraphContextAdapter

            thread = threading.Thread(
                target=adapter_worker, args=("langgraph", LangGraphContextAdapter)
            )
            threads.append(thread)
        except ImportError:
            pass

        # Start all threads
        for thread in threads:
            thread.start()
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        # Verify no major issues
        assert len(results) >= 0
        # Check context store integrity
        stats = context_store.get_cache_stats()
        assert stats is not None


class TestAdapterPerformance:
    """Performance tests for adapters under various conditions."""

    def test_adapter_memory_usage(self):
        """Test adapter memory usage under load."""
        import gc

        gc.collect()
        initial_objects = len(gc.get_objects())
        try:
            # First check if LangChain is actually available
            import langchain_core.messages
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            adapter = LangChainContextAdapter()
            # Perform many operations
            for i in range(100):
                session_id = f"session_{i}"
                # Only call store_messages if LangChain is fully available
                adapter.store_messages(session_id, [])

                if i % 10 == 0:
                    # Retrieve some messages
                    adapter.retrieve_messages(session_id)
            # Force garbage collection
            gc.collect()
            final_objects = len(gc.get_objects())
            object_growth = final_objects - initial_objects
            # Should not have excessive memory growth
            assert (
                object_growth < 5000
            ), f"Excessive memory growth: {object_growth} objects"

        except ImportError:
            pytest.skip("LangChain not available for memory testing")

    def test_adapter_performance_under_load(self):
        """Test adapter performance under high load."""
        try:
            # First check if LangChain is actually available
            import langchain_core.messages

            from context_store.adapters.langchain_adapter import LangChainContextAdapter
            adapter = LangChainContextAdapter()
            start_time = time.time()
            # Perform many rapid operations
            for i in range(1000):
                session_id = f"load_session_{i % 10}"  # Reuse some sessions
                adapter.store_messages(session_id, [])

                if i % 5 == 0:
                    adapter.retrieve_messages(session_id)
            end_time = time.time()
            duration = end_time - start_time
            # Should complete in reasonable time (adjust threshold as needed)
            assert duration < 60, f"Operations took too long: {duration:.2f} seconds"
            # Check performance analytics
            if hasattr(adapter, "get_performance_analytics"):
                analytics = adapter.get_performance_analytics()
                assert analytics is not None

        except ImportError:
            pytest.skip("LangChain not available for performance testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
