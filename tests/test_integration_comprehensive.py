#!/usr/bin/env python3
"""
Comprehensive Integration Tests

This module contains integration tests that verify the complete
system works correctly with all components working together.
"""

import pytest
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch

from context_store import (
    ContextReferenceStore,
    LargeContextState,
    MultimodalContent,
    MultimodalPart,
    CacheEvictionPolicy,
)


class TestFullSystemIntegration:
    """Integration tests for the complete system."""

    def test_complete_workflow_integration(self):
        """Test a complete workflow using all major components."""
        # Initialize core store with advanced features
        store = ContextReferenceStore(
            cache_size=100,
            eviction_policy=CacheEvictionPolicy.LRU,
            enable_compression=True,
            use_disk_storage=True,
            memory_threshold=0.8,
        )
        token_manager = None
        semantic_analyzer = None
        dashboard = None

        try:
            from context_store.optimization.token_manager import create_token_manager

            token_manager = create_token_manager(context_store=store)
            print("Token Manager initialized")
        except ImportError:
            print("Token Manager not available")

        try:
            from context_store.semantic.semantic_analyzer import (
                create_semantic_analyzer,
            )

            semantic_analyzer = create_semantic_analyzer(context_store=store)
            print("Semantic Analyzer initialized")
        except ImportError:
            print("Semantic Analyzer not available")

        try:
            from context_store.monitoring.tui_dashboard import create_dashboard

            dashboard = create_dashboard(
                store, token_manager=token_manager, semantic_analyzer=semantic_analyzer
            )
            print("TUI Dashboard initialized")
        except ImportError:
            print("TUI Dashboard not available")

        # Store diverse content types
        print("\nStoring diverse content")

        # Text content
        text_contexts = []
        for i in range(10):
            content = f"Integration test text content {i}: " + (
                "artificial intelligence and machine learning " * 20
            )
            context_id = store.store(content)
            text_contexts.append(context_id)

        # Binary content
        binary_contexts = []
        for i in range(5):
            binary_data = os.urandom(10240)  # 10KB random data
            context_id = store.store(binary_data)
            binary_contexts.append(context_id)

        # JSON content
        json_contexts = []
        for i in range(5):
            json_data = {
                "id": i,
                "metadata": {"created": time.time(), "type": "integration_test"},
                "data": list(range(i * 10, (i + 1) * 10)),
                "nested": {"level1": {"level2": f"value_{i}"}},
            }
            context_id = store.store(json_data)
            json_contexts.append(context_id)

        # Multimodal content
        multimodal_contexts = []
        for i in range(3):
            parts = [
                MultimodalPart(text=f"Multimodal text part {i}"),
                MultimodalPart(binary_data=os.urandom(1024)),
                MultimodalPart(json_data={"part": i, "type": "multimodal"}),
            ]
            multimodal = MultimodalContent(role="user", parts=parts)
            context_id = store.store(multimodal)
            multimodal_contexts.append(context_id)

        all_contexts = (
            text_contexts + binary_contexts + json_contexts + multimodal_contexts
        )
        print(f"Stored {len(all_contexts)} contexts of various types")

        # Test optimization features
        print("\nPhase 2: Testing optimization features")

        if token_manager:
            # Test token optimization
            try:
                from context_store.optimization.token_manager import (
                    TokenBudget,
                    OptimizationStrategy,
                )

                budget = TokenBudget(max_tokens=5000)
                optimization_result = token_manager.optimize_contexts(
                    context_ids=text_contexts,
                    token_budget=budget,
                    strategy=OptimizationStrategy.BALANCED,
                )

                print(
                    f"Token optimization: {len(optimization_result.selected_contexts)}/{len(text_contexts)} contexts selected"
                )
                print(f"Estimated tokens: {optimization_result.estimated_tokens}")

            except Exception as e:
                print(f"Token optimization error: {e}")

        if semantic_analyzer:
            try:
                if len(text_contexts) >= 2:
                    similar_matches = semantic_analyzer.find_similar_contexts(
                        text_contexts[0], text_contexts[1:5], similarity_threshold=0.5
                    )
                    print(f"Found {len(similar_matches)} similar contexts")

                # Test clustering
                if len(text_contexts) >= 3:
                    clusters = semantic_analyzer.cluster_contexts(
                        text_contexts[:5], num_clusters=2
                    )
                    print(f"Created {len(clusters)} clusters")

                # Test deduplication
                dedup_result = semantic_analyzer.deduplicate_contexts(
                    text_contexts[:5], similarity_threshold=0.9
                )
                if dedup_result:
                    print(
                        f"Deduplication: {len(dedup_result.deduplicated_contexts)}/{len(text_contexts[:5])} unique contexts"
                    )

            except Exception as e:
                print(f"Semantic analysis error: {e}")

        # Test adapter integrations
        print("\nPhase 3: Testing adapter integrations")

        # Test LangChain adapter
        try:
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            langchain_adapter = LangChainContextAdapter(context_store=store)

            # Store mock messages
            mock_messages = [
                Mock(
                    content="Hello",
                    additional_kwargs={},
                    __class__=Mock(__name__="HumanMessage"),
                ),
                Mock(
                    content="Hi there!",
                    additional_kwargs={},
                    __class__=Mock(__name__="AIMessage"),
                ),
            ]

            langchain_adapter.store_messages("integration_session", mock_messages)
            retrieved_messages = langchain_adapter.retrieve_messages(
                "integration_session"
            )

            print(
                f"LangChain adapter: Stored and retrieved {len(retrieved_messages)} messages"
            )

        except ImportError:
            print("LangChain adapter not available")
        except Exception as e:
            print(f"LangChain adapter error: {e}")

        # Test LangGraph adapter
        try:
            from context_store.adapters.langgraph_adapter import LangGraphContextAdapter

            langgraph_adapter = LangGraphContextAdapter(context_store=store)

            # Store mock state
            test_state = {
                "current_step": "integration_test",
                "data": {"processed": True, "timestamp": time.time()},
                "history": ["step1", "step2", "step3"],
            }

            langgraph_adapter.store_state("integration_thread", test_state)
            retrieved_state = langgraph_adapter.get_state("integration_thread")

            print(
                f"LangGraph adapter: Stored and retrieved state with {len(retrieved_state)} keys"
            )

        except ImportError:
            print("LangGraph adapter not available")
        except Exception as e:
            print(f"LangGraph adapter error: {e}")

        # Test LlamaIndex adapter
        try:
            from context_store.adapters.llamaindex_adapter import (
                LlamaIndexContextAdapter,
            )

            llamaindex_adapter = LlamaIndexContextAdapter(context_store=store)

            # Store mock documents
            mock_docs = [
                {"text": "Integration test document 1", "metadata": {"id": 1}},
                {"text": "Integration test document 2", "metadata": {"id": 2}},
            ]

            doc_collection = llamaindex_adapter.store_documents(
                mock_docs, "integration_collection"
            )
            retrieved_docs = llamaindex_adapter.retrieve_documents(
                "integration_collection"
            )

            print(
                f"LlamaIndex adapter: Stored and retrieved {len(retrieved_docs)} documents"
            )

        except ImportError:
            print("LlamaIndex adapter not available")
        except Exception as e:
            print(f"LlamaIndex adapter error: {e}")

        # Test Composio adapter
        try:
            from context_store.adapters.composio_adapter import ComposioContextAdapter

            composio_adapter = ComposioContextAdapter(context_store=store)

            # Test caching functionality
            mock_result = composio_adapter.execute_tool_with_caching(
                "integration_tool", "test_action", {"input": "integration_test_data"}
            )

            print("Composio adapter: Tool execution caching tested")

        except ImportError:
            print("Composio adapter not available")
        except Exception as e:
            print(f"Composio adapter error: {e}")

        # Test system performance and reliability
        print("\nSystem performance validation")

        # Test retrieval performance
        retrieval_start = time.time()
        successful_retrievals = 0

        for context_id in all_contexts:
            try:
                retrieved = store.retrieve(context_id)
                if retrieved is not None:
                    successful_retrievals += 1
            except Exception as e:
                print(f"Retrieval error: {e}")

        retrieval_time = time.time() - retrieval_start
        retrieval_rate = len(all_contexts) / retrieval_time
        success_rate = successful_retrievals / len(all_contexts)

        print(f"Retrieval performance: {retrieval_rate:.2f} retrievals/sec")
        print(f"Success rate: {success_rate:.2%}")

        # Get comprehensive statistics
        cache_stats = store.get_cache_stats()
        print(f"Cache statistics:")
        print(f"  - Total contexts: {cache_stats.get('total_contexts', 0)}")
        print(f"  - Cache hits: {cache_stats.get('total_hits', 0)}")
        print(f"  - Cache misses: {cache_stats.get('total_misses', 0)}")
        print(f"  - Evictions: {cache_stats.get('total_evictions', 0)}")
        print(f"  - Hit rate: {cache_stats.get('hit_rate', 0):.2%}")

        # Validate data integrity
        print("\nData integrity validation")

        integrity_errors = 0

        # Validate text contexts
        for i, context_id in enumerate(text_contexts):
            try:
                retrieved = store.retrieve(context_id)
                expected_content = f"Integration test text content {i}: " + (
                    "artificial intelligence and machine learning " * 20
                )
                if retrieved != expected_content:
                    integrity_errors += 1
            except Exception:
                integrity_errors += 1

        # Validate binary contexts
        for context_id in binary_contexts:
            try:
                retrieved = store.retrieve(context_id)
                if not isinstance(retrieved, bytes):
                    integrity_errors += 1
            except Exception:
                integrity_errors += 1

        # Validate JSON contexts
        for i, context_id in enumerate(json_contexts):
            try:
                retrieved = store.retrieve(context_id)
                if not isinstance(retrieved, dict) or retrieved.get("id") != i:
                    integrity_errors += 1
            except Exception:
                integrity_errors += 1

        # Validate multimodal contexts
        for context_id in multimodal_contexts:
            try:
                retrieved = store.retrieve(context_id)
                if (
                    not isinstance(retrieved, MultimodalContent)
                    or len(retrieved.parts) != 3
                ):
                    integrity_errors += 1
            except Exception:
                integrity_errors += 1

        integrity_rate = 1 - (integrity_errors / len(all_contexts))
        print(f"Data integrity: {integrity_rate:.2%} ({integrity_errors} errors)")

        # Final validation
        print("\nIntegration test results:")
        print(f"  - Total contexts stored: {len(all_contexts)}")
        print(f"  - Retrieval success rate: {success_rate:.2%}")
        print(f"  - Data integrity rate: {integrity_rate:.2%}")
        print(f"  - Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")

        # Assert overall system health
        assert (
            success_rate > 0.95
        ), f"Retrieval success rate too low: {success_rate:.2%}"
        assert (
            integrity_rate > 0.95
        ), f"Data integrity rate too low: {integrity_rate:.2%}"
        assert (
            retrieval_rate > 50
        ), f"Retrieval performance too low: {retrieval_rate:.2f}/sec"

    def test_adapter_interoperability(self):
        """Test that different adapters can work together."""
        store = ContextReferenceStore(cache_size=100)
        adapters = {}

        # Initialize available adapters
        try:
            from context_store.adapters.langchain_adapter import LangChainContextAdapter

            adapters["langchain"] = LangChainContextAdapter(context_store=store)
        except ImportError:
            pass

        try:
            from context_store.adapters.langgraph_adapter import LangGraphContextAdapter

            adapters["langgraph"] = LangGraphContextAdapter(context_store=store)
        except ImportError:
            pass

        try:
            from context_store.adapters.llamaindex_adapter import (
                LlamaIndexContextAdapter,
            )

            adapters["llamaindex"] = LlamaIndexContextAdapter(context_store=store)
        except ImportError:
            pass

        try:
            from context_store.adapters.composio_adapter import ComposioContextAdapter

            adapters["composio"] = ComposioContextAdapter(context_store=store)
        except ImportError:
            pass

        if not adapters:
            pytest.skip("No adapters available for interoperability test")

        print(
            f"Testing interoperability with {len(adapters)} adapters: {list(adapters.keys())}"
        )

        # Each adapter stores data
        adapter_data = {}

        for name, adapter in adapters.items():
            try:
                if name == "langchain":
                    mock_messages = [
                        Mock(
                            content=f"Message from {name}",
                            additional_kwargs={},
                            __class__=Mock(__name__="HumanMessage"),
                        )
                    ]
                    adapter.store_messages(f"{name}_session", mock_messages)
                    adapter_data[name] = f"{name}_session"

                elif name == "langgraph":
                    test_state = {"adapter": name, "data": f"test_data_{name}"}
                    adapter.store_state(f"{name}_thread", test_state)
                    adapter_data[name] = f"{name}_thread"

                elif name == "llamaindex":
                    docs = [
                        {"text": f"Document from {name}", "metadata": {"source": name}}
                    ]
                    adapter.store_documents(docs, f"{name}_collection")
                    adapter_data[name] = f"{name}_collection"

                elif name == "composio":
                    # Test auth state storage
                    auth_state = {
                        "app_name": f"{name}_app",
                        "auth_type": "test",
                        "credentials": {"token": f"{name}_token"},
                    }
                    adapter.store_auth_state(f"{name}_user", auth_state)
                    adapter_data[name] = f"{name}_user"

            except Exception as e:
                print(f"Error storing data for {name}: {e}")

        # Verify shared store statistics
        initial_stats = store.get_cache_stats()
        print(
            f"Shared store contexts after adapter operations: {initial_stats.get('total_contexts', 0)}"
        )

        # Each adapter retrieves its data
        retrieval_success = {}

        for name, adapter in adapters.items():
            try:
                if name == "langchain" and name in adapter_data:
                    messages = adapter.retrieve_messages(adapter_data[name])
                    retrieval_success[name] = len(messages) > 0

                elif name == "langgraph" and name in adapter_data:
                    state = adapter.get_state(adapter_data[name])
                    retrieval_success[name] = state.get("adapter") == name

                elif name == "llamaindex" and name in adapter_data:
                    docs = adapter.retrieve_documents(adapter_data[name])
                    retrieval_success[name] = len(docs) > 0

                elif name == "composio" and name in adapter_data:
                    auth_state = adapter.retrieve_auth_state(adapter_data[name])
                    retrieval_success[name] = (
                        auth_state.get("app_name") == f"{name}_app"
                    )

            except Exception as e:
                print(f"Error retrieving data for {name}: {e}")
                retrieval_success[name] = False

        # Test cross-adapter data isolation
        isolation_test_passed = True

        for name1, adapter1 in adapters.items():
            for name2, adapter2 in adapters.items():
                if name1 != name2 and name1 in adapter_data and name2 in adapter_data:
                    try:
                        # Try to access data from one adapter using another
                        if name1 == "langchain" and name2 == "langchain":
                            # Can't easily test cross-access for same adapter type
                            continue

                        # Data should be isolated (this test is conceptual)
                        # The actual isolation is handled by the adapters using different
                        # key naming schemes and data formats

                    except Exception:
                        # Expected - data should be isolated
                        pass

        print(f"Adapter retrieval success: {retrieval_success}")

        # Final store statistics
        final_stats = store.get_cache_stats()
        print(f"Final shared store statistics:")
        print(f"  - Total contexts: {final_stats.get('total_contexts', 0)}")
        print(f"  - Hit rate: {final_stats.get('hit_rate', 0):.2%}")

        # Verify adapters worked together successfully
        successful_adapters = sum(retrieval_success.values())
        total_adapters = len(adapters)

        assert (
            successful_adapters > 0
        ), "No adapters successfully stored and retrieved data"

        success_rate = successful_adapters / total_adapters
        print(f"Adapter interoperability success rate: {success_rate:.2%}")

    def test_end_to_end_workflow_simulation(self):
        """Simulate a realistic end-to-end workflow."""
        print("🎯 Simulating realistic end-to-end workflow")

        # Initialize system
        store = ContextReferenceStore(
            cache_size=50,
            eviction_policy=CacheEvictionPolicy.LRU,
            enable_compression=True,
            use_disk_storage=True,
        )

        # Simulate a multi-agent conversation system
        print("\n📝 Simulating multi-agent conversation system")

        # Agent 1: Data Analysis Agent
        analysis_contexts = []
        for i in range(5):
            data = {
                "dataset": f"dataset_{i}",
                "analysis": {
                    "mean": i * 10.5,
                    "std": i * 2.3,
                    "samples": list(range(i * 100, (i + 1) * 100)),
                },
                "conclusions": f"Analysis {i} shows positive trends in data quality",
            }
            context_id = store.store(data)
            analysis_contexts.append(context_id)

        # Agent 2: Document Processing Agent
        document_contexts = []
        for i in range(3):
            document = {
                "title": f"Research Paper {i}",
                "content": f"This paper discusses advanced topics in AI research " * 50,
                "metadata": {
                    "authors": [f"Author_{j}" for j in range(3)],
                    "citations": i * 10,
                    "year": 2020 + i,
                },
                "sections": [f"Section {j}: Content..." for j in range(5)],
            }
            context_id = store.store(document)
            document_contexts.append(context_id)

        # Agent 3: Multimodal Content Agent
        multimodal_contexts = []
        for i in range(3):
            # Simulate processing of images, text, and metadata
            parts = [
                MultimodalPart(text=f"Image analysis result {i}"),
                MultimodalPart(binary_data=os.urandom(5120)),  # Simulated image data
                MultimodalPart(
                    json_data={
                        "image_metadata": {
                            "width": 1920,
                            "height": 1080,
                            "format": "PNG",
                            "analysis": {"objects_detected": i * 5, "confidence": 0.95},
                        }
                    }
                ),
            ]
            multimodal = MultimodalContent(role="assistant", parts=parts)
            context_id = store.store(multimodal)
            multimodal_contexts.append(context_id)

        all_workflow_contexts = (
            analysis_contexts + document_contexts + multimodal_contexts
        )

        print(f"Stored {len(all_workflow_contexts)} contexts across 3 simulated agents")

        # Simulate cross-agent data sharing
        print("\n🔄 Simulating cross-agent data sharing")

        shared_analysis = []

        # Analysis agent shares findings with document agent
        for analysis_id in analysis_contexts:
            try:
                analysis_data = store.retrieve(analysis_id)
                # Document agent creates summary
                summary = {
                    "source": "analysis_agent",
                    "summary": f"Dataset {analysis_data.get('dataset')} analysis complete",
                    "key_metrics": {
                        "mean": analysis_data.get("analysis", {}).get("mean"),
                        "sample_count": len(
                            analysis_data.get("analysis", {}).get("samples", [])
                        ),
                    },
                }
                summary_id = store.store(summary)
                shared_analysis.append(summary_id)
            except Exception as e:
                print(f"Error in cross-agent sharing: {e}")

        # Simulate optimization during workflow
        print("\n⚡ Testing optimization during workflow")

        try:
            from context_store.optimization.token_manager import (
                create_token_manager,
                TokenBudget,
                OptimizationStrategy,
            )

            token_manager = create_token_manager(context_store=store)

            # Optimize for memory-constrained scenario
            budget = TokenBudget(max_tokens=3000)
            optimization = token_manager.optimize_contexts(
                context_ids=all_workflow_contexts,
                token_budget=budget,
                strategy=OptimizationStrategy.QUALITY_FIRST,
            )

            print(
                f"Optimization selected {len(optimization.selected_contexts)}/{len(all_workflow_contexts)} contexts"
            )
            print(
                f"Estimated token usage: {optimization.estimated_tokens}/{budget.max_tokens}"
            )

        except ImportError:
            print("Token optimization not available - skipping")
        except Exception as e:
            print(f"Optimization error: {e}")

        # Test semantic analysis on workflow data
        print("\n🧠 Testing semantic analysis on workflow data")

        try:
            from context_store.semantic.semantic_analyzer import (
                create_semantic_analyzer,
            )

            semantic_analyzer = create_semantic_analyzer(context_store=store)

            # Find related documents
            if len(document_contexts) >= 2:
                similar_docs = semantic_analyzer.find_similar_contexts(
                    document_contexts[0],
                    document_contexts[1:],
                    similarity_threshold=0.3,
                )
                print(f"Found {len(similar_docs)} semantically similar documents")

            # Cluster analysis results
            if len(analysis_contexts) >= 3:
                analysis_clusters = semantic_analyzer.cluster_contexts(
                    analysis_contexts, num_clusters=2
                )
                print(
                    f"Clustered analysis results into {len(analysis_clusters)} groups"
                )

        except ImportError:
            print("Semantic analysis not available - skipping")
        except Exception as e:
            print(f"Semantic analysis error: {e}")

        # Simulate workflow completion and cleanup
        print("\n🧹 Simulating workflow completion")

        # Get final statistics
        final_stats = store.get_cache_stats()

        # Simulate partial cleanup (remove some intermediate results)
        cleaned_contexts = 0
        for context_id in shared_analysis[:2]:  # Clean up some shared analysis
            try:
                store.delete(context_id)
                cleaned_contexts += 1
            except Exception as e:
                print(f"Cleanup error: {e}")

        print(f"Cleaned up {cleaned_contexts} intermediate contexts")

        # Final validation
        remaining_contexts = 0
        for context_id in analysis_contexts + document_contexts + multimodal_contexts:
            try:
                store.retrieve(context_id)
                remaining_contexts += 1
            except KeyError:
                pass  # Expected for cleaned contexts

        print(f"\n✅ Workflow simulation results:")
        print(f"  - Original contexts: {len(all_workflow_contexts)}")
        print(f"  - Remaining contexts: {remaining_contexts}")
        print(f"  - Cache hit rate: {final_stats.get('hit_rate', 0):.2%}")
        print(f"  - Total evictions: {final_stats.get('total_evictions', 0)}")

        # Verify workflow completed successfully
        assert (
            remaining_contexts >= len(all_workflow_contexts) - 5
        ), "Too many contexts lost during workflow"

        print("🎉 End-to-end workflow simulation completed successfully!")


class TestSystemRobustness:
    """Test system robustness under various conditions."""

    def test_graceful_degradation(self):
        """Test system behavior when components fail gracefully."""
        print("Testing graceful degradation scenarios")

        # Test with disabled features
        degraded_store = ContextReferenceStore(
            cache_size=10,
            enable_compression=False,  # Disable compression
            use_disk_storage=False,  # Disable disk storage
        )

        # Should still work with basic functionality
        test_content = "Graceful degradation test content"
        context_id = degraded_store.store(test_content)
        retrieved = degraded_store.retrieve(context_id)

        assert retrieved == test_content

        # Test with very constrained resources
        constrained_store = ContextReferenceStore(
            cache_size=2,  # Very small cache
            memory_threshold=0.1,  # Very low memory threshold
        )

        # Store more than cache can hold
        context_ids = []
        for i in range(10):
            content = f"Constrained test {i}"
            context_id = constrained_store.store(content)
            context_ids.append(context_id)

        # Should handle evictions gracefully
        stats = constrained_store.get_cache_stats()
        assert stats.get("total_evictions", 0) > 0

        # Recent contexts should still be accessible
        recent_accessible = 0
        for context_id in context_ids[-3:]:
            try:
                constrained_store.retrieve(context_id)
                recent_accessible += 1
            except KeyError:
                pass

        assert recent_accessible > 0, "No recent contexts accessible after evictions"

    def test_error_recovery(self):
        """Test system recovery from various error conditions."""
        print("Testing error recovery mechanisms")

        store = ContextReferenceStore()

        # Store valid data first
        valid_content = "Valid content before errors"
        valid_id = store.store(valid_content)

        # Test recovery from invalid operations
        error_scenarios = [
            ("empty_context_id", lambda: store.retrieve("")),
            ("none_context_id", lambda: store.retrieve(None)),
            ("invalid_context_id", lambda: store.retrieve("invalid_id_12345")),
            ("delete_nonexistent", lambda: store.delete("nonexistent_id")),
        ]

        recovered_operations = 0

        for scenario_name, operation in error_scenarios:
            try:
                operation()
                print(f"Unexpected success for {scenario_name}")
            except (KeyError, ValueError, TypeError) as e:
                # Expected errors
                recovered_operations += 1
                print(f"Recovered from {scenario_name}: {type(e).__name__}")
            except Exception as e:
                print(f"Unexpected error for {scenario_name}: {e}")

        # Verify system still works after errors
        post_error_content = "Content stored after error recovery"
        post_error_id = store.store(post_error_content)
        post_error_retrieved = store.retrieve(post_error_id)

        assert post_error_retrieved == post_error_content

        # Original data should still be accessible
        original_retrieved = store.retrieve(valid_id)
        assert original_retrieved == valid_content

        print(
            f"Error recovery: {recovered_operations}/{len(error_scenarios)} scenarios handled correctly"
        )

    def test_concurrent_robustness(self):
        """Test system robustness under concurrent access."""
        print("Testing concurrent robustness")

        import threading
        import time

        store = ContextReferenceStore(cache_size=50)

        # Shared state for testing
        results = {
            "successful_operations": 0,
            "failed_operations": 0,
            "data_corruption": 0,
        }

        lock = threading.Lock()

        def robustness_worker(worker_id, num_operations):
            """Worker that performs various operations concurrently."""
            local_context_ids = []

            for i in range(num_operations):
                try:
                    # Mix of operations
                    operation = i % 4

                    if operation == 0:  # Store
                        content = f"Worker {worker_id} operation {i} content with timestamp {time.time()}"
                        context_id = store.store(content)
                        local_context_ids.append((context_id, content))

                        with lock:
                            results["successful_operations"] += 1

                    elif operation == 1 and local_context_ids:  # Retrieve and verify
                        context_id, expected_content = local_context_ids[-1]
                        retrieved = store.retrieve(context_id)

                        if retrieved != expected_content:
                            with lock:
                                results["data_corruption"] += 1
                        else:
                            with lock:
                                results["successful_operations"] += 1

                    elif operation == 2 and local_context_ids:  # Delete
                        context_id, _ = local_context_ids.pop()
                        store.delete(context_id)

                        with lock:
                            results["successful_operations"] += 1

                    elif operation == 3:  # Get stats (read-only)
                        stats = store.get_cache_stats()
                        if isinstance(stats, dict):
                            with lock:
                                results["successful_operations"] += 1

                except Exception as e:
                    with lock:
                        results["failed_operations"] += 1

                # Small delay to increase concurrency
                time.sleep(0.001)

        # Run concurrent workers
        num_workers = 20
        operations_per_worker = 50

        threads = []
        start_time = time.time()

        for i in range(num_workers):
            thread = threading.Thread(
                target=robustness_worker, args=(i, operations_per_worker)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout

        end_time = time.time()

        total_operations = (
            results["successful_operations"] + results["failed_operations"]
        )
        success_rate = (
            results["successful_operations"] / total_operations
            if total_operations > 0
            else 0
        )
        corruption_rate = (
            results["data_corruption"] / total_operations if total_operations > 0 else 0
        )

        print(f"Concurrent robustness results:")
        print(f"  - Total operations: {total_operations}")
        print(f"  - Success rate: {success_rate:.2%}")
        print(f"  - Corruption rate: {corruption_rate:.2%}")
        print(f"  - Duration: {end_time - start_time:.2f}s")
        print(
            f"  - Operations/second: {total_operations / (end_time - start_time):.2f}"
        )

        # Verify robustness
        assert success_rate > 0.85, f"Success rate too low: {success_rate:.2%}"
        assert (
            corruption_rate < 0.01
        ), f"Data corruption detected: {corruption_rate:.2%}"

    def test_resource_management(self):
        """Test proper resource management and cleanup."""
        print("Testing resource management")

        import gc
        import tempfile

        # Test memory management
        initial_objects = len(gc.get_objects())

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create store with disk storage
            store = ContextReferenceStore(
                cache_size=20, use_disk_storage=True, binary_cache_dir=temp_dir
            )

            # Perform many operations
            context_ids = []
            for i in range(100):
                content = f"Resource management test {i}: " + ("data " * 1000)
                context_id = store.store(content)
                context_ids.append(context_id)

                # Occasionally retrieve and delete
                if i % 10 == 0 and context_ids:
                    old_id = context_ids.pop(0)
                    try:
                        store.retrieve(old_id)
                        store.delete(old_id)
                    except KeyError:
                        pass  # May have been evicted

            # Check disk usage
            disk_files = []
            for root, dirs, files in os.walk(temp_dir):
                disk_files.extend(files)

            print(f"Disk files created: {len(disk_files)}")

            # Force cleanup
            del store
            gc.collect()

        # Check memory usage after cleanup
        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects

        print(f"Object growth after cleanup: {object_growth}")

        # Should not have excessive memory growth
        assert object_growth < 5000, f"Excessive memory growth: {object_growth} objects"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
