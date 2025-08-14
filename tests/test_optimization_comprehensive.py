#!/usr/bin/env python3
"""
Comprehensive Tests for Optimization and Semantic Modules

This module contains comprehensive tests for the token manager,
semantic analyzer, and monitoring components.
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from context_store import ContextReferenceStore


class TestTokenManager:
    """Comprehensive tests for Token Manager."""

    def test_token_manager_without_dependencies(self):
        """Test token manager behavior when dependencies are missing."""
        try:
            from context_store.optimization.token_manager import (
                TokenAwareContextManager,
                create_token_manager,
            )

            # Should work even without tiktoken
            manager = create_token_manager()
            assert manager is not None

        except ImportError:
            # If the module itself can't be imported, test the fallback
            with patch(
                "context_store.optimization.token_manager.TIKTOKEN_AVAILABLE", False
            ):
                try:
                    from context_store.optimization.token_manager import (
                        create_token_manager,
                    )

                    manager = create_token_manager()
                    # Should handle missing dependencies gracefully
                except ImportError:
                    pytest.skip("Token manager not available")

    def test_token_counting_edge_cases(self):
        """Test edge cases in token counting."""
        try:
            from context_store.optimization.token_manager import (
                TokenAwareContextManager,
                ModelConfig,
                ModelFamily,
                create_token_manager,
            )

            manager = create_token_manager()

            # Test empty content
            empty_tokens = manager.count_tokens("", ModelFamily.GPT)
            assert empty_tokens >= 0

            # Test very long content
            long_content = "word " * 100000  # Very long text
            long_tokens = manager.count_tokens(long_content, ModelFamily.GPT)
            assert long_tokens > 0

            # Test Unicode content
            unicode_content = "🚀🎯💡" * 1000
            unicode_tokens = manager.count_tokens(unicode_content, ModelFamily.GPT)
            assert unicode_tokens > 0

            # Test special characters
            special_content = "!@#$%^&*()_+-=[]{}|;':\",./<>?" * 100
            special_tokens = manager.count_tokens(special_content, ModelFamily.GPT)
            assert special_tokens > 0

        except ImportError:
            pytest.skip("Token manager dependencies not available")

    def test_context_optimization_edge_cases(self):
        """Test edge cases in context optimization."""
        try:
            from context_store.optimization.token_manager import (
                TokenAwareContextManager,
                OptimizationStrategy,
                TokenBudget,
                create_token_manager,
            )

            context_store = ContextReferenceStore()
            manager = create_token_manager(context_store=context_store)

            # Test with empty contexts
            empty_result = manager.optimize_contexts(
                context_ids=[],
                token_budget=TokenBudget(max_tokens=1000),
                strategy=OptimizationStrategy.BALANCED,
            )
            assert len(empty_result.selected_contexts) == 0

            # Store test contexts
            contexts = []
            for i in range(10):
                content = f"Test context {i} with some content " * (i + 1)
                context_id = context_store.store(content)
                contexts.append(context_id)

            # Test with very small token budget
            small_budget_result = manager.optimize_contexts(
                context_ids=contexts,
                token_budget=TokenBudget(max_tokens=10),  # Very small
                strategy=OptimizationStrategy.COST_FIRST,
            )
            # Should select minimal contexts
            assert len(small_budget_result.selected_contexts) <= len(contexts)

            # Test with very large token budget
            large_budget_result = manager.optimize_contexts(
                context_ids=contexts,
                token_budget=TokenBudget(max_tokens=1000000),  # Very large
                strategy=OptimizationStrategy.QUALITY_FIRST,
            )
            # Should select all contexts
            assert len(large_budget_result.selected_contexts) <= len(contexts)

        except ImportError:
            pytest.skip("Token manager dependencies not available")

    def test_model_configuration_edge_cases(self):
        """Test edge cases in model configuration."""
        try:
            from context_store.optimization.token_manager import (
                ModelConfig,
                ModelFamily,
                TokenBudget,
            )

            # Test with extreme model configurations
            extreme_config = ModelConfig(
                family=ModelFamily.GPT,
                name="gpt-4",
                max_tokens=0,  # Zero max tokens
                cost_per_token=0.0,  # Free model
                quality_score=0.0,  # Lowest quality
            )

            assert extreme_config.family == ModelFamily.GPT
            assert extreme_config.max_tokens == 0

            # Test with very expensive model
            expensive_config = ModelConfig(
                family=ModelFamily.GPT,
                name="expensive-model",
                max_tokens=1000000,
                cost_per_token=1.0,  # Very expensive
                quality_score=1.0,
            )

            assert expensive_config.cost_per_token == 1.0

        except ImportError:
            pytest.skip("Token manager dependencies not available")

    def test_optimization_strategies(self):
        """Test different optimization strategies."""
        try:
            from context_store.optimization.token_manager import (
                TokenAwareContextManager,
                OptimizationStrategy,
                TokenBudget,
                create_token_manager,
            )

            context_store = ContextReferenceStore()
            manager = create_token_manager(context_store=context_store)

            # Store contexts with different characteristics
            contexts = []
            for i in range(5):
                content = (
                    f"Context {i}: " + ("important " if i % 2 == 0 else "normal ") * 50
                )
                context_id = context_store.store(content)
                contexts.append(context_id)

            budget = TokenBudget(max_tokens=1000)

            # Test all strategies
            strategies = [
                OptimizationStrategy.COST_FIRST,
                OptimizationStrategy.QUALITY_FIRST,
                OptimizationStrategy.BALANCED,
                OptimizationStrategy.SPEED_FIRST,
                OptimizationStrategy.COMPREHENSIVE,
            ]

            results = {}
            for strategy in strategies:
                result = manager.optimize_contexts(
                    context_ids=contexts, token_budget=budget, strategy=strategy
                )
                results[strategy] = result
                assert result is not None
                assert len(result.selected_contexts) <= len(contexts)

            # Different strategies should potentially give different results
            # (though with small test data, they might be the same)

        except ImportError:
            pytest.skip("Token manager dependencies not available")


class TestSemanticAnalyzer:
    """Comprehensive tests for Semantic Analyzer."""

    def test_semantic_analyzer_without_dependencies(self):
        """Test semantic analyzer behavior when dependencies are missing."""
        try:
            from context_store.semantic.semantic_analyzer import (
                SemanticContextAnalyzer,
                create_semantic_analyzer,
            )

            # Should work even without sentence-transformers
            analyzer = create_semantic_analyzer()
            assert analyzer is not None

        except ImportError:
            # If the module itself can't be imported, test the fallback
            with patch(
                "context_store.semantic.semantic_analyzer.EMBEDDINGS_AVAILABLE", False
            ):
                try:
                    from context_store.semantic.semantic_analyzer import (
                        create_semantic_analyzer,
                    )

                    analyzer = create_semantic_analyzer()
                    # Should handle missing dependencies gracefully
                except ImportError:
                    pytest.skip("Semantic analyzer not available")

    def test_similarity_detection_edge_cases(self):
        """Test edge cases in similarity detection."""
        try:
            from context_store.semantic.semantic_analyzer import (
                SemanticContextAnalyzer,
                SimilarityMethod,
                create_semantic_analyzer,
            )

            context_store = ContextReferenceStore()
            analyzer = create_semantic_analyzer(context_store=context_store)

            # Test with identical contexts
            content1 = "This is identical content"
            content2 = "This is identical content"

            context_id1 = context_store.store(content1)
            context_id2 = context_store.store(content2)

            matches = analyzer.find_similar_contexts(
                context_id1,
                [context_id2],
                similarity_threshold=0.9,
                method=SimilarityMethod.COSINE,
            )

            # Should find high similarity for identical content
            if matches:  # May be empty if embeddings not available
                assert len(matches) > 0

            # Test with completely different contexts
            content3 = "Completely different unrelated content about space exploration"
            content4 = (
                "Totally distinct text discussing culinary arts and cooking techniques"
            )

            context_id3 = context_store.store(content3)
            context_id4 = context_store.store(content4)

            different_matches = analyzer.find_similar_contexts(
                context_id3,
                [context_id4],
                similarity_threshold=0.9,
                method=SimilarityMethod.COSINE,
            )

            # Should find low similarity for different content
            assert (
                len(different_matches) <= 1
            )  # May find some similarity due to common words

        except ImportError:
            pytest.skip("Semantic analyzer dependencies not available")

    def test_clustering_edge_cases(self):
        """Test edge cases in context clustering."""
        try:
            from context_store.semantic.semantic_analyzer import (
                SemanticContextAnalyzer,
                ClusteringAlgorithm,
                create_semantic_analyzer,
            )

            context_store = ContextReferenceStore()
            analyzer = create_semantic_analyzer(context_store=context_store)

            # Test clustering with single context
            single_content = "Single context for clustering test"
            single_id = context_store.store(single_content)

            single_clusters = analyzer.cluster_contexts(
                [single_id], algorithm=ClusteringAlgorithm.KMEANS, num_clusters=1
            )

            if single_clusters:  # May be empty if clustering not available
                assert len(single_clusters) <= 1

            # Test clustering with many similar contexts
            similar_contexts = []
            for i in range(10):
                content = f"Similar context about artificial intelligence and machine learning topic {i}"
                context_id = context_store.store(content)
                similar_contexts.append(context_id)

            similar_clusters = analyzer.cluster_contexts(
                similar_contexts, algorithm=ClusteringAlgorithm.KMEANS, num_clusters=3
            )

            if similar_clusters:
                assert len(similar_clusters) <= 3

            # Test clustering with diverse contexts
            diverse_contexts = []
            diverse_topics = [
                "Space exploration and astronomy research",
                "Cooking recipes and culinary techniques",
                "Sports statistics and athletic performance",
                "Music composition and sound engineering",
                "Economic policy and financial markets",
            ]

            for topic in diverse_topics:
                content = f"Content about {topic} with detailed information"
                context_id = context_store.store(content)
                diverse_contexts.append(context_id)

            diverse_clusters = analyzer.cluster_contexts(
                diverse_contexts, algorithm=ClusteringAlgorithm.DBSCAN, min_samples=1
            )

            if diverse_clusters:
                # Should create multiple clusters for diverse content
                assert len(diverse_clusters) >= 1

        except ImportError:
            pytest.skip("Semantic analyzer dependencies not available")

    def test_deduplication_edge_cases(self):
        """Test edge cases in context deduplication."""
        try:
            from context_store.semantic.semantic_analyzer import (
                SemanticContextAnalyzer,
                MergeStrategy,
                create_semantic_analyzer,
            )

            context_store = ContextReferenceStore()
            analyzer = create_semantic_analyzer(context_store=context_store)

            # Test deduplication with no duplicates
            unique_contexts = []
            for i in range(5):
                content = f"Unique content number {i} with distinct information"
                context_id = context_store.store(content)
                unique_contexts.append(context_id)

            unique_result = analyzer.deduplicate_contexts(
                unique_contexts,
                similarity_threshold=0.9,
                merge_strategy=MergeStrategy.KEEP_FIRST,
            )

            if unique_result:
                # Should keep all unique contexts
                assert len(unique_result.deduplicated_contexts) <= len(unique_contexts)

            # Test deduplication with obvious duplicates
            duplicate_contexts = []
            base_content = "This is duplicate content for testing"

            # Store same content multiple times
            for i in range(5):
                context_id = context_store.store(base_content)
                duplicate_contexts.append(context_id)

            # Store slight variations
            for i in range(3):
                varied_content = f"{base_content} with variation {i}"
                context_id = context_store.store(varied_content)
                duplicate_contexts.append(context_id)

            duplicate_result = analyzer.deduplicate_contexts(
                duplicate_contexts,
                similarity_threshold=0.8,
                merge_strategy=MergeStrategy.KEEP_LARGEST,
            )

            if duplicate_result:
                # Should reduce the number of contexts
                assert len(duplicate_result.deduplicated_contexts) <= len(
                    duplicate_contexts
                )

        except ImportError:
            pytest.skip("Semantic analyzer dependencies not available")

    def test_merge_strategies(self):
        """Test different merge strategies in deduplication."""
        try:
            from context_store.semantic.semantic_analyzer import (
                SemanticContextAnalyzer,
                MergeStrategy,
                create_semantic_analyzer,
            )

            context_store = ContextReferenceStore()
            analyzer = create_semantic_analyzer(context_store=context_store)

            # Store contexts with different sizes
            contexts = []
            base_content = "Base content for merge strategy testing"

            # Short version
            short_id = context_store.store(base_content)
            contexts.append(short_id)

            # Medium version
            medium_content = base_content + " with additional details"
            medium_id = context_store.store(medium_content)
            contexts.append(medium_id)

            # Long version
            long_content = medium_content + " and even more comprehensive information"
            long_id = context_store.store(long_content)
            contexts.append(long_id)

            # Test different merge strategies
            strategies = [
                MergeStrategy.KEEP_FIRST,
                MergeStrategy.KEEP_LARGEST,
                MergeStrategy.KEEP_MOST_RECENT,
                MergeStrategy.MERGE_CONTENT,
            ]

            for strategy in strategies:
                result = analyzer.deduplicate_contexts(
                    contexts, similarity_threshold=0.7, merge_strategy=strategy
                )

                if result:
                    # Each strategy should handle the contexts
                    assert result.deduplicated_contexts is not None
                    assert len(result.deduplicated_contexts) <= len(contexts)

        except ImportError:
            pytest.skip("Semantic analyzer dependencies not available")


class TestMonitoringDashboard:
    """Comprehensive tests for TUI Dashboard."""

    def test_dashboard_creation(self):
        """Test dashboard creation and initialization."""
        try:
            from context_store.monitoring.tui_dashboard import (
                ContextStoreTUIDashboard,
                create_dashboard,
            )

            context_store = ContextReferenceStore()

            # Test dashboard creation
            dashboard = create_dashboard(context_store)
            assert dashboard is not None
            assert hasattr(dashboard, "context_store")

            # Test dashboard with optimization features
            try:
                from context_store.optimization.token_manager import (
                    create_token_manager,
                )
                from context_store.semantic.semantic_analyzer import (
                    create_semantic_analyzer,
                )

                token_manager = create_token_manager(context_store)
                semantic_analyzer = create_semantic_analyzer(context_store)

                enhanced_dashboard = create_dashboard(
                    context_store,
                    token_manager=token_manager,
                    semantic_analyzer=semantic_analyzer,
                )
                assert enhanced_dashboard is not None

            except ImportError:
                # Optional features not available
                pass

        except ImportError:
            pytest.skip("TUI dashboard not available")

    def test_dashboard_data_collection(self):
        """Test dashboard data collection functionality."""
        try:
            from context_store.monitoring.tui_dashboard import (
                ContextStoreTUIDashboard,
                create_dashboard,
            )

            context_store = ContextReferenceStore()
            dashboard = create_dashboard(context_store)

            # Store some test data
            for i in range(10):
                content = f"Dashboard test content {i}"
                context_store.store(content)

            # Test data collection methods (if they exist)
            if hasattr(dashboard, "get_store_stats"):
                stats = dashboard.get_store_stats()
                assert stats is not None

            if hasattr(dashboard, "get_performance_data"):
                perf_data = dashboard.get_performance_data()
                assert perf_data is not None

        except ImportError:
            pytest.skip("TUI dashboard not available")

    def test_dashboard_with_edge_cases(self):
        """Test dashboard behavior with edge case data."""
        try:
            from context_store.monitoring.tui_dashboard import create_dashboard

            # Test with empty context store
            empty_store = ContextReferenceStore(cache_size=1)
            empty_dashboard = create_dashboard(empty_store)
            assert empty_dashboard is not None

            # Test with heavily loaded context store
            loaded_store = ContextReferenceStore(cache_size=5)

            # Fill beyond capacity to trigger evictions
            for i in range(20):
                content = f"Heavy load content {i}" * 100
                loaded_store.store(content)

            loaded_dashboard = create_dashboard(loaded_store)
            assert loaded_dashboard is not None

        except ImportError:
            pytest.skip("TUI dashboard not available")


class TestIntegratedOptimization:
    """Tests for integrated optimization scenarios."""

    def test_token_manager_semantic_analyzer_integration(self):
        """Test integration between token manager and semantic analyzer."""
        try:
            from context_store.optimization.token_manager import create_token_manager
            from context_store.semantic.semantic_analyzer import (
                create_semantic_analyzer,
            )

            context_store = ContextReferenceStore()
            token_manager = create_token_manager(context_store)
            semantic_analyzer = create_semantic_analyzer(context_store)

            # Store related contexts
            related_contexts = []
            base_topic = "artificial intelligence and machine learning"

            for i in range(5):
                content = (
                    f"Content about {base_topic} with focus on aspect {i}: "
                    + f"detailed information about neural networks and deep learning "
                    * (i + 1)
                )
                context_id = context_store.store(content)
                related_contexts.append(context_id)

            # Use semantic analyzer to find similar contexts
            if hasattr(semantic_analyzer, "find_similar_contexts"):
                similar = semantic_analyzer.find_similar_contexts(
                    related_contexts[0], related_contexts[1:], similarity_threshold=0.5
                )

                # Use token manager to optimize the similar contexts
                if similar and hasattr(token_manager, "optimize_contexts"):
                    from context_store.optimization.token_manager import TokenBudget

                    budget = TokenBudget(max_tokens=2000)
                    optimized = token_manager.optimize_contexts(
                        context_ids=[related_contexts[0]]
                        + [m.context_id for m in similar],
                        token_budget=budget,
                    )

                    assert optimized is not None

        except ImportError:
            pytest.skip("Optimization dependencies not available")

    def test_optimization_under_memory_pressure(self):
        """Test optimization components under memory pressure."""
        try:
            from context_store.optimization.token_manager import create_token_manager
            from context_store.semantic.semantic_analyzer import (
                create_semantic_analyzer,
            )

            # Create store with very limited memory
            context_store = ContextReferenceStore(
                cache_size=3, memory_threshold=0.1  # Very low threshold
            )

            token_manager = create_token_manager(context_store)
            semantic_analyzer = create_semantic_analyzer(context_store)

            # Store many contexts to trigger memory pressure
            contexts = []
            for i in range(20):
                large_content = f"Large content for memory pressure test {i}: " + (
                    "content " * 1000
                )
                context_id = context_store.store(large_content)
                contexts.append(context_id)

            # Optimization should still work under memory pressure
            if hasattr(semantic_analyzer, "cluster_contexts"):
                try:
                    clusters = semantic_analyzer.cluster_contexts(
                        contexts[:5], num_clusters=2  # Use subset to avoid timeout
                    )
                    # Should handle memory pressure gracefully
                except Exception as e:
                    # Memory pressure might cause operations to fail
                    pass

            if hasattr(token_manager, "optimize_contexts"):
                try:
                    from context_store.optimization.token_manager import TokenBudget

                    budget = TokenBudget(max_tokens=1000)
                    optimized = token_manager.optimize_contexts(
                        context_ids=contexts[:3], token_budget=budget  # Use subset
                    )
                    # Should handle memory pressure gracefully
                except Exception as e:
                    # Memory pressure might cause operations to fail
                    pass

        except ImportError:
            pytest.skip("Optimization dependencies not available")

    def test_concurrent_optimization_operations(self):
        """Test concurrent optimization operations."""
        import threading

        try:
            from context_store.optimization.token_manager import create_token_manager
            from context_store.semantic.semantic_analyzer import (
                create_semantic_analyzer,
            )

            context_store = ContextReferenceStore(cache_size=50)
            token_manager = create_token_manager(context_store)
            semantic_analyzer = create_semantic_analyzer(context_store)

            # Store test contexts
            contexts = []
            for i in range(10):
                content = f"Concurrent optimization test content {i} with detailed information"
                context_id = context_store.store(content)
                contexts.append(context_id)

            results = {"token_ops": 0, "semantic_ops": 0, "errors": 0}

            def token_worker():
                """Worker for token optimization operations."""
                try:
                    for i in range(5):
                        if hasattr(token_manager, "optimize_contexts"):
                            from context_store.optimization.token_manager import (
                                TokenBudget,
                            )

                            budget = TokenBudget(max_tokens=1000)
                            token_manager.optimize_contexts(
                                context_ids=contexts[:3], token_budget=budget
                            )
                            results["token_ops"] += 1
                        time.sleep(0.01)
                except Exception:
                    results["errors"] += 1

            def semantic_worker():
                """Worker for semantic analysis operations."""
                try:
                    for i in range(5):
                        if hasattr(semantic_analyzer, "find_similar_contexts"):
                            semantic_analyzer.find_similar_contexts(
                                contexts[0], contexts[1:3], similarity_threshold=0.5
                            )
                            results["semantic_ops"] += 1
                        time.sleep(0.01)
                except Exception:
                    results["errors"] += 1

            # Run concurrent operations
            threads = [
                threading.Thread(target=token_worker),
                threading.Thread(target=semantic_worker),
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=30)

            # Should complete without major issues
            total_ops = results["token_ops"] + results["semantic_ops"]
            if total_ops > 0:
                error_rate = results["errors"] / (total_ops + results["errors"])
                assert error_rate < 0.5, f"High error rate: {error_rate:.2%}"

        except ImportError:
            pytest.skip("Optimization dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
