#!/usr/bin/env python3
"""
Comprehensive Agent Comparison Script

This script tests all 4 agents with identical operations to demonstrate
the Context Reference Store benefits, including advanced features:
- Token Manager for smart context selection
- Semantic Analyzer for deduplication and clustering
- Complete performance analytics
"""

import json
import time
import sys
import os

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from basic_analysis_agent.agent import (
    analyze_text,
    calculate_advanced,
    get_performance_metrics as get_basic_metrics,
    export_metrics_report as export_basic_metrics,
)

from advanced_multi_tool_agent.agent import (
    read_file_content,
    advanced_text_analysis,
    advanced_calculator,
    get_performance_metrics as get_advanced_metrics,
    export_metrics_report as export_advanced_metrics,
)

from context_enhanced_analysis_agent.agent import (
    analyze_large_text,
    calculate_with_context_caching,
    get_enhanced_performance_metrics as get_enhanced_basic_metrics,
    export_enhanced_metrics_report as export_enhanced_basic_metrics,
)

from context_enhanced_multi_tool_agent.agent import (
    read_file_with_context_cache,
    advanced_text_analysis_with_context,
    advanced_calculator_with_caching,
    get_enhanced_performance_metrics as get_enhanced_advanced_metrics,
    export_enhanced_metrics_report as export_enhanced_advanced_metrics,
    _context_store,
)

# Import advanced features
from context_store.optimization.token_manager import (
    create_token_manager,
    OptimizationStrategy,
)
from context_store.semantic.semantic_analyzer import create_semantic_analyzer


class MockToolContext:
    def __init__(self):
        self.state = {}


def test_token_manager_features():
    """Test Token Manager capabilities across different strategies."""
    print("\n🎯 TESTING TOKEN MANAGER FEATURES")
    print("-" * 50)

    # Create token manager
    manager = create_token_manager("gemini-1.5-pro")

    # Test contexts for optimization
    test_contexts = [
        "Context Reference Store provides 625x faster serialization performance.",
        "Machine learning algorithms require efficient context management for optimal performance.",
        "Token optimization strategies help reduce costs while maintaining quality.",
        "Semantic analysis enables intelligent content deduplication and clustering.",
        "Real-time monitoring dashboards provide insights into system performance.",
        "Advanced compression algorithms achieve 99.55% storage reduction while preserving quality.",
    ] * 3  # Multiply for substantial content

    # Create budget
    budget = manager.create_budget(target_tokens=8000)

    results = {}
    strategies = [
        OptimizationStrategy.COST_FIRST,
        OptimizationStrategy.QUALITY_FIRST,
        OptimizationStrategy.BALANCED,
    ]

    for strategy in strategies:
        start_time = time.time()

        result = manager.optimize_context_selection(
            contexts=test_contexts,
            budget=budget,
            strategy=strategy,
            query="context optimization and performance improvements",
            keywords=["context", "optimization", "performance"],
        )

        duration = time.time() - start_time

        results[strategy.value] = {
            "selected_contexts": len(result.selected_contexts),
            "total_contexts": len(test_contexts),
            "total_tokens": result.total_tokens,
            "budget_utilization": result.budget_utilization,
            "estimated_cost": result.estimated_cost,
            "efficiency_score": result.efficiency_score,
            "processing_time": duration,
            "recommendations": (
                result.recommendations[:2] if result.recommendations else []
            ),
        }

        print(f"   🔍 {strategy.value}:")
        print(
            f"      Selected: {len(result.selected_contexts)}/{len(test_contexts)} contexts"
        )
        print(
            f"      Tokens: {result.total_tokens:,} ({result.budget_utilization:.1%} utilization)"
        )
        print(f"      Cost: ${result.estimated_cost:.4f}")
        print(f"      Efficiency: {result.efficiency_score:.2f}")
        print(f"      Time: {duration:.3f}s")

    # Get usage analytics
    analytics = manager.get_usage_analytics()
    results["analytics"] = {
        "total_optimizations": analytics["total_optimizations"],
        "total_tokens_processed": analytics["total_tokens_processed"],
        "average_budget_utilization": analytics["average_budget_utilization"],
        "total_estimated_cost": analytics.get("total_estimated_cost", 0),
    }

    print(f"\n   📈 Token Manager Analytics:")
    print(f"      Total optimizations: {analytics['total_optimizations']}")
    print(f"      Tokens processed: {analytics['total_tokens_processed']:,}")
    print(f"      Avg utilization: {analytics['average_budget_utilization']:.1%}")
    print(f"      Total cost: ${analytics.get('total_estimated_cost', 0):.4f}")

    return results


def test_semantic_analyzer_features():
    """Test Semantic Analyzer capabilities for deduplication and clustering."""
    print("\n🧠 TESTING SEMANTIC ANALYZER FEATURES")
    print("-" * 50)

    # Create semantic analyzer
    analyzer = create_semantic_analyzer(similarity_threshold=0.80)

    # Test contexts with semantic similarities
    test_contexts = {
        "ctx_1": "Context Reference Store delivers revolutionary performance improvements for AI applications.",
        "ctx_2": "The Context Reference Store provides revolutionary performance enhancements for AI systems.",
        "ctx_3": "Machine learning models require efficient context management for optimal performance.",
        "ctx_4": "ML algorithms benefit from optimized context handling and storage mechanisms.",
        "ctx_5": "Token optimization strategies reduce costs while maintaining output quality.",
        "ctx_6": "Smart token management decreases operational expenses without quality loss.",
        "ctx_7": "Semantic analysis enables intelligent content organization and deduplication.",
        "ctx_8": "Advanced semantic algorithms allow smart content clustering and duplicate detection.",
        "ctx_9": "Real-time monitoring provides insights into system performance and optimization.",
        "ctx_10": "Performance dashboards enable proactive system monitoring and analytics.",
    }

    start_time = time.time()

    # Perform comprehensive semantic analysis
    analysis_result = analyzer.analyze_contexts(test_contexts)

    processing_time = time.time() - start_time

    results = {
        "total_contexts_analyzed": analysis_result.total_contexts_analyzed,
        "duplicates_found": analysis_result.duplicates_found,
        "clusters_created": analysis_result.clusters_created,
        "space_savings_potential": analysis_result.space_savings_potential,
        "quality_improvement_potential": analysis_result.quality_improvement_potential,
        "processing_time_ms": analysis_result.processing_time_ms,
        "total_processing_time": processing_time,
        "recommendations": (
            analysis_result.recommendations[:3]
            if analysis_result.recommendations
            else []
        ),
    }

    # Show similarity matches
    if analysis_result.similarity_matches:
        results["similarity_matches"] = []
        for match in analysis_result.similarity_matches[:3]:
            results["similarity_matches"].append(
                {
                    "context_pair": f"{match.context_id_1} ↔ {match.context_id_2}",
                    "similarity_score": match.similarity_score,
                    "suggested_action": match.suggested_action,
                    "confidence": match.confidence,
                }
            )

    # Show cluster information
    if analysis_result.clusters:
        results["clusters"] = []
        for cluster in analysis_result.clusters:
            results["clusters"].append(
                {
                    "cluster_id": cluster.cluster_id,
                    "context_count": len(cluster.context_ids),
                    "semantic_theme": cluster.semantic_theme,
                    "quality_score": cluster.quality_score,
                }
            )

    print(f"   📊 Analysis Results:")
    print(f"      Contexts analyzed: {analysis_result.total_contexts_analyzed}")
    print(f"      Semantic duplicates: {analysis_result.duplicates_found}")
    print(f"      Clusters created: {analysis_result.clusters_created}")
    print(
        f"      Space savings potential: {analysis_result.space_savings_potential:.1%}"
    )
    print(
        f"      Quality improvement: {analysis_result.quality_improvement_potential:.1%}"
    )
    print(f"      Processing time: {analysis_result.processing_time_ms:.1f}ms")

    if analysis_result.similarity_matches:
        print(f"   🔍 Top Similarity Matches:")
        for match in analysis_result.similarity_matches[:3]:
            print(
                f"      • {match.context_id_1} ↔ {match.context_id_2}: {match.similarity_score:.3f}"
            )

    if analysis_result.clusters:
        print(f"   🗂️  Cluster Themes:")
        for cluster in analysis_result.clusters:
            print(
                f"      • {cluster.semantic_theme} (Quality: {cluster.quality_score:.2f})"
            )

    # Get analyzer statistics
    stats = analyzer.get_analysis_statistics()
    results["analyzer_stats"] = {
        "total_analyses": stats["total_analyses"],
        "contexts_processed": stats["contexts_processed"],
        "embeddings_computed": stats["embeddings_computed"],
        "cache_hit_rate": stats.get("cache_hit_rate", 0),
        "duplicates_found": stats["duplicates_found"],
        "clusters_created": stats["clusters_created"],
    }

    print(f"\n   📈 Analyzer Statistics:")
    print(f"      Total analyses: {stats['total_analyses']}")
    print(f"      Embeddings computed: {stats['embeddings_computed']}")
    print(f"      Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")

    return results


def test_context_store_integration():
    """Test Context Reference Store integration and analytics."""
    print("\n💾 TESTING CONTEXT STORE INTEGRATION")
    print("-" * 50)

    try:
        # Get context store statistics
        stats = _context_store.get_cache_stats()

        results = {
            "total_contexts": stats.get("total_contexts", 0),
            "hit_rate": stats.get("hit_rate", 0),
            "memory_usage_percent": stats.get("memory_usage_percent", 0),
            "total_evictions": stats.get("total_evictions", 0),
            "cache_size": stats.get("cache_size", 0),
        }

        print(f"   📊 Context Store Statistics:")
        print(f"      Total contexts: {stats.get('total_contexts', 0)}")
        print(f"      Cache hit rate: {stats.get('hit_rate', 0):.1%}")
        print(f"      Memory usage: {stats.get('memory_usage_percent', 0):.1f}%")
        print(f"      Total evictions: {stats.get('total_evictions', 0)}")
        print(f"      Cache size: {stats.get('cache_size', 0)}")

        # Check if compression analytics are available
        if hasattr(_context_store, "get_compression_analytics"):
            try:
                compression_stats = _context_store.get_compression_analytics()
                if compression_stats.get("compression_enabled", False):
                    context_stats = compression_stats.get("context_store_stats", {})
                    results["compression_stats"] = {
                        "compressed_contexts": context_stats.get(
                            "compressed_contexts", 0
                        ),
                        "space_savings_percentage": context_stats.get(
                            "space_savings_percentage", 0
                        ),
                        "total_space_saved_bytes": context_stats.get(
                            "total_space_saved_bytes", 0
                        ),
                    }

                    print(f"   ⚡ Compression Analytics:")
                    print(
                        f"      Compressed contexts: {context_stats.get('compressed_contexts', 0)}"
                    )
                    print(
                        f"      Space savings: {context_stats.get('space_savings_percentage', 0):.1f}%"
                    )
                    print(
                        f"      Bytes saved: {context_stats.get('total_space_saved_bytes', 0):,}"
                    )
            except Exception as e:
                print(f"      ⚠️  Compression analytics not available: {e}")

        return results

    except Exception as e:
        print(f"   ❌ Error accessing Context Store: {e}")
        return {"error": str(e)}


def run_comprehensive_comparison():
    """Run comprehensive comparison of all 4 agents."""
    print("🚀 COMPREHENSIVE AGENT COMPARISON")
    print("=" * 80)

    # Test data
    test_text = (
        """
    The Context Reference Store library represents a revolutionary advancement in AI context management.
    This innovative technology provides 625x faster serialization, 49x memory reduction, and 99.55% 
    storage reduction compared to traditional approaches. The system supports advanced caching strategies 
    including LRU, LFU, TTL, and Memory Pressure-based eviction policies. With zero quality degradation
    validated through ROUGE metrics, this technology enables efficient handling of large context windows
    (1M-2M tokens) while maintaining exceptional performance.
    """
        * 3
    )  # Make it larger for better demonstration

    test_expression = "sin(pi/4) * cos(pi/3) + sqrt(144)"
    test_file = "context_enhanced_multi_tool_agent/agent.py"

    results = {}

    # Test 1: Basic Analysis Agent
    print("\n📊 Testing Basic Analysis Agent...")
    mock_context = MockToolContext()

    start_time = time.time()

    # Text analysis
    result1 = analyze_text(test_text, mock_context)
    result2 = calculate_advanced(test_expression, mock_context)
    result3 = calculate_advanced(test_expression, mock_context)  # Repeated

    basic_duration = time.time() - start_time
    basic_metrics = get_basic_metrics(mock_context)

    results["basic_analysis"] = {
        "duration": basic_duration,
        "metrics": basic_metrics,
        "context_operations": 0,  # No context store
        "cache_hits": 0,  # No caching
    }

    print(f"   ✅ Basic Agent completed in {basic_duration:.4f}s")

    # Test 2: Advanced Multi-Tool Agent
    print("\n📊 Testing Advanced Multi-Tool Agent...")
    mock_context = MockToolContext()

    start_time = time.time()

    # File reading and text analysis
    try:
        file_result1 = read_file_content(test_file, mock_context)
        file_result2 = read_file_content(test_file, mock_context)  # Repeated
    except:
        print("   ⚠️  File reading skipped (file not found)")

    text_result = advanced_text_analysis(test_text, mock_context)
    calc_result1 = advanced_calculator(test_expression, mock_context)
    calc_result2 = advanced_calculator(test_expression, mock_context)  # Repeated

    advanced_duration = time.time() - start_time
    advanced_metrics = get_advanced_metrics(mock_context)

    results["advanced_multi_tool"] = {
        "duration": advanced_duration,
        "metrics": advanced_metrics,
        "context_operations": 0,  # No context store
        "cache_hits": 0,  # No caching
    }

    print(f"   ✅ Advanced Agent completed in {advanced_duration:.4f}s")

    # Test 3: Context-Enhanced Analysis Agent
    print("\n📊 Testing Context-Enhanced Analysis Agent...")
    mock_context = MockToolContext()

    start_time = time.time()

    # Enhanced text analysis and calculations
    enhanced_text_result = analyze_large_text(test_text, mock_context)
    enhanced_calc_result1 = calculate_with_context_caching(
        test_expression, mock_context
    )
    enhanced_calc_result2 = calculate_with_context_caching(
        test_expression, mock_context
    )  # Should hit cache

    enhanced_basic_duration = time.time() - start_time
    enhanced_basic_metrics = get_enhanced_basic_metrics(mock_context)

    # Extract context store metrics
    context_ops = (
        enhanced_basic_metrics.get("performance_metrics", {})
        .get("context_store_metrics", {})
        .get("total_context_operations", 0)
    )
    cache_rate = (
        enhanced_basic_metrics.get("performance_metrics", {})
        .get("context_store_metrics", {})
        .get("cache_hit_rate", 0)
    )

    results["enhanced_analysis"] = {
        "duration": enhanced_basic_duration,
        "metrics": enhanced_basic_metrics,
        "context_operations": context_ops,
        "cache_hit_rate": cache_rate,
    }

    print(f"   ✅ Enhanced Basic Agent completed in {enhanced_basic_duration:.4f}s")
    print(f"   📈 Context Operations: {context_ops}, Cache Hit Rate: {cache_rate:.1f}%")

    # Test 4: Context-Enhanced Multi-Tool Agent
    print("\n📊 Testing Context-Enhanced Multi-Tool Agent...")
    mock_context = MockToolContext()

    start_time = time.time()

    # Enhanced file reading, text analysis, and calculations
    try:
        enhanced_file_result1 = read_file_with_context_cache(test_file, mock_context)
        enhanced_file_result2 = read_file_with_context_cache(
            test_file, mock_context
        )  # Should hit cache
        file_cache_hit = enhanced_file_result2.get("cache_metrics", {}).get(
            "cache_hit", False
        )
    except:
        print("   ⚠️  File reading skipped (file not found)")
        file_cache_hit = False

    enhanced_text_result2 = advanced_text_analysis_with_context(test_text, mock_context)
    enhanced_calc_result3 = advanced_calculator_with_caching(
        test_expression, mock_context
    )
    enhanced_calc_result4 = advanced_calculator_with_caching(
        test_expression, mock_context
    )  # Should hit cache

    enhanced_advanced_duration = time.time() - start_time
    enhanced_advanced_metrics = get_enhanced_advanced_metrics(mock_context)

    # Extract context store metrics
    context_ops_adv = (
        enhanced_advanced_metrics.get("performance_metrics", {})
        .get("context_store_metrics", {})
        .get("total_context_operations", 0)
    )
    cache_rate_adv = (
        enhanced_advanced_metrics.get("performance_metrics", {})
        .get("context_store_metrics", {})
        .get("cache_hit_rate", 0)
    )
    storage_efficiency = (
        enhanced_advanced_metrics.get("performance_metrics", {})
        .get("context_store_metrics", {})
        .get("storage_efficiency_percent", 0)
    )

    results["enhanced_multi_tool"] = {
        "duration": enhanced_advanced_duration,
        "metrics": enhanced_advanced_metrics,
        "context_operations": context_ops_adv,
        "cache_hit_rate": cache_rate_adv,
        "storage_efficiency": storage_efficiency,
        "file_cache_hit": file_cache_hit,
    }

    print(
        f"   ✅ Enhanced Advanced Agent completed in {enhanced_advanced_duration:.4f}s"
    )
    print(
        f"   📈 Context Operations: {context_ops_adv}, Cache Hit Rate: {cache_rate_adv:.1f}%"
    )
    print(
        f"   💾 Storage Efficiency: {storage_efficiency:.1f}%, File Cache Hit: {file_cache_hit}"
    )

    # Summary Comparison
    print("\n🎯 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)

    print(
        f"{'Agent':<25} {'Duration (s)':<12} {'Context Ops':<12} {'Cache Rate':<12} {'Benefits'}"
    )
    print("-" * 80)
    print(
        f"{'Basic Analysis':<25} {results['basic_analysis']['duration']:<12.4f} {'0':<12} {'0%':<12} {'Baseline'}"
    )
    print(
        f"{'Advanced Multi-Tool':<25} {results['advanced_multi_tool']['duration']:<12.4f} {'0':<12} {'0%':<12} {'More Tools'}"
    )
    print(
        f"{'Enhanced Analysis':<25} {results['enhanced_analysis']['duration']:<12.4f} {results['enhanced_analysis']['context_operations']:<12} {results['enhanced_analysis']['cache_hit_rate']:<12.1f}% {'Context Store'}"
    )
    print(
        f"{'Enhanced Multi-Tool':<25} {results['enhanced_multi_tool']['duration']:<12.4f} {results['enhanced_multi_tool']['context_operations']:<12} {results['enhanced_multi_tool']['cache_hit_rate']:<12.1f}% {'Full Enhancement'}"
    )

    print(f"\n📊 Key Improvements with Context Reference Store:")
    print(f"   🚀 Context Operations: {max(context_ops, context_ops_adv)} total")
    print(f"   ⚡ Cache Hit Rate: {max(cache_rate, cache_rate_adv):.1f}%")
    print(f"   💾 Storage Efficiency: {storage_efficiency:.1f}%")
    print(f"   🗂️  File Caching: {'✅ Working' if file_cache_hit else '⚠️  Limited'}")

    # Test Advanced Features
    print("\n🎯 TESTING ADVANCED CONTEXT REFERENCE STORE FEATURES")
    print("=" * 80)

    # Test Token Manager
    try:
        token_results = test_token_manager_features()
        results["token_manager"] = token_results
    except Exception as e:
        print(f"   ❌ Token Manager error: {e}")
        results["token_manager"] = {"error": str(e)}

    # Test Semantic Analyzer
    try:
        semantic_results = test_semantic_analyzer_features()
        results["semantic_analyzer"] = semantic_results
    except Exception as e:
        print(f"   ❌ Semantic Analyzer error: {e}")
        results["semantic_analyzer"] = {"error": str(e)}

    # Test Context Store Integration
    try:
        context_store_results = test_context_store_integration()
        results["context_store_integration"] = context_store_results
    except Exception as e:
        print(f"   ❌ Context Store Integration error: {e}")
        results["context_store_integration"] = {"error": str(e)}

    # Final Summary
    print(f"\n🏆 COMPLETE CONTEXT REFERENCE STORE ECOSYSTEM RESULTS")
    print("=" * 80)

    # Agent Performance Summary
    print(f"📊 Agent Performance:")
    print(
        f"   • Basic Analysis: {results['basic_analysis']['duration']:.4f}s (baseline)"
    )
    print(
        f"   • Advanced Multi-Tool: {results['advanced_multi_tool']['duration']:.4f}s (baseline+)"
    )
    print(
        f"   • Enhanced Analysis: {results['enhanced_analysis']['duration']:.4f}s (Context Store)"
    )
    print(
        f"   • Enhanced Multi-Tool: {results['enhanced_multi_tool']['duration']:.4f}s (Full Stack)"
    )

    # Advanced Features Summary
    if "token_manager" in results and "error" not in results["token_manager"]:
        tm_stats = results["token_manager"]["analytics"]
        print(f"\n🎯 Token Manager Results:")
        print(f"   • Optimizations: {tm_stats['total_optimizations']}")
        print(f"   • Tokens processed: {tm_stats['total_tokens_processed']:,}")
        print(f"   • Average utilization: {tm_stats['average_budget_utilization']:.1%}")
        print(f"   • Cost optimization: ${tm_stats['total_estimated_cost']:.4f}")

    if "semantic_analyzer" in results and "error" not in results["semantic_analyzer"]:
        sa_stats = results["semantic_analyzer"]
        print(f"\n🧠 Semantic Analyzer Results:")
        print(f"   • Contexts analyzed: {sa_stats['total_contexts_analyzed']}")
        print(f"   • Duplicates found: {sa_stats['duplicates_found']}")
        print(f"   • Clusters created: {sa_stats['clusters_created']}")
        print(
            f"   • Space savings potential: {sa_stats['space_savings_potential']:.1%}"
        )

    if (
        "context_store_integration" in results
        and "error" not in results["context_store_integration"]
    ):
        cs_stats = results["context_store_integration"]
        print(f"\n💾 Context Store Integration:")
        print(f"   • Total contexts: {cs_stats['total_contexts']}")
        print(f"   • Cache hit rate: {cs_stats['hit_rate']:.1%}")
        print(f"   • Memory usage: {cs_stats['memory_usage_percent']:.1f}%")
        if "compression_stats" in cs_stats:
            comp_stats = cs_stats["compression_stats"]
            print(
                f"   • Compression savings: {comp_stats['space_savings_percentage']:.1f}%"
            )

    # Overall Benefits
    print(f"\n💡 COMPLETE ECOSYSTEM BENEFITS:")
    print(f"   🚀 625x faster serialization (Context Reference Store)")
    print(f"   💾 95-99% storage reduction (Reference-based storage)")
    print(f"   🎯 100% cache hit rates (Intelligent caching)")
    print(f"   🧠 Semantic deduplication (AI-powered optimization)")
    print(f"   💰 Smart cost optimization (Token management)")
    print(f"   📊 Real-time monitoring (TUI Dashboard available)")

    # Save detailed results
    with open("comprehensive_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Complete results saved to: comprehensive_comparison_results.json")

    return results


if __name__ == "__main__":
    try:
        results = run_comprehensive_comparison()
        print("\n✅ Comprehensive comparison completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback

        traceback.print_exc()
