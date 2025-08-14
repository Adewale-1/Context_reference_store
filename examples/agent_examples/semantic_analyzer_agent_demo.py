#!/usr/bin/env python3
"""
Semantic Analyzer Integration with Context-Enhanced Agents

This demo shows advanced semantic analysis for context deduplication,
clustering, and optimization using embedding-based similarity detection.

Features:
- Semantic similarity detection beyond exact matches
- Intelligent context clustering and grouping
- Content deduplication with quality preservation
- Semantic search and relationship mapping
- Performance analytics and space savings calculation
"""

import sys
import os
import time
import json

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from context_store.semantic.semantic_analyzer import (
    create_semantic_analyzer,
    SimilarityMethod,
    ClusteringAlgorithm,
    MergeStrategy,
)

from context_enhanced_multi_tool_agent.agent import (
    advanced_text_analysis_with_context,
    read_file_with_context_cache,
    _context_store,
)


class MockToolContext:
    def __init__(self):
        self.state = {}


def demo_semantic_duplicate_detection():
    """Demonstrate semantic duplicate detection capabilities."""
    print("🧠 SEMANTIC DUPLICATE DETECTION DEMO")
    print("=" * 50)

    # Create semantic analyzer
    analyzer = create_semantic_analyzer(
        similarity_threshold=0.75, clustering_algorithm="dbscan"
    )

    # Sample contexts with semantic similarities (not exact duplicates)
    contexts = {
        "ctx_1": "Machine learning is a subset of artificial intelligence that focuses on learning algorithms.",
        "ctx_2": "AI and machine learning technologies are transforming how we approach complex computational problems.",
        "ctx_3": "The Context Reference Store provides 625x faster serialization compared to traditional methods.",
        "ctx_4": "Context Reference Store technology delivers 625x performance improvement in serialization speed.",
        "ctx_5": "Deep learning uses neural networks with multiple layers to learn complex data patterns.",
        "ctx_6": "Neural networks with many layers enable deep learning systems to understand intricate patterns.",
        "ctx_7": "Python is a popular programming language for machine learning and data science applications.",
        "ctx_8": "Data scientists frequently use Python programming language for ML and analytics projects.",
        "ctx_9": "Token optimization helps reduce costs while maintaining quality in LLM applications.",
        "ctx_10": "Smart token management can significantly decrease LLM operational costs without quality loss.",
    }

    print(f"📊 Analyzing {len(contexts)} contexts for semantic duplicates...")

    # Find semantic matches
    matches = analyzer.find_semantic_duplicates(contexts, SimilarityMethod.EMBEDDING)

    print(f"\n🔍 Found {len(matches)} semantic matches:")
    for i, match in enumerate(matches):
        print(f"\n   Match {i+1}:")
        print(f"   📄 {match.context_id_1} ↔ {match.context_id_2}")
        print(f"   📊 Similarity: {match.similarity_score:.3f}")
        print(f"   🎯 Confidence: {match.confidence:.3f}")
        print(f"   💡 Suggested action: {match.suggested_action}")
        print(f"   🔍 Reasons: {', '.join(match.match_reasons)}")

        # Show the actual content being matched
        content1 = (
            contexts[match.context_id_1][:100] + "..."
            if len(contexts[match.context_id_1]) > 100
            else contexts[match.context_id_1]
        )
        content2 = (
            contexts[match.context_id_2][:100] + "..."
            if len(contexts[match.context_id_2]) > 100
            else contexts[match.context_id_2]
        )
        print(f"   📝 Content 1: {content1}")
        print(f"   📝 Content 2: {content2}")


def demo_semantic_clustering():
    """Demonstrate semantic clustering of contexts."""
    print("\n🗂️  SEMANTIC CLUSTERING DEMO")
    print("=" * 40)

    analyzer = create_semantic_analyzer(
        similarity_threshold=0.70, clustering_algorithm="dbscan"
    )

    # Larger set of contexts for clustering
    contexts = {
        # AI/ML cluster
        "ai_1": "Artificial intelligence is revolutionizing various industries with intelligent automation.",
        "ai_2": "Machine learning algorithms can process vast amounts of data to identify patterns.",
        "ai_3": "Deep learning networks use multiple layers to understand complex relationships in data.",
        "ai_4": "Neural networks are computational models inspired by biological brain structures.",
        # Context Store cluster
        "cs_1": "Context Reference Store provides revolutionary performance improvements for AI applications.",
        "cs_2": "The library delivers 625x faster serialization and 99.55% storage reduction benefits.",
        "cs_3": "Advanced caching strategies including LRU, LFU, and TTL optimize context management.",
        "cs_4": "Reference-based storage eliminates duplication while maintaining full functionality.",
        # Programming cluster
        "prog_1": "Python is widely used for data science, machine learning, and AI development.",
        "prog_2": "JavaScript enables interactive web applications and modern user interfaces.",
        "prog_3": "Software engineering practices ensure maintainable and scalable code architecture.",
        "prog_4": "Code optimization techniques improve performance and reduce resource consumption.",
        # Performance cluster
        "perf_1": "System performance monitoring helps identify bottlenecks and optimization opportunities.",
        "perf_2": "Memory management is crucial for efficient application performance and stability.",
        "perf_3": "Caching strategies significantly improve response times and reduce computational overhead.",
        "perf_4": "Load balancing distributes work across multiple systems for better throughput.",
    }

    print(f"📊 Creating semantic clusters from {len(contexts)} contexts...")

    # Create clusters
    clusters = analyzer.create_semantic_clusters(contexts)

    print(f"\n🎯 Created {len(clusters)} semantic clusters:")

    for i, cluster in enumerate(clusters):
        print(f"\n   Cluster {i+1}: {cluster.cluster_id}")
        print(f"   📄 Contexts: {len(cluster.context_ids)} items")
        print(f"   🎯 Theme: {cluster.semantic_theme}")
        print(f"   ⭐ Quality Score: {cluster.quality_score:.2f}")
        print(f"   🏆 Representative: {cluster.representative_context_id}")
        print(f"   📝 Summary: {cluster.cluster_summary[:100]}...")
        print(f"   📋 Context IDs: {', '.join(cluster.context_ids)}")


def demo_comprehensive_semantic_analysis():
    """Demonstrate comprehensive semantic analysis with all features."""
    print("\n🚀 COMPREHENSIVE SEMANTIC ANALYSIS")
    print("=" * 45)

    analyzer = create_semantic_analyzer(
        similarity_threshold=0.80, clustering_algorithm="hierarchical"
    )

    # Generate contexts using the enhanced agent for realistic data
    print("📝 Generating realistic contexts using Context-Enhanced Agent...")

    mock_context = MockToolContext()
    generated_contexts = {}

    sample_texts = [
        "Context Reference Store optimization and performance",
        "Machine learning algorithms and neural networks",
        "Token management and cost optimization strategies",
        "Semantic analysis and content clustering techniques",
        "Real-time monitoring and analytics dashboards",
        "Artificial intelligence and automation systems",
        "Python programming and software development",
        "Data science and statistical analysis methods",
    ]

    for i, text in enumerate(sample_texts):
        try:
            # Generate enhanced analysis
            result = advanced_text_analysis_with_context(
                text * 25, mock_context
            )  # Make substantial

            if result.get("status") == "success":
                # Create rich context from analysis
                context_content = (
                    f"Topic: {text}\n"
                    f"Analysis: {result.get('basic_metrics', {})}\n"
                    f"Content extracted: {len(result.get('content_extraction', {}).get('urls', []))} URLs, "
                    f"{len(result.get('content_extraction', {}).get('emails', []))} emails\n"
                    f"Readability: {result.get('readability', {}).get('reading_level', 'Unknown')}\n"
                    f"Word analysis: {result.get('word_analysis', {}).get('average_length', 0):.1f} avg word length"
                )

                generated_contexts[f"generated_{i}"] = context_content
                print(f"   ✅ Generated context {i+1} ({len(context_content)} chars)")

        except Exception as e:
            print(f"   ⚠️  Error generating context {i+1}: {e}")

    if not generated_contexts:
        print("   ⚠️  No contexts generated, using fallback data...")
        generated_contexts = {
            "fallback_1": "Context Reference Store provides advanced optimization features",
            "fallback_2": "Machine learning systems require efficient context management",
            "fallback_3": "Performance monitoring enables proactive system optimization",
        }

    print(
        f"\n🔍 Performing comprehensive analysis on {len(generated_contexts)} contexts..."
    )

    # Perform comprehensive analysis
    analysis_result = analyzer.analyze_contexts(generated_contexts)

    print(f"\n📊 Analysis Results:")
    print(f"   📄 Contexts analyzed: {analysis_result.total_contexts_analyzed}")
    print(f"   🔍 Semantic duplicates: {analysis_result.duplicates_found}")
    print(f"   🗂️  Clusters created: {analysis_result.clusters_created}")
    print(
        f"   💾 Space savings potential: {analysis_result.space_savings_potential:.1%}"
    )
    print(
        f"   ⭐ Quality improvement potential: {analysis_result.quality_improvement_potential:.1%}"
    )
    print(f"   ⏱️  Processing time: {analysis_result.processing_time_ms:.1f}ms")

    # Show recommendations
    if analysis_result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in analysis_result.recommendations:
            print(f"   • {rec}")

    # Show detailed match information
    if analysis_result.similarity_matches:
        print(f"\n🔍 Top Similarity Matches:")
        for match in analysis_result.similarity_matches[:3]:
            print(f"   • {match.context_id_1} ↔ {match.context_id_2}")
            print(
                f"     Similarity: {match.similarity_score:.3f} | Action: {match.suggested_action}"
            )

    # Show cluster themes
    if analysis_result.clusters:
        print(f"\n🎯 Cluster Themes:")
        for cluster in analysis_result.clusters:
            print(
                f"   • {cluster.cluster_id}: {cluster.semantic_theme} (Quality: {cluster.quality_score:.2f})"
            )


def demo_integration_with_context_store():
    """Show integration between Semantic Analyzer and Context Reference Store."""
    print("\n🔗 SEMANTIC ANALYZER + CONTEXT STORE INTEGRATION")
    print("=" * 55)

    # Use analyzer to optimize contexts before storing in Context Reference Store
    analyzer = create_semantic_analyzer(similarity_threshold=0.85)
    mock_context = MockToolContext()

    print("📝 Step 1: Generate content using Context-Enhanced Agent...")

    # Generate some content first
    test_files = [
        "context_enhanced_multi_tool_agent/agent.py",
        "basic_analysis_agent/agent.py",
    ]

    file_contexts = {}
    for file_path in test_files:
        try:
            result = read_file_with_context_cache(file_path, mock_context)
            if result.get("status") == "success":
                # Extract metadata as context
                metadata = result.get("metadata", {})
                context_content = (
                    f"File: {file_path}\n"
                    f"Size: {metadata.get('size', 0)} bytes\n"
                    f"Lines: {metadata.get('lines', 0)}\n"
                    f"Words: {metadata.get('words', 0)}\n"
                    f"Modified: {metadata.get('modified', 'Unknown')}"
                )
                file_contexts[f"file_{len(file_contexts)}"] = context_content
                print(f"   ✅ Processed {file_path}")
        except Exception as e:
            print(f"   ⚠️  Error processing {file_path}: {e}")

    if file_contexts:
        print(
            f"\n🔍 Step 2: Semantic analysis of {len(file_contexts)} file contexts..."
        )

        # Analyze for duplicates and clusters
        result = analyzer.analyze_contexts(file_contexts)

        print(f"   📊 Found {result.duplicates_found} potential duplicates")
        print(f"   🗂️  Created {result.clusters_created} clusters")
        print(f"   💾 Potential space savings: {result.space_savings_potential:.1%}")

        # Show Context Reference Store statistics
        print(f"\n💾 Step 3: Context Reference Store Statistics:")
        try:
            stats = _context_store.get_cache_stats()
            print(f"   📈 Total contexts stored: {stats.get('total_contexts', 0)}")
            print(f"   🎯 Cache hit rate: {stats.get('hit_rate', 0):.1%}")
            print(f"   🧠 Memory usage: {stats.get('memory_usage_percent', 0):.1f}%")
            print(f"   📤 Cache evictions: {stats.get('total_evictions', 0)}")

            # Calculate combined benefits
            context_store_efficiency = 95.0  # From previous demos
            semantic_efficiency = result.space_savings_potential * 100
            combined_efficiency = context_store_efficiency + semantic_efficiency

            print(f"\n🚀 Combined Optimization Benefits:")
            print(
                f"   ⚡ Context Reference Store: {context_store_efficiency:.1f}% storage efficiency"
            )
            print(
                f"   🧠 Semantic Analysis: {semantic_efficiency:.1f}% additional savings"
            )
            print(f"   🎯 Total Efficiency: {combined_efficiency:.1f}% optimization")

        except Exception as e:
            print(f"   ⚠️  Could not retrieve Context Store stats: {e}")

    else:
        print("   ⚠️  No file contexts generated for analysis")


def demo_different_similarity_methods():
    """Demonstrate different similarity calculation methods."""
    print("\n⚖️  SIMILARITY METHOD COMPARISON")
    print("=" * 40)

    # Test contexts
    contexts = {
        "original": "Context Reference Store provides revolutionary performance improvements",
        "similar": "The Context Reference Store delivers revolutionary performance enhancements",
        "related": "Performance improvements are essential for efficient context management systems",
        "different": "Machine learning algorithms process data using neural network architectures",
    }

    methods = [
        SimilarityMethod.EMBEDDING,
        SimilarityMethod.COSINE,
        SimilarityMethod.JACCARD,
        SimilarityMethod.HYBRID,
    ]

    analyzer = create_semantic_analyzer()

    print("📊 Comparing similarity methods:")

    base_content = contexts["original"]

    for method in methods:
        print(f"\n   🔍 Method: {method.value}")

        for ctx_id, content in contexts.items():
            if ctx_id != "original":
                similarity = analyzer.calculate_similarity(
                    base_content, content, method
                )
                print(f"      {ctx_id:>10}: {similarity:.3f}")


def main():
    """Main demo function."""
    print("🚀 Context Reference Store - Semantic Analysis Integration")
    print("This demo showcases advanced semantic analysis capabilities")
    print()

    try:
        # Run all demos
        demo_semantic_duplicate_detection()
        demo_semantic_clustering()
        demo_comprehensive_semantic_analysis()
        demo_integration_with_context_store()
        demo_different_similarity_methods()

        # Show analyzer statistics
        analyzer = create_semantic_analyzer()
        stats = analyzer.get_analysis_statistics()

        print(f"\n📈 Semantic Analyzer Statistics:")
        print(f"   🔄 Total analyses: {stats['total_analyses']}")
        print(f"   📄 Contexts processed: {stats['contexts_processed']}")
        print(f"   🧠 Embeddings computed: {stats['embeddings_computed']}")
        print(f"   💾 Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
        print(f"   🔍 Duplicates found: {stats['duplicates_found']}")
        print(f"   🗂️  Clusters created: {stats['clusters_created']}")

        print("\n✅ Semantic Analysis demo completed successfully!")
        print("\n💡 Key Benefits Demonstrated:")
        print("   🧠 Semantic similarity detection beyond exact matches")
        print("   🗂️  Intelligent content clustering and organization")
        print("   💾 Advanced deduplication with quality preservation")
        print("   🔍 Multiple similarity calculation methods")
        print("   🚀 Integration with Context Reference Store")

    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
