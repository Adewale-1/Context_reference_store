#!/usr/bin/env python3
"""
Semantic Context Clustering & Deduplication Demo

Demonstrates advanced semantic analysis capabilities for intelligent context
management using embeddings, clustering algorithms, and similarity detection.

Features demonstrated:
- Semantic similarity detection beyond exact matches
- Intelligent context clustering and grouping
- Content deduplication using semantic understanding
- Semantic search and retrieval
- Context relationship mapping
- Quality-aware optimization strategies

Usage:
    python examples/semantic_analysis_demo.py
"""

import json
import time
from typing import Dict, List

try:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from context_store import (
        ContextReferenceStore,
        CacheEvictionPolicy,
        create_semantic_analyzer,
    )

    CONTEXT_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Context Reference Store not available: {e}")
    print("Run from the library root directory or install with: pip install -e .")
    CONTEXT_STORE_AVAILABLE = False


def generate_semantically_related_content() -> Dict[str, str]:
    """Generate content with various levels of semantic similarity for testing."""

    return {
        # AI/ML cluster - high semantic similarity
        "ai_intro_1": """
        Artificial Intelligence (AI) represents a revolutionary approach to computing where machines
        are designed to simulate human cognitive processes. Machine learning, a subset of AI,
        enables systems to automatically improve their performance through experience without
        being explicitly programmed for every scenario.
        """,
        "ai_intro_2": """
        Machine learning and artificial intelligence are transforming how computers process
        information and make decisions. These technologies allow systems to learn from data
        and improve their capabilities over time, mimicking aspects of human intelligence
        and reasoning.
        """,
        "ai_technical": """
        Deep learning neural networks utilize multiple hidden layers to extract hierarchical
        features from raw data. These architectures, inspired by biological neural systems,
        can automatically discover complex patterns and representations that traditional
        algorithms struggle to identify.
        """,
        # Software development cluster - medium semantic similarity
        "code_python_1": """
        def process_data(input_list):
            '''Process a list of data items and return results'''
            results = []
            for item in input_list:
                processed = transform_item(item)
                validated = validate_result(processed)
                if validated:
                    results.append(processed)
            return results
        """,
        "code_python_2": """
        class DataProcessor:
            def __init__(self, config):
                self.config = config
                
            def process_batch(self, data_batch):
                processed_items = []
                for item in data_batch:
                    result = self.transform(item)
                    if self.validate(result):
                        processed_items.append(result)
                return processed_items
        """,
        "code_javascript": """
        function processUserData(userData) {
            const results = [];
            userData.forEach(user => {
                const processed = transformUser(user);
                if (validateUser(processed)) {
                    results.push(processed);
                }
            });
            return results;
        }
        """,
        # Business/sales cluster - medium semantic similarity
        "sales_q1": """
        Q1 2024 Sales Performance Summary:
        Total Revenue: $2.5M
        New Customers: 150
        Customer Retention: 85%
        Top Products: AI Platform (60%), Analytics Tools (25%), Consulting (15%)
        Key Markets: North America (50%), Europe (30%), Asia-Pacific (20%)
        """,
        "sales_q2": """
        Second Quarter 2024 Business Results:
        Revenue Achievement: $3.2M (28% growth)
        Customer Acquisition: 180 new clients
        Retention Rate: 87%
        Product Performance: AI Solutions leading with 65% of sales
        Geographic Distribution: US/Canada 48%, EMEA 32%, APAC 20%
        """,
        "marketing_analysis": """
        Marketing Campaign Performance Analysis:
        Digital Advertising ROI: 350%
        Lead Generation: 2,400 qualified leads
        Conversion Rate: 12.5%
        Channel Performance: Social Media (40%), Email (35%), Content Marketing (25%)
        Customer Acquisition Cost: $150 per customer
        """,
        # Technical documentation cluster - lower semantic similarity
        "api_doc_1": """
        # API Endpoint Documentation
        
        ## POST /api/v1/users
        Create a new user account
        
        **Request Body:**
        ```json
        {
            "username": "string",
            "email": "string",
            "password": "string"
        }
        ```
        
        **Response:**
        ```json
        {
            "user_id": "uuid",
            "status": "created",
            "timestamp": "ISO-8601"
        }
        ```
        """,
        "system_requirements": """
        System Requirements and Specifications:
        
        Minimum Hardware:
        - CPU: Intel i5 or AMD Ryzen 5
        - RAM: 8GB DDR4
        - Storage: 256GB SSD
        - Network: Broadband Internet connection
        
        Software Dependencies:
        - Operating System: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
        - Runtime: Python 3.8+, Node.js 14+
        - Database: PostgreSQL 12+ or MySQL 8+
        """,
        # Unique/outlier content - low semantic similarity to others
        "recipe": """
        Classic Chocolate Chip Cookie Recipe:
        
        Ingredients:
        - 2(1/2) cups all-purpose flour
        - 1 tsp baking soda
        - 1 tsp salt
        - 1 cup butter, softened
        - 3/4 cup granulated sugar
        - 3/4 cup brown sugar
        - 2 large eggs
        - 2 tsp vanilla extract
        - 2 cups chocolate chips
        
        Instructions:
        1. Preheat oven to 375°F
        2. Mix dry ingredients in a bowl
        3. Cream butter and sugars, add eggs and vanilla
        4. Combine wet and dry ingredients, fold in chocolate chips
        5. Bake for 9-11 minutes until golden brown
        """,
        "weather_report": """
        Weather Forecast - San Francisco Bay Area:
        
        Today: Partly cloudy with temperatures reaching 72°F
        Tonight: Clear skies, low around 55°F
        Tomorrow: Sunny with light winds, high 75°F
        
        Weekend Outlook:
        Saturday: Sunny, 78°F
        Sunday: Partly cloudy, 74°F
        
        Air Quality: Good (AQI 45)
        UV Index: 6 (High)
        Humidity: 65%
        """,
        # Near-duplicates for deduplication testing
        "duplicate_test_1": """
        The quick brown fox jumps over the lazy dog. This pangram contains
        every letter of the alphabet and is commonly used for testing fonts
        and keyboard layouts. It's a classic example in typography and
        computer science education.
        """,
        "duplicate_test_2": """
        A quick brown fox leaps over the lazy dog. This sentence includes
        all letters of the alphabet and is frequently used for testing fonts
        and keyboard layouts. It serves as a classic example in typography
        and computer science training.
        """,
    }


def demonstrate_semantic_analysis():
    """Demonstrate comprehensive semantic analysis capabilities."""

    if not CONTEXT_STORE_AVAILABLE:
        print("Cannot run demonstration - Context Reference Store not available")
        return

    print("SEMANTIC CONTEXT CLUSTERING & DEDUPLICATION DEMONSTRATION")
    print("=" * 68)
    print()
    print("This demo showcases advanced semantic understanding capabilities:")
    print("Semantic similarity detection beyond exact text matching")
    print("Intelligent context clustering and grouping")
    print("Content deduplication using meaning-based analysis")
    print("Semantic search and relationship mapping")
    print()

    # Create context store with all features enabled
    print("Initializing Context Store with Full Feature Set...")
    context_store = ContextReferenceStore(
        cache_size=100,
        eviction_policy=CacheEvictionPolicy.LRU,
        enable_compression=True,  # Enable compression for efficiency
        enable_cache_warming=True,
        use_disk_storage=True,
    )

    # Generate and store semantically related content
    print("Generating semantically diverse content samples...")
    content_samples = generate_semantically_related_content()

    context_ids = []
    for content_name, content in content_samples.items():
        context_id = context_store.store(
            content,
            metadata={
                "content_name": content_name,
                "demo_category": content_name.split("_")[0],  # Extract category
                "semantic_demo": True,
            },
        )
        context_ids.append(context_id)

    print(f"Stored {len(content_samples)} semantically diverse contexts")

    # Demonstrate basic semantic insights
    print("\nAnalyzing Content Characteristics...")

    insights = context_store.get_semantic_insights()

    if "error" in insights:
        print(f"{insights['message']}")
        print("Note: Semantic features require sentence-transformers library")
        print("Install with: pip install sentence-transformers")
    else:
        stats = insights["content_statistics"]
        content_dist = insights["content_type_distribution"]

        print(f"Content Statistics:")
        print(f"Total contexts: {insights['total_contexts']}")
        print(f"Average length: {stats['average_length']:.0f} characters")
        print(
            f"Content range: {stats['min_length']}-{stats['max_length']} characters"
        )

        print(f"\nContent Type Distribution:")
        for content_type, count in content_dist.items():
            if count > 0:
                percentage = (count / insights["total_contexts"]) * 100
                print(
                    f"{content_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
                )

        if insights["semantic_insights"]:
            print(f"\nSemantic Insights:")
            for insight in insights["semantic_insights"]:
                print(f"{insight}")

    # Demonstrate semantic pattern analysis
    print("\nPerforming Deep Semantic Analysis...")

    semantic_analysis = context_store.analyze_semantic_patterns(
        similarity_threshold=0.75, clustering_algorithm="dbscan"
    )

    if "error" in semantic_analysis:
        print(f"{semantic_analysis['message']}")

        # Show fallback analysis using basic methods
        print("Demonstrating fallback semantic analysis...")

        # Create standalone semantic analyzer for demo
        try:
            analyzer = create_semantic_analyzer(similarity_threshold=0.75)
            result = analyzer.analyze_contexts(content_samples)

            print(f"Fallback Analysis Results:")
            print(f"Contexts analyzed: {result.total_contexts_analyzed}")
            print(f"Semantic matches found: {result.duplicates_found}")
            print(f"Clusters created: {result.clusters_created}")
            print(f"Processing time: {result.processing_time_ms:.1f}ms")

            if result.similarity_matches:
                print(f"\nTop Semantic Matches:")
                for i, match in enumerate(result.similarity_matches[:3], 1):
                    print(f"      {i}. {match.context_id_1} ↔ {match.context_id_2}")
                    print(f"         Similarity: {match.similarity_score:.2f}")
                    print(f"         Confidence: {match.confidence:.2f}")
                    print(f"         Reasons: {', '.join(match.match_reasons[:2])}")

            if result.clusters:
                print(f"\nSemantic Clusters:")
                for i, cluster in enumerate(result.clusters[:3], 1):
                    print(
                        f"{i}. {cluster.cluster_id}: {len(cluster.context_ids)} contexts"
                    )
                    print(f"Theme: {cluster.semantic_theme}")
                    print(f"Quality: {cluster.quality_score:.2f}")

        except Exception as e:
            print(f"Fallback analysis failed: {e}")

    else:
        print(f"Semantic Analysis Results:")
        print(
            f"Contexts analyzed: {semantic_analysis['total_contexts_analyzed']}"
        )
        print(
            f"Semantic duplicates: {semantic_analysis['semantic_duplicates_found']}"
        )
        print(f"Clusters created: {semantic_analysis['clusters_created']}")
        print(
            f"Space savings potential: {semantic_analysis['space_savings_potential']:.1%}"
        )
        print(
            f"Processing time: {semantic_analysis['processing_time_ms']:.1f}ms"
        )

        # Show similarity matches
        if semantic_analysis["similarity_matches"]:
            print(f"\nSemantic Similarity Matches:")
            for i, match in enumerate(semantic_analysis["similarity_matches"][:3], 1):
                print(f"      {i}. {match['context_id_1']} ↔ {match['context_id_2']}")
                print(f"         Similarity: {match['similarity_score']:.2f}")
                print(f"         Action: {match['suggested_action']}")
                print(f"         Reasons: {', '.join(match['match_reasons'][:2])}")

        # Show clusters
        if semantic_analysis["clusters"]:
            print(f"\nSemantic Clusters:")
            for i, cluster in enumerate(semantic_analysis["clusters"][:3], 1):
                print(f"      {i}. {cluster['cluster_id']}: {cluster['size']} contexts")
                print(f"         Theme: {cluster['semantic_theme']}")
                print(
                    f"         Representative: {cluster['representative_context_id']}"
                )
                print(f"         Quality: {cluster['quality_score']:.2f}")

    # Demonstrate semantic search
    print("\nTesting Semantic Search Capabilities...")

    search_queries = [
        "artificial intelligence and machine learning",
        "software development and programming",
        "sales performance and revenue",
        "chocolate chip cookies recipe",
    ]

    for query in search_queries:
        print(f"\nSearch Query: '{query}'")

        search_results = context_store.find_similar_contexts(
            query_context=query,
            similarity_threshold=0.3,  # Lower threshold to find more results
            max_results=3,
        )

        if "error" in search_results:
            print(f"{search_results['message']}")
        else:
            print(
                f"Found {search_results['returned_results']} similar contexts"
            )

            for i, result in enumerate(search_results["similar_contexts"], 1):
                print(
                    f"         {i}. {result['context_id']} (similarity: {result['similarity_score']:.2f})"
                )
                print(f"            Preview: {result['content_preview'][:80]}...")

    # Demonstrate semantic optimization
    print("\nSemantic Storage Optimization Analysis...")

    optimization_analysis = context_store.optimize_semantic_storage(
        similarity_threshold=0.85, dry_run=True  # Don't actually modify data
    )

    if "error" in optimization_analysis:
        print(f"{optimization_analysis['message']}")
    else:
        print(f"Optimization Analysis (Dry Run):")
        print(
            f"Contexts analyzed: {optimization_analysis['total_contexts_analyzed']}"
        )
        print(
            f"Duplicates identified: {optimization_analysis['duplicates_found']}"
        )
        print(
            f"Potential space savings: {optimization_analysis['potential_savings_percentage']:.1f}%"
        )
        print(
            f"Bytes that could be saved: {optimization_analysis['potential_savings_bytes']:,}"
        )

        if optimization_analysis["optimization_plan"]:
            print(f"\nOptimization Plan:")
            for i, plan in enumerate(optimization_analysis["optimization_plan"][:3], 1):
                print(f"{i}. {plan['action'].replace('_', ' ').title()}")
                print(f"Keep: {plan['keep_context_id']}")
                print(f"Remove: {plan['remove_context_id']}")
                print(f"Similarity: {plan['similarity_score']:.2f}")
                print(f"Space saved: {plan['space_saved_bytes']:,} bytes")

        # Show recommendations
        active_recommendations = [
            r for r in optimization_analysis["recommendations"] if r.strip()
        ]
        if active_recommendations:
            print(f"\nOptimization Recommendations:")
            for rec in active_recommendations[:3]:
                print(f"{rec}")

    # Performance comparison and summary
    print("\nPerformance Impact Summary...")

    print("Semantic Analysis Benefits:")
    print(
        "Beyond hash-based deduplication: Finds similar content even with different wording"
    )
    print(
        "Intelligent clustering: Groups related contexts for efficient organization"
    )
    print(
        "Semantic search: Find relevant content using natural language queries"
    )
    print(
        "Quality-aware optimization: Preserves the best version when merging duplicates"
    )
    print(
        "Content understanding: Analyzes patterns and provides actionable insights"
    )

    print("\nCombined Value Proposition:")
    print("Base Context Reference Store: 625x serialization speedup")
    print("+ Smart Compression: 10-50x additional storage reduction")
    print("+ Token Optimization: Intelligent LLM cost management")
    print("+ Semantic Analysis: Meaning-based deduplication and search")
    print("+ Real-time TUI Dashboard: Visual monitoring and insights")

    print("\nUse Cases Enabled:")
    print("Intelligent content deduplication beyond exact matches")
    print("Semantic search for knowledge management systems")
    print("Automated content clustering and organization")
    print("Quality-preserving content optimization")
    print("Context relationship discovery and mapping")

    print("\nSemantic analysis demonstration completed successfully!")
    print(
        "Your Context Reference Store now includes advanced semantic understanding!"
    )


def main():
    """Run the comprehensive semantic analysis demonstration."""
    demonstrate_semantic_analysis()


if __name__ == "__main__":
    main()
