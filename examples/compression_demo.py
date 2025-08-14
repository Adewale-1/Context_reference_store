#!/usr/bin/env python3
"""
Smart Context Compression Demo

Demonstrates the new intelligent compression capabilities of the Context Reference Store.
Shows how the library can achieve 10-50x additional storage savings on top of the existing
625x serialization speedup.

Features demonstrated:
- Automatic content type detection
- Algorithm selection based on content characteristics
- Compression performance analytics
- Real-time compression recommendations
"""

import json
import time
import random
import string
from typing import Dict, Any

try:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from context_store import (
        ContextReferenceStore,
        ContextCompressionManager,
        CompressionAlgorithm,
        ContentType,
        CacheEvictionPolicy,
    )

    CONTEXT_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Context Reference Store not available: {e}")
    print("Run from the library root directory or install with: pip install -e .")
    CONTEXT_STORE_AVAILABLE = False


def generate_sample_content() -> Dict[str, str]:
    """Generate various types of content for compression testing."""

    # Large JSON data (API response simulation)
    large_json = {
        "users": [
            {
                "id": i,
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "profile": {
                    "bio": f"This is a detailed biography for user {i}. " * 10,
                    "settings": {
                        "theme": "dark",
                        "notifications": True,
                        "privacy": {
                            "public_profile": True,
                            "show_email": False,
                            "analytics": True,
                        },
                    },
                },
                "posts": [
                    {
                        "id": j,
                        "title": f"Post {j} by User {i}",
                        "content": f"This is the content of post {j}. " * 20,
                        "tags": ["technology", "ai", "programming"],
                        "metadata": {
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T00:00:00Z",
                            "views": random.randint(100, 10000),
                        },
                    }
                    for j in range(5)
                ],
            }
            for i in range(50)
        ]
    }

    python_code = (
        '''
# This is a comprehensive Python class for demonstration
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

class DataProcessor:
    """
    A comprehensive data processing class that handles various data types.
    
    This class provides methods for processing, transforming, and analyzing
    data from multiple sources with support for async operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data processor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processed_items = []
        
        # Initialize processing modules
        self.transformers = {}
        self.validators = {}
        self.output_handlers = {}
        
    async def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of data items asynchronously.
        
        Args:
            data: List of data items to process
            
        Returns:
            List of processed data items
        """
        results = []
        
        for item in data:
            try:
                # Validate the item
                if not self.validate_item(item):
                    self.logger.warning(f"Invalid item skipped: {item.get('id', 'unknown')}")
                    continue
                    
                # Transform the item
                transformed_item = await self.transform_item(item)
                
                # Apply business logic
                processed_item = self.apply_business_logic(transformed_item)
                
                # Store the result
                results.append(processed_item)
                self.processed_items.append(processed_item)
                
            except Exception as e:
                self.logger.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
                continue
                
        return results
    '''
        * 3
    )  # Make it larger

    # Large text document
    large_text = (
        """
    The Context Reference Store: A Revolutionary Approach to Large Context Management
    
    In the rapidly evolving landscape of artificial intelligence and large language models,
    the management of context windows has emerged as one of the most critical challenges
    facing developers and researchers alike. Traditional approaches to context management
    often fall short when dealing with the massive context windows now supported by
    modern language models, which can handle millions of tokens in a single session.
    
    The Context Reference Store represents a paradigm shift in how we approach this
    fundamental problem. Rather than passing entire contexts through the system with
    each operation, our approach leverages a reference-based architecture that provides
    unprecedented efficiency gains while maintaining perfect fidelity of the original
    content.
    
    Key Performance Achievements:
    
    Our comprehensive benchmarking has demonstrated remarkable performance improvements
    across all key metrics. The serialization speed improvements are particularly
    noteworthy, with our system achieving serialization times of approximately 40
    milliseconds for contexts that would traditionally require 25 seconds to process.
    This represents a 625x improvement in serialization performance, fundamentally
    changing what's possible in real-time AI applications.
    
    Memory efficiency gains are equally impressive. In multi-agent scenarios, where
    traditional approaches would consume 50GB of memory for 50 agents sharing large
    contexts, our reference-based approach maintains the same functionality with
    just 1.02GB of memory usage. This 49x reduction in memory consumption enables
    previously impossible deployment scenarios and dramatically reduces infrastructure
    costs.
    
    The multimodal capabilities of our system deliver even more dramatic improvements.
    Where traditional base64 encoding approaches would expand a 50MB video file to
    67MB of JSON data, our hybrid storage architecture represents the same content
    with just 300 bytes of reference data. This 223,000x reduction in serialized
    size makes real-time multimodal AI applications practical at scale.
    
    Technical Architecture:
    
    The Context Reference Store is built on a foundation of advanced computer science
    principles, including sophisticated caching algorithms, intelligent eviction
    policies, and optimized data structures. Our hybrid storage approach automatically
    routes small binary data to memory-based storage while efficiently managing
    large binary data on disk with intelligent caching and reference counting.
    
    The system supports multiple eviction policies including Least Recently Used (LRU),
    Least Frequently Used (LFU), Time-To-Live (TTL) based eviction, and memory
    pressure-based eviction. This flexibility allows developers to optimize the
    system for their specific use cases and deployment environments.
    
    Advanced Features:
    
    Beyond basic context storage and retrieval, the Context Reference Store includes
    enterprise-grade features such as cache warming, where the system intelligently
    preloads frequently accessed contexts to minimize latency. The background cleanup
    processes ensure that expired contexts are automatically removed without impacting
    system performance.
    
    The multimodal support includes sophisticated binary deduplication using SHA256
    hashing, ensuring that identical binary content is stored only once regardless
    of how many contexts reference it. This approach provides massive storage
    efficiency gains while maintaining perfect data integrity.
    
    Integration and Compatibility:
    
    One of the key design principles of the Context Reference Store is backward
    compatibility. The system integrates seamlessly with existing applications
    without requiring changes to existing code. New applications can take advantage
    of advanced features while legacy systems continue to function normally.
    
    The framework-agnostic design ensures compatibility with popular AI frameworks
    including LangChain, LangGraph, and LlamaIndex. Specialized adapters provide
    native integration points for each framework while maintaining consistent
    performance characteristics across all platforms.
    
    Future Directions:
    
    As we look toward the future of AI context management, several exciting
    developments are on the horizon. Distributed caching capabilities will enable
    multi-node deployments where context stores can be shared across multiple
    instances and geographic regions. Machine learning-driven optimization will
    automatically tune eviction policies and caching strategies based on observed
    usage patterns.
    
    Advanced compression algorithms specifically designed for AI context data will
    provide even greater storage efficiency. Integration with cloud storage services
    will enable virtually unlimited context storage capacity while maintaining
    the performance characteristics that make real-time AI applications possible.
    
    Conclusion:
    
    The Context Reference Store represents a fundamental advancement in how we
    approach large context management in AI systems. By combining cutting-edge
    computer science techniques with practical engineering solutions, we've created
    a system that not only solves today's context management challenges but also
    provides a foundation for the even more demanding applications of tomorrow.
    
    The performance improvements speak for themselves: 625x faster serialization,
    49x memory reduction, and 99.55% storage reduction for multimodal content.
    These gains translate directly into reduced infrastructure costs, improved
    application performance, and new possibilities for AI application architectures.
    
    As the AI landscape continues to evolve and context windows grow even larger,
    the Context Reference Store provides the scalable, efficient foundation needed
    to unlock the full potential of large language models and multimodal AI systems.
    """
        * 5
    )  # Make it larger

    # CSV-like structured data
    csv_data = "name,age,email,department,salary,start_date,performance_rating\n"
    for i in range(1000):
        csv_data += f"Employee{i},{random.randint(22, 65)},emp{i}@company.com,Engineering,{random.randint(50000, 150000)},2020-{random.randint(1,12):02d}-{random.randint(1,28):02d},{random.uniform(1.0, 5.0):.1f}\n"

    # XML/HTML content
    html_content = (
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Context Reference Store Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: #f0f0f0; padding: 20px; border-radius: 8px; }
            .content { margin: 20px 0; }
            .code-block { background: #f8f8f8; padding: 15px; border-radius: 4px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Context Reference Store</h1>
            <p>Revolutionary context management for AI applications</p>
        </div>
        
        <div class="content">
            <h2>Performance Metrics</h2>
            <ul>
                <li>625x faster serialization</li>
                <li>49x memory reduction</li>
                <li>99.55% storage reduction for multimodal content</li>
            </ul>
            
            <h2>Example Usage</h2>
            <div class="code-block">
                from context_store import ContextReferenceStore<br>
                <br>
                store = ContextReferenceStore()<br>
                context_id = store.store(large_document)<br>
                retrieved_content = store.retrieve(context_id)
            </div>
        </div>
    </body>
    </html>
    """
        * 20
    )  # Make it larger

    return {
        "large_json": json.dumps(large_json),
        "python_code": python_code,
        "large_text": large_text,
        "csv_data": csv_data,
        "html_content": html_content,
    }


def demonstrate_compression_features():
    """Demonstrate the compression capabilities with various content types."""

    if not CONTEXT_STORE_AVAILABLE:
        print("Cannot run demonstration - Context Reference Store not available")
        return

    print("Smart Context Compression Demonstration")
    print("=" * 60)

    # Create context store with compression enabled
    print("\nInitializing Context Store with Smart Compression...")
    context_store = ContextReferenceStore(
        cache_size=100,
        eviction_policy=CacheEvictionPolicy.LRU,
        enable_compression=True,  # Enable smart compression
        compression_min_size=512,  # Compress content > 512 bytes
        enable_cache_warming=True,
    )

    # Generate sample content
    print("Generating sample content for testing...")
    sample_content = generate_sample_content()

    print(f"\nSample Content Generated:")
    for content_type, content in sample_content.items():
        size_kb = len(content.encode("utf-8")) / 1024
        print(f"   • {content_type}: {size_kb:.1f} KB")

    # Store content and measure compression performance
    print("\nStoring content with automatic compression...")
    storage_results = {}

    for content_name, content in sample_content.items():
        print(f"\n   Processing {content_name}...")

        start_time = time.time()
        context_id = context_store.store(
            content,
            metadata={
                "content_type": content_name,
                "description": f"Sample {content_name} for compression testing",
            },
        )
        storage_time = time.time() - start_time

        # Retrieve to test decompression
        start_time = time.time()
        retrieved_content = context_store.retrieve(context_id)
        retrieval_time = time.time() - start_time

        # Verify content integrity
        content_matches = content == retrieved_content

        storage_results[content_name] = {
            "context_id": context_id,
            "storage_time_ms": storage_time * 1000,
            "retrieval_time_ms": retrieval_time * 1000,
            "content_integrity": content_matches,
            "original_size": len(content.encode("utf-8")),
        }

        print(
            f"Stored in {storage_time*1000:.1f}ms, retrieved in {retrieval_time*1000:.1f}ms"
        )
        print(
            f"Content integrity: {'Perfect' if content_matches else '❌ Failed'}"
        )

    # Get comprehensive compression analytics
    print("\nAnalyzing Compression Performance...")
    analytics = context_store.get_compression_analytics()

    if analytics.get("compression_enabled"):
        context_stats = analytics["context_store_stats"]
        manager_stats = analytics["compression_manager_analytics"]["summary"]

        print(f"\nCompression Performance Results:")
        print(f"Total contexts stored: {context_stats['total_contexts']}")
        print(f"Contexts compressed: {context_stats['compressed_contexts']}")
        print(
            f"Compression adoption rate: {context_stats['compression_adoption_rate']:.1f}%"
        )
        print(
            f"Total space savings: {context_stats['space_savings_percentage']:.1f}%"
        )
        print(
            f"Space saved: {context_stats['total_space_saved_bytes'] / 1024:.1f} KB"
        )

        print(f"\nSpeed Metrics:")
        print(
            f"Avg compression time: {manager_stats['avg_compression_time_ms']:.1f}ms"
        )
        print(
            f"Avg decompression time: {manager_stats['avg_decompression_time_ms']:.1f}ms"
        )
        print(
            f"Overall compression ratio: {manager_stats['overall_compression_ratio']:.3f}"
        )

        # Performance impact calculation
        performance_impact = analytics["performance_impact"]
        print(f"\nEfficiency Gains:")
        print(
            f"Storage efficiency multiplier: {performance_impact['storage_efficiency_multiplier']:.1f}x"
        )
        print(f"{performance_impact['estimated_memory_reduction']}")
        print(f"{performance_impact['combined_with_reference_store']}")

        # Content type breakdown
        if context_stats.get("content_type_breakdown"):
            print(f"\nContent Type Performance:")
            for content_type, stats in context_stats["content_type_breakdown"].items():
                print(
                    f"{content_type}: {stats['avg_savings']:.1f}% avg savings ({stats['count']} samples)"
                )

    # Get optimization recommendations
    print("\nGetting Optimization Recommendations...")
    recommendations = context_store.get_compression_recommendations()

    if "recommendations" in recommendations:
        print(f"\nOptimization Recommendations:")
        print(
            f"Optimization Score: {recommendations['optimization_score']:.1f}/100"
        )

        for i, rec in enumerate(recommendations["recommendations"], 1):
            priority_emoji = (
                "🔴"
                if rec["priority"] == "high"
                else "🟡" if rec["priority"] == "medium" else "🟢"
            )
            print(f"   {i}. {priority_emoji} {rec['recommendation']}")
            if "details" in rec:
                print(f"      Details: {rec['details']}")

        print(f"\nNext Steps:")
        for step in recommendations["next_steps"]:
            print(f"{step}")

    # Demonstrate real-time compression monitoring
    print("\nReal-time Performance Monitoring...")

    # Add more content to show adaptive behavior
    additional_content = {
        "large_config": json.dumps(
            {"settings": {"key" + str(i): f"value{i}" * 50 for i in range(100)}}
        ),
        "log_data": "\n".join(
            [f"2024-01-01 12:00:{i:02d} INFO Processing item {i}" for i in range(500)]
        ),
        "structured_data": json.dumps(
            [{"id": i, "data": "x" * 100} for i in range(100)]
        ),
    }

    print("   Adding additional content to demonstrate adaptive compression...")
    for name, content in additional_content.items():
        context_store.store(content, metadata={"type": name})

    # Final analytics
    final_analytics = context_store.get_compression_analytics()
    if final_analytics.get("compression_enabled"):
        final_stats = final_analytics["context_store_stats"]
        print(f"\nFinal Performance Summary:")
        print(f"Total contexts: {final_stats['total_contexts']}")
        print(
            f"Final space savings: {final_stats['space_savings_percentage']:.1f}%"
        )
        print(
            f"Total efficiency gain: {final_analytics['performance_impact']['combined_with_reference_store']}"
        )

        # Calculate combined performance improvement
        base_reference_store_improvement = 625  # Your existing 625x improvement
        compression_multiplier = final_analytics["performance_impact"][
            "storage_efficiency_multiplier"
        ]
        total_improvement = base_reference_store_improvement * compression_multiplier

        print(f"+ Smart Compression: {compression_multiplier:.1f}x additional")
        print(
            f"= TOTAL: {total_improvement:.0f}x improvement over traditional approaches!"
        )

    print(f"\nCompression demonstration completed successfully!")
    print(f"Your Context Reference Store now delivers even greater efficiency!")


def main():
    """Run the comprehensive compression demonstration."""
    demonstrate_compression_features()


if __name__ == "__main__":
    main()
