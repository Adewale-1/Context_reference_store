#!/usr/bin/env python3
"""
Token Manager Integration with Context-Enhanced Agents

This demo shows how to use the Token Manager for intelligent context selection
within token budgets, optimizing for cost, quality, and performance.

Features:
- Smart token-aware context selection
- Multiple optimization strategies
- Cost estimation and budget management
- Context relevance scoring
- Performance analytics and recommendations
"""

import sys
import os
import time

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from context_store.optimization.token_manager import (
    create_token_manager,
    OptimizationStrategy,
    ModelFamily,
)

from context_enhanced_multi_tool_agent.agent import (
    advanced_text_analysis_with_context,
    _context_store,
)


class MockToolContext:
    def __init__(self):
        self.state = {}


def demo_token_aware_context_selection():
    """Demonstrate smart token-aware context selection."""
    print("🎯 SMART TOKEN-AWARE CONTEXT MANAGEMENT DEMO")
    print("=" * 60)

    # Create token manager for different models
    models_to_test = ["gemini-1.5-pro", "gpt-4", "claude-3-sonnet"]

    for model_name in models_to_test:
        print(f"\n🤖 Testing with {model_name}")
        print("-" * 40)

        # Create token manager
        manager = create_token_manager(model_name)

        # Sample contexts of varying relevance and size
        contexts = [
            "Context Reference Store provides 625x faster serialization performance.",
            "The Context Reference Store library represents a revolutionary advancement in AI context management. "
            "This innovative technology provides 625x faster serialization, 49x memory reduction, and 99.55% "
            "storage reduction compared to traditional approaches. The system supports advanced caching strategies "
            "including LRU, LFU, TTL, and Memory Pressure-based eviction policies."
            * 10,
            "Machine learning algorithms require efficient context management for optimal performance. "
            "Large language models benefit significantly from optimized context handling and storage."
            * 15,
            "AI systems process vast amounts of contextual information. Efficient storage and retrieval "
            "mechanisms are crucial for maintaining performance while managing memory constraints."
            * 8,
            "Token optimization strategies help reduce costs while maintaining quality. Smart context "
            "selection can dramatically improve efficiency in LLM applications." * 12,
            "Performance monitoring and analytics provide insights into system behavior. Real-time "
            "dashboards enable proactive optimization and cost management." * 6,
            "Semantic analysis and clustering identify duplicate content and optimize storage efficiency. "
            "Advanced algorithms can detect similar contexts and suggest consolidation strategies."
            * 20,
        ]

        # Create token budget
        budget = manager.create_budget(
            target_tokens=10000,  # 10K token budget
            system_prompt="You are a helpful AI assistant focused on context optimization.",
            reserved_output_tokens=1500,
        )

        print(
            f"   📊 Token Budget: {budget.context_tokens_available:,} tokens available"
        )
        print(f"   💰 Model: {model_name}")

        # Test different optimization strategies
        strategies = [
            OptimizationStrategy.COST_FIRST,
            OptimizationStrategy.QUALITY_FIRST,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.COMPREHENSIVE,
        ]

        for strategy in strategies:
            print(f"\n   🔍 Strategy: {strategy.value}")

            result = manager.optimize_context_selection(
                contexts=contexts,
                budget=budget,
                strategy=strategy,
                query="context optimization and performance improvements",
                keywords=["context", "optimization", "performance", "efficiency"],
            )

            print(f"      ✅ Selected: {len(result.selected_contexts)} contexts")
            print(f"      📏 Total tokens: {result.total_tokens:,}")
            print(f"      📊 Budget utilization: {result.budget_utilization:.1%}")
            print(f"      💵 Estimated cost: ${result.estimated_cost:.4f}")
            print(f"      ⭐ Efficiency score: {result.efficiency_score:.2f}")

            if result.recommendations:
                print(f"      💡 Top recommendation: {result.recommendations[0]}")

    # Show usage analytics
    print(f"\n📈 Token Manager Analytics:")
    analytics = manager.get_usage_analytics()
    print(f"   🔄 Total optimizations: {analytics['total_optimizations']}")
    print(
        f"   📊 Average budget utilization: {analytics['average_budget_utilization']:.1%}"
    )
    print(
        f"   💰 Total estimated cost: ${analytics.get('total_estimated_cost', 0):.4f}"
    )

    # Model upgrade suggestions
    suggestions = manager.suggest_model_upgrade(analytics)
    if suggestions["suggestions"]:
        print(f"\n🚀 Model Recommendations:")
        for suggestion in suggestions["suggestions"]:
            print(f"   • {suggestion['recommendation']}")
            if "potential_savings_percent" in suggestion:
                print(
                    f"     💰 Potential savings: {suggestion['potential_savings_percent']:.1f}%"
                )


def demo_token_integration_with_context_store():
    """Show how Token Manager integrates with Context Reference Store."""
    print("\n🔗 TOKEN MANAGER + CONTEXT STORE INTEGRATION")
    print("=" * 55)

    # Create token manager
    manager = create_token_manager("gemini-1.5-pro")
    mock_context = MockToolContext()

    # Generate some content using context-enhanced agent
    print("📝 Generating context using Context-Enhanced Agent...")

    test_texts = [
        "Context Reference Store optimization techniques",
        "Machine learning performance improvements through efficient context management",
        "Token-aware optimization strategies for large language models",
        "Real-time analytics and monitoring for AI systems",
    ]

    enhanced_contexts = []
    for text in test_texts:
        try:
            result = advanced_text_analysis_with_context(
                text * 50, mock_context
            )  # Make longer
            if result.get("status") == "success":
                # Extract the enhanced analysis as context
                context_content = f"Analysis: {result.get('basic_metrics', {})}"
                enhanced_contexts.append(context_content)
                print(
                    f"   ✅ Generated enhanced context ({len(context_content)} chars)"
                )
        except Exception as e:
            print(f"   ⚠️  Error generating context: {e}")

    if enhanced_contexts:
        print(f"\n🎯 Optimizing {len(enhanced_contexts)} enhanced contexts...")

        # Create budget
        budget = manager.create_budget(target_tokens=5000)

        # Optimize selection
        result = manager.optimize_context_selection(
            contexts=enhanced_contexts,
            budget=budget,
            strategy=OptimizationStrategy.BALANCED,
            query="context analysis and optimization",
        )

        print(f"   📊 Results:")
        print(f"      Selected contexts: {len(result.selected_contexts)}")
        print(f"      Token utilization: {result.budget_utilization:.1%}")
        print(f"      Efficiency score: {result.efficiency_score:.2f}")

        # Show context store stats
        print(f"\n💾 Context Store Stats:")
        try:
            stats = _context_store.get_cache_stats()
            print(f"   📈 Total contexts: {stats.get('total_contexts', 0)}")
            print(f"   🎯 Cache hit rate: {stats.get('hit_rate', 0):.1%}")
            print(f"   🧠 Memory usage: {stats.get('memory_usage_percent', 0):.1f}%")
        except Exception as e:
            print(f"   ⚠️  Could not retrieve stats: {e}")

    else:
        print("   ⚠️  No enhanced contexts generated for optimization")


def demo_cost_optimization_scenarios():
    """Demonstrate different cost optimization scenarios."""
    print("\n💰 COST OPTIMIZATION SCENARIOS")
    print("=" * 40)

    scenarios = [
        {
            "name": "High-Volume Processing",
            "contexts": 50,
            "target_tokens": 50000,
            "strategy": OptimizationStrategy.COST_FIRST,
        },
        {
            "name": "Quality-Critical Analysis",
            "contexts": 20,
            "target_tokens": 20000,
            "strategy": OptimizationStrategy.QUALITY_FIRST,
        },
        {
            "name": "Balanced Production Use",
            "contexts": 30,
            "target_tokens": 30000,
            "strategy": OptimizationStrategy.BALANCED,
        },
    ]

    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")

        # Generate contexts for scenario
        base_context = "Context optimization and performance analysis. " * 20
        contexts = [
            f"{base_context} Variation {i}" for i in range(scenario["contexts"])
        ]

        manager = create_token_manager("gemini-1.5-pro")
        budget = manager.create_budget(target_tokens=scenario["target_tokens"])

        result = manager.optimize_context_selection(
            contexts=contexts, budget=budget, strategy=scenario["strategy"]
        )

        print(f"   📊 Processed {len(contexts)} contexts")
        print(f"   ✅ Selected {len(result.selected_contexts)} contexts")
        print(
            f"   📏 Used {result.total_tokens:,} / {budget.context_tokens_available:,} tokens"
        )
        print(f"   💵 Estimated cost: ${result.estimated_cost:.4f}")
        print(f"   ⭐ Efficiency: {result.efficiency_score:.2f}")

        if result.recommendations:
            print(f"   💡 Recommendation: {result.recommendations[0]}")


def main():
    """Main demo function."""
    print("🚀 Context Reference Store - Token Management Integration")
    print("This demo showcases intelligent token-aware context optimization")
    print()

    try:
        # Run all demos
        demo_token_aware_context_selection()
        demo_token_integration_with_context_store()
        demo_cost_optimization_scenarios()

        print("\n✅ Token Manager demo completed successfully!")
        print("\n💡 Key Benefits Demonstrated:")
        print("   🎯 Smart context selection within token budgets")
        print("   💰 Cost optimization across different strategies")
        print("   ⚡ Integration with Context Reference Store")
        print("   📊 Real-time analytics and recommendations")
        print("   🤖 Multi-model support and configuration")

    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
