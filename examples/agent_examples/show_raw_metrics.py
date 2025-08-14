#!/usr/bin/env python3
"""
Show the raw structured output from get_enhanced_performance_metrics function
"""

import json
from context_enhanced_analysis_agent.agent import get_enhanced_performance_metrics


class MockToolContext:
    def __init__(self):
        self.state = {}


def show_raw_metrics():
    """Display the raw structured metrics output."""
    print("🔍 Raw Enhanced Performance Metrics Structure")
    print("=" * 60)

    # Get raw metrics directly from the function
    mock_context = MockToolContext()
    raw_metrics = get_enhanced_performance_metrics(mock_context)

    # Pretty print the structured data
    print("\n📊 STRUCTURED OUTPUT:")
    print(json.dumps(raw_metrics, indent=2, default=str))

    print("\n🎯 KEY SECTIONS:")
    if raw_metrics.get("status") == "success":
        print(f"   ✅ Status: {raw_metrics['status']}")
        print(f"   🤖 Agent: {raw_metrics['agent_context']['agent_name']}")
        print(
            f"   📅 Timestamp: {raw_metrics['agent_context']['measurement_timestamp']}"
        )

        metrics = raw_metrics["performance_metrics"]
        context_store = metrics.get("context_store_metrics", {})

        print(f"\n⚡ Context Store Metrics:")
        print(
            f"   - Total Operations: {context_store.get('total_context_operations', 0)}"
        )
        print(
            f"   - Storage Efficiency: {context_store.get('storage_efficiency_percent', 0):.2f}%"
        )
        print(f"   - Cache Hit Rate: {context_store.get('cache_hit_rate', 0):.1f}%")
        print(
            f"   - Avg Serialization: {context_store.get('average_serialization_time', 0):.6f}s"
        )

    print("\n💡 This is the actual structured data the function returns!")
    print("   The ADK agent interprets this and gives you a summary.")


if __name__ == "__main__":
    show_raw_metrics()
