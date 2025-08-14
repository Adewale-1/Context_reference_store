# Getting Started with Context Reference Store

This guide will help you get up and running with Context Reference Store quickly.

## Table of Contents

- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [Your First Context Store](#your-first-context-store)
- [Working with Different Data Types](#working-with-different-data-types)
- [Performance Optimization](#performance-optimization)
- [Next Steps](#next-steps)

## Installation

### Basic Installation

```bash
pip install context-reference-store
```

### Framework-Specific Installation

Choose the frameworks you need:

```bash
# For Agent Development Kit
pip install context-reference-store[adk]

# For LangChain projects
pip install context-reference-store[langchain]

# For LangGraph workflows
pip install context-reference-store[langgraph]

# For LlamaIndex applications
pip install context-reference-store[llamaindex]

# For Composio tool integration
pip install context-reference-store[composio]

# For everything
pip install context-reference-store[full]
```

## Basic Concepts

### Context Reference Store

The Context Reference Store is the core component that manages your application's context data. Instead of keeping large context objects in memory, it stores them efficiently and provides lightweight references.

### Key Benefits

- **Memory Efficiency**: 49x reduction in memory usage
- **Speed**: 625x faster serialization
- **Storage**: 99.8% smaller storage footprint
- **Scalability**: Handles large contexts with minimal overhead

### Context References

When you store context, you get back a lightweight reference (context_id) that you can use to retrieve the full context later.

## Your First Context Store

Let's create your first context store:

```python
from context_store import ContextReferenceStore

# Create a store with basic configuration
store = ContextReferenceStore(
    cache_size=100,        # Keep 100 items in memory
    use_compression=True   # Enable compression for storage efficiency
)

# Store some context
context_id = store.store("This is my first context!")

# Retrieve it later
retrieved_context = store.retrieve(context_id)
print(retrieved_context)  # Output: "This is my first context!"

# Check performance statistics
stats = store.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Total contexts stored: {stats['total_contexts']}")
```

## Working with Different Data Types

### Text Content

```python
# Simple text
text_id = store.store("Large text content here...")

# Retrieve text
text_content = store.retrieve(text_id)
```

### Structured Data

```python
# Dictionary/JSON data
data = {
    "user_id": "12345",
    "conversation": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ],
    "metadata": {
        "timestamp": "2024-01-01T10:00:00Z",
        "session_id": "abc123"
    }
}

data_id = store.store(data)
retrieved_data = store.retrieve(data_id)
```

### Lists and Arrays

```python
# Store list of items
items = ["item1", "item2", "item3", {"complex": "object"}]
list_id = store.store(items)

# Retrieve list
retrieved_items = store.retrieve(list_id)
```

### Binary Data

```python
# Store binary data (images, files, etc.)
with open("image.jpg", "rb") as f:
    binary_data = f.read()

binary_id = store.store(binary_data)
retrieved_binary = store.retrieve(binary_id)
```

## Performance Optimization

### Cache Configuration

```python
from context_store import CacheEvictionPolicy

# Configure for high-performance scenarios
store = ContextReferenceStore(
    cache_size=1000,                          # Larger cache for frequently accessed data
    eviction_policy=CacheEvictionPolicy.LRU, # Least Recently Used eviction
    use_compression=True,                     # Enable compression
    compression_algorithm="lz4"              # Fast compression
)
```

### Memory Management

```python
# Configure memory thresholds
store = ContextReferenceStore(
    cache_size=500,
    memory_threshold_mb=100,  # Move to disk when memory usage exceeds 100MB
    use_disk_storage=True,
    disk_cache_dir="./context_cache"
)
```

### Batch Operations

```python
# Store multiple contexts efficiently
contexts = ["context1", "context2", "context3"]
context_ids = []

for context in contexts:
    context_ids.append(store.store(context))

# Or use batch operations if available
# context_ids = store.batch_store(contexts)
```

## Async Operations

For high-performance applications, use async operations:

```python
import asyncio
from context_store import AsyncContextReferenceStore

async def async_example():
    async with AsyncContextReferenceStore() as store:
        # Store contexts asynchronously
        context_ids = await store.batch_store_async([
            "Context 1",
            "Context 2",
            "Context 3"
        ])

        # Retrieve contexts asynchronously
        contexts = await store.batch_retrieve_async(context_ids)

        return contexts

# Run async example
contexts = asyncio.run(async_example())
```

## Monitoring Performance

```python
# Get detailed performance statistics
stats = store.get_detailed_stats()

print(f"""
Performance Metrics:
- Cache Hit Rate: {stats['hit_rate']:.2%}
- Average Retrieval Time: {stats['avg_retrieval_time_ms']}ms
- Memory Usage: {stats['memory_usage_mb']}MB
- Contexts Stored: {stats['total_contexts']}
- Storage Efficiency: {stats['compression_ratio']:.2f}x compression
""")
```

## Error Handling

```python
try:
    # Store context
    context_id = store.store(large_context)

    # Retrieve context
    retrieved = store.retrieve(context_id)

except ContextStoreError as e:
    print(f"Context store error: {e}")

except KeyError as e:
    print(f"Context not found: {e}")

except MemoryError as e:
    print(f"Memory error: {e}")
    # Consider enabling disk storage or reducing cache size
```

## Best Practices

### 1. Choose Appropriate Cache Size

```python
# For small applications
store = ContextReferenceStore(cache_size=50)

# For medium applications
store = ContextReferenceStore(cache_size=500)

# For large applications
store = ContextReferenceStore(cache_size=5000)
```

### 2. Use Compression for Large Contexts

```python
# Always enable compression for large contexts
store = ContextReferenceStore(
    use_compression=True,
    compression_algorithm="lz4"  # Fast compression
    # or "zstd" for better compression ratio
)
```

### 3. Monitor Performance

```python
# Regular performance monitoring
def monitor_performance(store):
    stats = store.get_cache_stats()

    if stats['hit_rate'] < 0.8:  # Less than 80% hit rate
        print("Consider increasing cache size")

    if stats['memory_usage_mb'] > 500:  # More than 500MB
        print("Consider enabling disk storage")

    return stats

# Call periodically
stats = monitor_performance(store)
```

### 4. Context Lifecycle Management

```python
class ContextManager:
    def __init__(self):
        self.store = ContextReferenceStore(cache_size=1000)
        self.active_contexts = {}

    def create_context(self, key: str, data):
        """Create and track a context"""
        context_id = self.store.store(data)
        self.active_contexts[key] = context_id
        return context_id

    def get_context(self, key: str):
        """Retrieve context by key"""
        if key in self.active_contexts:
            return self.store.retrieve(self.active_contexts[key])
        return None

    def cleanup_context(self, key: str):
        """Clean up context when no longer needed"""
        if key in self.active_contexts:
            # Context will be automatically garbage collected
            # when removed from cache
            del self.active_contexts[key]

# Usage
manager = ContextManager()
manager.create_context("user_session", {"user_id": "123", "data": "..."})
context = manager.get_context("user_session")
manager.cleanup_context("user_session")
```

## Next Steps

Now that you understand the basics, explore these advanced topics:

1. **[Building AI Agents](tutorials/building-agents.md)** - Learn to build intelligent agents with context management
2. **[Framework Integrations](integrations/)** - Integrate with LangChain, LangGraph, LlamaIndex, and more
3. **[Performance Optimization](guides/performance.md)** - Advanced performance tuning techniques
4. **[API Reference](api/)** - Complete API documentation

## Common Issues and Solutions

### Issue: High Memory Usage

```python
# Solution: Enable disk storage
store = ContextReferenceStore(
    use_disk_storage=True,
    memory_threshold_mb=100,
    disk_cache_dir="./cache"
)
```

### Issue: Slow Retrieval

```python
# Solution: Increase cache size or optimize compression
store = ContextReferenceStore(
    cache_size=2000,  # Increase cache size
    compression_algorithm="lz4"  # Use faster compression
)
```

### Issue: Context Not Found

```python
# Solution: Implement proper error handling
def safe_retrieve(store, context_id, default=None):
    try:
        return store.retrieve(context_id)
    except KeyError:
        return default

# Usage
context = safe_retrieve(store, context_id, "Default context")
```

## Support

If you need help:

- Check the [API Reference](api/) for detailed documentation
- Visit [GitHub Issues](https://github.com/Adewale-1/Context_reference_store/issues) for bug reports
- Join [GitHub Discussions](https://github.com/Adewale-1/Context_reference_store/discussions) for questions

Happy coding with Context Reference Store!
