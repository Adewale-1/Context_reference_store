# Core API Reference

This document provides a comprehensive reference for the core Context Reference Store API.

## Table of Contents

- [Core API Reference](#core-api-reference)
  - [Table of Contents](#table-of-contents)
  - [ContextReferenceStore](#contextreferencestore)
    - [Constructor](#constructor)
    - [Core Methods](#core-methods)
    - [Configuration Methods](#configuration-methods)
    - [Monitoring Methods](#monitoring-methods)
    - [Utility Methods](#utility-methods)
  - [AsyncContextReferenceStore](#asynccontextreferencestore)
    - [Constructor](#constructor-1)
    - [Async Methods](#async-methods)
    - [Context Manager Support](#context-manager-support)
  - [Cache Eviction Policies](#cache-eviction-policies)
  - [Data Types and Structures](#data-types-and-structures)
  - [Exceptions](#exceptions)
  - [Configuration Classes](#configuration-classes)
  - [Examples](#examples)

## ContextReferenceStore

The main class for managing context references with intelligent caching and compression.

### Constructor

```python
class ContextReferenceStore:
    def __init__(
        self,
        cache_size: int = 1000,
        eviction_policy: Union[str, CacheEvictionPolicy] = "LRU",
        use_compression: bool = False,
        compression_algorithm: str = "lz4",
        compression_level: int = 1,
        use_disk_storage: bool = False,
        disk_cache_dir: str = "./context_cache",
        memory_threshold_mb: int = 500,
        ttl_seconds: Optional[int] = None,
        max_context_size_mb: int = 100,
        enable_metrics: bool = True
    )
```

**Parameters:**

- **cache_size** (int, default=1000): Maximum number of contexts to keep in memory cache
- **eviction_policy** (str|CacheEvictionPolicy, default="LRU"): Cache eviction strategy
  - "LRU": Least Recently Used
  - "LFU": Least Frequently Used
  - "TTL": Time To Live
  - "MEMORY_PRESSURE": Based on memory pressure
- **use_compression** (bool, default=False): Enable context compression
- **compression_algorithm** (str, default="lz4"): Compression algorithm ("lz4", "zstd")
- **compression_level** (int, default=1): Compression level (1-9, higher = better compression)
- **use_disk_storage** (bool, default=False): Enable disk-based storage for large contexts
- **disk_cache_dir** (str, default="./context_cache"): Directory for disk cache
- **memory_threshold_mb** (int, default=500): Memory threshold for moving to disk
- **ttl_seconds** (int, optional): Time-to-live for contexts (TTL policy only)
- **max_context_size_mb** (int, default=100): Maximum size for individual contexts
- **enable_metrics** (bool, default=True): Enable performance metrics collection

### Core Methods

#### store

```python
def store(self, context: Any, metadata: Optional[Dict[str, Any]] = None) -> str
```

Store context data and return a unique reference ID.

**Parameters:**

- **context** (Any): The context data to store (any serializable object)
- **metadata** (Dict[str, Any], optional): Additional metadata to associate with context

**Returns:**

- **str**: Unique context reference ID

**Example:**

```python
store = ContextReferenceStore()

# Store simple text
text_id = store.store("Hello, World!")

# Store structured data with metadata
data = {"user": "john", "action": "login"}
data_id = store.store(data, metadata={"priority": "high", "timestamp": time.time()})
```

#### retrieve

```python
def retrieve(self, context_id: str) -> Any
```

Retrieve context data by reference ID.

**Parameters:**

- **context_id** (str): The context reference ID

**Returns:**

- **Any**: The original context data

**Raises:**

- **KeyError**: If context_id is not found
- **ContextStoreError**: If context retrieval fails

**Example:**

```python
# Retrieve stored context
context_data = store.retrieve(text_id)
print(context_data)  # "Hello, World!"
```

#### exists

```python
def exists(self, context_id: str) -> bool
```

Check if a context exists in the store.

**Parameters:**

- **context_id** (str): The context reference ID

**Returns:**

- **bool**: True if context exists, False otherwise

**Example:**

```python
if store.exists(context_id):
    data = store.retrieve(context_id)
```

#### delete

```python
def delete(self, context_id: str) -> bool
```

Delete a context from the store.

**Parameters:**

- **context_id** (str): The context reference ID

**Returns:**

- **bool**: True if context was deleted, False if not found

**Example:**

```python
success = store.delete(context_id)
print(f"Deleted: {success}")
```

#### clear

```python
def clear(self) -> int
```

Clear all contexts from the store.

**Returns:**

- **int**: Number of contexts that were cleared

**Example:**

```python
cleared_count = store.clear()
print(f"Cleared {cleared_count} contexts")
```

### Configuration Methods

#### configure

```python
def configure(self, **config_options) -> None
```

Update store configuration dynamically.

**Parameters:**

- **config_options**: Configuration options to update

**Example:**

```python
# Update cache size and enable compression
store.configure(
    cache_size=2000,
    use_compression=True,
    compression_algorithm="zstd"
)
```

#### get_configuration

```python
def get_configuration(self) -> Dict[str, Any]
```

Get current store configuration.

**Returns:**

- **Dict[str, Any]**: Current configuration options

**Example:**

```python
config = store.get_configuration()
print(f"Cache size: {config['cache_size']}")
print(f"Compression: {config['use_compression']}")
```

#### set_eviction_policy

```python
def set_eviction_policy(self, policy: Union[str, CacheEvictionPolicy]) -> None
```

Change the cache eviction policy.

**Parameters:**

- **policy**: New eviction policy

**Example:**

```python
store.set_eviction_policy("LFU")
```

### Monitoring Methods

#### get_cache_stats

```python
def get_cache_stats(self) -> Dict[str, Any]
```

Get basic cache statistics.

**Returns:**

- **Dict[str, Any]**: Cache statistics including:
  - `hit_rate`: Cache hit rate (0.0 to 1.0)
  - `total_contexts`: Total number of stored contexts
  - `cache_size`: Current cache size
  - `memory_usage_mb`: Current memory usage in MB

**Example:**

```python
stats = store.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory usage: {stats['memory_usage_mb']} MB")
```

#### get_detailed_stats

```python
def get_detailed_stats(self) -> Dict[str, Any]
```

Get comprehensive statistics.

**Returns:**

- **Dict[str, Any]**: Detailed statistics including:
  - All basic stats plus:
  - `avg_retrieval_time_ms`: Average retrieval time in milliseconds
  - `compression_ratio`: Compression ratio if enabled
  - `disk_usage_mb`: Disk usage in MB if disk storage enabled
  - `total_operations`: Total number of operations
  - `error_count`: Number of errors encountered

**Example:**

```python
detailed_stats = store.get_detailed_stats()
print(f"Avg retrieval time: {detailed_stats['avg_retrieval_time_ms']:.2f}ms")
print(f"Compression ratio: {detailed_stats['compression_ratio']:.2f}x")
```

#### get_performance_metrics

```python
def get_performance_metrics(self, include_history: bool = False) -> Dict[str, Any]
```

Get performance metrics with optional history.

**Parameters:**

- **include_history** (bool, default=False): Include historical metrics

**Returns:**

- **Dict[str, Any]**: Performance metrics

**Example:**

```python
metrics = store.get_performance_metrics(include_history=True)
print(f"Recent performance: {metrics['recent_avg_time_ms']:.2f}ms")
```

#### reset_stats

```python
def reset_stats(self) -> None
```

Reset all statistics counters.

**Example:**

```python
store.reset_stats()
```

### Utility Methods

#### get_context_info

```python
def get_context_info(self, context_id: str) -> Dict[str, Any]
```

Get information about a specific context.

**Parameters:**

- **context_id** (str): The context reference ID

**Returns:**

- **Dict[str, Any]**: Context information including:
  - `size_bytes`: Size in bytes
  - `created_at`: Creation timestamp
  - `last_accessed`: Last access timestamp
  - `access_count`: Number of times accessed
  - `is_compressed`: Whether context is compressed
  - `storage_location`: "memory" or "disk"

**Example:**

```python
info = store.get_context_info(context_id)
print(f"Size: {info['size_bytes']} bytes")
print(f"Accessed {info['access_count']} times")
```

#### list_contexts

```python
def list_contexts(
    self,
    limit: Optional[int] = None,
    sort_by: str = "created_at",
    reverse: bool = True
) -> List[str]
```

List stored context IDs.

**Parameters:**

- **limit** (int, optional): Maximum number of IDs to return
- **sort_by** (str, default="created_at"): Sort criteria
- **reverse** (bool, default=True): Sort in reverse order

**Returns:**

- **List[str]**: List of context IDs

**Example:**

```python
# Get 10 most recently created contexts
recent_contexts = store.list_contexts(limit=10, sort_by="created_at")

# Get 5 most accessed contexts
popular_contexts = store.list_contexts(limit=5, sort_by="access_count")
```

#### optimize

```python
def optimize(self) -> Dict[str, Any]
```

Perform optimization operations (cleanup, defragmentation, etc.).

**Returns:**

- **Dict[str, Any]**: Optimization results

**Example:**

```python
results = store.optimize()
print(f"Freed {results['freed_memory_mb']} MB")
print(f"Cleaned {results['expired_contexts']} expired contexts")
```

#### backup

```python
def backup(self, backup_path: str, include_data: bool = True) -> bool
```

Create a backup of the store.

**Parameters:**

- **backup_path** (str): Path for backup file
- **include_data** (bool, default=True): Include context data in backup

**Returns:**

- **bool**: True if backup successful

**Example:**

```python
success = store.backup("./backup_20240101.db", include_data=True)
print(f"Backup created: {success}")
```

#### restore

```python
def restore(self, backup_path: str, merge: bool = False) -> bool
```

Restore from a backup.

**Parameters:**

- **backup_path** (str): Path to backup file
- **merge** (bool, default=False): Merge with existing data instead of replacing

**Returns:**

- **bool**: True if restore successful

**Example:**

```python
success = store.restore("./backup_20240101.db", merge=True)
print(f"Restore completed: {success}")
```

## AsyncContextReferenceStore

Asynchronous version of ContextReferenceStore for high-performance applications.

### Constructor

```python
class AsyncContextReferenceStore:
    def __init__(self, **kwargs)
```

Takes the same parameters as `ContextReferenceStore`.

### Async Methods

#### store_async

```python
async def store_async(
    self,
    context: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

Asynchronously store context data.

**Example:**

```python
async_store = AsyncContextReferenceStore()
context_id = await async_store.store_async(large_data)
```

#### retrieve_async

```python
async def retrieve_async(self, context_id: str) -> Any
```

Asynchronously retrieve context data.

**Example:**

```python
data = await async_store.retrieve_async(context_id)
```

#### batch_store_async

```python
async def batch_store_async(
    self,
    contexts: List[Any],
    metadata_list: Optional[List[Dict[str, Any]]] = None
) -> List[str]
```

Store multiple contexts in parallel.

**Parameters:**

- **contexts** (List[Any]): List of context data to store
- **metadata_list** (List[Dict], optional): List of metadata for each context

**Returns:**

- **List[str]**: List of context IDs

**Example:**

```python
contexts = ["context1", "context2", "context3"]
context_ids = await async_store.batch_store_async(contexts)
```

#### batch_retrieve_async

```python
async def batch_retrieve_async(self, context_ids: List[str]) -> List[Any]
```

Retrieve multiple contexts in parallel.

**Parameters:**

- **context_ids** (List[str]): List of context IDs to retrieve

**Returns:**

- **List[Any]**: List of retrieved context data

**Example:**

```python
contexts = await async_store.batch_retrieve_async(context_ids)
```

### Context Manager Support

```python
async with AsyncContextReferenceStore() as store:
    context_id = await store.store_async(data)
    retrieved = await store.retrieve_async(context_id)
```

## Cache Eviction Policies

### CacheEvictionPolicy Enum

```python
from enum import Enum

class CacheEvictionPolicy(Enum):
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    MEMORY_PRESSURE = "memory"     # Based on memory pressure
    RANDOM = "random"              # Random eviction
```

### Policy Descriptions

#### LRU (Least Recently Used)

- Evicts the least recently accessed contexts
- Good for most general-purpose applications
- Balanced performance and memory usage

#### LFU (Least Frequently Used)

- Evicts contexts with the lowest access frequency
- Good for applications with clear access patterns
- Keeps frequently used contexts in memory longer

#### TTL (Time To Live)

- Evicts contexts after a specified time period
- Good for time-sensitive data
- Requires `ttl_seconds` parameter

#### MEMORY_PRESSURE

- Evicts contexts based on memory pressure
- Dynamic eviction based on available memory
- Good for resource-constrained environments

## Data Types and Structures

### Context Metadata

```python
class ContextMetadata:
    """Metadata associated with stored contexts"""

    context_id: str
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    is_compressed: bool
    storage_location: str  # "memory" or "disk"
    compression_ratio: Optional[float]
    metadata: Dict[str, Any]
```

### Cache Statistics

```python
class CacheStats:
    """Cache performance statistics"""

    hit_rate: float                    # Cache hit rate (0.0 to 1.0)
    miss_rate: float                   # Cache miss rate (0.0 to 1.0)
    total_contexts: int                # Total number of contexts
    cache_size: int                    # Current cache size
    memory_usage_mb: float             # Memory usage in MB
    disk_usage_mb: float               # Disk usage in MB (if enabled)
    avg_retrieval_time_ms: float       # Average retrieval time
    compression_ratio: float           # Overall compression ratio
    total_operations: int              # Total number of operations
    error_count: int                   # Number of errors
```

### Performance Metrics

```python
class PerformanceMetrics:
    """Detailed performance metrics"""

    operations_per_second: float
    avg_store_time_ms: float
    avg_retrieve_time_ms: float
    p95_retrieve_time_ms: float
    p99_retrieve_time_ms: float
    memory_efficiency: float
    storage_efficiency: float
    cache_efficiency: float
```

## Exceptions

### ContextStoreError

```python
class ContextStoreError(Exception):
    """Base exception for context store operations"""
    pass
```

### ContextNotFoundError

```python
class ContextNotFoundError(ContextStoreError):
    """Raised when a context ID is not found"""
    pass
```

### ContextTooLargeError

```python
class ContextTooLargeError(ContextStoreError):
    """Raised when context exceeds maximum size limit"""
    pass
```

### CompressionError

```python
class CompressionError(ContextStoreError):
    """Raised when compression/decompression fails"""
    pass
```

### StorageError

```python
class StorageError(ContextStoreError):
    """Raised when storage operations fail"""
    pass
```

## Configuration Classes

### StoreConfig

```python
@dataclass
class StoreConfig:
    """Configuration for ContextReferenceStore"""

    cache_size: int = 1000
    eviction_policy: str = "LRU"
    use_compression: bool = False
    compression_algorithm: str = "lz4"
    compression_level: int = 1
    use_disk_storage: bool = False
    disk_cache_dir: str = "./context_cache"
    memory_threshold_mb: int = 500
    ttl_seconds: Optional[int] = None
    max_context_size_mb: int = 100
    enable_metrics: bool = True

    # Advanced configuration
    max_concurrent_operations: int = 100
    background_cleanup_interval: int = 300
    metrics_collection_interval: int = 60
    disk_sync_interval: int = 30
```

### CompressionConfig

```python
@dataclass
class CompressionConfig:
    """Configuration for compression settings"""

    algorithm: str = "lz4"          # "lz4", "zstd"
    level: int = 1                  # 1-9, higher = better compression
    min_size_bytes: int = 1024      # Minimum size to compress
    max_size_bytes: int = 100*1024*1024  # Maximum size to compress
    enable_parallel: bool = True    # Enable parallel compression
```

## Examples

### Basic Usage

```python
from context_store import ContextReferenceStore

# Create store with basic configuration
store = ContextReferenceStore(
    cache_size=500,
    use_compression=True
)

# Store and retrieve data
data = {"user": "alice", "session": "abc123"}
context_id = store.store(data)
retrieved_data = store.retrieve(context_id)

print(f"Original: {data}")
print(f"Retrieved: {retrieved_data}")

# Check statistics
stats = store.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Advanced Configuration

```python
from context_store import ContextReferenceStore, CacheEvictionPolicy

# Advanced configuration
store = ContextReferenceStore(
    cache_size=2000,
    eviction_policy=CacheEvictionPolicy.LFU,
    use_compression=True,
    compression_algorithm="zstd",
    compression_level=3,
    use_disk_storage=True,
    disk_cache_dir="./advanced_cache",
    memory_threshold_mb=200,
    max_context_size_mb=50
)

# Store large context with metadata
large_context = {"data": "x" * 10000}  # Large string
metadata = {
    "type": "large_data",
    "priority": "high",
    "created_by": "advanced_example"
}

context_id = store.store(large_context, metadata=metadata)

# Get detailed information
info = store.get_context_info(context_id)
print(f"Context size: {info['size_bytes']} bytes")
print(f"Compressed: {info['is_compressed']}")
print(f"Storage: {info['storage_location']}")

# Get detailed statistics
detailed_stats = store.get_detailed_stats()
print(f"Compression ratio: {detailed_stats['compression_ratio']:.2f}x")
print(f"Average retrieval time: {detailed_stats['avg_retrieval_time_ms']:.2f}ms")
```

### Async Usage

```python
import asyncio
from context_store import AsyncContextReferenceStore

async def async_example():
    async with AsyncContextReferenceStore(cache_size=1000) as store:
        # Store multiple contexts concurrently
        contexts = [f"context_{i}" for i in range(100)]
        context_ids = await store.batch_store_async(contexts)

        # Retrieve all contexts
        retrieved_contexts = await store.batch_retrieve_async(context_ids)

        print(f"Stored and retrieved {len(retrieved_contexts)} contexts")

        # Get statistics
        stats = await store.get_cache_stats_async()
        print(f"Async hit rate: {stats['hit_rate']:.2%}")

# Run async example
asyncio.run(async_example())
```

### Error Handling

```python
from context_store import (
    ContextReferenceStore,
    ContextNotFoundError,
    ContextTooLargeError
)

store = ContextReferenceStore(max_context_size_mb=1)

try:
    # Store normal context
    context_id = store.store("normal data")

    # Try to store oversized context
    huge_data = "x" * (2 * 1024 * 1024)  # 2MB
    store.store(huge_data)

except ContextTooLargeError as e:
    print(f"Context too large: {e}")

try:
    # Try to retrieve non-existent context
    missing_data = store.retrieve("non_existent_id")

except ContextNotFoundError as e:
    print(f"Context not found: {e}")

# Safe retrieval with default
def safe_retrieve(store, context_id, default=None):
    try:
        return store.retrieve(context_id)
    except ContextNotFoundError:
        return default

result = safe_retrieve(store, "might_not_exist", "default_value")
```

### Monitoring and Optimization

```python
from context_store import ContextReferenceStore
import time

store = ContextReferenceStore(cache_size=100, enable_metrics=True)

# Store some test data
for i in range(150):
    store.store(f"test_data_{i}")

# Monitor performance
def monitor_performance():
    stats = store.get_detailed_stats()

    print(f"Cache Statistics:")
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    print(f"  Memory Usage: {stats['memory_usage_mb']:.2f} MB")
    print(f"  Total Contexts: {stats['total_contexts']}")
    print(f"  Average Retrieval Time: {stats['avg_retrieval_time_ms']:.2f} ms")

    # Check if optimization is needed
    if stats['hit_rate'] < 0.8:
        print("⚠️  Low hit rate - consider increasing cache size")

    if stats['memory_usage_mb'] > 100:
        print("⚠️  High memory usage - consider enabling disk storage")

    if stats['avg_retrieval_time_ms'] > 10:
        print("⚠️  Slow retrieval - consider optimization")

monitor_performance()

# Perform optimization
optimization_results = store.optimize()
print(f"Optimization freed {optimization_results['freed_memory_mb']} MB")

# Reset statistics for fresh monitoring
store.reset_stats()
```

This completes the Core API Reference documentation. It provides comprehensive coverage of all methods, parameters, return values, and practical examples for using the Context Reference Store effectively.
