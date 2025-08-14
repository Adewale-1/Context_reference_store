# Utilities API Reference

This document provides comprehensive API reference for Context Reference Store utility functions and helper classes.

## Table of Contents

- [Compression Utilities](#compression-utilities)
- [Serialization Utilities](#serialization-utilities)
- [Context Utilities](#context-utilities)
- [Performance Utilities](#performance-utilities)
- [Validation Utilities](#validation-utilities)
- [Debug Utilities](#debug-utilities)

## Compression Utilities

### `CompressionManager`

Manage compression and decompression of context data.

```python
from context_store.utils import CompressionManager

compression_manager = CompressionManager()
```

#### Methods

##### `compress(data: bytes, algorithm: str = "lz4") -> bytes`

Compress data using specified algorithm.

**Parameters:**
- `data`: Data to compress
- `algorithm`: Compression algorithm ("lz4", "gzip", "zstd")

**Returns:** Compressed data bytes

**Example:**
```python
original_data = b"This is some text to compress"
compressed = compression_manager.compress(original_data, "lz4")
print(f"Compression ratio: {len(compressed) / len(original_data):.2f}")
```

##### `decompress(compressed_data: bytes, algorithm: str = "lz4") -> bytes`

Decompress data using specified algorithm.

**Parameters:**
- `compressed_data`: Compressed data
- `algorithm`: Compression algorithm used

**Returns:** Decompressed data bytes

##### `get_compression_ratio(original_size: int, compressed_size: int) -> float`

Calculate compression ratio.

**Parameters:**
- `original_size`: Original data size
- `compressed_size`: Compressed data size

**Returns:** Compression ratio (0.0 to 1.0)

##### `benchmark_algorithms(data: bytes) -> Dict[str, Dict[str, float]]`

Benchmark different compression algorithms.

**Parameters:**
- `data`: Test data

**Returns:** Benchmark results for each algorithm

**Example:**
```python
test_data = b"Large text data for testing compression..."
results = compression_manager.benchmark_algorithms(test_data)

for algo, metrics in results.items():
    print(f"{algo}: ratio={metrics['ratio']:.2f}, time={metrics['time_ms']:.2f}ms")
```

##### `auto_select_algorithm(data_size: int, compression_priority: str = "balanced") -> str`

Automatically select best compression algorithm.

**Parameters:**
- `data_size`: Size of data to compress
- `compression_priority`: Priority ("speed", "ratio", "balanced")

**Returns:** Recommended algorithm name

## Serialization Utilities

### `SerializationManager`

Handle serialization and deserialization of various data types.

```python
from context_store.utils import SerializationManager

serializer = SerializationManager()
```

#### Methods

##### `serialize(obj: Any, format: str = "pickle") -> bytes`

Serialize object to bytes.

**Parameters:**
- `obj`: Object to serialize
- `format`: Serialization format ("pickle", "json", "msgpack")

**Returns:** Serialized data bytes

**Example:**
```python
data = {"key": "value", "numbers": [1, 2, 3]}
serialized = serializer.serialize(data, "json")
```

##### `deserialize(data: bytes, format: str = "pickle") -> Any`

Deserialize bytes to object.

**Parameters:**
- `data`: Serialized data
- `format`: Serialization format used

**Returns:** Deserialized object

##### `is_serializable(obj: Any, format: str = "pickle") -> bool`

Check if object can be serialized.

**Parameters:**
- `obj`: Object to check
- `format`: Target serialization format

**Returns:** True if serializable

##### `estimate_serialized_size(obj: Any, format: str = "pickle") -> int`

Estimate serialized size without actually serializing.

**Parameters:**
- `obj`: Object to estimate
- `format`: Serialization format

**Returns:** Estimated size in bytes

##### `register_custom_serializer(type_class: type, serializer_func: callable, deserializer_func: callable) -> None`

Register custom serialization for specific types.

**Parameters:**
- `type_class`: Class to register serializer for
- `serializer_func`: Function to serialize instances
- `deserializer_func`: Function to deserialize data

**Example:**
```python
def serialize_custom_class(obj):
    return {"custom_data": obj.data}

def deserialize_custom_class(data):
    return CustomClass(data["custom_data"])

serializer.register_custom_serializer(
    CustomClass,
    serialize_custom_class,
    deserialize_custom_class
)
```

## Context Utilities

### `ContextValidator`

Validate context data and metadata.

```python
from context_store.utils import ContextValidator

validator = ContextValidator()
```

#### Methods

##### `validate_context_data(data: Any) -> Tuple[bool, List[str]]`

Validate context data.

**Parameters:**
- `data`: Data to validate

**Returns:** Tuple of (is_valid, list_of_errors)

**Example:**
```python
is_valid, errors = validator.validate_context_data({"key": "value"})
if not is_valid:
    print(f"Validation errors: {errors}")
```

##### `validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]`

Validate metadata structure.

**Parameters:**
- `metadata`: Metadata to validate

**Returns:** Tuple of (is_valid, list_of_errors)

##### `sanitize_context_id(context_id: str) -> str`

Sanitize context ID string.

**Parameters:**
- `context_id`: Context ID to sanitize

**Returns:** Sanitized context ID

##### `validate_context_type(context_type: str) -> bool`

Validate context type string.

**Parameters:**
- `context_type`: Context type to validate

**Returns:** True if valid

##### `set_validation_rules(rules: Dict[str, Any]) -> None`

Set custom validation rules.

**Parameters:**
- `rules`: Validation rules configuration

### `ContextTransformer`

Transform context data between different formats.

```python
from context_store.utils import ContextTransformer

transformer = ContextTransformer()
```

#### Methods

##### `transform_to_standard_format(data: Any, source_format: str) -> Dict[str, Any]`

Transform data to standard context format.

**Parameters:**
- `data`: Data to transform
- `source_format`: Source data format

**Returns:** Standardized context data

##### `extract_metadata(context_data: Any) -> Dict[str, Any]`

Extract metadata from context data.

**Parameters:**
- `context_data`: Context data

**Returns:** Extracted metadata

##### `normalize_text_content(text: str) -> str`

Normalize text content for consistent storage.

**Parameters:**
- `text`: Text to normalize

**Returns:** Normalized text

##### `detect_content_type(data: Any) -> str`

Auto-detect content type of data.

**Parameters:**
- `data`: Data to analyze

**Returns:** Detected content type

## Performance Utilities

### `PerformanceProfiler`

Profile performance of context operations.

```python
from context_store.utils import PerformanceProfiler

profiler = PerformanceProfiler()
```

#### Methods

##### `profile_operation(operation_func: callable, *args, **kwargs) -> Dict[str, Any]`

Profile a single operation.

**Parameters:**
- `operation_func`: Function to profile
- `*args, **kwargs`: Arguments for the function

**Returns:** Performance metrics

**Example:**
```python
def slow_operation():
    time.sleep(0.1)
    return "result"

metrics = profiler.profile_operation(slow_operation)
print(f"Execution time: {metrics['execution_time_ms']:.2f}ms")
```

##### `start_profiling_session(session_name: str) -> None`

Start profiling session.

**Parameters:**
- `session_name`: Name for the profiling session

##### `end_profiling_session() -> Dict[str, Any]`

End profiling session and get results.

**Returns:** Profiling session results

##### `profile_memory_usage(func: callable, *args, **kwargs) -> Dict[str, Any]`

Profile memory usage of a function.

**Parameters:**
- `func`: Function to profile
- `*args, **kwargs`: Function arguments

**Returns:** Memory usage metrics

##### `benchmark_bulk_operations(operation_func: callable, data_sizes: List[int], iterations: int = 10) -> Dict[str, List[float]]`

Benchmark operation performance across different data sizes.

**Parameters:**
- `operation_func`: Function to benchmark
- `data_sizes`: List of data sizes to test
- `iterations`: Number of iterations per size

**Returns:** Benchmark results

### `CacheOptimizer`

Optimize cache performance and configuration.

```python
from context_store.utils import CacheOptimizer

optimizer = CacheOptimizer()
```

#### Methods

##### `analyze_access_patterns(access_log: List[str]) -> Dict[str, Any]`

Analyze context access patterns.

**Parameters:**
- `access_log`: List of context IDs in access order

**Returns:** Access pattern analysis

##### `recommend_cache_size(usage_stats: Dict[str, Any]) -> int`

Recommend optimal cache size.

**Parameters:**
- `usage_stats`: Cache usage statistics

**Returns:** Recommended cache size

##### `optimize_eviction_policy(access_patterns: Dict[str, Any]) -> str`

Recommend optimal eviction policy.

**Parameters:**
- `access_patterns`: Access pattern analysis

**Returns:** Recommended eviction policy

##### `calculate_cache_efficiency(hit_rate: float, memory_usage: float, target_performance: Dict[str, float]) -> float`

Calculate cache efficiency score.

**Parameters:**
- `hit_rate`: Cache hit rate
- `memory_usage`: Memory usage in MB
- `target_performance`: Target performance metrics

**Returns:** Efficiency score (0.0 to 1.0)

## Validation Utilities

### `DataValidator`

Comprehensive data validation utilities.

```python
from context_store.utils import DataValidator

validator = DataValidator()
```

#### Methods

##### `validate_data_size(data: Any, max_size_bytes: int = None) -> Tuple[bool, int]`

Validate data size constraints.

**Parameters:**
- `data`: Data to validate
- `max_size_bytes`: Maximum allowed size

**Returns:** Tuple of (is_valid, actual_size)

##### `validate_data_type(data: Any, allowed_types: List[type]) -> bool`

Validate data type.

**Parameters:**
- `data`: Data to validate
- `allowed_types`: List of allowed types

**Returns:** True if valid type

##### `validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]`

Validate data against JSON schema.

**Parameters:**
- `data`: Data to validate
- `schema`: JSON schema

**Returns:** Tuple of (is_valid, validation_errors)

##### `sanitize_input(data: Any, sanitization_rules: Dict[str, Any] = None) -> Any`

Sanitize input data.

**Parameters:**
- `data`: Data to sanitize
- `sanitization_rules`: Custom sanitization rules

**Returns:** Sanitized data

##### `check_security_constraints(data: Any) -> Tuple[bool, List[str]]`

Check data for security constraints.

**Parameters:**
- `data`: Data to check

**Returns:** Tuple of (is_safe, security_warnings)

### `ConfigValidator`

Validate configuration settings.

```python
from context_store.utils import ConfigValidator

config_validator = ConfigValidator()
```

#### Methods

##### `validate_store_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]`

Validate context store configuration.

**Parameters:**
- `config`: Configuration to validate

**Returns:** Tuple of (is_valid, validation_errors)

##### `validate_adapter_config(adapter_type: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]`

Validate adapter configuration.

**Parameters:**
- `adapter_type`: Type of adapter
- `config`: Configuration to validate

**Returns:** Tuple of (is_valid, validation_errors)

##### `recommend_config_optimizations(current_config: Dict[str, Any], usage_stats: Dict[str, Any]) -> List[str]`

Recommend configuration optimizations.

**Parameters:**
- `current_config`: Current configuration
- `usage_stats`: Usage statistics

**Returns:** List of optimization recommendations

## Debug Utilities

### `DebugHelper`

Debugging and troubleshooting utilities.

```python
from context_store.utils import DebugHelper

debug_helper = DebugHelper()
```

#### Methods

##### `enable_debug_logging(level: str = "DEBUG") -> None`

Enable debug logging.

**Parameters:**
- `level`: Logging level

##### `trace_context_operations(context_store) -> None`

Enable operation tracing.

**Parameters:**
- `context_store`: Context store instance to trace

##### `get_debug_info(context_store) -> Dict[str, Any]`

Get comprehensive debug information.

**Parameters:**
- `context_store`: Context store instance

**Returns:** Debug information

**Example:**
```python
debug_info = debug_helper.get_debug_info(store)
print(f"Store state: {debug_info['store_state']}")
print(f"Recent operations: {debug_info['recent_operations']}")
```

##### `diagnose_performance_issues(context_store) -> List[str]`

Diagnose potential performance issues.

**Parameters:**
- `context_store`: Context store instance

**Returns:** List of identified issues

##### `export_debug_report(context_store, report_path: str) -> None`

Export comprehensive debug report.

**Parameters:**
- `context_store`: Context store instance
- `report_path`: Path to save report

##### `verify_data_integrity(context_store) -> Dict[str, Any]`

Verify data integrity.

**Parameters:**
- `context_store`: Context store instance

**Returns:** Integrity check results

### `LogAnalyzer`

Analyze context store logs for insights.

```python
from context_store.utils import LogAnalyzer

log_analyzer = LogAnalyzer()
```

#### Methods

##### `parse_log_file(log_path: str) -> List[Dict[str, Any]]`

Parse context store log file.

**Parameters:**
- `log_path`: Path to log file

**Returns:** List of parsed log entries

##### `analyze_error_patterns(log_entries: List[Dict[str, Any]]) -> Dict[str, Any]`

Analyze error patterns in logs.

**Parameters:**
- `log_entries`: Parsed log entries

**Returns:** Error pattern analysis

##### `generate_usage_report(log_entries: List[Dict[str, Any]], time_range: str = "24h") -> Dict[str, Any]`

Generate usage report from logs.

**Parameters:**
- `log_entries`: Parsed log entries
- `time_range`: Time range for analysis

**Returns:** Usage report

##### `detect_anomalies(log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]`

Detect anomalies in operation patterns.

**Parameters:**
- `log_entries`: Parsed log entries

**Returns:** List of detected anomalies

## Utility Functions

### Helper Functions

Collection of standalone utility functions.

```python
from context_store.utils import (
    generate_context_id,
    estimate_object_size,
    format_bytes,
    parse_time_duration,
    create_hash
)
```

#### `generate_context_id(prefix: str = "", length: int = 16) -> str`

Generate unique context ID.

**Parameters:**
- `prefix`: Optional prefix for ID
- `length`: Length of random portion

**Returns:** Generated context ID

#### `estimate_object_size(obj: Any) -> int`

Estimate object size in bytes.

**Parameters:**
- `obj`: Object to estimate

**Returns:** Estimated size in bytes

#### `format_bytes(num_bytes: int) -> str`

Format bytes into human-readable string.

**Parameters:**
- `num_bytes`: Number of bytes

**Returns:** Formatted string (e.g., "1.2 MB")

#### `parse_time_duration(duration_str: str) -> int`

Parse time duration string to seconds.

**Parameters:**
- `duration_str`: Duration string (e.g., "1h", "30m", "5s")

**Returns:** Duration in seconds

#### `create_hash(data: bytes, algorithm: str = "sha256") -> str`

Create hash of data.

**Parameters:**
- `data`: Data to hash
- `algorithm`: Hash algorithm

**Returns:** Hex digest of hash

## Integration with Core System

### Using Utilities with Context Store

```python
from context_store import ContextReferenceStore
from context_store.utils import (
    CompressionManager,
    PerformanceProfiler,
    DebugHelper
)

# Create store with utility integration
store = ContextReferenceStore(
    cache_size=1000,
    use_compression=True
)

# Set up performance profiling
profiler = PerformanceProfiler()
profiler.start_profiling_session("production_analysis")

# Enable debug tracing
debug_helper = DebugHelper()
debug_helper.trace_context_operations(store)

# Use compression utilities
compression_manager = CompressionManager()
data = b"Large context data..."
compressed_data = compression_manager.compress(data, "lz4")

# Store with utilities
context_id = store.store(compressed_data)

# Get profiling results
results = profiler.end_profiling_session()
print(f"Session performance: {results}")
```

### Custom Utility Integration

```python
# Create custom utility class
class CustomContextProcessor:
    def __init__(self, context_store):
        self.store = context_store
        self.validator = DataValidator()
        self.profiler = PerformanceProfiler()
    
    def safe_store(self, data, validate=True):
        if validate:
            is_valid, errors = self.validator.validate_data_size(data)
            if not is_valid:
                raise ValueError(f"Data validation failed: {errors}")
        
        return self.profiler.profile_operation(
            self.store.store,
            data
        )
```

For more advanced utility usage examples, see the [Getting Started Guide](../getting-started.md) and [Performance Optimization Guide](../guides/performance.md).
