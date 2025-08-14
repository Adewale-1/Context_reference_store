# Monitoring API Reference

This document provides comprehensive API reference for Context Reference Store monitoring and performance tracking capabilities.

## Table of Contents

- [Performance Monitor](#performance-monitor)
- [Cache Monitor](#cache-monitor)
- [Memory Monitor](#memory-monitor)
- [Context Analytics](#context-analytics)
- [Real-time Metrics](#real-time-metrics)
- [Custom Monitoring](#custom-monitoring)

## Performance Monitor

### `PerformanceMonitor`

Monitor performance metrics for context operations.

```python
from context_store.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(context_store)
```

#### Methods

##### `get_performance_stats() -> Dict[str, Any]`

Get comprehensive performance statistics.

**Returns:** Dictionary containing performance metrics

**Example:**
```python
stats = monitor.get_performance_stats()
print(f"Average store time: {stats['avg_store_time_ms']:.2f}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

**Sample Output:**
```python
{
    "avg_store_time_ms": 1.23,
    "avg_retrieve_time_ms": 0.87,
    "cache_hit_rate": 0.85,
    "total_operations": 10000,
    "error_rate": 0.001,
    "memory_usage_mb": 128.5
}
```

##### `start_performance_tracking() -> None`

Start continuous performance tracking.

**Example:**
```python
monitor.start_performance_tracking()
# Performance metrics will be collected automatically
```

##### `stop_performance_tracking() -> Dict[str, Any]`

Stop tracking and get final statistics.

**Returns:** Final performance statistics

##### `get_operation_metrics(operation_type: str) -> Dict[str, Any]`

Get metrics for specific operation type.

**Parameters:**
- `operation_type`: Type of operation ("store", "retrieve", "search", etc.)

**Returns:** Operation-specific metrics

**Example:**
```python
store_metrics = monitor.get_operation_metrics("store")
retrieve_metrics = monitor.get_operation_metrics("retrieve")
```

##### `set_performance_thresholds(thresholds: Dict[str, float]) -> None`

Set performance alert thresholds.

**Parameters:**
- `thresholds`: Dictionary of metric thresholds

**Example:**
```python
monitor.set_performance_thresholds({
    "max_store_time_ms": 100.0,
    "min_cache_hit_rate": 0.8,
    "max_memory_usage_mb": 500.0
})
```

##### `get_performance_alerts() -> List[Dict[str, Any]]`

Get current performance alerts.

**Returns:** List of active performance alerts

## Cache Monitor

### `CacheMonitor`

Monitor cache performance and behavior.

```python
from context_store.monitoring import CacheMonitor

cache_monitor = CacheMonitor(context_store)
```

#### Methods

##### `get_cache_stats() -> Dict[str, Any]`

Get detailed cache statistics.

**Returns:** Cache performance metrics

**Example:**
```python
cache_stats = cache_monitor.get_cache_stats()
print(f"Cache size: {cache_stats['current_size']}")
print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
print(f"Evictions: {cache_stats['eviction_count']}")
```

**Sample Output:**
```python
{
    "current_size": 1500,
    "max_size": 2000,
    "hit_rate": 0.87,
    "miss_rate": 0.13,
    "eviction_count": 45,
    "eviction_policy": "LRU",
    "memory_usage_bytes": 67108864
}
```

##### `get_cache_efficiency() -> float`

Calculate cache efficiency score.

**Returns:** Efficiency score between 0.0 and 1.0

##### `get_hot_contexts(limit: int = 10) -> List[Dict[str, Any]]`

Get most frequently accessed contexts.

**Parameters:**
- `limit`: Maximum number of contexts to return

**Returns:** List of hot context information

##### `get_cache_distribution() -> Dict[str, int]`

Get distribution of cached context types.

**Returns:** Dictionary mapping context types to counts

##### `track_cache_pattern(pattern_name: str, context_id: str) -> None`

Track custom cache access patterns.

**Parameters:**
- `pattern_name`: Name of the pattern
- `context_id`: Context being accessed

##### `get_cache_recommendations() -> List[str]`

Get cache optimization recommendations.

**Returns:** List of optimization suggestions

## Memory Monitor

### `MemoryMonitor`

Monitor memory usage and optimization.

```python
from context_store.monitoring import MemoryMonitor

memory_monitor = MemoryMonitor(context_store)
```

#### Methods

##### `get_memory_usage() -> Dict[str, float]`

Get current memory usage statistics.

**Returns:** Memory usage metrics in MB

**Example:**
```python
memory_stats = memory_monitor.get_memory_usage()
print(f"Total memory: {memory_stats['total_mb']:.2f}MB")
print(f"Cache memory: {memory_stats['cache_mb']:.2f}MB")
print(f"Context memory: {memory_stats['contexts_mb']:.2f}MB")
```

##### `track_memory_trend(duration_minutes: int = 60) -> List[Dict[str, Any]]`

Track memory usage over time.

**Parameters:**
- `duration_minutes`: Tracking duration

**Returns:** List of memory usage snapshots

##### `get_memory_efficiency() -> float`

Calculate memory usage efficiency.

**Returns:** Efficiency score between 0.0 and 1.0

##### `get_largest_contexts(limit: int = 10) -> List[Dict[str, Any]]`

Get contexts consuming most memory.

**Parameters:**
- `limit`: Maximum number of contexts

**Returns:** List of memory-heavy contexts

##### `estimate_memory_savings(compression_level: str = "medium") -> Dict[str, float]`

Estimate potential memory savings with compression.

**Parameters:**
- `compression_level`: Compression level ("low", "medium", "high")

**Returns:** Estimated savings metrics

##### `trigger_memory_cleanup() -> Dict[str, int]`

Trigger memory cleanup and optimization.

**Returns:** Cleanup statistics

## Context Analytics

### `ContextAnalytics`

Advanced analytics for context usage patterns.

```python
from context_store.monitoring import ContextAnalytics

analytics = ContextAnalytics(context_store)
```

#### Methods

##### `get_usage_patterns() -> Dict[str, Any]`

Analyze context usage patterns.

**Returns:** Usage pattern analysis

**Example:**
```python
patterns = analytics.get_usage_patterns()
print(f"Peak usage hour: {patterns['peak_hour']}")
print(f"Most active context type: {patterns['top_context_type']}")
```

##### `get_context_lifecycle_stats() -> Dict[str, Any]`

Get context lifecycle statistics.

**Returns:** Lifecycle metrics (creation, access, deletion patterns)

##### `analyze_retrieval_patterns(time_window_hours: int = 24) -> Dict[str, Any]`

Analyze context retrieval patterns.

**Parameters:**
- `time_window_hours`: Analysis time window

**Returns:** Retrieval pattern analysis

##### `get_semantic_clusters(min_cluster_size: int = 5) -> List[Dict[str, Any]]`

Identify semantic clusters in stored contexts.

**Parameters:**
- `min_cluster_size`: Minimum contexts per cluster

**Returns:** List of semantic clusters

##### `get_user_behavior_insights(user_id: str = None) -> Dict[str, Any]`

Get user behavior insights.

**Parameters:**
- `user_id`: Specific user ID (optional)

**Returns:** User behavior analysis

##### `predict_storage_needs(forecast_days: int = 30) -> Dict[str, float]`

Predict future storage requirements.

**Parameters:**
- `forecast_days`: Forecast period

**Returns:** Storage predictions

## Real-time Metrics

### `RealTimeMetrics`

Real-time monitoring and alerting.

```python
from context_store.monitoring import RealTimeMetrics

real_time = RealTimeMetrics(context_store)
```

#### Methods

##### `start_real_time_monitoring(update_interval_seconds: int = 5) -> None`

Start real-time metric collection.

**Parameters:**
- `update_interval_seconds`: Metric update frequency

##### `stop_real_time_monitoring() -> None`

Stop real-time monitoring.

##### `get_current_metrics() -> Dict[str, Any]`

Get current real-time metrics.

**Returns:** Current system metrics

##### `set_metric_alerts(alerts_config: Dict[str, Dict]) -> None`

Configure metric-based alerts.

**Parameters:**
- `alerts_config`: Alert configuration

**Example:**
```python
alerts_config = {
    "high_latency": {
        "metric": "avg_retrieve_time_ms",
        "threshold": 100.0,
        "operator": "greater_than"
    },
    "low_cache_hit": {
        "metric": "cache_hit_rate", 
        "threshold": 0.7,
        "operator": "less_than"
    }
}
real_time.set_metric_alerts(alerts_config)
```

##### `get_active_alerts() -> List[Dict[str, Any]]`

Get currently active alerts.

**Returns:** List of active alerts

##### `register_alert_callback(callback_func: callable) -> None`

Register callback for alert notifications.

**Parameters:**
- `callback_func`: Function to call when alerts trigger

**Example:**
```python
def alert_handler(alert):
    print(f"ALERT: {alert['metric']} = {alert['value']}")
    
real_time.register_alert_callback(alert_handler)
```

##### `get_metrics_dashboard_data() -> Dict[str, Any]`

Get data formatted for dashboard display.

**Returns:** Dashboard-ready metrics data

## Custom Monitoring

### `CustomMonitor`

Create custom monitoring solutions.

```python
from context_store.monitoring import CustomMonitor

custom_monitor = CustomMonitor(context_store)
```

#### Methods

##### `register_custom_metric(metric_name: str, calculation_func: callable) -> None`

Register a custom metric calculation.

**Parameters:**
- `metric_name`: Name of the custom metric
- `calculation_func`: Function to calculate the metric

**Example:**
```python
def calculate_efficiency_ratio(store):
    stats = store.get_cache_stats()
    return stats["total_contexts"] / stats["memory_usage_mb"]

custom_monitor.register_custom_metric(
    "efficiency_ratio",
    calculate_efficiency_ratio
)
```

##### `get_custom_metrics() -> Dict[str, float]`

Get all custom metric values.

**Returns:** Dictionary of custom metric values

##### `create_custom_dashboard(metrics: List[str], refresh_interval: int = 30) -> str`

Create custom monitoring dashboard.

**Parameters:**
- `metrics`: List of metrics to include
- `refresh_interval`: Dashboard refresh interval in seconds

**Returns:** Dashboard identifier

##### `export_metrics(format: str = "json", time_range: str = "24h") -> str`

Export metrics data.

**Parameters:**
- `format`: Export format ("json", "csv", "prometheus")
- `time_range`: Time range for export

**Returns:** Exported data or file path

##### `register_monitoring_plugin(plugin_class: type) -> None`

Register custom monitoring plugin.

**Parameters:**
- `plugin_class`: Custom monitoring plugin class

## Monitoring Configuration

### Global Monitoring Settings

```python
from context_store.monitoring import configure_monitoring

# Configure global monitoring settings
configure_monitoring({
    "enable_performance_tracking": True,
    "enable_cache_monitoring": True,
    "enable_memory_monitoring": True,
    "metric_retention_days": 30,
    "alert_cooldown_minutes": 5,
    "dashboard_auto_refresh": True
})
```

### Integration with External Systems

#### Prometheus Integration

```python
from context_store.monitoring import PrometheusExporter

prometheus_exporter = PrometheusExporter(context_store)
prometheus_exporter.start_server(port=8000)
```

#### Grafana Integration

```python
from context_store.monitoring import GrafanaDashboard

grafana = GrafanaDashboard(context_store)
grafana.create_standard_dashboard()
grafana.export_dashboard_json("context_store_dashboard.json")
```

## Best Practices

### 1. Performance Monitoring

```python
# Set up comprehensive performance monitoring
monitor = PerformanceMonitor(context_store)
monitor.set_performance_thresholds({
    "max_store_time_ms": 50.0,
    "max_retrieve_time_ms": 20.0,
    "min_cache_hit_rate": 0.85
})
monitor.start_performance_tracking()
```

### 2. Memory Optimization

```python
# Monitor and optimize memory usage
memory_monitor = MemoryMonitor(context_store)
memory_stats = memory_monitor.get_memory_usage()

if memory_stats["total_mb"] > 1000:  # 1GB threshold
    cleanup_stats = memory_monitor.trigger_memory_cleanup()
    print(f"Cleaned up {cleanup_stats['contexts_removed']} contexts")
```

### 3. Real-time Alerting

```python
# Set up real-time alerting
real_time = RealTimeMetrics(context_store)
real_time.set_metric_alerts({
    "memory_warning": {
        "metric": "memory_usage_mb",
        "threshold": 800.0,
        "operator": "greater_than"
    }
})

def handle_alert(alert):
    if alert["severity"] == "critical":
        # Trigger emergency procedures
        emergency_cleanup()
    
real_time.register_alert_callback(handle_alert)
real_time.start_real_time_monitoring()
```

### 4. Analytics and Insights

```python
# Regular analytics review
analytics = ContextAnalytics(context_store)
weekly_report = analytics.get_usage_patterns()
storage_forecast = analytics.predict_storage_needs(30)

print(f"Weekly context creation rate: {weekly_report['contexts_per_day']}")
print(f"Predicted storage in 30 days: {storage_forecast['estimated_mb']:.2f}MB")
```

For complete examples and advanced monitoring patterns, see the [Performance Optimization Guide](../guides/performance.md).
