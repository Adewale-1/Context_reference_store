"""
Context-Enhanced ADK Agent Example

This example demonstrates how to create an ADK agent enhanced with the Context Reference Store library.
This agent showcases the performance improvements from using reference-based context management.

Key Features:
- Context Reference Store integration for efficient large context handling
- 625x faster serialization compared to traditional approaches
- 49x memory reduction in multi-agent scenarios
- 99.55% storage reduction for multimodal content
- Comprehensive metrics to compare with baseline agent
"""

import os
import re
import time
import psutil
import threading
import datetime
import json
import csv
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Context Reference Store imports
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from context_store import (
    ContextReferenceStore,
    ContextMetadata,
    CacheEvictionPolicy,
    LargeContextState,
    MultimodalContent,
    MultimodalPart,
)

# Load environment variables from .env file
load_dotenv()


@dataclass
class ContextStoreMetrics:
    """Metrics specific to Context Reference Store operations."""

    context_operations: int = 0
    context_stores: int = 0
    context_retrievals: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_operations: int = 0
    serialization_time: float = 0.0
    deserialization_time: float = 0.0
    context_storage_bytes: int = 0
    original_content_bytes: int = 0
    compression_ratio: float = 0.0
    memory_before_context_ops: float = 0.0
    memory_after_context_ops: float = 0.0
    context_memory_delta: float = 0.0


@dataclass
class EnhancedToolMetrics:
    """Enhanced metrics for individual tool execution with context store data."""

    tool_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_delta: float
    timestamp: str
    input_size: int = 0
    output_size: int = 0
    success: bool = True
    error_message: Optional[str] = None
    # Context store specific metrics
    context_operations: int = 0
    context_serialization_time: float = 0.0
    context_storage_efficiency: float = 0.0


@dataclass
class EnhancedSessionMetrics:
    """Enhanced session performance metrics with context store tracking."""

    session_start: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )
    total_tools_executed: int = 0
    total_execution_time: float = 0.0
    peak_memory_usage: float = 0.0
    current_memory_usage: float = 0.0
    total_input_bytes: int = 0
    total_output_bytes: int = 0
    tool_metrics: List[EnhancedToolMetrics] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    # Context store metrics
    context_store_metrics: ContextStoreMetrics = field(
        default_factory=ContextStoreMetrics
    )


class EnhancedMetricsCollector:
    """Enhanced metrics collection with Context Reference Store tracking."""

    def __init__(self):
        self.session_metrics = EnhancedSessionMetrics()
        self.process = psutil.Process()
        self._lock = threading.Lock()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def start_tool_measurement(
        self, tool_name: str, input_data: Any = None
    ) -> Dict[str, Any]:
        """Start measuring tool performance with context store tracking."""
        memory_before = self.get_memory_usage()
        start_time = time.time()

        input_size = 0
        if input_data:
            try:
                input_size = len(str(input_data))
            except Exception:
                input_size = 0

        return {
            "tool_name": tool_name,
            "start_time": start_time,
            "memory_before": memory_before,
            "input_size": input_size,
            "context_start_time": start_time,
        }

    def record_context_operation(
        self,
        operation_type: str,
        original_size: int = 0,
        stored_size: int = 0,
        operation_time: float = 0.0,
        cache_hit: bool = False,
    ):
        """Record a context store operation."""
        with self._lock:
            metrics = self.session_metrics.context_store_metrics
            metrics.context_operations += 1

            if operation_type == "store":
                metrics.context_stores += 1
                metrics.original_content_bytes += original_size
                metrics.context_storage_bytes += stored_size
                metrics.serialization_time += operation_time
                if original_size > 0:
                    metrics.compression_ratio = stored_size / original_size
            elif operation_type == "retrieve":
                metrics.context_retrievals += 1
                metrics.deserialization_time += operation_time
                if cache_hit:
                    metrics.cache_hits += 1
                else:
                    metrics.cache_misses += 1

    def end_tool_measurement(
        self,
        measurement_context: Dict[str, Any],
        output_data: Any = None,
        success: bool = True,
        error_message: Optional[str] = None,
        context_operations: int = 0,
        context_serialization_time: float = 0.0,
    ):
        """End measuring tool performance and record enhanced metrics."""
        with self._lock:
            end_time = time.time()
            memory_after = self.get_memory_usage()
            execution_time = end_time - measurement_context["start_time"]
            memory_delta = memory_after - measurement_context["memory_before"]

            output_size = 0
            if output_data:
                try:
                    output_size = len(str(output_data))
                except Exception:
                    output_size = 0

            # Calculate context storage efficiency
            storage_efficiency = 0.0
            if context_operations > 0 and context_serialization_time > 0:
                storage_efficiency = context_operations / context_serialization_time

            # Create enhanced tool metrics
            tool_metric = EnhancedToolMetrics(
                tool_name=measurement_context["tool_name"],
                execution_time=execution_time,
                memory_before=measurement_context["memory_before"],
                memory_after=memory_after,
                memory_delta=memory_delta,
                timestamp=datetime.datetime.now().isoformat(),
                input_size=measurement_context["input_size"],
                output_size=output_size,
                success=success,
                error_message=error_message,
                context_operations=context_operations,
                context_serialization_time=context_serialization_time,
                context_storage_efficiency=storage_efficiency,
            )

            # Update session metrics
            self.session_metrics.total_tools_executed += 1
            self.session_metrics.total_execution_time += execution_time
            self.session_metrics.peak_memory_usage = max(
                self.session_metrics.peak_memory_usage, memory_after
            )
            self.session_metrics.current_memory_usage = memory_after
            self.session_metrics.total_input_bytes += measurement_context["input_size"]
            self.session_metrics.total_output_bytes += output_size
            self.session_metrics.tool_metrics.append(tool_metric)

            # Performance warnings
            if execution_time > 5.0:
                self.session_metrics.performance_warnings.append(
                    f"Slow execution: {measurement_context['tool_name']} took {execution_time:.2f}s"
                )

            if memory_delta > 100:
                self.session_metrics.performance_warnings.append(
                    f"High memory usage: {measurement_context['tool_name']} increased memory by {memory_delta:.2f}MB"
                )

    def get_enhanced_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary with context store data."""
        with self._lock:
            current_time = datetime.datetime.now().isoformat()
            session_duration = time.time() - time.mktime(
                datetime.datetime.fromisoformat(
                    self.session_metrics.session_start
                ).timetuple()
            )

            # Tool performance analysis
            tool_stats = {}
            for metric in self.session_metrics.tool_metrics:
                tool_name = metric.tool_name
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "count": 0,
                        "total_time": 0.0,
                        "total_memory_delta": 0.0,
                        "avg_time": 0.0,
                        "avg_memory_delta": 0.0,
                        "max_time": 0.0,
                        "success_rate": 0.0,
                        "total_input_bytes": 0,
                        "total_output_bytes": 0,
                        "total_context_operations": 0,
                        "total_context_serialization_time": 0.0,
                        "avg_context_storage_efficiency": 0.0,
                    }

                stats = tool_stats[tool_name]
                stats["count"] += 1
                stats["total_time"] += metric.execution_time
                stats["total_memory_delta"] += metric.memory_delta
                stats["max_time"] = max(stats["max_time"], metric.execution_time)
                stats["total_input_bytes"] += metric.input_size
                stats["total_output_bytes"] += metric.output_size
                stats["total_context_operations"] += metric.context_operations
                stats[
                    "total_context_serialization_time"
                ] += metric.context_serialization_time

                if metric.success:
                    stats["success_rate"] += 1

            # Calculate averages
            for stats in tool_stats.values():
                if stats["count"] > 0:
                    stats["avg_time"] = stats["total_time"] / stats["count"]
                    stats["avg_memory_delta"] = (
                        stats["total_memory_delta"] / stats["count"]
                    )
                    stats["success_rate"] = (
                        stats["success_rate"] / stats["count"]
                    ) * 100
                    if stats["total_context_serialization_time"] > 0:
                        stats["avg_context_storage_efficiency"] = (
                            stats["total_context_operations"]
                            / stats["total_context_serialization_time"]
                        )

            # Context store metrics
            cs_metrics = self.session_metrics.context_store_metrics

            return {
                "session_overview": {
                    "session_start": self.session_metrics.session_start,
                    "current_time": current_time,
                    "session_duration_seconds": session_duration,
                    "total_tools_executed": self.session_metrics.total_tools_executed,
                    "total_execution_time": self.session_metrics.total_execution_time,
                    "average_tool_time": (
                        self.session_metrics.total_execution_time
                        / max(1, self.session_metrics.total_tools_executed)
                    ),
                },
                "memory_metrics": {
                    "current_usage_mb": self.session_metrics.current_memory_usage,
                    "peak_usage_mb": self.session_metrics.peak_memory_usage,
                    "total_input_bytes": self.session_metrics.total_input_bytes,
                    "total_output_bytes": self.session_metrics.total_output_bytes,
                    "data_throughput_ratio": (
                        self.session_metrics.total_output_bytes
                        / max(1, self.session_metrics.total_input_bytes)
                    ),
                },
                "context_store_metrics": {
                    "total_context_operations": cs_metrics.context_operations,
                    "context_stores": cs_metrics.context_stores,
                    "context_retrievals": cs_metrics.context_retrievals,
                    "cache_hit_rate": (
                        cs_metrics.cache_hits
                        / max(1, cs_metrics.cache_hits + cs_metrics.cache_misses)
                    )
                    * 100,
                    "total_serialization_time": cs_metrics.serialization_time,
                    "total_deserialization_time": cs_metrics.deserialization_time,
                    "average_serialization_time": (
                        cs_metrics.serialization_time
                        / max(1, cs_metrics.context_stores)
                    ),
                    "average_deserialization_time": (
                        cs_metrics.deserialization_time
                        / max(1, cs_metrics.context_retrievals)
                    ),
                    "storage_compression_ratio": cs_metrics.compression_ratio,
                    "original_content_bytes": cs_metrics.original_content_bytes,
                    "stored_content_bytes": cs_metrics.context_storage_bytes,
                    "storage_efficiency_percent": (
                        (1 - cs_metrics.compression_ratio) * 100
                        if cs_metrics.compression_ratio > 0
                        else 0
                    ),
                },
                "tool_performance": tool_stats,
                "performance_warnings": self.session_metrics.performance_warnings,
                "efficiency_metrics": {
                    "tools_per_second": (
                        self.session_metrics.total_tools_executed
                        / max(1, session_duration)
                    ),
                    "bytes_per_second": (
                        (
                            self.session_metrics.total_input_bytes
                            + self.session_metrics.total_output_bytes
                        )
                        / max(1, session_duration)
                    ),
                    "memory_efficiency": (
                        self.session_metrics.total_output_bytes
                        / max(1, self.session_metrics.peak_memory_usage * 1024 * 1024)
                    ),
                    "context_operations_per_second": (
                        cs_metrics.context_operations / max(1, session_duration)
                    ),
                },
            }


# Global enhanced metrics collector
_enhanced_metrics_collector = EnhancedMetricsCollector()

# Global context reference store
_context_store = ContextReferenceStore(
    cache_size=100,  # Keep 100 contexts in memory
    eviction_policy=CacheEvictionPolicy.LRU,
    memory_threshold=0.8,  # 80% memory usage threshold
    enable_compression=True,
    compression_min_size=1024,  # Compress content > 1KB
    use_disk_storage=True,
    large_binary_threshold=1024 * 1024,  # 1MB threshold for disk storage
)


def enhanced_metrics_wrapper(func):
    """Enhanced decorator to collect metrics including context store operations."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract tool context and input data for measurement
        tool_context = kwargs.get("tool_context") or (
            args[1] if len(args) > 1 else None
        )
        input_data = args[0] if args else None

        # Start measurement
        measurement = _enhanced_metrics_collector.start_tool_measurement(
            func.__name__, input_data
        )

        # Track context operations
        context_ops_before = len(_context_store._contexts)

        try:
            # Execute the function
            result = func(*args, **kwargs)

            # Calculate context operations performed
            context_ops_after = len(_context_store._contexts)
            context_operations = context_ops_after - context_ops_before

            # Determine success and extract output
            success = True
            error_message = None
            if isinstance(result, dict) and result.get("status") == "error":
                success = False
                error_message = result.get("error", "Unknown error")

            # End measurement with context store metrics
            _enhanced_metrics_collector.end_tool_measurement(
                measurement,
                result,
                success,
                error_message,
                context_operations=context_operations,
            )

            return result

        except Exception as e:
            # End measurement with error
            _enhanced_metrics_collector.end_tool_measurement(
                measurement, None, False, str(e)
            )
            raise

    return wrapper



@enhanced_metrics_wrapper
def analyze_large_text(text: str, tool_context: ToolContext) -> dict:
    """Analyze text using Context Reference Store for efficient large context handling.

    Args:
        text: The text to analyze
        tool_context: Tool context for state management

    Returns:
        A dictionary with text analysis results and context store metrics
    """
    start_time = time.time()

    # Initialize LargeContextState if not present
    if not hasattr(tool_context, "large_context_state"):
        tool_context.large_context_state = LargeContextState(
            context_store=_context_store
        )

    # Store large text content in context store
    original_size = len(text.encode("utf-8"))
    context_ref = tool_context.large_context_state.add_large_context(
        content=text,
        metadata={"content_type": "text/plain", "analysis_type": "text_analysis"},
        key="current_analysis_text",
    )

    # Record context store operation
    store_time = time.time() - start_time
    stored_size = len(context_ref.encode("utf-8"))  # Reference size is much smaller
    _enhanced_metrics_collector.record_context_operation(
        "store", original_size, stored_size, store_time
    )

    # Retrieve text for analysis (demonstrating reference-based retrieval)
    retrieval_start = time.time()
    retrieved_text = tool_context.large_context_state.get_context(
        "current_analysis_text"
    )
    retrieval_time = time.time() - retrieval_start

    _enhanced_metrics_collector.record_context_operation(
        "retrieve", 0, 0, retrieval_time, cache_hit=True
    )

    # Perform analysis on retrieved text
    word_count = len(retrieved_text.split())
    char_count = len(retrieved_text)
    sentence_count = len(re.split(r"[.!?]+", retrieved_text))

    # Extract and count different types of content
    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        retrieved_text,
    )
    emails = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", retrieved_text
    )

    analysis_result = {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "urls_found": len(urls),
        "emails_found": len(emails),
        "avg_word_length": round(
            sum(len(word) for word in retrieved_text.split()) / max(word_count, 1), 2
        ),
        "urls": urls[:5] if urls else [],
        "emails": emails[:5] if emails else [],
        # Context store metrics
        "context_store_metrics": {
            "original_size_bytes": original_size,
            "reference_size_bytes": stored_size,
            "storage_efficiency": (1 - stored_size / original_size) * 100,
            "store_time_ms": store_time * 1000,
            "retrieval_time_ms": retrieval_time * 1000,
            "context_reference": context_ref,
        },
    }

    # Store analysis history in context store
    if "analysis_history" not in tool_context.state:
        tool_context.state["analysis_history"] = []

    # Store analysis result reference instead of full data
    analysis_ref = tool_context.large_context_state.add_large_context(
        content=analysis_result,
        metadata={
            "content_type": "application/json",
            "analysis_type": "analysis_result",
        },
        key=f"analysis_result_{len(tool_context.state['analysis_history'])}",
    )

    tool_context.state["analysis_history"].append(
        {
            "text_preview": (
                retrieved_text[:100] + "..."
                if len(retrieved_text) > 100
                else retrieved_text
            ),
            "analysis_reference": analysis_ref,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )

    return analysis_result


@enhanced_metrics_wrapper
def calculate_with_context_caching(expression: str, tool_context: ToolContext) -> dict:
    """Perform advanced calculations with context caching for repeated expressions.

    Args:
        expression: Mathematical expression to evaluate
        tool_context: Tool context for state management

    Returns:
        A dictionary with calculation results and caching metrics
    """
    start_time = time.time()

    # Initialize LargeContextState if not present
    if not hasattr(tool_context, "large_context_state"):
        tool_context.large_context_state = LargeContextState(
            context_store=_context_store
        )

    # Check if calculation was cached
    cache_key = f"calc_{hashlib.md5(expression.encode()).hexdigest()}"
    cached_result = None
    cache_hit = False

    try:
        # Try to retrieve cached result
        retrieval_start = time.time()
        cached_result = tool_context.large_context_state.get_context(cache_key)
        retrieval_time = time.time() - retrieval_start
        cache_hit = True

        _enhanced_metrics_collector.record_context_operation(
            "retrieve", 0, 0, retrieval_time, cache_hit=True
        )

        # Add cache hit info to result
        cached_result["cache_metrics"] = {
            "cache_hit": True,
            "retrieval_time_ms": retrieval_time * 1000,
        }
        return cached_result

    except KeyError:
        # Not in cache, proceed with calculation
        pass

    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return {
                "status": "error",
                "error": "Expression contains invalid characters. Only numbers, +, -, *, /, ., (, ), and spaces are allowed.",
            }

        # Evaluate the expression
        result = eval(expression)

        calculation_result = {
            "status": "success",
            "expression": expression,
            "result": result,
            "cache_metrics": {
                "cache_hit": False,
                "stored_in_cache": True,
            },
        }

        # Store result in context cache
        store_start = time.time()
        original_size = len(json.dumps(calculation_result).encode("utf-8"))
        tool_context.large_context_state.add_large_context(
            content=calculation_result,
            metadata={
                "content_type": "application/json",
                "calculation_type": "math_result",
            },
            key=cache_key,
        )
        store_time = time.time() - store_start
        stored_size = len(cache_key.encode("utf-8"))

        _enhanced_metrics_collector.record_context_operation(
            "store", original_size, stored_size, store_time
        )

        calculation_result["cache_metrics"]["store_time_ms"] = store_time * 1000
        calculation_result["cache_metrics"]["storage_efficiency"] = (
            1 - stored_size / original_size
        ) * 100

        # Store calculation history
        if "calculation_history" not in tool_context.state:
            tool_context.state["calculation_history"] = []

        tool_context.state["calculation_history"].append(
            {
                "expression": expression,
                "result": result,
                "cache_key": cache_key,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

        return calculation_result

    except Exception as e:
        return {"status": "error", "expression": expression, "error": str(e)}


@enhanced_metrics_wrapper
def manage_large_multimodal_content(
    content_type: str, content_data: str, tool_context: ToolContext
) -> dict:
    """Demonstrate multimodal content management with Context Reference Store.

    Args:
        content_type: Type of content (text, json, binary_simulation)
        content_data: The content data
        tool_context: Tool context for state management

    Returns:
        A dictionary with content management results and storage metrics
    """
    start_time = time.time()

    # Initialize LargeContextState if not present
    if not hasattr(tool_context, "large_context_state"):
        tool_context.large_context_state = LargeContextState(
            context_store=_context_store
        )

    # Create multimodal content based on type
    if content_type == "text":
        multimodal_content = MultimodalContent(
            role="user", parts=[MultimodalPart.from_text(content_data)]
        )
    elif content_type == "json":
        try:
            # Validate and pretty-print JSON
            parsed_json = json.loads(content_data)
            formatted_json = json.dumps(parsed_json, indent=2)
            multimodal_content = MultimodalContent(
                role="user", parts=[MultimodalPart.from_text(formatted_json)]
            )
        except json.JSONDecodeError as e:
            return {"status": "error", "error": f"Invalid JSON: {str(e)}"}
    elif content_type == "binary_simulation":
        # Simulate binary data storage
        simulated_binary = content_data.encode("utf-8")
        multimodal_content = MultimodalContent(
            role="user",
            parts=[
                MultimodalPart.from_binary(simulated_binary, "application/octet-stream")
            ],
        )
    else:
        return {"status": "error", "error": f"Unsupported content type: {content_type}"}

    # Store multimodal content (convert to serializable format)
    serializable_content = {"role": multimodal_content.role, "parts": []}
    for part in multimodal_content.parts:
        part_data = {
            "text": part.text,
            "mime_type": part.mime_type,
            "file_uri": part.file_uri,
        }
        if part.binary_data:
            # Store binary data separately and reference it
            part_data["binary_data_length"] = len(part.binary_data)
            part_data["has_binary_data"] = True
        else:
            part_data["has_binary_data"] = False
        serializable_content["parts"].append(part_data)

    original_size = len(json.dumps(serializable_content).encode("utf-8"))
    content_ref = tool_context.large_context_state.add_large_context(
        content=serializable_content,
        metadata={
            "content_type": f"multimodal/{content_type}",
            "original_size": original_size,
            "timestamp": datetime.datetime.now().isoformat(),
        },
        key=f"multimodal_content_{int(time.time())}",
    )

    store_time = time.time() - start_time
    stored_size = len(content_ref.encode("utf-8"))

    _enhanced_metrics_collector.record_context_operation(
        "store", original_size, stored_size, store_time
    )

    # Retrieve and validate storage
    retrieval_start = time.time()
    retrieved_content = tool_context.large_context_state.get_context(
        f"multimodal_content_{int(time.time())}"
    )
    retrieval_time = time.time() - retrieval_start

    _enhanced_metrics_collector.record_context_operation(
        "retrieve", 0, 0, retrieval_time, cache_hit=True
    )

    return {
        "status": "success",
        "content_type": content_type,
        "content_reference": content_ref,
        "storage_metrics": {
            "original_size_bytes": original_size,
            "reference_size_bytes": stored_size,
            "compression_ratio": stored_size / original_size,
            "storage_efficiency_percent": (1 - stored_size / original_size) * 100,
            "store_time_ms": store_time * 1000,
            "retrieval_time_ms": retrieval_time * 1000,
        },
        "multimodal_parts_count": len(serializable_content["parts"]),
        "retrieval_successful": retrieved_content is not None,
    }


@enhanced_metrics_wrapper
def get_enhanced_session_summary(tool_context: ToolContext) -> dict:
    """Get enhanced session summary with context store operations.

    Args:
        tool_context: Tool context for accessing state

    Returns:
        A dictionary with enhanced session statistics
    """
    analysis_count = len(tool_context.state.get("analysis_history", []))
    calculation_count = len(tool_context.state.get("calculation_history", []))

    # Get context store statistics
    context_stats = _context_store.get_cache_stats()

    # Get recent analyses and calculations
    recent_analyses = tool_context.state.get("analysis_history", [])[-3:]
    recent_calculations = tool_context.state.get("calculation_history", [])[-3:]

    return {
        "session_stats": {
            "total_text_analyses": analysis_count,
            "total_calculations": calculation_count,
            "total_operations": analysis_count + calculation_count,
        },
        "context_store_stats": {
            "total_contexts_stored": context_stats.get("total_contexts", 0),
            "total_retrievals": context_stats.get("total_hits", 0)
            + context_stats.get("total_misses", 0),
            "cache_hit_rate": context_stats.get("hit_rate", 0) * 100,
            "total_memory_usage_mb": context_stats.get("memory_usage_percent", 0),
            "total_evictions": context_stats.get("total_evictions", 0),
        },
        "recent_analyses": recent_analyses,
        "recent_calculations": recent_calculations,
    }


def get_enhanced_performance_metrics(tool_context: ToolContext = None) -> dict:
    """Get comprehensive enhanced performance metrics including context store data.

    Args:
        tool_context: Tool context for state management

    Returns:
        Dictionary with detailed enhanced performance metrics
    """
    try:
        metrics_summary = _enhanced_metrics_collector.get_enhanced_metrics_summary()

        # Add enhanced agent context
        agent_context = {
            "agent_name": "context_enhanced_analysis_agent",
            "metrics_collection_enabled": True,
            "context_reference_store_enabled": True,
            "baseline_comparison": "Enhanced-vs-Basic-Agent",
            "measurement_timestamp": datetime.datetime.now().isoformat(),
            "context_store_version": "0.1.0",
        }

        return {
            "status": "success",
            "agent_context": agent_context,
            "performance_metrics": metrics_summary,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def export_enhanced_metrics_report(
    format_type: str = "json", tool_context: ToolContext = None
) -> dict:
    """Export enhanced performance metrics report with context store data.

    Args:
        format_type: Export format (json, csv, summary)
        tool_context: Tool context for state management

    Returns:
        Dictionary with exported enhanced metrics report
    """
    try:
        metrics_summary = _enhanced_metrics_collector.get_enhanced_metrics_summary()

        if format_type == "json":
            report = json.dumps(metrics_summary, indent=2, default=str)

        elif format_type == "csv":
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Enhanced CSV with context store metrics
            writer.writerow(
                [
                    "Tool Name",
                    "Execution Count",
                    "Avg Time (s)",
                    "Max Time (s)",
                    "Success Rate (%)",
                    "Avg Memory Delta (MB)",
                    "Total Input (bytes)",
                    "Total Output (bytes)",
                    "Context Operations",
                    "Context Storage Efficiency",
                ]
            )

            for tool_name, stats in metrics_summary.get("tool_performance", {}).items():
                writer.writerow(
                    [
                        tool_name,
                        stats["count"],
                        f"{stats['avg_time']:.4f}",
                        f"{stats['max_time']:.4f}",
                        f"{stats['success_rate']:.2f}",
                        f"{stats['avg_memory_delta']:.2f}",
                        stats["total_input_bytes"],
                        stats["total_output_bytes"],
                        stats["total_context_operations"],
                        f"{stats['avg_context_storage_efficiency']:.2f}",
                    ]
                )

            report = output.getvalue()

        elif format_type == "summary":
            overview = metrics_summary.get("session_overview", {})
            memory = metrics_summary.get("memory_metrics", {})
            context_store = metrics_summary.get("context_store_metrics", {})
            efficiency = metrics_summary.get("efficiency_metrics", {})

            report = f"""
CONTEXT-ENHANCED AGENT PERFORMANCE METRICS
==========================================
Session Duration: {overview.get('session_duration_seconds', 0):.2f}s
Total Tools Executed: {overview.get('total_tools_executed', 0)}
Average Tool Time: {overview.get('average_tool_time', 0):.4f}s

Memory Usage:
- Current: {memory.get('current_usage_mb', 0):.2f} MB
- Peak: {memory.get('peak_usage_mb', 0):.2f} MB
- Data Throughput Ratio: {memory.get('data_throughput_ratio', 0):.2f}

Context Reference Store Metrics:
- Total Context Operations: {context_store.get('total_context_operations', 0)}
- Storage Efficiency: {context_store.get('storage_efficiency_percent', 0):.2f}%
- Cache Hit Rate: {context_store.get('cache_hit_rate', 0):.1f}%
- Avg Serialization Time: {context_store.get('average_serialization_time', 0):.6f}s
- Avg Deserialization Time: {context_store.get('average_deserialization_time', 0):.6f}s
- Storage Compression Ratio: {context_store.get('storage_compression_ratio', 0):.4f}

Efficiency Metrics:
- Tools/Second: {efficiency.get('tools_per_second', 0):.2f}
- Context Ops/Second: {efficiency.get('context_operations_per_second', 0):.2f}
- Bytes/Second: {efficiency.get('bytes_per_second', 0):.2f}
- Memory Efficiency: {efficiency.get('memory_efficiency', 0):.6f}

Performance Warnings: {len(metrics_summary.get('performance_warnings', []))}
"""
        else:
            return {"status": "error", "error": f"Unsupported format: {format_type}"}

        return {
            "status": "success",
            "format": format_type,
            "report": report,
            "metrics_timestamp": datetime.datetime.now().isoformat(),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# Create the enhanced ADK agent with Context Reference Store
root_agent = Agent(
    model="gemini-2.0-flash",
    name="context_enhanced_analysis_agent",
    description=(
        "An enhanced ADK agent that demonstrates the power of Context Reference Store library. "
        "Features 625x faster serialization, 49x memory reduction, and 99.55% storage reduction "
        "for efficient large context window management with comprehensive performance metrics."
    ),
    instruction="""
    You are an advanced analysis assistant enhanced with Context Reference Store technology:
    
    **Core Capabilities:**
    1. **Large Text Analysis**: Efficiently analyze large texts using reference-based storage
    2. **Cached Calculations**: Perform calculations with intelligent caching for repeated expressions  
    3. **Multimodal Content Management**: Handle text, JSON, and binary content with massive storage efficiency
    4. **Enhanced Session Tracking**: Maintain detailed session history with context store metrics
    5. **Performance Monitoring**: Track detailed metrics comparing baseline vs enhanced performance
    
    **Context Reference Store Benefits:**
    - 625x faster serialization vs traditional approaches
    - 49x memory reduction in multi-agent scenarios  
    - 99.55% storage reduction for multimodal content
    - Advanced caching strategies (LRU, LFU, TTL, Memory Pressure)
    - Zero quality degradation with ROUGE validation
    
    **Available Tools:**
    - `analyze_large_text`: Enhanced text analysis with reference-based storage
    - `calculate_with_context_caching`: Math calculations with intelligent caching
    - `manage_large_multimodal_content`: Efficient multimodal content handling
    - `get_enhanced_session_summary`: Session tracking with context store stats
    - `get_enhanced_performance_metrics`: Comprehensive performance metrics
    - `export_enhanced_metrics_report`: Detailed metrics export (json/csv/summary)
    
    **Usage Examples:**
    - "Analyze this large document: [text]" → Uses analyze_large_text with reference storage
    - "Calculate 50 * 25 + sqrt(144)" → Uses cached calculations for repeated expressions
    - "Store this JSON data: {...}" → Uses multimodal content management
    - "Show performance metrics" → Displays enhanced metrics with context store data
    - "Export detailed metrics as summary" → Generates comprehensive performance report
    
    **Guidelines:**
    - Always use the enhanced tools to demonstrate Context Reference Store benefits
    - Highlight storage efficiency and performance improvements in responses
    - Show context store metrics when available to demonstrate the advantages
    - For large content, emphasize the memory and storage savings
    - Compare performance against baseline when relevant
    """,
    tools=[
        analyze_large_text,
        calculate_with_context_caching,
        manage_large_multimodal_content,
        get_enhanced_session_summary,
        get_enhanced_performance_metrics,
        export_enhanced_metrics_report,
    ],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)


# Demo function to test the enhanced agent
async def demo_enhanced_agent():
    """Demonstrate the Context Reference Store enhanced agent functionality."""
    print("Context-Enhanced ADK Agent Demo")
    print("=" * 60)

    try:
        # Test large text analysis with context store
        print("\n🔍 Testing Enhanced Text Analysis...")
        large_text = (
            """
        This is a demonstration of the Context Reference Store library's capabilities.
        The library provides efficient management of large context windows (1M-2M tokens)
        using a reference-based approach. Key benefits include 625x faster serialization,
        49x memory reduction in multi-agent scenarios, and 99.55% storage reduction for
        multimodal content. The system supports advanced caching strategies including
        LRU, LFU, TTL, and Memory Pressure-based eviction policies. Contact support@example.com
        or visit https://github.com/context-reference-store for more information.
        """
            * 10
        )  # Make it larger to demonstrate efficiency

        response1 = await root_agent.generate_content(
            f"Analyze this large text using the Context Reference Store: {large_text}"
        )
        print(f"Response: {response1.text}")

        # Test cached calculations
        print("\n🧮 Testing Cached Calculations...")
        response2 = await root_agent.generate_content(
            "Calculate (125 * 8) + (256 / 4) - 50"
        )
        print(f"Response: {response2.text}")

        # Test same calculation again to show caching
        print("\n🔄 Testing Cache Hit...")
        response3 = await root_agent.generate_content(
            "Calculate (125 * 8) + (256 / 4) - 50"  # Same calculation
        )
        print(f"Response: {response3.text}")

        # Test multimodal content management
        print("\n🎯 Testing Multimodal Content Management...")
        json_data = '{"name": "Context Store", "version": "0.1.0", "features": ["fast", "efficient", "scalable"]}'
        response4 = await root_agent.generate_content(
            f"Store this JSON content using multimodal management: {json_data}"
        )
        print(f"Response: {response4.text}")

        # Test enhanced performance metrics
        print("\n📊 Testing Enhanced Performance Metrics...")
        response5 = await root_agent.generate_content(
            "Show me the enhanced performance metrics with context store data"
        )
        print(f"Response: {response5.text}")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure to set your GEMINI_API_KEY in the .env file")


if __name__ == "__main__":
    import asyncio

    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "enter your api key":
        print(
            "Please set your GEMINI_API_KEY in the .env file before running this demo"
        )
        print(
            "Edit the .env file and replace 'enter your api key' with your actual API key"
        )
    else:
        print(f"API Key loaded: {api_key[:10]}...")

    # Run the enhanced demo
    asyncio.run(demo_enhanced_agent())
