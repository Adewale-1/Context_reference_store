"""
Advanced Multi-Tool ADK Agent

A sophisticated ADK agent with extensive tool capabilities across multiple domains:
- File and system operations
- Data analysis and visualization
- Web and API interactions
- Text processing and NLP
- Mathematical and scientific computing
- Productivity and automation tools
- Security and validation utilities
"""

import os
import re
import json
import csv
import base64
import hashlib
import datetime
import random
import string
import urllib.parse
import urllib.request
import math
import statistics
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv()

@dataclass
class ToolMetrics:
    """Metrics for individual tool execution."""

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


@dataclass
class SessionMetrics:
    """Overall session performance metrics."""

    session_start: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )
    total_tools_executed: int = 0
    total_execution_time: float = 0.0
    peak_memory_usage: float = 0.0
    current_memory_usage: float = 0.0
    total_input_bytes: int = 0
    total_output_bytes: int = 0
    tool_metrics: List[ToolMetrics] = field(default_factory=list)
    context_operations: Dict[str, int] = field(default_factory=dict)
    performance_warnings: List[str] = field(default_factory=list)


class MetricsCollector:
    """Centralized metrics collection and analysis."""

    def __init__(self):
        self.session_metrics = SessionMetrics()
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
        """Start measuring tool performance."""
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
        }

    def end_tool_measurement(
        self,
        measurement_context: Dict[str, Any],
        output_data: Any = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """End measuring tool performance and record metrics."""
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

            # Create tool metrics
            tool_metric = ToolMetrics(
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

            # Track context operations
            if "context" in measurement_context["tool_name"].lower():
                op_type = measurement_context["tool_name"]
                self.session_metrics.context_operations[op_type] = (
                    self.session_metrics.context_operations.get(op_type, 0) + 1
                )

            # Performance warnings
            if execution_time > 5.0:  # > 5 seconds
                self.session_metrics.performance_warnings.append(
                    f"Slow execution: {measurement_context['tool_name']} took {execution_time:.2f}s"
                )

            if memory_delta > 100:  # > 100MB increase
                self.session_metrics.performance_warnings.append(
                    f"High memory usage: {measurement_context['tool_name']} increased memory by {memory_delta:.2f}MB"
                )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
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
                    }

                stats = tool_stats[tool_name]
                stats["count"] += 1
                stats["total_time"] += metric.execution_time
                stats["total_memory_delta"] += metric.memory_delta
                stats["max_time"] = max(stats["max_time"], metric.execution_time)
                stats["total_input_bytes"] += metric.input_size
                stats["total_output_bytes"] += metric.output_size

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
                "tool_performance": tool_stats,
                "context_operations": self.session_metrics.context_operations,
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
                },
            }


# Global metrics collector
_metrics_collector = MetricsCollector()


def metrics_wrapper(func):
    """Decorator to automatically collect metrics for tool functions."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract tool context and input data for measurement
        tool_context = kwargs.get("tool_context") or (
            args[1] if len(args) > 1 else None
        )
        input_data = args[0] if args else None

        # Start measurement
        measurement = _metrics_collector.start_tool_measurement(
            func.__name__, input_data
        )
        try:
            # Execute the function
            result = func(*args, **kwargs)

            # Determine success and extract output
            success = True
            error_message = None
            if isinstance(result, dict) and result.get("status") == "error":
                success = False
                error_message = result.get("error", "Unknown error")

            # End measurement
            _metrics_collector.end_tool_measurement(
                measurement, result, success, error_message
            )
            return result

        except Exception as e:
            # End measurement with error
            _metrics_collector.end_tool_measurement(measurement, None, False, str(e))
            raise

    return wrapper




@metrics_wrapper
def read_file_content(file_path: str, tool_context: ToolContext) -> dict:
    """Read content from a file safely.

    Args:
        file_path: Path to the file to read
        tool_context: Tool context for state management

    Returns:
        Dictionary with file content and metadata
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "error": f"File {file_path} does not exist"}

        if path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
            return {"status": "error", "error": "File too large (>10MB)"}

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        metadata = {
            "size": path.stat().st_size,
            "modified": datetime.datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat(),
            "lines": len(content.splitlines()),
            "words": len(content.split()),
            "characters": len(content),
        }
        # Store in state
        if "file_operations" not in tool_context.state:
            tool_context.state["file_operations"] = []

        tool_context.state["file_operations"].append(
            {
                "operation": "read",
                "file": str(path),
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata": metadata,
            }
        )
        return {
            "status": "success",
            "content": content,
            "metadata": metadata,
            "file_path": str(path),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def write_file_content(
    file_path: str, content: str, append: bool = False, tool_context: ToolContext = None
) -> dict:
    """Write content to a file safely.

    Args:
        file_path: Path to the file to write
        content: Content to write
        append: Whether to append or overwrite
        tool_context: Tool context for state management

    Returns:
        Dictionary with operation result
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)

        # Store in state
        if tool_context and "file_operations" not in tool_context.state:
            tool_context.state["file_operations"] = []

        if tool_context:
            tool_context.state["file_operations"].append(
                {
                    "operation": "write" + ("_append" if append else ""),
                    "file": str(path),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "content_length": len(content),
                }
            )

        return {
            "status": "success",
            "file_path": str(path),
            "operation": "append" if append else "write",
            "bytes_written": len(content.encode("utf-8")),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def list_directory(
    directory_path: str, pattern: str = "*", tool_context: ToolContext = None
) -> dict:
    """List files and directories with pattern matching.

    Args:
        directory_path: Path to directory
        pattern: Glob pattern for filtering
        tool_context: Tool context for state management

    Returns:
        Dictionary with directory listing
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return {
                "status": "error",
                "error": f"Directory {directory_path} does not exist",
            }

        if not path.is_dir():
            return {"status": "error", "error": f"{directory_path} is not a directory"}

        items = []
        for item in path.glob(pattern):
            stat = item.stat()
            items.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else 0,
                    "modified": datetime.datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(),
                    "permissions": oct(stat.st_mode)[-3:],
                }
            )

        items.sort(key=lambda x: (x["type"], x["name"]))

        return {
            "status": "success",
            "directory": str(path),
            "pattern": pattern,
            "items": items,
            "total_files": sum(1 for item in items if item["type"] == "file"),
            "total_directories": sum(
                1 for item in items if item["type"] == "directory"
            ),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}



@metrics_wrapper
def analyze_csv_data(file_path: str, tool_context: ToolContext) -> dict:
    """Analyze CSV data and provide statistics.

    Args:
        file_path: Path to CSV file
        tool_context: Tool context for state management

    Returns:
        Dictionary with data analysis results
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "error": f"File {file_path} does not exist"}

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            for row in reader:
                rows.append(row)

        if not rows:
            return {"status": "error", "error": "CSV file is empty or has no data rows"}

        # Basic statistics
        num_rows = len(rows)
        num_columns = len(headers)

        # Analyze each column
        column_analysis = {}
        for header in headers:
            values = [row[header] for row in rows if row[header].strip()]

            # Try to detect data type and compute statistics
            numeric_values = []
            for value in values:
                try:
                    numeric_values.append(float(value))
                except ValueError:
                    pass

            analysis = {
                "total_values": len(values),
                "empty_values": num_rows - len(values),
                "unique_values": len(set(values)),
                "is_numeric": len(numeric_values) > len(values) * 0.8,  # 80% threshold
            }

            if analysis["is_numeric"] and numeric_values:
                analysis.update(
                    {
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "mean": statistics.mean(numeric_values),
                        "median": statistics.median(numeric_values),
                        "std_dev": (
                            statistics.stdev(numeric_values)
                            if len(numeric_values) > 1
                            else 0
                        ),
                    }
                )
            else:
                # Text analysis
                analysis.update(
                    {
                        "sample_values": list(set(values))[:10],
                        "avg_length": (
                            statistics.mean([len(str(v)) for v in values])
                            if values
                            else 0
                        ),
                    }
                )

            column_analysis[header] = analysis

        result = {
            "status": "success",
            "file_path": str(path),
            "rows": num_rows,
            "columns": num_columns,
            "headers": headers,
            "column_analysis": column_analysis,
            "sample_data": rows[:5],  # First 5 rows as sample
        }

        # Store in state
        if "data_analysis" not in tool_context.state:
            tool_context.state["data_analysis"] = []

        tool_context.state["data_analysis"].append(
            {
                "operation": "csv_analysis",
                "file": str(path),
                "timestamp": datetime.datetime.now().isoformat(),
                "summary": f"{num_rows} rows, {num_columns} columns",
            }
        )
        return result

    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def generate_statistical_report(
    data_points: List[float], tool_context: ToolContext
) -> dict:
    """Generate comprehensive statistical analysis of numerical data.

    Args:
        data_points: List of numerical values
        tool_context: Tool context for state management

    Returns:
        Dictionary with statistical analysis
    """
    try:
        if not data_points:
            return {"status": "error", "error": "No data points provided"}

        # Convert to float and filter valid numbers
        valid_points = []
        for point in data_points:
            try:
                valid_points.append(float(point))
            except (ValueError, TypeError):
                pass

        if not valid_points:
            return {"status": "error", "error": "No valid numerical data points found"}

        n = len(valid_points)
        valid_points.sort()

        # Basic statistics
        stats = {
            "count": n,
            "min": min(valid_points),
            "max": max(valid_points),
            "range": max(valid_points) - min(valid_points),
            "sum": sum(valid_points),
            "mean": statistics.mean(valid_points),
            "median": statistics.median(valid_points),
            "mode": (
                statistics.mode(valid_points) if len(set(valid_points)) < n else None
            ),
        }

        # Advanced statistics
        if n > 1:
            stats.update(
                {
                    "variance": statistics.variance(valid_points),
                    "std_deviation": statistics.stdev(valid_points),
                    "coefficient_of_variation": statistics.stdev(valid_points)
                    / statistics.mean(valid_points)
                    * 100,
                }
            )

        # Percentiles
        if n >= 4:
            stats.update(
                {
                    "q1": statistics.quantiles(valid_points, n=4)[0],
                    "q3": statistics.quantiles(valid_points, n=4)[2],
                    "iqr": statistics.quantiles(valid_points, n=4)[2]
                    - statistics.quantiles(valid_points, n=4)[0],
                }
            )

        # Distribution analysis
        stats["outliers"] = []
        if n >= 4 and "iqr" in stats:
            q1, q3 = stats["q1"], stats["q3"]
            iqr = stats["iqr"]
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            stats["outliers"] = [
                x for x in valid_points if x < lower_bound or x > upper_bound
            ]

        # Data quality metrics
        stats["data_quality"] = {
            "completeness": len(valid_points) / len(data_points) * 100,
            "has_outliers": len(stats["outliers"]) > 0,
            "distribution_symmetry": (
                "symmetric"
                if abs(stats["mean"] - stats["median"])
                < stats.get("std_deviation", 0) * 0.1
                else "skewed"
            ),
        }

        return {"status": "success", "statistics": stats}

    except Exception as e:
        return {"status": "error", "error": str(e)}




@metrics_wrapper
def advanced_text_analysis(text: str, tool_context: ToolContext) -> dict:
    """Perform comprehensive text analysis including linguistic features.

    Args:
        text: Text to analyze
        tool_context: Tool context for state management

    Returns:
        Dictionary with advanced text analysis results
    """
    try:
        if not text.strip():
            return {"status": "error", "error": "Empty text provided"}

        # Basic metrics
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Character analysis
        char_counts = {
            "total": len(text),
            "alphabetic": sum(1 for c in text if c.isalpha()),
            "numeric": sum(1 for c in text if c.isdigit()),
            "whitespace": sum(1 for c in text if c.isspace()),
            "punctuation": sum(1 for c in text if c in ".,!?;:\"'()-[]{}"),
            "uppercase": sum(1 for c in text if c.isupper()),
            "lowercase": sum(1 for c in text if c.islower()),
        }

        # Word analysis
        word_lengths = [len(word.strip(".,!?;:\"'()-[]{}")) for word in words]
        unique_words = set(word.lower().strip(".,!?;:\"'()-[]{}") for word in words)

        # Sentence analysis
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]

        # Readability metrics (simplified)
        avg_sentence_length = (
            statistics.mean(sentence_lengths) if sentence_lengths else 0
        )
        avg_word_length = statistics.mean(word_lengths) if word_lengths else 0

        # Flesch Reading Ease (simplified approximation)
        if sentence_lengths and word_lengths:
            flesch_score = (
                206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            )
        else:
            flesch_score = 0

        # Content extraction
        urls = re.findall(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            text,
        )
        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
        )
        phone_numbers = re.findall(
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b", text
        )
        hashtags = re.findall(r"#\w+", text)
        mentions = re.findall(r"@\w+", text)

        # Word frequency (top 10)
        word_freq = {}
        for word in words:
            clean_word = word.lower().strip(".,!?;:\"'()-[]{}")
            if clean_word and len(clean_word) > 2:  # Skip short words
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        # Language patterns
        patterns = {
            "questions": len(re.findall(r"\?", text)),
            "exclamations": len(re.findall(r"!", text)),
            "quotes": len(re.findall(r'["\']', text)) // 2,
            "capitalized_words": sum(1 for word in words if word[0].isupper() if word),
            "all_caps_words": sum(
                1 for word in words if word.isupper() and len(word) > 1
            ),
        }

        result = {
            "status": "success",
            "basic_metrics": {
                "character_count": len(text),
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "paragraph_count": len(paragraphs),
                "unique_words": len(unique_words),
                "lexical_diversity": len(unique_words) / len(words) if words else 0,
            },
            "character_analysis": char_counts,
            "word_analysis": {
                "average_length": avg_word_length,
                "longest_word": max(words, key=len) if words else "",
                "shortest_word": min(words, key=len) if words else "",
                "top_words": top_words,
            },
            "sentence_analysis": {
                "average_length": avg_sentence_length,
                "longest_sentence": (
                    max(sentences, key=lambda x: len(x.split())) if sentences else ""
                ),
                "shortest_sentence": (
                    min(sentences, key=lambda x: len(x.split())) if sentences else ""
                ),
            },
            "readability": {
                "flesch_score": flesch_score,
                "reading_level": (
                    "Very Easy"
                    if flesch_score >= 90
                    else (
                        "Easy"
                        if flesch_score >= 80
                        else (
                            "Fairly Easy"
                            if flesch_score >= 70
                            else (
                                "Standard"
                                if flesch_score >= 60
                                else (
                                    "Fairly Difficult"
                                    if flesch_score >= 50
                                    else (
                                        "Difficult"
                                        if flesch_score >= 30
                                        else "Very Difficult"
                                    )
                                )
                            )
                        )
                    )
                ),
            },
            "content_extraction": {
                "urls": urls,
                "emails": emails,
                "phone_numbers": phone_numbers,
                "hashtags": hashtags,
                "mentions": mentions,
            },
            "language_patterns": patterns,
        }
        return result

    except Exception as e:
        return {"status": "error", "error": str(e)}



@metrics_wrapper
def fetch_url_content(
    url: str, timeout: int = 10, tool_context: ToolContext = None
) -> dict:
    """Fetch content from a URL safely.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        tool_context: Tool context for state management

    Returns:
        Dictionary with URL content and metadata
    """
    try:
        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return {
                "status": "error",
                "error": "URL must start with http:// or https://",
            }

        # Create request with user agent
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (ADK Agent Bot)"}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()

            # Try to decode as text
            try:
                text_content = content.decode("utf-8")
                content_type = "text"
            except UnicodeDecodeError:
                text_content = base64.b64encode(content).decode("ascii")
                content_type = "binary"

            headers = dict(response.headers)

            result = {
                "status": "success",
                "url": url,
                "content": text_content,
                "content_type": content_type,
                "size": len(content),
                "status_code": response.getcode(),
                "headers": headers,
                "encoding": response.headers.get_content_charset() or "unknown",
            }

            # Content analysis for text
            if content_type == "text":
                result["analysis"] = {
                    "lines": len(text_content.splitlines()),
                    "words": len(text_content.split()),
                    "contains_html": "<html" in text_content.lower(),
                    "contains_json": text_content.strip().startswith(("{", "[")),
                    "title": (
                        re.search(
                            r"<title[^>]*>([^<]+)</title>", text_content, re.IGNORECASE
                        ).group(1)
                        if re.search(
                            r"<title[^>]*>([^<]+)</title>", text_content, re.IGNORECASE
                        )
                        else None
                    ),
                }
            return result

    except urllib.error.URLError as e:
        return {"status": "error", "error": f"URL error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def encode_decode_data(
    data: str,
    operation: str,
    encoding: str = "base64",
    tool_context: ToolContext = None,
) -> dict:
    """Encode or decode data using various encoding schemes.

    Args:
        data: Data to encode/decode
        operation: 'encode' or 'decode'
        encoding: Encoding scheme (base64, url, html)
        tool_context: Tool context for state management

    Returns:
        Dictionary with encoded/decoded result
    """
    try:
        if operation not in ["encode", "decode"]:
            return {
                "status": "error",
                "error": "Operation must be 'encode' or 'decode'",
            }

        if encoding == "base64":
            if operation == "encode":
                result = base64.b64encode(data.encode("utf-8")).decode("ascii")
            else:
                result = base64.b64decode(data).decode("utf-8")

        elif encoding == "url":
            if operation == "encode":
                result = urllib.parse.quote(data)
            else:
                result = urllib.parse.unquote(data)

        elif encoding == "html":
            if operation == "encode":
                result = (
                    data.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&#x27;")
                )
            else:
                result = (
                    data.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&quot;", '"')
                    .replace("&#x27;", "'")
                )

        else:
            return {"status": "error", "error": f"Unsupported encoding: {encoding}"}

        return {
            "status": "success",
            "operation": operation,
            "encoding": encoding,
            "input": data,
            "result": result,
            "input_length": len(data),
            "result_length": len(result),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}



@metrics_wrapper
def advanced_calculator(expression: str, tool_context: ToolContext) -> dict:
    """Advanced calculator with mathematical functions.

    Args:
        expression: Mathematical expression to evaluate
        tool_context: Tool context for state management

    Returns:
        Dictionary with calculation results
    """
    try:
        # Safe mathematical namespace
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "divmod": divmod,
            "math": math,
            "pi": math.pi,
            "e": math.e,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "sqrt": math.sqrt,
            "factorial": math.factorial,
            "ceil": math.ceil,
            "floor": math.floor,
            "degrees": math.degrees,
            "radians": math.radians,
            "gcd": math.gcd,
        }

        # Pre-process expression for common mathematical notation
        expression = expression.replace("^", "**") 
        expression = re.sub(
            r"(\d+)([a-zA-Z])", r"\1*\2", expression
        ) 
        # Evaluate expression
        result = eval(expression, safe_dict)

        # Additional analysis
        is_integer = isinstance(result, int) or (
            isinstance(result, float) and result.is_integer()
        )

        analysis = {
            "result": result,
            "type": type(result).__name__,
            "is_integer": is_integer,
            "is_positive": result > 0 if isinstance(result, (int, float)) else None,
            "scientific_notation": (
                f"{result:.3e}" if isinstance(result, (int, float)) else None
            ),
        }

        # Mathematical properties
        if isinstance(result, (int, float)) and result > 0:
            analysis.update(
                {
                    "square_root": math.sqrt(result),
                    "natural_log": math.log(result),
                    "base_10_log": math.log10(result),
                    "sine": math.sin(result),
                    "cosine": math.cos(result),
                }
            )

        # Store in calculation history
        if "calculations" not in tool_context.state:
            tool_context.state["calculations"] = []

        tool_context.state["calculations"].append(
            {
                "expression": expression,
                "result": result,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

        return {"status": "success", "expression": expression, "analysis": analysis}

    except Exception as e:
        return {"status": "error", "expression": expression, "error": str(e)}


@metrics_wrapper
def number_theory_analysis(number: int, tool_context: ToolContext) -> dict:
    """Analyze mathematical properties of a number.

    Args:
        number: Integer to analyze
        tool_context: Tool context for state management

    Returns:
        Dictionary with number theory analysis
    """
    try:
        n = int(number)
        if n <= 0:
            return {"status": "error", "error": "Number must be positive"}

        # Prime factorization
        def prime_factors(n):
            factors = []
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors.append(d)
                    n //= d
                d += 1
            if n > 1:
                factors.append(n)
            return factors

        # Check if prime
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True

        # Get divisors
        def get_divisors(n):
            divisors = []
            for i in range(1, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    divisors.append(i)
                    if i != n // i:
                        divisors.append(n // i)
            return sorted(divisors)

        # Perfect number check
        def is_perfect(n):
            return sum(d for d in get_divisors(n) if d < n) == n

        factors = prime_factors(n)
        divisors = get_divisors(n)

        properties = {
            "number": n,
            "is_prime": is_prime(n),
            "is_perfect": is_perfect(n),
            "is_even": n % 2 == 0,
            "is_square": int(math.sqrt(n)) ** 2 == n,
            "is_cube": round(n ** (1 / 3)) ** 3 == n,
            "prime_factors": factors,
            "unique_prime_factors": list(set(factors)),
            "divisors": divisors,
            "divisor_count": len(divisors),
            "sum_of_divisors": sum(divisors),
            "digital_root": n % 9 if n % 9 != 0 else 9,
            "digit_sum": sum(int(digit) for digit in str(n)),
            "binary": bin(n),
            "octal": oct(n),
            "hexadecimal": hex(n),
        }

        # Special number classifications
        if n > 1:
            properties["is_fibonacci"] = n in [
                1,
                1,
                2,
                3,
                5,
                8,
                13,
                21,
                34,
                55,
                89,
                144,
                233,
                377,
                610,
                987,
                1597,
            ]

        return {"status": "success", "analysis": properties}

    except Exception as e:
        return {"status": "error", "error": str(e)}



@metrics_wrapper
def generate_hash(
    data: str, algorithm: str = "sha256", tool_context: ToolContext = None
) -> dict:
    """Generate hash of data using various algorithms.

    Args:
        data: Data to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        tool_context: Tool context for state management

    Returns:
        Dictionary with hash results
    """
    try:
        algorithms = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }

        if algorithm not in algorithms:
            return {"status": "error", "error": f"Unsupported algorithm: {algorithm}"}

        hash_obj = algorithms[algorithm]()
        hash_obj.update(data.encode("utf-8"))
        hash_value = hash_obj.hexdigest()

        return {
            "status": "success",
            "data": data,
            "algorithm": algorithm,
            "hash": hash_value,
            "length": len(hash_value),
            "input_length": len(data),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def generate_password(
    length: int = 12, include_special: bool = True, tool_context: ToolContext = None
) -> dict:
    """Generate a secure random password.

    Args:
        length: Password length
        include_special: Whether to include special characters
        tool_context: Tool context for state management

    Returns:
        Dictionary with password and strength analysis
    """
    try:
        if length < 4:
            return {"status": "error", "error": "Password length must be at least 4"}

        if length > 100:
            return {"status": "error", "error": "Password length must be 100 or less"}

        # Character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*()-_=+[]{}|;:,.<>?"

        # Ensure at least one character from each required set
        password_chars = [
            random.choice(lowercase),
            random.choice(uppercase),
            random.choice(digits),
        ]

        if include_special:
            password_chars.append(random.choice(special))
            all_chars = lowercase + uppercase + digits + special
        else:
            all_chars = lowercase + uppercase + digits

        # Fill remaining length
        for _ in range(length - len(password_chars)):
            password_chars.append(random.choice(all_chars))

        # Shuffle the password
        random.shuffle(password_chars)
        password = "".join(password_chars)

        # Strength analysis
        strength_score = 0
        criteria = {
            "has_lowercase": any(c.islower() for c in password),
            "has_uppercase": any(c.isupper() for c in password),
            "has_digits": any(c.isdigit() for c in password),
            "has_special": any(c in special for c in password),
            "length_8_plus": len(password) >= 8,
            "length_12_plus": len(password) >= 12,
        }

        strength_score = sum(criteria.values())

        if strength_score >= 5:
            strength = "Very Strong"
        elif strength_score >= 4:
            strength = "Strong"
        elif strength_score >= 3:
            strength = "Medium"
        else:
            strength = "Weak"

        return {
            "status": "success",
            "password": password,
            "length": len(password),
            "strength": strength,
            "criteria": criteria,
            "entropy_bits": length * math.log2(len(all_chars)),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def validate_input(
    data: str, validation_type: str, tool_context: ToolContext = None
) -> dict:
    """Validate input data against various patterns.

    Args:
        data: Data to validate
        validation_type: Type of validation (email, url, ip, phone, credit_card)
        tool_context: Tool context for state management

    Returns:
        Dictionary with validation results
    """
    try:
        patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "url": r"^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$",
            "ipv4": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
            "phone": r"^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$",
            "credit_card": r"^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})$",
        }

        if validation_type not in patterns:
            return {
                "status": "error",
                "error": f"Unsupported validation type: {validation_type}",
            }

        pattern = patterns[validation_type]
        is_valid = bool(re.match(pattern, data.strip()))

        # Additional context-specific validation
        additional_info = {}

        if validation_type == "email":
            parts = data.split("@")
            if len(parts) == 2:
                additional_info = {
                    "local_part": parts[0],
                    "domain": parts[1],
                    "tld": parts[1].split(".")[-1] if "." in parts[1] else None,
                }
        elif validation_type == "url":
            try:
                from urllib.parse import urlparse

                parsed = urlparse(data)
                additional_info = {
                    "scheme": parsed.scheme,
                    "domain": parsed.netloc,
                    "path": parsed.path,
                    "has_query": bool(parsed.query),
                    "has_fragment": bool(parsed.fragment),
                }
            except:
                pass
        elif validation_type == "credit_card":
            # Luhn algorithm check
            def luhn_check(card_num):
                digits = [int(d) for d in card_num if d.isdigit()]
                for i in range(len(digits) - 2, -1, -2):
                    digits[i] *= 2
                    if digits[i] > 9:
                        digits[i] -= 9
                return sum(digits) % 10 == 0

            card_type = "Unknown"
            if data.startswith("4"):
                card_type = "Visa"
            elif data.startswith(("51", "52", "53", "54", "55")):
                card_type = "MasterCard"
            elif data.startswith(("34", "37")):
                card_type = "American Express"

            additional_info = {
                "card_type": card_type,
                "passes_luhn": luhn_check(data),
                "length": len([d for d in data if d.isdigit()]),
            }

        return {
            "status": "success",
            "data": data,
            "validation_type": validation_type,
            "is_valid": is_valid,
            "pattern_used": pattern,
            "additional_info": additional_info,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def datetime_operations(
    operation: str,
    datetime_str: Optional[str] = None,
    format_str: str = "%Y-%m-%d %H:%M:%S",
    tool_context: ToolContext = None,
) -> dict:
    """Perform various datetime operations.

    Args:
        operation: Operation to perform (now, parse, format, add_days, timestamp)
        datetime_str: Datetime string to parse (if applicable)
        format_str: Format string for parsing/formatting
        tool_context: Tool context for state management

    Returns:
        Dictionary with datetime operation results
    """
    try:
        now = datetime.datetime.now()

        if operation == "now":
            result = {
                "current_datetime": now.isoformat(),
                "formatted": now.strftime(format_str),
                "timestamp": now.timestamp(),
                "weekday": now.strftime("%A"),
                "month": now.strftime("%B"),
                "year": now.year,
                "day_of_year": now.timetuple().tm_yday,
                "week_number": now.isocalendar()[1],
                "timezone": str(now.astimezone().tzinfo),
            }
        elif operation == "parse":
            if not datetime_str:
                return {
                    "status": "error",
                    "error": "datetime_str required for parse operation",
                }
            parsed = datetime.datetime.strptime(datetime_str, format_str)
            result = {
                "input": datetime_str,
                "format": format_str,
                "parsed": parsed.isoformat(),
                "timestamp": parsed.timestamp(),
                "weekday": parsed.strftime("%A"),
                "month": parsed.strftime("%B"),
                "is_weekend": parsed.weekday() >= 5,
            }
        elif operation == "timestamp":
            if datetime_str:
                try:
                    # Try to parse as timestamp
                    ts = float(datetime_str)
                    dt = datetime.datetime.fromtimestamp(ts)
                except ValueError:
                    # Try to parse as datetime string
                    dt = datetime.datetime.fromisoformat(
                        datetime_str.replace("Z", "+00:00")
                    )
            else:
                dt = now

            result = {
                "datetime": dt.isoformat(),
                "timestamp": dt.timestamp(),
                "formatted": dt.strftime(format_str),
                "utc": dt.utctimetuple(),
                "local": dt.timetuple(),
            }
        else:
            return {"status": "error", "error": f"Unsupported operation: {operation}"}

        return {"status": "success", "operation": operation, "result": result}

    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def format_data(
    data: str, output_format: str, tool_context: ToolContext = None
) -> dict:
    """Format data into various output formats.

    Args:
        data: Data to format as a string. For structured data, pass JSON (object or array)
        output_format: Output format (json, csv, table, xml)
        tool_context: Tool context for state management

    Returns:
        Dictionary with formatted data
    """
    try:
        # Attempt to parse JSON from the input string when useful for table/csv/json formats
        parsed_from_json = None
        try:
            parsed_from_json = json.loads(data)
        except json.JSONDecodeError:
            parsed_from_json = None

        if output_format == "json":
            if parsed_from_json is not None:
                formatted = json.dumps(parsed_from_json, indent=2, ensure_ascii=False)
            else:
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
        elif output_format == "table":
            rows_source = (
                parsed_from_json if isinstance(parsed_from_json, list) else None
            )
            if rows_source and rows_source and isinstance(rows_source[0], dict):
                # List of dictionaries -> table
                headers = list(rows_source[0].keys())
                rows = []

                # Calculate column widths
                col_widths = {h: len(h) for h in headers}
                for row in rows_source:
                    for header in headers:
                        col_widths[header] = max(
                            col_widths[header], len(str(row.get(header, "")))
                        )

                # Build table
                header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
                separator = "-+-".join("-" * col_widths[h] for h in headers)

                rows.append(header_line)
                rows.append(separator)

                for row in rows_source:
                    row_line = " | ".join(
                        str(row.get(h, "")).ljust(col_widths[h]) for h in headers
                    )
                    rows.append(row_line)

                formatted = "\n".join(rows)
            else:
                formatted = str(data)
        elif output_format == "csv":
            rows_source = (
                parsed_from_json if isinstance(parsed_from_json, list) else None
            )
            if rows_source and rows_source and isinstance(rows_source[0], dict):
                import io

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=rows_source[0].keys())
                writer.writeheader()
                writer.writerows(rows_source)
                formatted = output.getvalue()
            else:
                formatted = str(data)
        else:
            return {"status": "error", "error": f"Unsupported format: {output_format}"}

        return {
            "status": "success",
            "input_type": "str",
            "output_format": output_format,
            "formatted": formatted,
            "size": len(formatted),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}



def get_session_analytics(tool_context: ToolContext = None) -> dict:
    """Get comprehensive analytics about the current session.

    Args:
        tool_context: Tool context for accessing state

    Returns:
        Dictionary with session analytics
    """
    try:
        # Safely obtain state dictionary
        state = getattr(tool_context, "state", {}) if tool_context is not None else {}
        if state is None:
            state = {}

        # Count operations by type
        operation_counts = {}
        total_operations = 0

        for key in list(state.keys()):
            try:
                if isinstance(state[key], list):
                    count = len(state[key])
                    operation_counts[key] = count
                    total_operations += count
            except Exception:
                continue

        # File operations analysis
        file_ops = state.get("file_operations", []) if isinstance(state, dict) else []
        file_stats = {
            "total_files_accessed": len(set(op.get("file", "") for op in file_ops)),
            "read_operations": sum(
                1 for op in file_ops if op.get("operation", "").startswith("read")
            ),
            "write_operations": sum(
                1 for op in file_ops if op.get("operation", "").startswith("write")
            ),
        }
        # Data analysis operations
        data_ops = state.get("data_analysis", []) if isinstance(state, dict) else []
        data_stats = {
            "csv_analyses": sum(
                1 for op in data_ops if op.get("operation") == "csv_analysis"
            ),
            "files_analyzed": len(set(op.get("file", "") for op in data_ops)),
        }
        # Calculation history
        calculations = state.get("calculations", []) if isinstance(state, dict) else []
        calc_stats = {
            "total_calculations": len(calculations),
            "recent_calculations": calculations[-5:] if calculations else [],
        }

        # Session timing (if available)
        session_start = None
        if isinstance(state, dict):
            raw_session_start = state.get("session_start")
            try:
                if isinstance(raw_session_start, (datetime.datetime, datetime.date)):
                    session_start = raw_session_start.isoformat()
                elif raw_session_start is not None:
                    session_start = str(raw_session_start)
            except Exception:
                session_start = None
        session_info = {
            "total_operations": total_operations,
            "operation_breakdown": operation_counts,
            "session_start": session_start,
            "current_time": datetime.datetime.now().isoformat(),
        }

        return {
            "status": "success",
            "session_info": session_info,
            "file_operations": file_stats,
            "data_analysis": data_stats,
            "calculations": calc_stats,
            "state_keys": (
                [str(k) for k in list(state.keys())] if isinstance(state, dict) else []
            ),
            "memory_usage": (
                len(str(state)) if state is not None else 0
            ),  # Rough estimate
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@metrics_wrapper
def clear_session_data(
    data_type: str = "all", tool_context: ToolContext = None
) -> dict:
    """Clear specific types of session data.

    Args:
        data_type: Type of data to clear (all, file_operations, calculations, data_analysis)
        tool_context: Tool context for state management

    Returns:
        Dictionary with clearing operation results
    """
    try:
        if data_type == "all":
            cleared_keys = list(tool_context.state.keys())
            tool_context.state.clear()
        elif data_type in tool_context.state:
            cleared_keys = [data_type]
            del tool_context.state[data_type]
        else:
            return {"status": "error", "error": f"Data type '{data_type}' not found"}

        return {
            "status": "success",
            "cleared_data_types": cleared_keys,
            "remaining_keys": list(tool_context.state.keys()),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_performance_metrics(tool_context: ToolContext = None) -> dict:
    """Get comprehensive performance metrics for the session.

    Args:
        tool_context: Tool context for state management

    Returns:
        Dictionary with detailed performance metrics
    """
    try:
        metrics_summary = _metrics_collector.get_metrics_summary()

        # Add context about the agent and current state
        agent_context = {
            "agent_name": "advanced_multi_tool_agent",
            "metrics_collection_enabled": True,
            "baseline_measurement": "Pre-Context-Reference-Store",
            "measurement_timestamp": datetime.datetime.now().isoformat(),
        }

        return {
            "status": "success",
            "agent_context": agent_context,
            "performance_metrics": metrics_summary,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def export_metrics_report(
    format_type: str = "json", tool_context: ToolContext = None
) -> dict:
    """Export detailed performance metrics report.

    Args:
        format_type: Export format (json, csv, summary)
        tool_context: Tool context for state management

    Returns:
        Dictionary with exported metrics report
    """
    try:
        metrics_summary = _metrics_collector.get_metrics_summary()

        if format_type == "json":
            report = json.dumps(metrics_summary, indent=2, default=str)

        elif format_type == "csv":
            # Convert tool performance to CSV format
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Header
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
                ]
            )

            # Data rows
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
                    ]
                )

            report = output.getvalue()

        elif format_type == "summary":
            overview = metrics_summary.get("session_overview", {})
            memory = metrics_summary.get("memory_metrics", {})
            efficiency = metrics_summary.get("efficiency_metrics", {})

            report = f"""
PERFORMANCE METRICS SUMMARY
============================
Session Duration: {overview.get('session_duration_seconds', 0):.2f}s
Total Tools Executed: {overview.get('total_tools_executed', 0)}
Average Tool Time: {overview.get('average_tool_time', 0):.4f}s

Memory Usage:
- Current: {memory.get('current_usage_mb', 0):.2f} MB
- Peak: {memory.get('peak_usage_mb', 0):.2f} MB
- Data Throughput Ratio: {memory.get('data_throughput_ratio', 0):.2f}

Efficiency:
- Tools/Second: {efficiency.get('tools_per_second', 0):.2f}
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




# Create the advanced multi-tool agent
root_agent = Agent(
    model="gemini-2.0-flash",
    name="advanced_multi_tool_agent",
    description=(
        "An advanced AI agent with comprehensive tool capabilities across multiple domains: "
        "file operations, data analysis, text processing, web utilities, mathematical computing, "
        "security tools, datetime operations, and session management. This agent can handle "
        "complex workflows and provides detailed analysis and reporting capabilities."
    ),
    instruction="""
    You are an advanced AI assistant with extensive tool capabilities across multiple domains. 
    You can help with a wide variety of tasks including:
    
    **FILE & SYSTEM OPERATIONS:**
    - Read and write files safely
    - List and analyze directory contents
    - Manage file operations with detailed metadata
    
    **DATA ANALYSIS & PROCESSING:**
    - Analyze CSV data with comprehensive statistics
    - Generate statistical reports for numerical data
    - Process and transform data in various formats
    
    **TEXT PROCESSING & NLP:**
    - Perform advanced text analysis (readability, linguistics, content extraction)
    - Extract URLs, emails, phone numbers, hashtags, mentions
    - Analyze word frequency, patterns, and language characteristics
    
    **WEB & API UTILITIES:**
    - Fetch content from URLs safely
    - Encode/decode data (base64, URL, HTML)
    - Process web content and APIs
    
    **MATHEMATICAL & SCIENTIFIC COMPUTING:**
    - Advanced calculator with mathematical functions
    - Number theory analysis (prime factors, divisors, special properties)
    - Statistical computations and analysis
    
    **SECURITY & VALIDATION:**
    - Generate secure passwords with strength analysis
    - Create hashes using various algorithms (MD5, SHA1, SHA256, SHA512)
    - Validate input data (emails, URLs, IP addresses, phone numbers, credit cards)
    
    **DATETIME & FORMATTING:**
    - Parse, format, and manipulate datetime values
    - Convert between timestamps and human-readable formats
    - Format data into JSON, CSV, tables, and other formats
    
    **SESSION MANAGEMENT:**
    - Track all operations and provide detailed analytics
    - Manage session state and data
    - Clear and organize session information
    
    **PERFORMANCE MONITORING:**
    - Real-time metrics collection for all tool executions
    - Memory usage tracking and analysis
    - Performance warnings for slow operations
    - Comprehensive performance reports in multiple formats
    
    **GUIDELINES:**
    - Always use the most appropriate tool for each task
    - Provide comprehensive analysis and insights
    - Handle errors gracefully and suggest alternatives
    - Maintain detailed session history for reference
    - When working with files, always check permissions and safety
    - For complex tasks, break them down into logical steps
    - Provide context and explanations for technical operations
    
    **SAFETY & SECURITY:**
    - Never execute unsafe operations or arbitrary code
    - Validate all inputs before processing
    - Respect file system boundaries and permissions
    - Handle sensitive data appropriately
    - Use secure methods for all operations
    
    You can handle complex multi-step workflows by combining multiple tools and provide 
    detailed reports and analysis for all operations performed.
    """,
    tools=[
        # File System & Data Operations
        read_file_content,
        write_file_content,
        list_directory,
        analyze_csv_data,
        generate_statistical_report,
        # Text Processing & NLP
        advanced_text_analysis,
        # Web & API Utilities
        fetch_url_content,
        encode_decode_data,
        # Mathematical & Scientific Computing
        advanced_calculator,
        number_theory_analysis,
        # Security & Validation
        generate_hash,
        generate_password,
        validate_input,
        # DateTime & Formatting
        datetime_operations,
        format_data,
        # Session Management
        get_session_analytics,
        clear_session_data,
        # Performance Monitoring
        get_performance_metrics,
        export_metrics_report,
    ],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)
