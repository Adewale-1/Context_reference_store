"""
Basic ADK Agent Example

This example demonstrates how to create a simple ADK agent using the basic Agent class.
This agent does NOT use the context-reference-store library - it's a pure ADK example.

The agent provides:
- Text analysis capabilities
- Basic calculation functions
- Information storage in agent state
"""

import os
import re
import time
import psutil
import threading
import datetime
import json
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# METRICS & PERFORMANCE MONITORING
# =============================================================================


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


# =============================================================================
# TOOL FUNCTIONS
# =============================================================================


@metrics_wrapper
def analyze_text(text: str, tool_context: ToolContext) -> dict:
    """Analyze text and provide statistics.

    Args:
        text: The text to analyze
        tool_context: Tool context for state management

    Returns:
        A dictionary with text analysis results
    """
    # Basic text analysis
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = len(re.split(r"[.!?]+", text))

    # Extract and count different types of content
    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        text,
    )
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)

    analysis_result = {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "urls_found": len(urls),
        "emails_found": len(emails),
        "avg_word_length": round(
            sum(len(word) for word in text.split()) / max(word_count, 1), 2
        ),
        "urls": urls[:5] if urls else [],  # Show first 5 URLs found
        "emails": emails[:5] if emails else [],  # Show first 5 emails found
    }

    # Store in agent state for future reference
    if "analysis_history" not in tool_context.state:
        tool_context.state["analysis_history"] = []

    tool_context.state["analysis_history"].append(
        {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "analysis": analysis_result,
        }
    )

    return analysis_result


@metrics_wrapper
def calculate_advanced(expression: str, tool_context: ToolContext) -> dict:
    """Perform advanced calculations safely.

    Args:
        expression: Mathematical expression to evaluate
        tool_context: Tool context for state management

    Returns:
        A dictionary with calculation results
    """
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

        # Store calculation history
        if "calculation_history" not in tool_context.state:
            tool_context.state["calculation_history"] = []

        tool_context.state["calculation_history"].append(
            {"expression": expression, "result": result}
        )

        return {"status": "success", "expression": expression, "result": result}

    except Exception as e:
        return {"status": "error", "expression": expression, "error": str(e)}


@metrics_wrapper
def get_session_summary(tool_context: ToolContext) -> dict:
    """Get a summary of the current session.

    Args:
        tool_context: Tool context for accessing state

    Returns:
        A dictionary with session statistics
    """
    analysis_count = len(tool_context.state.get("analysis_history", []))
    calculation_count = len(tool_context.state.get("calculation_history", []))

    recent_analyses = tool_context.state.get("analysis_history", [])[
        -3:
    ]  # Last 3 analyses
    recent_calculations = tool_context.state.get("calculation_history", [])[
        -3:
    ]  # Last 3 calculations

    return {
        "session_stats": {
            "total_text_analyses": analysis_count,
            "total_calculations": calculation_count,
            "total_operations": analysis_count + calculation_count,
        },
        "recent_analyses": recent_analyses,
        "recent_calculations": recent_calculations,
    }


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
            "agent_name": "basic_analysis_agent",
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
BASIC AGENT PERFORMANCE METRICS
===============================
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


# Create the ADK agent
root_agent = Agent(
    model="gemini-2.0-flash",
    name="basic_analysis_agent",
    description=(
        "A basic ADK agent that can analyze text, perform calculations, "
        "and maintain session state. This demonstrates core ADK functionality "
        "without any external context management libraries."
    ),
    instruction="""
    You are a helpful analysis assistant with the following capabilities:
    
    1. **Text Analysis**: You can analyze any text to provide statistics like word count, 
       character count, sentence count, and detect URLs and email addresses.
    
    2. **Advanced Calculations**: You can perform mathematical calculations including 
       complex expressions with parentheses.
    
    3. **Session Management**: You maintain a history of all analyses and calculations 
       performed in this session and can provide summaries.
    
    4. **Performance Monitoring**: You can track and report detailed performance metrics
       including execution times, memory usage, and efficiency statistics.
    
    **Guidelines:**
    - Always use the appropriate tool for each task
    - For text analysis, call the analyze_text tool with the text to analyze
    - For calculations, call the calculate_advanced tool with the mathematical expression
    - To see session history, call the get_session_summary tool
    - To view performance metrics, call the get_performance_metrics tool
    - To export performance reports, call the export_metrics_report tool
    - Be helpful and provide insights based on the tool results
    - If a user asks for multiple operations, you can call multiple tools in parallel
    
    **Examples of what you can do:**
    - "Analyze this paragraph: [text]" → Use analyze_text tool
    - "Calculate 25 * 4 + (100 / 5)" → Use calculate_advanced tool  
    - "What have we done in this session?" → Use get_session_summary tool
    - "Show performance metrics" → Use get_performance_metrics tool
    - "Export summary report" → Use export_metrics_report tool with format "summary"
    - "Analyze this text and calculate 50 + 30" → Use both tools in parallel
    """,
    tools=[
        analyze_text,
        calculate_advanced,
        get_session_summary,
        get_performance_metrics,
        export_metrics_report,
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


# Demo function to test the agent
async def demo_agent():
    """Demonstrate the basic ADK agent functionality."""
    print("Basic ADK Agent Demo")
    print("=" * 50)

    try:
        # Test text analysis
        print("\nTesting Text Analysis...")
        response1 = await root_agent.generate_content(
            "Analyze this text: 'Hello world! This is a test message with some content. "
            "Contact us at test@example.com or visit https://example.com for more info.'"
        )
        print(f"Response: {response1.text}")

        # Test calculation
        print("\nTesting Calculation...")
        response2 = await root_agent.generate_content(
            "Calculate the result of: (25 * 4) + (100 / 5) - 10"
        )
        print(f"Response: {response2.text}")

        # Test session summary
        print("\nTesting Session Summary...")
        response3 = await root_agent.generate_content(
            "Can you give me a summary of what we've done in this session?"
        )
        print(f"Response: {response3.text}")

        # Test parallel operations
        print("\nTesting Parallel Operations...")
        response4 = await root_agent.generate_content(
            "Please analyze this text: 'The quick brown fox jumps over the lazy dog' "
            "and also calculate 15 * 8 + 42"
        )
        print(f"Response: {response4.text}")

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

    # Run the demo
    asyncio.run(demo_agent())
