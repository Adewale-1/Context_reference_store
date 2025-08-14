"""
Terminal User Interface (TUI) Analytics Dashboard

A beautiful, real-time terminal dashboard for monitoring Context Reference Store
performance, compression analytics, and system health.

Features:
- Real-time performance metrics display
- Interactive compression analytics
- Memory usage monitoring
- Cache performance visualization
- Color-coded alerts and recommendations
- Keyboard navigation and controls
"""

import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import sys
import os
import signal

try:
    import curses
    import curses.textpad

    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

try:
    from ..core.context_reference_store import ContextReferenceStore

    CONTEXT_STORE_AVAILABLE = True
except ImportError:
    CONTEXT_STORE_AVAILABLE = False


@dataclass
class DashboardMetrics:
    """Container for dashboard metrics."""

    timestamp: float
    total_contexts: int
    compressed_contexts: int
    cache_hit_rate: float
    memory_usage_percent: float
    compression_ratio: float
    space_savings_percent: float
    avg_compression_time_ms: float
    avg_decompression_time_ms: float
    total_space_saved_bytes: int
    evictions: int
    efficiency_multiplier: float


class TUIColors:
    """Color schemes for the TUI dashboard."""

    @staticmethod
    def init_colors():
        """Initialize color pairs for the dashboard."""
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1) 
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Warning
        curses.init_pair(3, curses.COLOR_RED, -1)  # Error/Critical
        curses.init_pair(4, curses.COLOR_BLUE, -1)  # Info/Header
        curses.init_pair(5, curses.COLOR_CYAN, -1)  # Accent
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)
        curses.init_pair(7, curses.COLOR_WHITE, -1)

    GREEN = 1
    YELLOW = 2
    RED = 3
    BLUE = 4
    CYAN = 5
    MAGENTA = 6
    WHITE = 7


class ContextStoreTUIDashboard:
    """
    Terminal User Interface Dashboard for Context Reference Store monitoring.

    Provides real-time visualization of performance metrics, compression analytics,
    and system health in a beautiful terminal interface.
    """

    def __init__(
        self, context_store: ContextReferenceStore, update_interval: float = 1.0
    ):
        """
        Initialize the TUI dashboard.

        Args:
            context_store: The ContextReferenceStore instance to monitor
            update_interval: Update frequency in seconds
        """
        if not CURSES_AVAILABLE:
            raise ImportError(
                "curses library not available. TUI dashboard requires curses."
            )

        self.context_store = context_store
        self.update_interval = update_interval
        self.running = False
        self.screen = None

        # Metrics history for charts
        self.metrics_history: deque = deque(maxlen=100)
        self.current_metrics: Optional[DashboardMetrics] = None

        # Dashboard state
        self.current_tab = 0
        self.tabs = ["Overview", "Compression", "Cache", "Memory", "Alerts"]
        self.scroll_position = 0

        # Update thread
        self.update_thread = None
        self.update_lock = threading.Lock()

        # Alerts system
        self.alerts: List[Dict[str, Any]] = []
        self.max_alerts = 10

    def start(self):
        """Start the TUI dashboard."""
        if not CURSES_AVAILABLE:
            print("❌ TUI Dashboard requires curses library")
            return

        try:
            # Initialize curses
            self.screen = curses.initscr()
            curses.cbreak()
            curses.noecho()
            self.screen.keypad(True)
            self.screen.nodelay(True)

            # Initialize colors
            TUIColors.init_colors()

            # Start monitoring
            self.running = True
            self.update_thread = threading.Thread(target=self._update_metrics_loop)
            self.update_thread.daemon = True
            self.update_thread.start()

            # Main display loop
            self._main_loop()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Stop the dashboard and cleanup."""
        self.running = False

        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1)

        if self.screen:
            curses.nocbreak()
            self.screen.keypad(False)
            curses.echo()
            curses.endwin()

    def _update_metrics_loop(self):
        """Background thread for updating metrics."""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                self._add_alert("error", f"Metrics collection failed: {str(e)}")
                time.sleep(self.update_interval * 2)  # Slow down on errors

    def _collect_metrics(self):
        """Collect current metrics from the context store."""
        try:
            # Get basic cache stats
            cache_stats = self.context_store.get_cache_stats()

            # Get compression analytics if available
            compression_analytics = None
            if hasattr(self.context_store, "get_compression_analytics"):
                compression_analytics = self.context_store.get_compression_analytics()

            # Calculate metrics
            timestamp = time.time()
            total_contexts = cache_stats.get("total_contexts", 0)
            hit_rate = cache_stats.get("hit_rate", 0.0)
            memory_usage = cache_stats.get("memory_usage_percent", 0.0)
            evictions = cache_stats.get("total_evictions", 0)

            # Compression metrics
            compressed_contexts = 0
            compression_ratio = 1.0
            space_savings_percent = 0.0
            avg_compression_time = 0.0
            avg_decompression_time = 0.0
            total_space_saved = 0
            efficiency_multiplier = 1.0

            if compression_analytics and compression_analytics.get(
                "compression_enabled", False
            ):

                context_stats = compression_analytics.get("context_store_stats", {})
                manager_stats = compression_analytics.get(
                    "compression_manager_analytics", {}
                ).get("summary", {})
                performance = compression_analytics.get("performance_impact", {})

                compressed_contexts = context_stats.get("compressed_contexts", 0)
                space_savings_percent = context_stats.get(
                    "space_savings_percentage", 0.0
                )
                total_space_saved = context_stats.get("total_space_saved_bytes", 0)
                compression_ratio = manager_stats.get("overall_compression_ratio", 1.0)
                avg_compression_time = manager_stats.get("avg_compression_time_ms", 0.0)
                avg_decompression_time = manager_stats.get(
                    "avg_decompression_time_ms", 0.0
                )
                efficiency_multiplier = performance.get(
                    "storage_efficiency_multiplier", 1.0
                )

            # Create metrics object
            metrics = DashboardMetrics(
                timestamp=timestamp,
                total_contexts=total_contexts,
                compressed_contexts=compressed_contexts,
                cache_hit_rate=hit_rate,
                memory_usage_percent=memory_usage,
                compression_ratio=compression_ratio,
                space_savings_percent=space_savings_percent,
                avg_compression_time_ms=avg_compression_time,
                avg_decompression_time_ms=avg_decompression_time,
                total_space_saved_bytes=total_space_saved,
                evictions=evictions,
                efficiency_multiplier=efficiency_multiplier,
            )

            with self.update_lock:
                self.current_metrics = metrics
                self.metrics_history.append(metrics)

                # Check for alerts
                self._check_alerts(metrics)

        except Exception as e:
            self._add_alert("error", f"Failed to collect metrics: {str(e)}")

    def _check_alerts(self, metrics: DashboardMetrics):
        """Check metrics for alert conditions."""
        # Memory usage alerts
        if metrics.memory_usage_percent > 90:
            self._add_alert(
                "critical", f"High memory usage: {metrics.memory_usage_percent:.1f}%"
            )
        elif metrics.memory_usage_percent > 80:
            self._add_alert(
                "warning", f"Elevated memory usage: {metrics.memory_usage_percent:.1f}%"
            )

        # Cache performance alerts
        if metrics.cache_hit_rate < 0.5:
            self._add_alert(
                "warning", f"Low cache hit rate: {metrics.cache_hit_rate:.1%}"
            )

        # Compression performance alerts
        if metrics.space_savings_percent > 80:
            self._add_alert(
                "info",
                f"Excellent compression: {metrics.space_savings_percent:.1f}% savings",
            )
        elif metrics.space_savings_percent < 20 and metrics.compressed_contexts > 0:
            self._add_alert(
                "warning",
                f"Poor compression: {metrics.space_savings_percent:.1f}% savings",
            )

    def _add_alert(self, level: str, message: str):
        """Add an alert to the alerts list."""
        alert = {"timestamp": time.time(), "level": level, "message": message}

        # Avoid duplicate alerts
        for existing_alert in self.alerts:
            if existing_alert["message"] == message:
                return

        self.alerts.insert(0, alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop()

    def _main_loop(self):
        """Main display and input handling loop."""
        while self.running:
            try:
                # Clear screen and draw dashboard
                self.screen.clear()
                self._draw_dashboard()
                self.screen.refresh()

                # Handle input
                key = self.screen.getch()
                if key != -1:
                    self._handle_input(key)

                time.sleep(0.1)  # Small delay to prevent excessive CPU usage

            except curses.error:
                # Handle terminal resize or other curses errors
                pass
            except Exception as e:
                self._add_alert("error", f"Display error: {str(e)}")
                time.sleep(1)

    def _handle_input(self, key: int):
        """Handle keyboard input."""
        if key == ord("q") or key == ord("Q"):
            self.running = False
        elif key == curses.KEY_LEFT and self.current_tab > 0:
            self.current_tab -= 1
            self.scroll_position = 0
        elif key == curses.KEY_RIGHT and self.current_tab < len(self.tabs) - 1:
            self.current_tab += 1
            self.scroll_position = 0
        elif key == curses.KEY_UP and self.scroll_position > 0:
            self.scroll_position -= 1
        elif key == curses.KEY_DOWN:
            self.scroll_position += 1
        elif key == ord("r") or key == ord("R"):
            # Force refresh
            self._collect_metrics()
        elif key == ord("c") or key == ord("C"):
            # Clear alerts
            self.alerts.clear()

    def _draw_dashboard(self):
        """Draw the complete dashboard."""
        if not self.current_metrics:
            self._draw_loading_screen()
            return

        height, width = self.screen.getmaxyx()

        # Draw header
        self._draw_header(width)

        # Draw tab bar
        self._draw_tab_bar(width, 2)

        # Draw content based on current tab
        content_start_y = 4
        content_height = height - content_start_y - 2

        if self.tabs[self.current_tab] == "Overview":
            self._draw_overview_tab(content_start_y, content_height, width)
        elif self.tabs[self.current_tab] == "Compression":
            self._draw_compression_tab(content_start_y, content_height, width)
        elif self.tabs[self.current_tab] == "Cache":
            self._draw_cache_tab(content_start_y, content_height, width)
        elif self.tabs[self.current_tab] == "Memory":
            self._draw_memory_tab(content_start_y, content_height, width)
        elif self.tabs[self.current_tab] == "Alerts":
            self._draw_alerts_tab(content_start_y, content_height, width)

        # Draw footer
        self._draw_footer(height - 1, width)

    def _draw_header(self, width: int):
        """Draw the dashboard header."""
        title = "🚀 CONTEXT REFERENCE STORE - REAL-TIME ANALYTICS DASHBOARD"
        self._draw_centered_text(0, width, title, TUIColors.BLUE, curses.A_BOLD)

        # Current time and status
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        status = f"⚡ LIVE | {current_time}"
        self.screen.addstr(
            1, width - len(status) - 1, status, curses.color_pair(TUIColors.CYAN)
        )

    def _draw_tab_bar(self, width: int, y: int):
        """Draw the navigation tab bar."""
        tab_width = width // len(self.tabs)

        for i, tab in enumerate(self.tabs):
            x_start = i * tab_width

            # Highlight current tab
            if i == self.current_tab:
                attr = curses.color_pair(TUIColors.WHITE) | curses.A_REVERSE
            else:
                attr = curses.color_pair(TUIColors.WHITE)

            # Draw tab
            tab_text = f" {tab} "
            if x_start + len(tab_text) < width:
                self.screen.addstr(y, x_start, tab_text.ljust(tab_width), attr)

        # Draw separator line
        self.screen.addstr(y + 1, 0, "─" * width, curses.color_pair(TUIColors.BLUE))

    def _draw_overview_tab(self, start_y: int, height: int, width: int):
        """Draw the overview tab content."""
        metrics = self.current_metrics
        y = start_y

        # Performance summary box
        self._draw_box(y, 0, 8, width // 2 - 1, "📊 PERFORMANCE OVERVIEW")
        y_box = y + 2

        # Calculate total efficiency gain
        base_improvement = 625  # Your existing improvement
        total_improvement = base_improvement * metrics.efficiency_multiplier

        self.screen.addstr(
            y_box,
            2,
            f"Total Contexts: {metrics.total_contexts:,}",
            curses.color_pair(TUIColors.WHITE),
        )
        self.screen.addstr(
            y_box + 1,
            2,
            f"Compressed: {metrics.compressed_contexts:,}",
            curses.color_pair(TUIColors.GREEN),
        )
        self.screen.addstr(
            y_box + 2,
            2,
            f"Cache Hit Rate: {metrics.cache_hit_rate:.1%}",
            self._get_performance_color(metrics.cache_hit_rate, 0.8, 0.5),
        )
        self.screen.addstr(
            y_box + 3,
            2,
            f"Space Savings: {metrics.space_savings_percent:.1f}%",
            self._get_performance_color(metrics.space_savings_percent / 100, 0.6, 0.3),
        )
        self.screen.addstr(
            y_box + 4,
            2,
            f"Efficiency: {total_improvement:.0f}x",
            curses.color_pair(TUIColors.MAGENTA) | curses.A_BOLD,
        )

        # Compression metrics box
        self._draw_box(y, width // 2, 8, width // 2 - 1, "⚡ COMPRESSION METRICS")
        y_box = y + 2
        x_box = width // 2 + 2

        self.screen.addstr(
            y_box,
            x_box,
            f"Compression Ratio: {metrics.compression_ratio:.3f}",
            curses.color_pair(TUIColors.CYAN),
        )
        self.screen.addstr(
            y_box + 1,
            x_box,
            f"Avg Compress Time: {metrics.avg_compression_time_ms:.1f}ms",
            curses.color_pair(TUIColors.YELLOW),
        )
        self.screen.addstr(
            y_box + 2,
            x_box,
            f"Avg Decompress Time: {metrics.avg_decompression_time_ms:.1f}ms",
            curses.color_pair(TUIColors.YELLOW),
        )
        self.screen.addstr(
            y_box + 3,
            x_box,
            f"Space Saved: {self._format_bytes(metrics.total_space_saved_bytes)}",
            curses.color_pair(TUIColors.GREEN),
        )
        self.screen.addstr(
            y_box + 4,
            x_box,
            f"Efficiency Multiplier: {metrics.efficiency_multiplier:.1f}x",
            curses.color_pair(TUIColors.MAGENTA),
        )

        # Mini chart
        chart_y = y + 10
        if chart_y + 10 < start_y + height:
            self._draw_mini_chart(chart_y, width, "Cache Hit Rate History")

    def _draw_compression_tab(self, start_y: int, height: int, width: int):
        """Draw the compression analytics tab."""
        y = start_y

        if not hasattr(self.context_store, "get_compression_analytics"):
            self._draw_centered_text(
                y + height // 2,
                width,
                "Compression analytics not available",
                TUIColors.YELLOW,
            )
            return

        try:
            analytics = self.context_store.get_compression_analytics()

            if not analytics.get("compression_enabled", False):
                self._draw_centered_text(
                    y + height // 2, width, "Compression is disabled", TUIColors.YELLOW
                )
                return

            # Compression performance breakdown
            context_stats = analytics.get("context_store_stats", {})
            content_breakdown = context_stats.get("content_type_breakdown", {})

            self._draw_box(
                y,
                0,
                min(height - 2, len(content_breakdown) + 4),
                width,
                "📋 CONTENT TYPE COMPRESSION PERFORMANCE",
            )

            y_content = y + 2
            for i, (content_type, stats) in enumerate(content_breakdown.items()):
                if y_content + i < start_y + height - 3:
                    savings = stats["avg_savings"]
                    count = stats["count"]
                    color = self._get_performance_color(savings / 100, 0.6, 0.3)

                    self.screen.addstr(
                        y_content + i,
                        2,
                        f"{content_type.ljust(15)}: {savings:5.1f}% avg savings ({count:3d} samples)",
                        curses.color_pair(color),
                    )

        except Exception as e:
            self._draw_centered_text(
                y + height // 2,
                width,
                f"Error loading compression data: {str(e)}",
                TUIColors.RED,
            )

    def _draw_cache_tab(self, start_y: int, height: int, width: int):
        """Draw the cache performance tab."""
        y = start_y
        metrics = self.current_metrics

        # Cache statistics
        self._draw_box(y, 0, 10, width, "💾 CACHE PERFORMANCE")
        y_content = y + 2

        cache_stats = self.context_store.get_cache_stats()

        stats_to_show = [
            (
                "Hit Rate",
                f"{metrics.cache_hit_rate:.1%}",
                self._get_performance_color(metrics.cache_hit_rate, 0.8, 0.5),
            ),
            ("Total Contexts", f"{metrics.total_contexts:,}", TUIColors.WHITE),
            ("Cache Evictions", f"{metrics.evictions:,}", TUIColors.YELLOW),
            (
                "Memory Usage",
                f"{metrics.memory_usage_percent:.1f}%",
                self._get_performance_color(
                    1 - metrics.memory_usage_percent / 100, 0.5, 0.2
                ),
            ),
        ]

        for i, (label, value, color) in enumerate(stats_to_show):
            if y_content + i < start_y + height - 3:
                self.screen.addstr(
                    y_content + i,
                    2,
                    f"{label.ljust(20)}: {value}",
                    curses.color_pair(color),
                )

    def _draw_memory_tab(self, start_y: int, height: int, width: int):
        """Draw memory usage and system information."""
        y = start_y

        self._draw_box(y, 0, height - 2, width, "🧠 MEMORY & SYSTEM INFORMATION")
        y_content = y + 2

        # System memory info
        try:
            import psutil

            memory = psutil.virtual_memory()

            memory_info = [
                f"System Memory Total: {self._format_bytes(memory.total)}",
                f"System Memory Used: {self._format_bytes(memory.used)} ({memory.percent:.1f}%)",
                f"System Memory Available: {self._format_bytes(memory.available)}",
                "",
                f"Context Store Memory: {self.current_metrics.memory_usage_percent:.1f}%",
                f"Total Space Saved: {self._format_bytes(self.current_metrics.total_space_saved_bytes)}",
                f"Compression Efficiency: {self.current_metrics.efficiency_multiplier:.1f}x multiplier",
            ]

            for i, info in enumerate(memory_info):
                if y_content + i < start_y + height - 3:
                    color = (
                        TUIColors.GREEN if "saved" in info.lower() else TUIColors.WHITE
                    )
                    self.screen.addstr(y_content + i, 2, info, curses.color_pair(color))

        except ImportError:
            self.screen.addstr(
                y_content,
                2,
                "psutil not available for system memory info",
                curses.color_pair(TUIColors.YELLOW),
            )

    def _draw_alerts_tab(self, start_y: int, height: int, width: int):
        """Draw the alerts and notifications tab."""
        y = start_y

        self._draw_box(
            y, 0, height - 2, width, f"🚨 ALERTS & NOTIFICATIONS ({len(self.alerts)})"
        )
        y_content = y + 2

        if not self.alerts:
            self._draw_centered_text(
                y_content + 5,
                width,
                "No alerts - System operating normally",
                TUIColors.GREEN,
            )
            return

        # Show alerts with color coding
        for i, alert in enumerate(self.alerts):
            if y_content + i >= start_y + height - 3:
                break

            timestamp = time.strftime("%H:%M:%S", time.localtime(alert["timestamp"]))
            level = alert["level"]
            message = alert["message"]

            # Color based on alert level
            if level == "critical":
                color = TUIColors.RED
                icon = "🔴"
            elif level == "warning":
                color = TUIColors.YELLOW
                icon = "🟡"
            elif level == "error":
                color = TUIColors.RED
                icon = "❌"
            else:  # info
                color = TUIColors.CYAN
                icon = "ℹ️ "

            alert_text = f"{timestamp} {icon} {message}"
            if len(alert_text) > width - 4:
                alert_text = alert_text[: width - 7] + "..."

            self.screen.addstr(y_content + i, 2, alert_text, curses.color_pair(color))

    def _draw_footer(self, y: int, width: int):
        """Draw the dashboard footer with controls."""
        controls = (
            "Controls: ←→ Switch Tabs | ↑↓ Scroll | R Refresh | C Clear Alerts | Q Quit"
        )
        if len(controls) <= width:
            self._draw_centered_text(y, width, controls, TUIColors.BLUE)

    def _draw_loading_screen(self):
        """Draw a loading screen when no metrics are available."""
        height, width = self.screen.getmaxyx()

        loading_text = "⏳ Loading Context Reference Store metrics..."
        self._draw_centered_text(height // 2, width, loading_text, TUIColors.YELLOW)

        subtext = "Please wait while we collect performance data..."
        self._draw_centered_text(height // 2 + 2, width, subtext, TUIColors.WHITE)

    def _draw_mini_chart(self, y: int, width: int, title: str):
        """Draw a simple ASCII chart of historical data."""
        if len(self.metrics_history) < 2:
            return

        chart_height = 8
        chart_width = min(60, width - 4)

        self._draw_box(y, 0, chart_height + 2, chart_width + 4, title)

        # Get hit rate history
        hit_rates = [
            m.cache_hit_rate for m in list(self.metrics_history)[-chart_width:]
        ]

        if not hit_rates:
            return

        # Normalize to chart height
        min_val = min(hit_rates)
        max_val = max(hit_rates)
        val_range = max_val - min_val if max_val > min_val else 1

        # Draw chart
        for i, rate in enumerate(hit_rates):
            if i >= chart_width:
                break

            normalized = int(((rate - min_val) / val_range) * (chart_height - 1))
            chart_y = y + chart_height - normalized + 1

            if chart_y >= y + 2 and chart_y < y + chart_height + 1:
                self.screen.addstr(
                    chart_y, i + 2, "█", curses.color_pair(TUIColors.GREEN)
                )

        # Draw scale
        self.screen.addstr(
            y + chart_height + 1,
            2,
            f"Min: {min_val:.1%}",
            curses.color_pair(TUIColors.WHITE),
        )
        self.screen.addstr(
            y + chart_height + 1,
            chart_width - 8,
            f"Max: {max_val:.1%}",
            curses.color_pair(TUIColors.WHITE),
        )

    def _draw_box(self, y: int, x: int, height: int, width: int, title: str = ""):
        """Draw a box with optional title."""
        # Draw corners and edges
        self.screen.addch(y, x, "┌")
        self.screen.addch(y, x + width - 1, "┐")
        self.screen.addch(y + height - 1, x, "└")
        self.screen.addch(y + height - 1, x + width - 1, "┘")

        # Draw horizontal lines
        for i in range(1, width - 1):
            self.screen.addch(y, x + i, "─")
            self.screen.addch(y + height - 1, x + i, "─")

        # Draw vertical lines
        for i in range(1, height - 1):
            self.screen.addch(y + i, x, "│")
            self.screen.addch(y + i, x + width - 1, "│")

        # Draw title if provided
        if title and len(title) + 4 < width:
            title_text = f" {title} "
            title_x = x + (width - len(title_text)) // 2
            self.screen.addstr(
                y,
                title_x,
                title_text,
                curses.color_pair(TUIColors.BLUE) | curses.A_BOLD,
            )

    def _draw_centered_text(
        self, y: int, width: int, text: str, color: int, attr: int = 0
    ):
        """Draw centered text on a line."""
        if len(text) <= width:
            x = (width - len(text)) // 2
            self.screen.addstr(y, x, text, curses.color_pair(color) | attr)

    def _get_performance_color(
        self, value: float, good_threshold: float, poor_threshold: float
    ) -> int:
        """Get color based on performance value."""
        if value >= good_threshold:
            return TUIColors.GREEN
        elif value >= poor_threshold:
            return TUIColors.YELLOW
        else:
            return TUIColors.RED

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"


def create_dashboard(
    context_store: ContextReferenceStore, update_interval: float = 1.0
) -> ContextStoreTUIDashboard:
    """
    Create a TUI dashboard for monitoring a Context Reference Store.

    Args:
        context_store: The ContextReferenceStore instance to monitor
        update_interval: Update frequency in seconds

    Returns:
        Configured TUI dashboard instance

    Example:
        >>> from context_store import ContextReferenceStore
        >>> from context_store.monitoring import create_dashboard
        >>>
        >>> store = ContextReferenceStore(enable_compression=True)
        >>> dashboard = create_dashboard(store)
        >>> dashboard.start()  # This will block and show the TUI
    """
    return ContextStoreTUIDashboard(context_store, update_interval)


def main():
    """Demo function to show the dashboard in action."""
    if not CONTEXT_STORE_AVAILABLE:
        print("❌ Context Reference Store not available")
        return

    if not CURSES_AVAILABLE:
        print("❌ TUI Dashboard requires curses library")
        return

    # Create a demo context store
    from ..core.context_reference_store import (
        ContextReferenceStore,
        CacheEvictionPolicy,
    )

    store = ContextReferenceStore(
        cache_size=100,
        eviction_policy=CacheEvictionPolicy.LRU,
        enable_compression=True,
        enable_cache_warming=True,
    )

    # Add some demo content
    demo_content = [
        "This is a test context for demonstration purposes.",
        {"key": "value", "data": [1, 2, 3, 4, 5]},
        "A" * 1000,  # Trigger compression
        json.dumps({"large_object": {"nested": {"data": "x" * 500}}}),
    ]

    for i, content in enumerate(demo_content):
        store.store(content, metadata={"demo": True, "index": i})

    # Start dashboard
    dashboard = create_dashboard(store)
    dashboard.start()


if __name__ == "__main__":
    main()
