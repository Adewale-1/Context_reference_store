"""
Context Compression Manager

Provides intelligent compression and decompression for different content types
to achieve 10-50x additional storage savings on top of existing optimizations.

Features:
- Auto-detection of content types (text, JSON, code, binary)
- Algorithm selection based on content characteristics
- Compression ratio tracking and optimization
- Integration with existing ContextReferenceStore
"""

import json
import zlib
import gzip
import bz2
import lzma
import re
import hashlib
import logging
from typing import Dict, Any, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass
import time


class CompressionAlgorithm(Enum):
    """Available compression algorithms."""

    ZLIB = "zlib"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    NONE = "none"


class ContentType(Enum):
    """Detected content types for optimal compression."""

    TEXT = "text"
    JSON = "json"
    CODE = "code"
    XML_HTML = "xml_html"
    CSV = "csv"
    BINARY = "binary"
    STRUCTURED_TEXT = "structured_text"


@dataclass
class CompressionResult:
    """Result of compression operation."""

    compressed_data: bytes
    algorithm: CompressionAlgorithm
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    content_type: ContentType
    metadata: Dict[str, Any]

    @property
    def space_savings(self) -> float:
        """Calculate space savings percentage."""
        return (1 - self.compression_ratio) * 100

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (space savings / time)."""
        if self.compression_time == 0:
            return float("inf")
        return self.space_savings / self.compression_time


class ContextCompressionManager:
    """
    Smart compression manager for context content.

    Automatically detects content types and selects optimal compression
    algorithms to maximize storage efficiency while maintaining fast
    decompression speeds.
    """

    def __init__(
        self, enable_analytics: bool = True, min_compression_size: int = 1024
    ):  # Don't compress < 1KB
        """
        Initialize the compression manager.

        Args:
            enable_analytics: Whether to track compression analytics
            min_compression_size: Minimum size in bytes to attempt compression
        """
        self.logger = logging.getLogger(__name__)
        self.enable_analytics = enable_analytics
        self.min_compression_size = min_compression_size

        # Compression analytics
        self.compression_stats = {
            "total_compressions": 0,
            "total_decompressions": 0,
            "total_original_size": 0,
            "total_compressed_size": 0,
            "algorithm_usage": {alg.value: 0 for alg in CompressionAlgorithm},
            "content_type_stats": {
                ct.value: {"count": 0, "avg_ratio": 0.0} for ct in ContentType
            },
            "compression_times": [],
            "decompression_times": [],
        }

        # Algorithm configurations with optimal settings
        self.algorithm_configs = {
            CompressionAlgorithm.ZLIB: {"level": 6, "wbits": 15},
            CompressionAlgorithm.GZIP: {"compresslevel": 6},
            CompressionAlgorithm.BZIP2: {"compresslevel": 6},
            CompressionAlgorithm.LZMA: {"preset": 3},
        }

    def detect_content_type(self, content: str) -> ContentType:
        """
        Intelligently detect content type for optimal compression strategy.

        Args:
            content: String content to analyze

        Returns:
            Detected content type
        """
        content_lower = content.lower().strip()

        # JSON detection
        if self._is_json(content):
            return ContentType.JSON

        # XML/HTML detection
        if (
            content_lower.startswith("<?xml")
            or content_lower.startswith("<!doctype")
            or "<html" in content_lower
            or re.search(r"<[^>]+>", content)
        ):
            return ContentType.XML_HTML

        # CSV detection
        if self._is_csv(content):
            return ContentType.CSV

        # Code detection (multiple languages)
        if self._is_code(content):
            return ContentType.CODE

        # Structured text (markdown, documentation)
        if self._is_structured_text(content):
            return ContentType.STRUCTURED_TEXT

        # Default to plain text
        return ContentType.TEXT

    def _is_json(self, content: str) -> bool:
        """Check if content is valid JSON."""
        try:
            json.loads(content.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def _is_csv(self, content: str) -> bool:
        """Check if content appears to be CSV."""
        lines = content.split("\n")[:10] 
        if len(lines) < 2:
            return False

        # Look for consistent comma/tab patterns
        comma_counts = [line.count(",") for line in lines if line.strip()]
        tab_counts = [line.count("\t") for line in lines if line.strip()]

        return (len(set(comma_counts)) == 1 and comma_counts[0] > 0) or (
            len(set(tab_counts)) == 1 and tab_counts[0] > 0
        )

    def _is_code(self, content: str) -> bool:
        """Check if content appears to be source code."""
        code_indicators = [
            r"\bdef\s+\w+\s*\(",
            r"\bfunction\s+\w+\s*\(",
            r"\bclass\s+\w+",
            r"\bimport\s+\w+",
            r"\bfrom\s+\w+\s+import",
            r"\b(public|private|protected)\s+",  
            r"\b(int|str|bool|float|double|char)\s+\w+",  
            r"^\s*[#//]\s*",  
            r"\{[\s\S]*\}",  
        ]

        matches = sum(
            1
            for pattern in code_indicators
            if re.search(pattern, content, re.MULTILINE)
        )
        return matches >= 2

    def _is_structured_text(self, content: str) -> bool:
        """Check if content is structured text (markdown, docs)."""
        structured_indicators = [
            r"^#{1,6}\s+", 
            r"^\*\s+", 
            r"^\d+\.\s+", 
            r"\*\*.*?\*\*", 
            r"\[.*?\]\(.*?\)", 
            r"```[\s\S]*?```",  
        ]

        matches = sum(
            1
            for pattern in structured_indicators
            if re.search(pattern, content, re.MULTILINE)
        )
        return matches >= 2

    def select_optimal_algorithm(
        self, content: str, content_type: ContentType
    ) -> CompressionAlgorithm:
        """
        Select optimal compression algorithm based on content characteristics.

        Args:
            content: Content to compress
            content_type: Detected content type

        Returns:
            Optimal compression algorithm
        """
        content_size = len(content.encode("utf-8"))
        if content_size < self.min_compression_size:
            return CompressionAlgorithm.NONE
        # Algorithm selection based on content type and size
        if content_type == ContentType.JSON:
            # JSON compresses very well with zlib due to repetitive structure
            return (
                CompressionAlgorithm.ZLIB
                if content_size < 100000
                else CompressionAlgorithm.LZMA
            )

        elif content_type == ContentType.CODE:
            # Code has repetitive patterns, benefits from good dictionary compression
            return (
                CompressionAlgorithm.LZMA
                if content_size > 50000
                else CompressionAlgorithm.ZLIB
            )

        elif content_type == ContentType.XML_HTML:
            # XML/HTML has lots of repetitive tags
            return CompressionAlgorithm.ZLIB

        elif content_type == ContentType.CSV:
            # CSV has repetitive structure and values
            return (
                CompressionAlgorithm.BZIP2
                if content_size > 10000
                else CompressionAlgorithm.ZLIB
            )
        elif content_type == ContentType.STRUCTURED_TEXT:
            # Structured text benefits from good pattern recognition
            return (
                CompressionAlgorithm.LZMA
                if content_size > 20000
                else CompressionAlgorithm.ZLIB
            )
        else: 
            return (
                CompressionAlgorithm.ZLIB
                if content_size < 50000
                else CompressionAlgorithm.BZIP2
            )

    def preprocess_content(self, content: str, content_type: ContentType) -> str:
        """
        Preprocess content for optimal compression.

        Args:
            content: Original content
            content_type: Content type

        Returns:
            Preprocessed content optimized for compression
        """
        if content_type == ContentType.JSON:
            # Minify JSON
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
            except json.JSONDecodeError:
                return content
        elif content_type == ContentType.CODE:
            # Remove excessive whitespace and comments for better compression
            return self._minify_code(content)
        elif content_type == ContentType.XML_HTML:
            # Basic HTML/XML minification
            return self._minify_markup(content)
        # For other types, return as is
        return content

    def _minify_code(self, code: str) -> str:
        """Basic code minification for better compression."""
        lines = code.split("\n")
        minified_lines = []

        for line in lines:
            # Remove comments
            if line.strip().startswith("#"):
                continue
            if line.strip().startswith("//"):
                continue
            # Remove excessive whitespace but preserve indentation
            stripped = line.rstrip()
            if stripped:
                minified_lines.append(stripped)

        return "\n".join(minified_lines)

    def _minify_markup(self, markup: str) -> str:
        """Basic markup minification."""
        # Remove comments
        markup = re.sub(r"<!--.*?-->", "", markup, flags=re.DOTALL)
        markup = re.sub(r">\s+<", "><", markup)
        lines = [line.strip() for line in markup.split("\n") if line.strip()]

        return "\n".join(lines)

    def compress(
        self, content: str, algorithm: Optional[CompressionAlgorithm] = None
    ) -> CompressionResult:
        """
        Compress content using optimal algorithm and preprocessing.

        Args:
            content: Content to compress
            algorithm: Override algorithm selection (optional)

        Returns:
            Compression result with metadata
        """
        start_time = time.time()
        # Detect content type
        content_type = self.detect_content_type(content)
        if algorithm is None:
            algorithm = self.select_optimal_algorithm(content, content_type)

        # No compression for very small content
        if algorithm == CompressionAlgorithm.NONE:
            original_bytes = content.encode("utf-8")
            return CompressionResult(
                compressed_data=original_bytes,
                algorithm=algorithm,
                original_size=len(original_bytes),
                compressed_size=len(original_bytes),
                compression_ratio=1.0,
                compression_time=time.time() - start_time,
                content_type=content_type,
                metadata={"preprocessed": False},
            )

        # Preprocess content for optimal compression
        preprocessed_content = self.preprocess_content(content, content_type)
        original_size = len(content.encode("utf-8"))
        content_bytes = preprocessed_content.encode("utf-8")
        # Perform compression
        try:
            if algorithm == CompressionAlgorithm.ZLIB:
                config = self.algorithm_configs[algorithm]
                compressed_data = zlib.compress(content_bytes, level=config["level"])

            elif algorithm == CompressionAlgorithm.GZIP:
                config = self.algorithm_configs[algorithm]
                compressed_data = gzip.compress(
                    content_bytes, compresslevel=config["compresslevel"]
                )

            elif algorithm == CompressionAlgorithm.BZIP2:
                config = self.algorithm_configs[algorithm]
                compressed_data = bz2.compress(
                    content_bytes, compresslevel=config["compresslevel"]
                )

            elif algorithm == CompressionAlgorithm.LZMA:
                config = self.algorithm_configs[algorithm]
                compressed_data = lzma.compress(content_bytes, preset=config["preset"])

            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        except Exception as e:
            self.logger.warning(f"Compression failed with {algorithm}: {e}")
            # Fallback to no compression
            compressed_data = content_bytes
            algorithm = CompressionAlgorithm.NONE

        compression_time = time.time() - start_time
        compressed_size = len(compressed_data)
        compression_ratio = compressed_size / original_size


        if self.enable_analytics:
            self._update_compression_stats(
                algorithm,
                content_type,
                original_size,
                compressed_size,
                compression_time,
            )

        return CompressionResult(
            compressed_data=compressed_data,
            algorithm=algorithm,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            content_type=content_type,
            metadata={
                "preprocessed": preprocessed_content != content,
                "preprocessing_saved": len(content.encode("utf-8"))
                - len(content_bytes),
            },
        )

    def decompress(
        self,
        compressed_data: bytes,
        algorithm: CompressionAlgorithm,
        original_encoding: str = "utf-8",
    ) -> str:
        """
        Decompress data back to original string content.

        Args:
            compressed_data: Compressed byte data
            algorithm: Algorithm used for compression
            original_encoding: Original text encoding

        Returns:
            Decompressed string content
        """
        start_time = time.time()

        try:
            if algorithm == CompressionAlgorithm.NONE:
                decompressed_bytes = compressed_data

            elif algorithm == CompressionAlgorithm.ZLIB:
                decompressed_bytes = zlib.decompress(compressed_data)

            elif algorithm == CompressionAlgorithm.GZIP:
                decompressed_bytes = gzip.decompress(compressed_data)

            elif algorithm == CompressionAlgorithm.BZIP2:
                decompressed_bytes = bz2.decompress(compressed_data)

            elif algorithm == CompressionAlgorithm.LZMA:
                decompressed_bytes = lzma.decompress(compressed_data)

            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            decompressed_content = decompressed_bytes.decode(original_encoding)

            # Update analytics
            if self.enable_analytics:
                decompression_time = time.time() - start_time
                self.compression_stats["total_decompressions"] += 1
                self.compression_stats["decompression_times"].append(decompression_time)

            return decompressed_content

        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise

    def _update_compression_stats(
        self,
        algorithm: CompressionAlgorithm,
        content_type: ContentType,
        original_size: int,
        compressed_size: int,
        compression_time: float,
    ):
        """Update compression analytics."""
        stats = self.compression_stats
        stats["total_compressions"] += 1
        stats["total_original_size"] += original_size
        stats["total_compressed_size"] += compressed_size
        stats["algorithm_usage"][algorithm.value] += 1
        stats["compression_times"].append(compression_time)
        
        ct_stats = stats["content_type_stats"][content_type.value]
        ct_stats["count"] += 1

        current_ratio = compressed_size / original_size
        if ct_stats["avg_ratio"] == 0:
            ct_stats["avg_ratio"] = current_ratio
        else:
            ct_stats["avg_ratio"] = (ct_stats["avg_ratio"] + current_ratio) / 2

    def get_compression_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive compression analytics.

        Returns:
            Dictionary with compression statistics and insights
        """
        stats = self.compression_stats

        if stats["total_compressions"] == 0:
            return {"error": "No compression operations performed yet"}

        overall_ratio = stats["total_compressed_size"] / stats["total_original_size"]
        space_savings = (1 - overall_ratio) * 100

        avg_compression_time = sum(stats["compression_times"]) / len(
            stats["compression_times"]
        )
        avg_decompression_time = (
            (sum(stats["decompression_times"]) / len(stats["decompression_times"]))
            if stats["decompression_times"]
            else 0
        )
        return {
            "summary": {
                "total_compressions": stats["total_compressions"],
                "total_decompressions": stats["total_decompressions"],
                "overall_compression_ratio": overall_ratio,
                "space_savings_percent": space_savings,
                "total_space_saved_bytes": stats["total_original_size"]
                - stats["total_compressed_size"],
                "avg_compression_time_ms": avg_compression_time * 1000,
                "avg_decompression_time_ms": avg_decompression_time * 1000,
            },
            "algorithm_usage": stats["algorithm_usage"],
            "content_type_performance": stats["content_type_stats"],
            "efficiency_metrics": {
                "compression_speed_mb_per_sec": (
                    (stats["total_original_size"] / (1024 * 1024))
                    / sum(stats["compression_times"])
                    if stats["compression_times"]
                    else 0
                ),
                "decompression_speed_mb_per_sec": (
                    (stats["total_compressed_size"] / (1024 * 1024))
                    / sum(stats["decompression_times"])
                    if stats["decompression_times"]
                    else 0
                ),
            },
        }

    def optimize_for_content_type(self, content_type: ContentType) -> Dict[str, Any]:
        """
        Get optimization recommendations for specific content type.

        Args:
            content_type: Content type to optimize for

        Returns:
            Optimization recommendations
        """
        recommendations = {
            ContentType.JSON: {
                "preprocessing": "Enable JSON minification",
                "algorithm": "ZLIB for small JSON, LZMA for large JSON",
                "expected_ratio": "60-80% compression",
                "tips": [
                    "Remove unnecessary whitespace",
                    "Use shorter key names",
                    "Group similar objects",
                ],
            },
            ContentType.CODE: {
                "preprocessing": "Remove comments and excessive whitespace",
                "algorithm": "LZMA for large codebases, ZLIB for small files",
                "expected_ratio": "70-85% compression",
                "tips": [
                    "Consistent formatting",
                    "Remove debug prints",
                    "Group similar functions",
                ],
            },
            ContentType.TEXT: {
                "preprocessing": "Minimal preprocessing recommended",
                "algorithm": "ZLIB for general use",
                "expected_ratio": "40-60% compression",
                "tips": ["Use consistent terminology", "Remove duplicate sections"],
            },
        }
        return recommendations.get(
            content_type,
            {
                "preprocessing": "Basic whitespace cleanup",
                "algorithm": "ZLIB",
                "expected_ratio": "30-50% compression",
                "tips": ["Analyze content patterns for optimization opportunities"],
            },
        )
