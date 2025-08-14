"""
Semantic Analysis Module

Provides advanced semantic understanding capabilities for Context Reference Store,
including context clustering, semantic deduplication, and intelligent content analysis.
"""

from .semantic_analyzer import (
    SemanticContextAnalyzer,
    SimilarityMethod,
    ClusteringAlgorithm,
    MergeStrategy,
    SemanticMatch,
    ContextCluster,
    SemanticAnalysisResult,
    create_semantic_analyzer,
)

__all__ = [
    "SemanticContextAnalyzer",
    "SimilarityMethod",
    "ClusteringAlgorithm",
    "MergeStrategy",
    "SemanticMatch",
    "ContextCluster",
    "SemanticAnalysisResult",
    "create_semantic_analyzer",
]
