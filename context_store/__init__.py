"""
Context Reference Store - Efficient Large Context Window Management

A high-performance library for managing large context windows (1M-2M tokens)
with reference-based approach, multimodal support, and framework adapters.

Key Features:
- 625x faster serialization compared to traditional approaches
- 49x memory reduction in multi-agent scenarios
- 99.55% storage reduction for multimodal content
- Framework adapters for LangChain, LangGraph, LlamaIndex
- Advanced caching strategies (LRU, LFU, TTL, Memory Pressure)
- Zero quality degradation (validated with ROUGE metrics)
"""

__version__ = "1.0.5"
__author__ = "Adewale-Young Adenle"
__email__ = "waleadenle1@gmail.com"

from .core.context_reference_store import (
    ContextReferenceStore,
    ContextMetadata,
    CacheEvictionPolicy,
    MultimodalContent,
    MultimodalPart,
)
from .core.large_context_state import LargeContextState
from .core.base_state import BaseState
from .core.compression_manager import (
    ContextCompressionManager,
    CompressionAlgorithm,
    CompressionResult,
    ContentType,
)

__all__ = [
    "ContextReferenceStore",
    "ContextMetadata",
    "CacheEvictionPolicy",
    "MultimodalContent",
    "MultimodalPart",
    "LargeContextState",
    "BaseState",
    "ContextCompressionManager",
    "CompressionAlgorithm",
    "CompressionResult",
    "ContentType",
]

try:
    from .adapters.langchain_adapter import LangChainContextAdapter

    __all__.append("LangChainContextAdapter")
except ImportError:
    pass

try:
    from .adapters.langgraph_adapter import LangGraphContextAdapter

    __all__.append("LangGraphContextAdapter")
except ImportError:
    pass

try:
    from .adapters.llamaindex_adapter import LlamaIndexContextAdapter

    __all__.append("LlamaIndexContextAdapter")
except ImportError:
    pass

try:
    from .adapters.composio_adapter import ComposioContextAdapter

    __all__.append("ComposioContextAdapter")
except ImportError:
    pass

try:
    from .monitoring import ContextStoreTUIDashboard, create_dashboard

    __all__.extend(["ContextStoreTUIDashboard", "create_dashboard"])
except ImportError:
    pass

try:
    from .optimization import (
        TokenAwareContextManager,
        ModelConfig,
        ModelFamily,
        OptimizationStrategy,
        TokenBudget,
        create_token_manager,
    )

    __all__.extend(
        [
            "TokenAwareContextManager",
            "ModelConfig",
            "ModelFamily",
            "OptimizationStrategy",
            "TokenBudget",
            "create_token_manager",
        ]
    )
except ImportError:
    pass

try:
    from .semantic import (
        SemanticContextAnalyzer,
        SimilarityMethod,
        ClusteringAlgorithm,
        MergeStrategy,
        SemanticMatch,
        ContextCluster,
        create_semantic_analyzer,
    )

    __all__.extend(
        [
            "SemanticContextAnalyzer",
            "SimilarityMethod",
            "ClusteringAlgorithm",
            "MergeStrategy",
            "SemanticMatch",
            "ContextCluster",
            "create_semantic_analyzer",
        ]
    )
except ImportError:
    pass
