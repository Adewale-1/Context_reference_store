"""Framework adapters for integrating Context Reference Store with popular Agentic frameworks."""

__all__ = []


try:
    from .langchain_adapter import LangChainContextAdapter

    __all__.append("LangChainContextAdapter")
except ImportError:
    pass

try:
    from .langgraph_adapter import LangGraphContextAdapter

    __all__.append("LangGraphContextAdapter")
except ImportError:
    pass

try:
    from .llamaindex_adapter import LlamaIndexContextAdapter

    __all__.append("LlamaIndexContextAdapter")
except ImportError:
    pass

try:
    from .composio_adapter import ComposioContextAdapter

    __all__.append("ComposioContextAdapter")
except ImportError:
    pass
