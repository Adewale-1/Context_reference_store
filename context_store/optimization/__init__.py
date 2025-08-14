"""
Optimization Module

Provides intelligent optimization capabilities for Context Reference Store,
including token-aware management, cost optimization, and performance tuning.
"""

from .token_manager import (
    TokenAwareContextManager,
    ModelConfig,
    ModelFamily,
    OptimizationStrategy,
    TokenBudget,
    ContextCandidate,
    TokenOptimizationResult,
    create_token_manager,
)

__all__ = [
    "TokenAwareContextManager",
    "ModelConfig",
    "ModelFamily",
    "OptimizationStrategy",
    "TokenBudget",
    "ContextCandidate",
    "TokenOptimizationResult",
    "create_token_manager",
]
