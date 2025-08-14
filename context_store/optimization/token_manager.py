"""
Smart Token-Aware Context Management

Intelligent token counting, budget optimization, and cost-aware context selection
for LLM applications. Provides automatic token limit compliance and cost optimization.

Features:
- Accurate token counting for multiple LLM models
- Dynamic context selection within token budgets
- Cost optimization strategies
- Relevance-based context ranking
- Token usage analytics and forecasting
- Model-specific optimization profiles
"""

import re
import json
import math
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class ModelFamily(Enum):
    """Supported LLM model families."""

    GEMINI = "gemini"
    GPT = "gpt"
    CLAUDE = "claude"
    LLAMA = "llama"
    CUSTOM = "custom"


class OptimizationStrategy(Enum):
    """Token optimization strategies."""

    COST_FIRST = "cost_first"  # Minimize cost while meeting requirements
    QUALITY_FIRST = "quality_first"  # Maximize context quality within budget
    BALANCED = "balanced"  # Balance cost and quality
    SPEED_FIRST = "speed_first"  # Optimize for fastest processing
    COMPREHENSIVE = "comprehensive"  # Include maximum relevant context


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""

    name: str
    family: ModelFamily
    max_tokens: int
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    encoding_name: Optional[str] = None
    context_window_efficiency: float = 0.95  # Use 95% of max tokens for safety
    supports_system_prompt: bool = True
    average_tokens_per_char: float = 0.25  # Approximate for fallback counting


@dataclass
class TokenBudget:
    """Token budget configuration."""

    total_tokens: int
    system_prompt_tokens: int = 0
    reserved_output_tokens: int = 1000
    context_tokens_available: int = field(init=False)

    def __post_init__(self):
        self.context_tokens_available = (
            self.total_tokens - self.system_prompt_tokens - self.reserved_output_tokens
        )


@dataclass
class ContextCandidate:
    """A context candidate for selection."""

    context_id: str
    content: str
    token_count: int
    relevance_score: float
    priority: int = 5
    timestamp: float = field(default_factory=time.time)
    content_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenOptimizationResult:
    """Result of token optimization process."""

    selected_contexts: List[ContextCandidate]
    total_tokens: int
    budget_utilization: float
    estimated_cost: float
    optimization_strategy: OptimizationStrategy
    excluded_contexts: List[ContextCandidate]
    efficiency_score: float
    recommendations: List[str]


class TokenCounter(ABC):
    """Abstract base class for token counting implementations."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Provide a fast token estimate."""
        pass


class TikTokenCounter(TokenCounter):
    """Token counter using OpenAI's tiktoken library."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for accurate token counting")

        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken."""
        return len(self.encoding.encode(text))

    def estimate_tokens(self, text: str) -> int:
        """For tiktoken, estimate and actual count are the same."""
        return self.count_tokens(text)


class FallbackTokenCounter(TokenCounter):
    """Fallback token counter using character-based estimation."""

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count_tokens(self, text: str) -> int:
        """Estimate tokens based on character count."""
        return max(1, int(len(text) / self.chars_per_token))

    def estimate_tokens(self, text: str) -> int:
        """Fast estimation using character count."""
        return self.count_tokens(text)


class RelevanceScorer:
    """Calculates relevance scores for context selection."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_relevance(
        self, context: str, query: str = "", keywords: List[str] = None
    ) -> float:
        """
        Calculate relevance score for a context.

        Args:
            context: The context content
            query: Optional query to match against
            keywords: Optional list of keywords to match

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not context:
            return 0.0

        score = 0.0
        factors = 0

        # Query matching
        if query:
            query_score = self._calculate_query_match(context, query)
            score += query_score
            factors += 1

        # Keyword matching
        if keywords:
            keyword_score = self._calculate_keyword_match(context, keywords)
            score += keyword_score
            factors += 1

        # Content quality indicators
        quality_score = self._calculate_content_quality(context)
        score += quality_score
        factors += 1

        # Recency bias (favor newer content)
        recency_score = 0.5  # Default neutral score
        score += recency_score
        factors += 1

        return min(1.0, score / factors) if factors > 0 else 0.5

    def _calculate_query_match(self, context: str, query: str) -> float:
        """Calculate how well context matches a query."""
        if not query:
            return 0.5

        context_lower = context.lower()
        query_lower = query.lower()

        # Exact phrase match
        if query_lower in context_lower:
            return 1.0

        # Word overlap
        query_words = set(query_lower.split())
        context_words = set(context_lower.split())

        if not query_words:
            return 0.5

        overlap = len(query_words.intersection(context_words))
        return overlap / len(query_words)

    def _calculate_keyword_match(self, context: str, keywords: List[str]) -> float:
        """Calculate keyword matching score."""
        if not keywords:
            return 0.5

        context_lower = context.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in context_lower)

        return matches / len(keywords)

    def _calculate_content_quality(self, context: str) -> float:
        """Assess content quality based on various indicators."""
        if not context:
            return 0.0

        quality_indicators = []

        # Length indicator (prefer moderate length content)
        length = len(context)
        if 100 <= length <= 5000:  # Sweet spot for most contexts
            quality_indicators.append(0.8)
        elif length < 50:  # Too short
            quality_indicators.append(0.3)
        else:  # Very long content
            quality_indicators.append(0.6)

        # Structure indicators
        has_sentences = "." in context or "!" in context or "?" in context
        has_paragraphs = "\n\n" in context or "\n" in context
        has_formatting = any(char in context for char in ["*", "#", "-", "="])

        structure_score = sum([has_sentences, has_paragraphs, has_formatting]) / 3
        quality_indicators.append(structure_score)

        # Information density (avoid overly repetitive content)
        words = context.split()
        unique_words = set(word.lower() for word in words)
        if words:
            diversity_score = len(unique_words) / len(words)
            quality_indicators.append(min(1.0, diversity_score * 2))  # Scale up

        return (
            sum(quality_indicators) / len(quality_indicators)
            if quality_indicators
            else 0.5
        )


class TokenAwareContextManager:
    """
    Smart token-aware context management system.

    Provides intelligent context selection based on token budgets, costs,
    and relevance scoring for optimal LLM usage.
    """

    # Predefined model configurations
    MODEL_CONFIGS = {
        "gemini-1.5-pro": ModelConfig(
            name="gemini-1.5-pro",
            family=ModelFamily.GEMINI,
            max_tokens=2000000,
            cost_per_1k_input_tokens=0.0035,
            cost_per_1k_output_tokens=0.0105,
            encoding_name="cl100k_base",  # Approximate
            context_window_efficiency=0.98,
        ),
        "gemini-1.5-flash": ModelConfig(
            name="gemini-1.5-flash",
            family=ModelFamily.GEMINI,
            max_tokens=1000000,
            cost_per_1k_input_tokens=0.00035,
            cost_per_1k_output_tokens=0.00105,
            encoding_name="cl100k_base",
            context_window_efficiency=0.98,
        ),
        "gpt-4": ModelConfig(
            name="gpt-4",
            family=ModelFamily.GPT,
            max_tokens=128000,
            cost_per_1k_input_tokens=0.03,
            cost_per_1k_output_tokens=0.06,
            encoding_name="cl100k_base",
            context_window_efficiency=0.95,
        ),
        "gpt-4-turbo": ModelConfig(
            name="gpt-4-turbo",
            family=ModelFamily.GPT,
            max_tokens=128000,
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
            encoding_name="cl100k_base",
            context_window_efficiency=0.95,
        ),
        "gpt-3.5-turbo": ModelConfig(
            name="gpt-3.5-turbo",
            family=ModelFamily.GPT,
            max_tokens=16385,
            cost_per_1k_input_tokens=0.0015,
            cost_per_1k_output_tokens=0.002,
            encoding_name="cl100k_base",
            context_window_efficiency=0.9,
        ),
        "claude-3-opus": ModelConfig(
            name="claude-3-opus",
            family=ModelFamily.CLAUDE,
            max_tokens=200000,
            cost_per_1k_input_tokens=0.015,
            cost_per_1k_output_tokens=0.075,
            encoding_name="cl100k_base",  # Approximate
            context_window_efficiency=0.95,
        ),
        "claude-3-sonnet": ModelConfig(
            name="claude-3-sonnet",
            family=ModelFamily.CLAUDE,
            max_tokens=200000,
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            encoding_name="cl100k_base",
            context_window_efficiency=0.95,
        ),
    }

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the token-aware context manager.

        Args:
            model_name: Name of the LLM model to optimize for
        """
        self.logger = logging.getLogger(__name__)

        # Model configuration
        if model_name in self.MODEL_CONFIGS:
            self.model_config = self.MODEL_CONFIGS[model_name]
        else:
            self.logger.warning(f"Unknown model {model_name}, using default config")
            self.model_config = ModelConfig(
                name=model_name,
                family=ModelFamily.CUSTOM,
                max_tokens=8192,
                cost_per_1k_input_tokens=0.01,
                cost_per_1k_output_tokens=0.03,
            )

        # Initialize token counter
        self.token_counter = self._initialize_token_counter()

        # Initialize relevance scorer
        self.relevance_scorer = RelevanceScorer()

        # Usage statistics
        self.usage_stats = {
            "total_optimizations": 0,
            "total_tokens_processed": 0,
            "total_cost_saved": 0.0,
            "optimization_strategies_used": {
                strategy.value: 0 for strategy in OptimizationStrategy
            },
            "average_budget_utilization": 0.0,
        }

    def _initialize_token_counter(self) -> TokenCounter:
        """Initialize the appropriate token counter."""
        if TIKTOKEN_AVAILABLE and self.model_config.encoding_name:
            try:
                return TikTokenCounter(self.model_config.encoding_name)
            except Exception as e:
                self.logger.warning(f"Failed to initialize tiktoken: {e}")

        # Fallback to character-based counting
        chars_per_token = 1.0 / self.model_config.average_tokens_per_char
        return FallbackTokenCounter(chars_per_token)

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        return self.token_counter.count_tokens(text)

    def estimate_tokens(self, text: str) -> int:
        """Provide a fast token estimate."""
        return self.token_counter.estimate_tokens(text)

    def create_budget(
        self,
        target_tokens: Optional[int] = None,
        system_prompt: str = "",
        reserved_output_tokens: int = 1000,
    ) -> TokenBudget:
        """
        Create a token budget for optimization.

        Args:
            target_tokens: Target token count (uses model max if not specified)
            system_prompt: System prompt to account for
            reserved_output_tokens: Tokens to reserve for model output

        Returns:
            Configured token budget
        """
        if target_tokens is None:
            target_tokens = int(
                self.model_config.max_tokens
                * self.model_config.context_window_efficiency
            )

        system_prompt_tokens = self.count_tokens(system_prompt) if system_prompt else 0

        return TokenBudget(
            total_tokens=target_tokens,
            system_prompt_tokens=system_prompt_tokens,
            reserved_output_tokens=reserved_output_tokens,
        )

    def optimize_context_selection(
        self,
        contexts: List[str],
        budget: TokenBudget,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        query: str = "",
        keywords: List[str] = None,
        priorities: List[int] = None,
    ) -> TokenOptimizationResult:
        """
        Optimize context selection based on token budget and strategy.

        Args:
            contexts: List of context strings to choose from
            budget: Token budget constraints
            strategy: Optimization strategy to use
            query: Optional query for relevance scoring
            keywords: Optional keywords for relevance scoring
            priorities: Optional priority scores for each context

        Returns:
            Optimization result with selected contexts and metadata
        """
        start_time = time.time()

        # Create context candidates
        candidates = []
        for i, context in enumerate(contexts):
            token_count = self.count_tokens(context)
            relevance_score = self.relevance_scorer.calculate_relevance(
                context, query, keywords
            )
            priority = priorities[i] if priorities and i < len(priorities) else 5

            candidates.append(
                ContextCandidate(
                    context_id=f"ctx_{i}",
                    content=context,
                    token_count=token_count,
                    relevance_score=relevance_score,
                    priority=priority,
                )
            )

        # Apply optimization strategy
        selected_contexts, excluded_contexts = self._apply_optimization_strategy(
            candidates, budget, strategy
        )

        # Calculate results
        total_tokens = sum(ctx.token_count for ctx in selected_contexts)
        budget_utilization = total_tokens / budget.context_tokens_available
        estimated_cost = self._calculate_cost(total_tokens)
        efficiency_score = self._calculate_efficiency_score(
            selected_contexts, excluded_contexts, budget_utilization
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            selected_contexts, excluded_contexts, budget, strategy
        )

        # Update statistics
        self._update_usage_stats(strategy, total_tokens, budget_utilization)

        optimization_time = time.time() - start_time
        self.logger.info(f"Token optimization completed in {optimization_time:.3f}s")

        return TokenOptimizationResult(
            selected_contexts=selected_contexts,
            total_tokens=total_tokens,
            budget_utilization=budget_utilization,
            estimated_cost=estimated_cost,
            optimization_strategy=strategy,
            excluded_contexts=excluded_contexts,
            efficiency_score=efficiency_score,
            recommendations=recommendations,
        )

    def _apply_optimization_strategy(
        self,
        candidates: List[ContextCandidate],
        budget: TokenBudget,
        strategy: OptimizationStrategy,
    ) -> Tuple[List[ContextCandidate], List[ContextCandidate]]:
        """Apply the specified optimization strategy."""

        if strategy == OptimizationStrategy.COST_FIRST:
            return self._optimize_for_cost(candidates, budget)
        elif strategy == OptimizationStrategy.QUALITY_FIRST:
            return self._optimize_for_quality(candidates, budget)
        elif strategy == OptimizationStrategy.SPEED_FIRST:
            return self._optimize_for_speed(candidates, budget)
        elif strategy == OptimizationStrategy.COMPREHENSIVE:
            return self._optimize_comprehensive(candidates, budget)
        else:  # BALANCED
            return self._optimize_balanced(candidates, budget)

    def _optimize_for_cost(
        self, candidates: List[ContextCandidate], budget: TokenBudget
    ) -> Tuple[List[ContextCandidate], List[ContextCandidate]]:
        """Optimize for minimum cost while meeting basic requirements."""
        # Sort by cost efficiency (relevance per token)
        candidates.sort(key=lambda x: x.relevance_score / x.token_count, reverse=True)

        selected = []
        total_tokens = 0

        for candidate in candidates:
            if total_tokens + candidate.token_count <= budget.context_tokens_available:
                selected.append(candidate)
                total_tokens += candidate.token_count
            else:
                break

        excluded = [c for c in candidates if c not in selected]
        return selected, excluded

    def _optimize_for_quality(
        self, candidates: List[ContextCandidate], budget: TokenBudget
    ) -> Tuple[List[ContextCandidate], List[ContextCandidate]]:
        """Optimize for maximum relevance and quality."""
        # Sort by relevance score and priority
        candidates.sort(key=lambda x: (x.relevance_score, x.priority), reverse=True)

        selected = []
        total_tokens = 0

        for candidate in candidates:
            if total_tokens + candidate.token_count <= budget.context_tokens_available:
                selected.append(candidate)
                total_tokens += candidate.token_count
            else:
                break

        excluded = [c for c in candidates if c not in selected]
        return selected, excluded

    def _optimize_for_speed(
        self, candidates: List[ContextCandidate], budget: TokenBudget
    ) -> Tuple[List[ContextCandidate], List[ContextCandidate]]:
        """Optimize for fastest processing (fewer tokens)."""
        # Sort by token count (ascending) and relevance
        candidates.sort(key=lambda x: (x.token_count, -x.relevance_score))

        selected = []
        total_tokens = 0

        for candidate in candidates:
            if total_tokens + candidate.token_count <= budget.context_tokens_available:
                selected.append(candidate)
                total_tokens += candidate.token_count
            else:
                break

        excluded = [c for c in candidates if c not in selected]
        return selected, excluded

    def _optimize_comprehensive(
        self, candidates: List[ContextCandidate], budget: TokenBudget
    ) -> Tuple[List[ContextCandidate], List[ContextCandidate]]:
        """Include maximum relevant context within budget."""
        # Sort by relevance, then fit as many as possible
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        # Use knapsack-like algorithm for optimal selection
        return self._knapsack_optimization(candidates, budget)

    def _optimize_balanced(
        self, candidates: List[ContextCandidate], budget: TokenBudget
    ) -> Tuple[List[ContextCandidate], List[ContextCandidate]]:
        """Balance cost, quality, and comprehensiveness."""
        # Calculate composite score
        for candidate in candidates:
            efficiency = candidate.relevance_score / max(1, candidate.token_count / 100)
            priority_weight = candidate.priority / 10.0
            candidate.composite_score = (
                (efficiency * 0.6)
                + (candidate.relevance_score * 0.3)
                + (priority_weight * 0.1)
            )

        # Sort by composite score
        candidates.sort(key=lambda x: x.composite_score, reverse=True)

        selected = []
        total_tokens = 0

        for candidate in candidates:
            if total_tokens + candidate.token_count <= budget.context_tokens_available:
                selected.append(candidate)
                total_tokens += candidate.token_count
            else:
                break

        excluded = [c for c in candidates if c not in selected]
        return selected, excluded

    def _knapsack_optimization(
        self, candidates: List[ContextCandidate], budget: TokenBudget
    ) -> Tuple[List[ContextCandidate], List[ContextCandidate]]:
        """Use dynamic programming for optimal context selection."""
        n = len(candidates)
        W = budget.context_tokens_available

        # Create DP table
        dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

        # Build table in bottom-up manner
        for i in range(1, n + 1):
            candidate = candidates[i - 1]
            weight = candidate.token_count
            value = int(candidate.relevance_score * 1000)  # Scale for integer math

            for w in range(1, W + 1):
                if weight <= w:
                    dp[i][w] = max(value + dp[i - 1][w - weight], dp[i - 1][w])
                else:
                    dp[i][w] = dp[i - 1][w]

        # Backtrack to find selected items
        selected_indices = []
        w = W
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_indices.append(i - 1)
                w -= candidates[i - 1].token_count

        selected = [candidates[i] for i in selected_indices]
        excluded = [c for i, c in enumerate(candidates) if i not in selected_indices]

        return selected, excluded

    def _calculate_cost(self, token_count: int) -> float:
        """Calculate estimated cost for the given token count."""
        return (token_count / 1000) * self.model_config.cost_per_1k_input_tokens

    def _calculate_efficiency_score(
        self,
        selected: List[ContextCandidate],
        excluded: List[ContextCandidate],
        budget_utilization: float,
    ) -> float:
        """Calculate overall efficiency score."""
        if not selected:
            return 0.0

        # Average relevance of selected contexts
        avg_relevance = sum(c.relevance_score for c in selected) / len(selected)

        # Budget utilization score (prefer high utilization)
        utilization_score = min(
            1.0, budget_utilization * 1.2
        )  # Bonus for high utilization

        # Penalty for high-value excluded contexts
        if excluded:
            avg_excluded_relevance = sum(c.relevance_score for c in excluded) / len(
                excluded
            )
            exclusion_penalty = avg_excluded_relevance * 0.3
        else:
            exclusion_penalty = 0.0

        efficiency = (
            (avg_relevance * 0.6) + (utilization_score * 0.4) - exclusion_penalty
        )
        return max(0.0, min(1.0, efficiency))

    def _generate_recommendations(
        self,
        selected: List[ContextCandidate],
        excluded: List[ContextCandidate],
        budget: TokenBudget,
        strategy: OptimizationStrategy,
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if not selected:
            recommendations.append(
                "No contexts fit within token budget - consider increasing budget"
            )
            return recommendations

        budget_utilization = (
            sum(c.token_count for c in selected) / budget.context_tokens_available
        )

        # Budget utilization recommendations
        if budget_utilization < 0.5:
            recommendations.append(
                "Low budget utilization - consider including more contexts"
            )
        elif budget_utilization > 0.95:
            recommendations.append(
                "Very high budget utilization - consider optimizing for cost"
            )

        # Strategy-specific recommendations
        if strategy == OptimizationStrategy.COST_FIRST and budget_utilization < 0.8:
            recommendations.append(
                "Cost optimization successful - could include more context if quality is priority"
            )
        elif strategy == OptimizationStrategy.QUALITY_FIRST and excluded:
            high_relevance_excluded = [c for c in excluded if c.relevance_score > 0.7]
            if high_relevance_excluded:
                recommendations.append(
                    f"Consider increasing budget to include {len(high_relevance_excluded)} high-relevance contexts"
                )

        # Content quality recommendations
        avg_relevance = sum(c.relevance_score for c in selected) / len(selected)
        if avg_relevance < 0.5:
            recommendations.append(
                "Selected contexts have low relevance - consider improving query or keywords"
            )

        return recommendations

    def _update_usage_stats(
        self, strategy: OptimizationStrategy, tokens: int, budget_utilization: float
    ):
        """Update usage statistics."""
        self.usage_stats["total_optimizations"] += 1
        self.usage_stats["total_tokens_processed"] += tokens
        self.usage_stats["optimization_strategies_used"][strategy.value] += 1

        # Update rolling average
        current_avg = self.usage_stats["average_budget_utilization"]
        total_ops = self.usage_stats["total_optimizations"]
        self.usage_stats["average_budget_utilization"] = (
            current_avg * (total_ops - 1) + budget_utilization
        ) / total_ops

    def get_usage_analytics(self) -> Dict[str, Any]:
        """Get comprehensive usage analytics."""
        stats = self.usage_stats.copy()

        # Add model information
        stats["model_config"] = {
            "name": self.model_config.name,
            "family": self.model_config.family.value,
            "max_tokens": self.model_config.max_tokens,
            "cost_per_1k_input": self.model_config.cost_per_1k_input_tokens,
            "cost_per_1k_output": self.model_config.cost_per_1k_output_tokens,
        }

        # Calculate additional metrics
        if stats["total_optimizations"] > 0:
            stats["avg_tokens_per_optimization"] = (
                stats["total_tokens_processed"] / stats["total_optimizations"]
            )

            total_cost = (
                stats["total_tokens_processed"] / 1000
            ) * self.model_config.cost_per_1k_input_tokens
            stats["total_estimated_cost"] = total_cost

            # Estimate cost savings (assuming 20% reduction vs naive approach)
            naive_cost = total_cost * 1.2
            stats["estimated_cost_savings"] = naive_cost - total_cost

        return stats

    def suggest_model_upgrade(self, current_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest model upgrades based on usage patterns."""
        suggestions = []

        avg_tokens = current_usage.get("avg_tokens_per_optimization", 0)
        total_cost = current_usage.get("total_estimated_cost", 0)

        # Check if current model is suitable
        if avg_tokens > self.model_config.max_tokens * 0.8:
            suggestions.append(
                {
                    "type": "capacity_upgrade",
                    "recommendation": "Consider upgrading to a model with larger context window",
                    "reason": f"Average usage ({avg_tokens:,.0f} tokens) is close to model limit",
                    "suggested_models": [
                        name
                        for name, config in self.MODEL_CONFIGS.items()
                        if config.max_tokens > self.model_config.max_tokens
                    ],
                }
            )

        # Cost optimization suggestions
        cheaper_models = [
            (name, config)
            for name, config in self.MODEL_CONFIGS.items()
            if config.cost_per_1k_input_tokens
            < self.model_config.cost_per_1k_input_tokens
            and config.max_tokens >= avg_tokens * 1.2  # Ensure sufficient capacity
        ]

        if cheaper_models and total_cost > 100:  # Only suggest if significant cost
            best_alternative = min(
                cheaper_models, key=lambda x: x[1].cost_per_1k_input_tokens
            )
            potential_savings = (
                (
                    self.model_config.cost_per_1k_input_tokens
                    - best_alternative[1].cost_per_1k_input_tokens
                )
                / self.model_config.cost_per_1k_input_tokens
                * 100
            )

            suggestions.append(
                {
                    "type": "cost_optimization",
                    "recommendation": f"Consider {best_alternative[0]} for cost savings",
                    "potential_savings_percent": potential_savings,
                    "estimated_monthly_savings": total_cost * (potential_savings / 100),
                }
            )

        return {
            "current_model": self.model_config.name,
            "suggestions": suggestions,
            "analysis_date": time.time(),
        }


def create_token_manager(
    model_name: str = "gemini-1.5-pro",
) -> TokenAwareContextManager:
    """
    Create a token-aware context manager for the specified model.

    Args:
        model_name: Name of the LLM model to optimize for

    Returns:
        Configured TokenAwareContextManager instance

    Example:
        >>> manager = create_token_manager("gemini-1.5-pro")
        >>> budget = manager.create_budget(target_tokens=50000)
        >>> result = manager.optimize_context_selection(contexts, budget)
    """
    return TokenAwareContextManager(model_name)


def main():
    """Demo function showing token management capabilities."""
    print("🎯 Smart Token-Aware Context Management Demo")
    print("=" * 50)

    # Create token manager
    manager = create_token_manager("gemini-1.5-pro")

    # Sample contexts of varying relevance and size
    contexts = [
        "This is a short context about AI.",
        "Artificial Intelligence is revolutionizing how we approach complex problems. "
        * 50,
        "Machine learning algorithms can process vast amounts of data efficiently. "
        * 30,
        "The future of AI includes advanced reasoning capabilities. " * 20,
        "Context management is crucial for large language model applications. " * 40,
    ]

    # Create budget
    budget = manager.create_budget(target_tokens=5000)

    # Test different optimization strategies
    strategies = [
        OptimizationStrategy.COST_FIRST,
        OptimizationStrategy.QUALITY_FIRST,
        OptimizationStrategy.BALANCED,
    ]

    for strategy in strategies:
        print(f"\n📊 Testing {strategy.value} strategy:")

        result = manager.optimize_context_selection(
            contexts=contexts,
            budget=budget,
            strategy=strategy,
            query="AI and machine learning applications",
        )

        print(f"   Selected contexts: {len(result.selected_contexts)}")
        print(f"   Total tokens: {result.total_tokens:,}")
        print(f"   Budget utilization: {result.budget_utilization:.1%}")
        print(f"   Estimated cost: ${result.estimated_cost:.4f}")
        print(f"   Efficiency score: {result.efficiency_score:.2f}")

    # Show analytics
    analytics = manager.get_usage_analytics()
    print(f"\n📈 Usage Analytics:")
    print(f"   Total optimizations: {analytics['total_optimizations']}")
    print(
        f"   Average budget utilization: {analytics['average_budget_utilization']:.1%}"
    )


if __name__ == "__main__":
    main()
