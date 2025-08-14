"""
Semantic Context Clustering & Deduplication

Advanced semantic analysis for intelligent context management using embeddings,
clustering algorithms, and similarity detection to optimize context storage
and retrieval beyond traditional hash-based deduplication.

Features:
- Semantic similarity detection using embeddings
- Intelligent context clustering and grouping
- Content deduplication beyond exact matches
- Semantic search and retrieval
- Context relationship mapping
- Quality-aware merging strategies
"""

import logging
import hashlib
import json
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SimilarityMethod(Enum):
    """Available similarity calculation methods."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    JACCARD = "jaccard"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


class ClusteringAlgorithm(Enum):
    """Available clustering algorithms."""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    SEMANTIC_GROUPS = "semantic_groups"


class MergeStrategy(Enum):
    """Strategies for merging similar contexts."""

    KEEP_LONGEST = "keep_longest"
    KEEP_HIGHEST_QUALITY = "keep_highest_quality"
    MERGE_CONTENT = "merge_content"
    KEEP_MOST_RECENT = "keep_most_recent"
    REFERENCE_ORIGINAL = "reference_original"


@dataclass
class SemanticMatch:
    """Represents a semantic match between contexts."""

    context_id_1: str
    context_id_2: str
    similarity_score: float
    method_used: SimilarityMethod
    confidence: float
    match_reasons: List[str]
    suggested_action: str


@dataclass
class ContextCluster:
    """Represents a cluster of semantically similar contexts."""

    cluster_id: str
    context_ids: List[str]
    centroid_embedding: Optional[np.ndarray] = None
    representative_context_id: str = ""
    cluster_summary: str = ""
    semantic_theme: str = ""
    quality_score: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class SemanticAnalysisResult:
    """Result of semantic analysis operation."""

    total_contexts_analyzed: int
    duplicates_found: int
    clusters_created: int
    similarity_matches: List[SemanticMatch]
    clusters: List[ContextCluster]
    space_savings_potential: float
    quality_improvement_potential: float
    processing_time_ms: float
    recommendations: List[str]


class FallbackSemanticAnalyzer:
    """Fallback semantic analyzer using basic text analysis."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def encode_text(self, text: str) -> np.ndarray:
        """Create a basic feature vector from text."""
        if not NUMPY_AVAILABLE:
            # Return dummy embedding if numpy not available
            return [0.0] * 50

        # Basic text features
        features = []

        # Length features
        features.append(len(text) / 10000)  # Normalized length
        features.append(len(text.split()) / 1000)  # Word count
        features.append(len(set(text.split())) / 1000)  # Unique words

        # Character-based features
        features.append(text.count(".") / len(text) if text else 0)  # Sentence density
        features.append(text.count("\n") / len(text) if text else 0)  # Line breaks
        features.append(text.count(" ") / len(text) if text else 0)  # Space density

        # Content type indicators
        features.append(1.0 if "{" in text and "}" in text else 0.0)  # JSON-like
        features.append(1.0 if "<" in text and ">" in text else 0.0)  # HTML/XML-like
        features.append(1.0 if "def " in text or "class " in text else 0.0)  # Code-like

        # Keywords density (basic)
        keywords = ["the", "and", "or", "but", "data", "system", "user", "function"]
        for keyword in keywords:
            density = text.lower().count(keyword) / len(text.split()) if text else 0
            features.append(min(density, 1.0))

        # Pad to fixed size
        while len(features) < 50:
            features.append(0.0)

        return np.array(features[:50])

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        if not NUMPY_AVAILABLE:
            return 0.5  # Default similarity

        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class SemanticContextAnalyzer:
    """
    Advanced semantic analyzer for context clustering and deduplication.

    Uses state-of-the-art embedding models and clustering algorithms to identify
    semantically similar contexts and optimize storage efficiency.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.DBSCAN,
        enable_fallback: bool = True,
    ):
        """
        Initialize the semantic analyzer.

        Args:
            embedding_model: Name of the sentence transformer model to use
            similarity_threshold: Threshold for considering contexts similar
            clustering_algorithm: Algorithm to use for clustering
            enable_fallback: Whether to enable fallback analysis if libraries are unavailable
        """
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = similarity_threshold
        self.clustering_algorithm = clustering_algorithm
        self.enable_fallback = enable_fallback

        # Initialize embedding model
        self.embedding_model = None
        self.fallback_analyzer = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.logger.info(f"Initialized embedding model: {embedding_model}")
            except Exception as e:
                self.logger.warning(f"Failed to load embedding model: {e}")
                if enable_fallback:
                    self.fallback_analyzer = FallbackSemanticAnalyzer()
        elif enable_fallback:
            self.fallback_analyzer = FallbackSemanticAnalyzer()
            self.logger.info("Using fallback semantic analyzer")

        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}

        # Analysis statistics
        self.analysis_stats = {
            "total_analyses": 0,
            "contexts_processed": 0,
            "embeddings_computed": 0,
            "cache_hits": 0,
            "duplicates_found": 0,
            "clusters_created": 0,
        }

    def encode_context(self, content: str) -> np.ndarray:
        """
        Generate embedding for context content.

        Args:
            content: Context content to encode

        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.embedding_cache:
            self.analysis_stats["cache_hits"] += 1
            return self.embedding_cache[content_hash]

        # Generate embedding
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode([content])[0]
                self.embedding_cache[content_hash] = embedding
                self.analysis_stats["embeddings_computed"] += 1
                return embedding
            except Exception as e:
                self.logger.warning(f"Embedding generation failed: {e}")

        # Fallback to basic analysis
        if self.fallback_analyzer:
            embedding = self.fallback_analyzer.encode_text(content)
            self.embedding_cache[content_hash] = embedding
            self.analysis_stats["embeddings_computed"] += 1
            return embedding

        # Return zero vector if no methods available
        if NUMPY_AVAILABLE:
            return np.zeros(384)  # Default embedding size
        else:
            return [0.0] * 384

    def calculate_similarity(
        self,
        content1: str,
        content2: str,
        method: SimilarityMethod = SimilarityMethod.EMBEDDING,
    ) -> float:
        """
        Calculate similarity between two contexts.

        Args:
            content1: First context content
            content2: Second context content
            method: Similarity calculation method

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if method == SimilarityMethod.EMBEDDING:
            return self._calculate_embedding_similarity(content1, content2)
        elif method == SimilarityMethod.COSINE:
            return self._calculate_text_cosine_similarity(content1, content2)
        elif method == SimilarityMethod.JACCARD:
            return self._calculate_jaccard_similarity(content1, content2)
        elif method == SimilarityMethod.EUCLIDEAN:
            return self._calculate_euclidean_similarity(content1, content2)
        elif method == SimilarityMethod.HYBRID:
            return self._calculate_hybrid_similarity(content1, content2)
        else:
            return self._calculate_embedding_similarity(content1, content2)

    def _calculate_embedding_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity using embeddings."""
        embedding1 = self.encode_context(content1)
        embedding2 = self.encode_context(content2)

        if self.embedding_model or self.fallback_analyzer:
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                return float(similarity)
            elif NUMPY_AVAILABLE:
                # Manual cosine similarity calculation
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)

                if norm1 == 0 or norm2 == 0:
                    return 0.0

                return float(dot_product / (norm1 * norm2))

        return 0.5  # Default similarity

    def _calculate_text_cosine_similarity(self, content1: str, content2: str) -> float:
        """Calculate text-based cosine similarity."""
        # Simple word-based cosine similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union_size = math.sqrt(len(words1)) * math.sqrt(len(words2))

        return intersection / union_size if union_size > 0 else 0.0

    def _calculate_jaccard_similarity(self, content1: str, content2: str) -> float:
        """Calculate Jaccard similarity between word sets."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_euclidean_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity based on euclidean distance of embeddings."""
        embedding1 = self.encode_context(content1)
        embedding2 = self.encode_context(content2)

        if NUMPY_AVAILABLE:
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert distance to similarity (0-1 range)
            max_distance = math.sqrt(len(embedding1))  # Maximum possible distance
            similarity = max(0.0, 1.0 - (distance / max_distance))
            return similarity

        return 0.5

    def _calculate_hybrid_similarity(self, content1: str, content2: str) -> float:
        """Calculate hybrid similarity using multiple methods."""
        embedding_sim = self._calculate_embedding_similarity(content1, content2)
        jaccard_sim = self._calculate_jaccard_similarity(content1, content2)
        cosine_sim = self._calculate_text_cosine_similarity(content1, content2)

        # Weighted combination
        hybrid_score = embedding_sim * 0.6 + jaccard_sim * 0.25 + cosine_sim * 0.15

        return hybrid_score

    def find_semantic_duplicates(
        self,
        contexts: Dict[str, str],
        method: SimilarityMethod = SimilarityMethod.EMBEDDING,
    ) -> List[SemanticMatch]:
        """
        Find semantically similar contexts that could be duplicates.

        Args:
            contexts: Dictionary of context_id -> content
            method: Similarity calculation method to use

        Returns:
            List of semantic matches found
        """
        start_time = time.time()
        matches = []
        context_items = list(contexts.items())

        self.logger.info(
            f"Analyzing {len(context_items)} contexts for semantic duplicates"
        )

        # Compare all pairs
        for i in range(len(context_items)):
            for j in range(i + 1, len(context_items)):
                context_id_1, content_1 = context_items[i]
                context_id_2, content_2 = context_items[j]

                # Skip if contents are identical (handled by hash-based deduplication)
                if content_1 == content_2:
                    continue

                similarity = self.calculate_similarity(content_1, content_2, method)

                if similarity >= self.similarity_threshold:
                    confidence = min(
                        1.0, similarity * 1.2
                    )  # Boost confidence for high similarity

                    match_reasons = self._analyze_match_reasons(
                        content_1, content_2, similarity
                    )
                    suggested_action = self._suggest_merge_action(
                        content_1, content_2, similarity
                    )

                    match = SemanticMatch(
                        context_id_1=context_id_1,
                        context_id_2=context_id_2,
                        similarity_score=similarity,
                        method_used=method,
                        confidence=confidence,
                        match_reasons=match_reasons,
                        suggested_action=suggested_action,
                    )

                    matches.append(match)
                    self.analysis_stats["duplicates_found"] += 1

        processing_time = (time.time() - start_time) * 1000
        self.logger.info(
            f"Found {len(matches)} semantic matches in {processing_time:.2f}ms"
        )

        return matches

    def create_semantic_clusters(
        self, contexts: Dict[str, str], num_clusters: Optional[int] = None
    ) -> List[ContextCluster]:
        """
        Create semantic clusters from contexts.

        Args:
            contexts: Dictionary of context_id -> content
            num_clusters: Number of clusters to create (auto-detected if None)

        Returns:
            List of context clusters
        """
        if not contexts:
            return []

        start_time = time.time()
        self.logger.info(f"Creating semantic clusters for {len(contexts)} contexts")

        # Generate embeddings for all contexts
        context_items = list(contexts.items())
        embeddings = []

        for context_id, content in context_items:
            embedding = self.encode_context(content)
            embeddings.append(embedding)

        if not embeddings:
            return []

        embeddings_array = np.array(embeddings) if NUMPY_AVAILABLE else embeddings

        # Perform clustering
        clusters = self._perform_clustering(embeddings_array, num_clusters)

        # Create cluster objects
        context_clusters = []

        for cluster_id, context_indices in clusters.items():
            if not context_indices:
                continue

            cluster_context_ids = [context_items[i][0] for i in context_indices]
            cluster_contents = [context_items[i][1] for i in context_indices]

            # Calculate centroid
            if NUMPY_AVAILABLE and embeddings_array.size > 0:
                cluster_embeddings = embeddings_array[context_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
            else:
                centroid = None

            # Find representative context (longest or highest quality)
            representative_idx = max(
                context_indices, key=lambda i: len(context_items[i][1])
            )
            representative_context_id = context_items[representative_idx][0]

            # Generate cluster summary
            cluster_summary = self._generate_cluster_summary(cluster_contents)
            semantic_theme = self._extract_semantic_theme(cluster_contents)
            quality_score = self._calculate_cluster_quality(cluster_contents)

            cluster = ContextCluster(
                cluster_id=f"cluster_{cluster_id}",
                context_ids=cluster_context_ids,
                centroid_embedding=centroid,
                representative_context_id=representative_context_id,
                cluster_summary=cluster_summary,
                semantic_theme=semantic_theme,
                quality_score=quality_score,
            )

            context_clusters.append(cluster)

        processing_time = (time.time() - start_time) * 1000
        self.analysis_stats["clusters_created"] += len(context_clusters)

        self.logger.info(
            f"Created {len(context_clusters)} clusters in {processing_time:.2f}ms"
        )

        return context_clusters

    def _perform_clustering(
        self, embeddings: np.ndarray, num_clusters: Optional[int] = None
    ) -> Dict[int, List[int]]:
        """Perform clustering on embeddings."""
        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            # Fallback: create single cluster with all contexts
            return {0: list(range(len(embeddings)))}

        n_samples = len(embeddings)

        if n_samples <= 1:
            return {0: list(range(n_samples))}

        try:
            if self.clustering_algorithm == ClusteringAlgorithm.KMEANS:
                if num_clusters is None:
                    num_clusters = min(max(2, n_samples // 3), 10)  # Heuristic

                clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(embeddings)

            elif self.clustering_algorithm == ClusteringAlgorithm.DBSCAN:
                # Use DBSCAN for automatic cluster detection
                clusterer = DBSCAN(eps=0.3, min_samples=2)
                labels = clusterer.fit_predict(embeddings)

            elif self.clustering_algorithm == ClusteringAlgorithm.HIERARCHICAL:
                if num_clusters is None:
                    num_clusters = min(max(2, n_samples // 4), 8)

                clusterer = AgglomerativeClustering(n_clusters=num_clusters)
                labels = clusterer.fit_predict(embeddings)

            else:  # SEMANTIC_GROUPS
                # Custom semantic grouping based on similarity
                labels = self._semantic_grouping(embeddings)

            # Group indices by cluster label
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                if label != -1:  # -1 is noise in DBSCAN
                    clusters[label].append(idx)

            return dict(clusters)

        except Exception as e:
            self.logger.warning(f"Clustering failed: {e}")
            # Fallback: single cluster
            return {0: list(range(n_samples))}

    def _semantic_grouping(self, embeddings: np.ndarray) -> np.ndarray:
        """Custom semantic grouping algorithm."""
        n_samples = len(embeddings)
        labels = np.arange(n_samples)  # Start with each item in its own cluster

        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)

        # Merge highly similar contexts
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if similarities[i][j] >= self.similarity_threshold:
                    # Merge clusters
                    labels[labels == labels[j]] = labels[i]

        # Renumber labels to be consecutive
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[label] for label in labels])

    def _analyze_match_reasons(
        self, content1: str, content2: str, similarity: float
    ) -> List[str]:
        """Analyze why two contexts were considered similar."""
        reasons = []

        # Length similarity
        len_ratio = min(len(content1), len(content2)) / max(
            len(content1), len(content2)
        )
        if len_ratio > 0.8:
            reasons.append(f"Similar length (ratio: {len_ratio:.2f})")

        # Word overlap
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if words1 and words2:
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            if overlap > 0.5:
                reasons.append(f"High word overlap ({overlap:.1%})")

        # Common patterns
        if self._has_common_patterns(content1, content2):
            reasons.append("Similar structural patterns detected")

        # High semantic similarity
        if similarity > 0.9:
            reasons.append("Very high semantic similarity")
        elif similarity > 0.85:
            reasons.append("High semantic similarity")

        return reasons if reasons else ["Semantic similarity detected"]

    def _has_common_patterns(self, content1: str, content2: str) -> bool:
        """Check if two contents have common structural patterns."""
        patterns = [
            r"\{.*\}",  # JSON-like
            r"<.*>",  # HTML/XML-like
            r"def\s+\w+",  # Function definitions
            r"class\s+\w+",  # Class definitions
            r"^\d+\.",  # Numbered lists
            r"^[*-]\s+",  # Bullet lists
        ]

        for pattern in patterns:
            if re.search(pattern, content1) and re.search(pattern, content2):
                return True

        return False

    def _suggest_merge_action(
        self, content1: str, content2: str, similarity: float
    ) -> str:
        """Suggest the best merge action for similar contexts."""
        if similarity > 0.95:
            return "MERGE_OR_DEDUPLICATE"
        elif similarity > 0.9:
            return "CONSIDER_MERGE"
        elif similarity > 0.85:
            return "REFERENCE_SIMILAR"
        else:
            return "MONITOR"

    def _generate_cluster_summary(self, contents: List[str]) -> str:
        """Generate a summary for a cluster of contexts."""
        if not contents:
            return "Empty cluster"

        # Use the shortest representative content as summary
        representative = min(contents, key=len)

        # Truncate if too long
        if len(representative) > 200:
            return representative[:200] + "..."

        return representative

    def _extract_semantic_theme(self, contents: List[str]) -> str:
        """Extract the semantic theme of a cluster."""
        # Simple keyword extraction
        all_words = []
        for content in contents:
            words = [
                word.lower()
                for word in content.split()
                if len(word) > 3 and word.isalpha()
            ]
            all_words.extend(words)

        # Count word frequencies
        word_counts = defaultdict(int)
        for word in all_words:
            word_counts[word] += 1

        # Get top keywords
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        if top_words:
            theme_words = [word for word, count in top_words]
            return ", ".join(theme_words)

        return "Mixed content"

    def _calculate_cluster_quality(self, contents: List[str]) -> float:
        """Calculate quality score for a cluster."""
        if not contents:
            return 0.0

        # Factors affecting quality
        scores = []

        # Content diversity (lower is better for clusters)
        lengths = [len(content) for content in contents]
        if lengths:
            length_std = np.std(lengths) if NUMPY_AVAILABLE else 0
            length_consistency = max(0.0, 1.0 - (length_std / max(lengths)))
            scores.append(length_consistency)

        # Average content length (moderate length preferred)
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        length_score = (
            min(1.0, avg_length / 1000)
            if avg_length < 1000
            else max(0.3, 1000 / avg_length)
        )
        scores.append(length_score)

        # Cluster size (moderate size preferred)
        size_score = (
            min(1.0, len(contents) / 5)
            if len(contents) < 5
            else max(0.5, 5 / len(contents))
        )
        scores.append(size_score)

        return sum(scores) / len(scores) if scores else 0.5

    def analyze_contexts(self, contexts: Dict[str, str]) -> SemanticAnalysisResult:
        """
        Perform comprehensive semantic analysis on contexts.

        Args:
            contexts: Dictionary of context_id -> content

        Returns:
            Complete analysis result with matches, clusters, and recommendations
        """
        start_time = time.time()
        self.analysis_stats["total_analyses"] += 1
        self.analysis_stats["contexts_processed"] += len(contexts)

        self.logger.info(f"Starting semantic analysis of {len(contexts)} contexts")

        # Find semantic duplicates
        similarity_matches = self.find_semantic_duplicates(contexts)

        # Create clusters
        clusters = self.create_semantic_clusters(contexts)

        # Calculate space savings potential
        space_savings = self._calculate_space_savings(
            contexts, similarity_matches, clusters
        )

        # Calculate quality improvement potential
        quality_improvement = self._calculate_quality_improvement(
            similarity_matches, clusters
        )

        # Generate recommendations
        recommendations = self._generate_analysis_recommendations(
            contexts, similarity_matches, clusters, space_savings
        )

        processing_time = (time.time() - start_time) * 1000

        result = SemanticAnalysisResult(
            total_contexts_analyzed=len(contexts),
            duplicates_found=len(similarity_matches),
            clusters_created=len(clusters),
            similarity_matches=similarity_matches,
            clusters=clusters,
            space_savings_potential=space_savings,
            quality_improvement_potential=quality_improvement,
            processing_time_ms=processing_time,
            recommendations=recommendations,
        )

        self.logger.info(f"Semantic analysis completed in {processing_time:.2f}ms")

        return result

    def _calculate_space_savings(
        self,
        contexts: Dict[str, str],
        matches: List[SemanticMatch],
        clusters: List[ContextCluster],
    ) -> float:
        """Calculate potential space savings from deduplication."""
        if not contexts:
            return 0.0

        total_size = sum(len(content) for content in contexts.values())

        # Savings from duplicate removal
        duplicate_savings = 0
        processed_pairs = set()

        for match in matches:
            pair_key = tuple(sorted([match.context_id_1, match.context_id_2]))
            if pair_key not in processed_pairs and match.similarity_score > 0.9:
                # Assume we keep the longer content
                size1 = len(contexts[match.context_id_1])
                size2 = len(contexts[match.context_id_2])
                duplicate_savings += min(size1, size2)
                processed_pairs.add(pair_key)

        # Additional savings from clustering (representative contexts)
        cluster_savings = 0
        for cluster in clusters:
            if len(cluster.context_ids) > 1:
                cluster_sizes = [
                    len(contexts[ctx_id]) for ctx_id in cluster.context_ids
                ]
                # Keep representative, save the rest
                cluster_savings += sum(cluster_sizes) - max(cluster_sizes)

        total_savings = max(duplicate_savings, cluster_savings)  # Don't double-count

        return (
            min(total_savings / total_size, 0.8) if total_size > 0 else 0.0
        )  # Cap at 80%

    def _calculate_quality_improvement(
        self, matches: List[SemanticMatch], clusters: List[ContextCluster]
    ) -> float:
        """Calculate potential quality improvement from semantic organization."""
        if not matches and not clusters:
            return 0.0

        # Quality improvement from removing near-duplicates
        high_similarity_matches = [m for m in matches if m.similarity_score > 0.95]
        duplicate_improvement = min(len(high_similarity_matches) * 0.1, 0.3)

        # Quality improvement from clustering
        avg_cluster_quality = (
            sum(c.quality_score for c in clusters) / len(clusters) if clusters else 0
        )
        cluster_improvement = avg_cluster_quality * 0.2

        return min(duplicate_improvement + cluster_improvement, 0.5)

    def _generate_analysis_recommendations(
        self,
        contexts: Dict[str, str],
        matches: List[SemanticMatch],
        clusters: List[ContextCluster],
        space_savings: float,
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Duplicate-related recommendations
        high_similarity_matches = [m for m in matches if m.similarity_score > 0.95]
        if high_similarity_matches:
            recommendations.append(
                f"Consider deduplicating {len(high_similarity_matches)} pairs of highly similar contexts"
            )

        moderate_similarity_matches = [
            m for m in matches if 0.85 <= m.similarity_score < 0.95
        ]
        if moderate_similarity_matches:
            recommendations.append(
                f"Review {len(moderate_similarity_matches)} pairs of moderately similar contexts for potential merging"
            )

        # Clustering recommendations
        large_clusters = [c for c in clusters if len(c.context_ids) > 5]
        if large_clusters:
            recommendations.append(
                f"Optimize {len(large_clusters)} large clusters by using representative contexts"
            )

        # Space savings recommendations
        if space_savings > 0.3:
            recommendations.append(
                f"Significant space savings possible: {space_savings:.1%} reduction through semantic optimization"
            )
        elif space_savings > 0.1:
            recommendations.append(
                f"Moderate space savings available: {space_savings:.1%} reduction through deduplication"
            )

        # Performance recommendations
        if len(contexts) > 100:
            recommendations.append(
                "Consider implementing semantic search indices for improved retrieval performance"
            )

        # Configuration recommendations
        if not self.embedding_model:
            recommendations.append(
                "Install sentence-transformers for improved semantic analysis: pip install sentence-transformers"
            )

        return (
            recommendations
            if recommendations
            else ["No immediate optimizations identified"]
        )

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics."""
        stats = self.analysis_stats.copy()

        # Add configuration information
        stats["configuration"] = {
            "embedding_model_available": self.embedding_model is not None,
            "fallback_analyzer_enabled": self.fallback_analyzer is not None,
            "similarity_threshold": self.similarity_threshold,
            "clustering_algorithm": self.clustering_algorithm.value,
            "libraries_available": {
                "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE,
                "numpy": NUMPY_AVAILABLE,
            },
        }

        # Calculate derived metrics
        if stats["total_analyses"] > 0:
            stats["avg_contexts_per_analysis"] = (
                stats["contexts_processed"] / stats["total_analyses"]
            )
            stats["avg_duplicates_per_analysis"] = (
                stats["duplicates_found"] / stats["total_analyses"]
            )
            stats["avg_clusters_per_analysis"] = (
                stats["clusters_created"] / stats["total_analyses"]
            )

        if stats["embeddings_computed"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / (
                stats["cache_hits"] + stats["embeddings_computed"]
            )

        return stats


def create_semantic_analyzer(
    embedding_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.85,
    clustering_algorithm: str = "dbscan",
) -> SemanticContextAnalyzer:
    """
    Create a semantic context analyzer with specified configuration.

    Args:
        embedding_model: Name of the sentence transformer model
        similarity_threshold: Threshold for considering contexts similar
        clustering_algorithm: Algorithm to use for clustering

    Returns:
        Configured SemanticContextAnalyzer instance
    """
    algorithm_map = {
        "kmeans": ClusteringAlgorithm.KMEANS,
        "dbscan": ClusteringAlgorithm.DBSCAN,
        "hierarchical": ClusteringAlgorithm.HIERARCHICAL,
        "semantic_groups": ClusteringAlgorithm.SEMANTIC_GROUPS,
    }

    algorithm = algorithm_map.get(clustering_algorithm, ClusteringAlgorithm.DBSCAN)

    return SemanticContextAnalyzer(
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        clustering_algorithm=algorithm,
    )


def main():
    """Demo function showing semantic analysis capabilities."""
    print("🧠 Semantic Context Clustering & Deduplication Demo")
    print("=" * 55)

    # Sample contexts with semantic similarities
    contexts = {
        "ctx_1": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "ctx_2": "AI and machine learning are transforming how we process data and make decisions.",
        "ctx_3": "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "ctx_4": "Neural networks are computational models inspired by biological neural systems.",
        "ctx_5": "Data science combines statistics, programming, and domain expertise for insights.",
        "ctx_6": "Statistical analysis and programming skills are essential for data scientists.",
        "ctx_7": "Python is a popular programming language for machine learning and data analysis.",
        "ctx_8": "Python programming language is widely used in AI and data science applications.",
    }

    # Create semantic analyzer
    analyzer = create_semantic_analyzer(
        similarity_threshold=0.7, clustering_algorithm="dbscan"
    )

    # Perform analysis
    result = analyzer.analyze_contexts(contexts)

    print(f"\n📊 Analysis Results:")
    print(f"   • Contexts analyzed: {result.total_contexts_analyzed}")
    print(f"   • Semantic duplicates found: {result.duplicates_found}")
    print(f"   • Clusters created: {result.clusters_created}")
    print(f"   • Space savings potential: {result.space_savings_potential:.1%}")
    print(f"   • Processing time: {result.processing_time_ms:.1f}ms")

    # Show similarity matches
    if result.similarity_matches:
        print(f"\n🔍 Semantic Matches:")
        for match in result.similarity_matches[:3]:  # Show first 3
            print(f"   • {match.context_id_1} ↔ {match.context_id_2}")
            print(f"     Similarity: {match.similarity_score:.2f}")
            print(f"     Suggestion: {match.suggested_action}")

    # Show clusters
    if result.clusters:
        print(f"\n🗂️  Semantic Clusters:")
        for cluster in result.clusters[:3]:  # Show first 3
            print(f"   • {cluster.cluster_id}: {len(cluster.context_ids)} contexts")
            print(f"     Theme: {cluster.semantic_theme}")
            print(f"     Quality: {cluster.quality_score:.2f}")

    # Show statistics
    stats = analyzer.get_analysis_statistics()
    print(f"\n📈 Analysis Statistics:")
    print(f"   • Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
    print(f"   • Embeddings computed: {stats['embeddings_computed']}")


if __name__ == "__main__":
    main()
