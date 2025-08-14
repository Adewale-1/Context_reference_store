# LlamaIndex Integration

This guide demonstrates how to integrate Context Reference Store with LlamaIndex for building efficient document-centric AI applications with enhanced retrieval and indexing capabilities.

## Table of Contents

- [LlamaIndex Integration](#llamaindex-integration)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Basic LlamaIndex Integration](#basic-llamaindex-integration)
  - [Document Management](#document-management)
  - [Enhanced Query Engines](#enhanced-query-engines)
  - [Vector Store Integration](#vector-store-integration)
  - [Index Management](#index-management)
  - [Retrieval Optimization](#retrieval-optimization)
  - [Multi-Index Systems](#multi-index-systems)
  - [Performance Optimization](#performance-optimization)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)

## Overview

LlamaIndex integration with Context Reference Store provides:

- **Efficient Document Storage**: 99.8% storage reduction for large document collections
- **Fast Index Operations**: Accelerated document indexing and retrieval
- **Smart Query Caching**: Intelligent caching of query results and embeddings
- **Scalable Vector Storage**: Efficient management of high-dimensional embeddings
- **Context-Aware Retrieval**: Enhanced retrieval with conversation and session context

## Installation

```bash
# Install with LlamaIndex support
pip install context-reference-store[llamaindex]

# Or install specific components
pip install context-reference-store llama-index
```

## Basic LlamaIndex Integration

### LlamaIndex Adapter Setup

```python
from context_store.adapters import LlamaIndexAdapter
from context_store import ContextReferenceStore
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.embeddings import OpenAIEmbedding
import time

# Initialize context store and adapter
context_store = ContextReferenceStore(
    cache_size=5000,
    use_compression=True,
    eviction_policy="LRU"
)

llamaindex_adapter = LlamaIndexAdapter(context_store)

# Create documents
documents = [
    Document(
        text="Artificial Intelligence is transforming various industries...",
        metadata={"source": "ai_overview.pdf", "page": 1}
    ),
    Document(
        text="Machine Learning is a subset of AI that focuses on algorithms...",
        metadata={"source": "ml_guide.pdf", "page": 1}
    ),
    Document(
        text="Deep Learning uses neural networks with multiple layers...",
        metadata={"source": "dl_tutorial.pdf", "page": 1}
    )
]

# Store documents with context management
collection_id = llamaindex_adapter.store_document_collection(
    documents=documents,
    collection_name="ai_knowledge_base",
    metadata={
        "created_at": time.time(),
        "domain": "artificial_intelligence",
        "document_count": len(documents)
    }
)

print(f"Stored {len(documents)} documents in collection {collection_id}")

# Create index with context integration
index = llamaindex_adapter.create_context_aware_index(
    collection_id=collection_id,
    index_type="vector",
    embedding_model="text-embedding-ada-002"
)

# Query with context awareness
query_engine = index.as_query_engine()
response = query_engine.query("What is machine learning?")
print(f"Response: {response}")
```

### Enhanced Document Processing

```python
from llama_index.core.schema import NodeWithScore
from typing import List, Dict, Any, Optional

class ContextAwareDocumentProcessor:
    """Enhanced document processor with context store integration."""

    def __init__(self, llamaindex_adapter: LlamaIndexAdapter):
        self.adapter = llamaindex_adapter
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=1024,
            chunk_overlap=200
        )
        self.processing_history = {}

    def process_documents(self, documents: List[Document], processing_config: Dict[str, Any]) -> str:
        """Process documents with enhanced context tracking."""

        processing_session_id = f"processing_{int(time.time())}"

        # Store processing context
        processing_context = {
            "session_id": processing_session_id,
            "document_count": len(documents),
            "config": processing_config,
            "started_at": time.time(),
            "chunks_created": 0,
            "embeddings_generated": 0
        }

        processing_context_id = self.adapter.store_processing_context(
            processing_context,
            context_type="document_processing"
        )

        # Parse documents into nodes
        all_nodes = []
        for doc_idx, document in enumerate(documents):
            # Parse document into chunks
            nodes = self.node_parser.get_nodes_from_documents([document])

            # Enhance nodes with context
            for node_idx, node in enumerate(nodes):
                # Store node context
                node_context_id = self.adapter.store_node_context(
                    node_data={
                        "text": node.text,
                        "metadata": node.metadata,
                        "document_index": doc_idx,
                        "node_index": node_idx,
                        "processing_session": processing_session_id
                    },
                    parent_document_id=document.doc_id,
                    processing_context_id=processing_context_id
                )

                # Add context reference to node
                node.metadata["context_reference_id"] = node_context_id
                all_nodes.append(node)

        # Update processing context
        processing_context["chunks_created"] = len(all_nodes)
        processing_context["completed_at"] = time.time()
        processing_context["processing_duration"] = processing_context["completed_at"] - processing_context["started_at"]

        # Store final processing context
        final_context_id = self.adapter.store_processing_context(
            processing_context,
            context_type="document_processing_complete"
        )

        # Store processed nodes collection
        collection_id = self.adapter.store_node_collection(
            nodes=all_nodes,
            collection_metadata={
                "processing_session_id": processing_session_id,
                "processing_context_id": final_context_id,
                "total_nodes": len(all_nodes),
                "source_documents": len(documents)
            }
        )

        self.processing_history[processing_session_id] = {
            "collection_id": collection_id,
            "processing_context_id": final_context_id,
            "node_count": len(all_nodes)
        }

        return collection_id

    def get_processing_stats(self, processing_session_id: str) -> Dict[str, Any]:
        """Get statistics for a processing session."""

        if processing_session_id not in self.processing_history:
            return {}

        session_info = self.processing_history[processing_session_id]
        processing_context = self.adapter.retrieve_processing_context(
            session_info["processing_context_id"]
        )

        return {
            "session_id": processing_session_id,
            "documents_processed": processing_context.get("document_count", 0),
            "chunks_created": processing_context.get("chunks_created", 0),
            "processing_duration": processing_context.get("processing_duration", 0),
            "collection_id": session_info["collection_id"]
        }

# Usage example
processor = ContextAwareDocumentProcessor(llamaindex_adapter)

# Process documents with configuration
processing_config = {
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "generate_embeddings": True,
    "extract_metadata": True
}

collection_id = processor.process_documents(documents, processing_config)
print(f"Document processing completed: {collection_id}")
```

## Document Management

### Advanced Document Storage

```python
class ContextAwareDocumentStore:
    """Enhanced document store with context management."""

    def __init__(self, llamaindex_adapter: LlamaIndexAdapter):
        self.adapter = llamaindex_adapter
        self.document_collections = {}
        self.metadata_index = {}

    def create_document_collection(self, collection_name: str, description: str = "") -> str:
        """Create a new document collection."""

        collection_metadata = {
            "name": collection_name,
            "description": description,
            "created_at": time.time(),
            "document_count": 0,
            "total_size_bytes": 0,
            "last_updated": time.time()
        }

        collection_id = self.adapter.store_collection_metadata(
            collection_metadata,
            collection_type="document_collection"
        )

        self.document_collections[collection_name] = collection_id

        return collection_id

    def add_documents_to_collection(self, collection_name: str, documents: List[Document],
                                   processing_options: Dict[str, Any] = None) -> List[str]:
        """Add documents to a collection with context tracking."""

        if collection_name not in self.document_collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection_id = self.document_collections[collection_name]
        processing_options = processing_options or {}

        # Store documents with enhanced metadata
        document_ids = []
        total_size = 0

        for doc in documents:
            # Calculate document size
            doc_size = len(doc.text.encode('utf-8'))
            total_size += doc_size

            # Enhance document metadata
            enhanced_metadata = {
                **doc.metadata,
                "collection_id": collection_id,
                "collection_name": collection_name,
                "added_at": time.time(),
                "size_bytes": doc_size,
                "processing_options": processing_options
            }

            # Store document with context
            doc_context_id = self.adapter.store_document_with_context(
                document_text=doc.text,
                document_metadata=enhanced_metadata,
                collection_id=collection_id
            )

            document_ids.append(doc_context_id)

            # Update metadata index
            doc_id = doc.doc_id or doc_context_id
            self.metadata_index[doc_id] = {
                "context_id": doc_context_id,
                "collection_id": collection_id,
                "collection_name": collection_name,
                "size_bytes": doc_size
            }

        # Update collection metadata
        collection_metadata = self.adapter.retrieve_collection_metadata(collection_id)
        collection_metadata["document_count"] += len(documents)
        collection_metadata["total_size_bytes"] += total_size
        collection_metadata["last_updated"] = time.time()

        self.adapter.update_collection_metadata(collection_id, collection_metadata)

        return document_ids

    def query_collection(self, collection_name: str, query: str,
                        query_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query documents in a collection with context awareness."""

        if collection_name not in self.document_collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection_id = self.document_collections[collection_name]
        query_options = query_options or {}

        # Store query context
        query_context = {
            "query": query,
            "collection_id": collection_id,
            "collection_name": collection_name,
            "query_options": query_options,
            "timestamp": time.time()
        }

        query_context_id = self.adapter.store_query_context(query_context)

        # Get collection documents
        collection_documents = self.adapter.get_collection_documents(collection_id)

        # Perform semantic search (simplified implementation)
        # In practice, this would use proper embeddings and similarity search
        relevant_docs = []
        for doc_context_id in collection_documents:
            doc_data = self.adapter.retrieve_document_context(doc_context_id)

            # Simple keyword matching (replace with proper semantic search)
            if any(word.lower() in doc_data["text"].lower() for word in query.split()):
                relevant_docs.append({
                    "document_id": doc_context_id,
                    "text": doc_data["text"][:500] + "...",  # Truncated preview
                    "metadata": doc_data["metadata"],
                    "relevance_score": 0.8  # Placeholder score
                })

        # Store query results
        query_results = {
            "query": query,
            "collection_name": collection_name,
            "results_count": len(relevant_docs),
            "results": relevant_docs,
            "query_context_id": query_context_id,
            "executed_at": time.time()
        }

        self.adapter.store_query_results(query_context_id, query_results)

        return query_results

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a document collection."""

        if collection_name not in self.document_collections:
            return {}

        collection_id = self.document_collections[collection_name]
        collection_metadata = self.adapter.retrieve_collection_metadata(collection_id)

        # Get additional statistics
        query_history = self.adapter.get_collection_query_history(collection_id)

        return {
            **collection_metadata,
            "query_count": len(query_history),
            "avg_query_frequency": len(query_history) / max(1, (time.time() - collection_metadata["created_at"]) / 3600),  # queries per hour
            "collection_id": collection_id
        }

# Usage example
doc_store = ContextAwareDocumentStore(llamaindex_adapter)

# Create collection
collection_id = doc_store.create_document_collection(
    "technical_docs",
    "Technical documentation and guides"
)

# Add documents
doc_ids = doc_store.add_documents_to_collection(
    "technical_docs",
    documents,
    processing_options={"extract_code_blocks": True}
)

# Query collection
results = doc_store.query_collection(
    "technical_docs",
    "machine learning algorithms",
    query_options={"max_results": 5}
)

print(f"Found {results['results_count']} relevant documents")
```

## Enhanced Query Engines

### Context-Aware Query Engine

```python
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response

class ContextAwareQueryEngine(BaseQueryEngine):
    """Query engine with context store integration."""

    def __init__(self, llamaindex_adapter: LlamaIndexAdapter, base_query_engine,
                 context_config: Dict[str, Any] = None):
        self.adapter = llamaindex_adapter
        self.base_engine = base_query_engine
        self.context_config = context_config or {}
        self.query_history = []

    def _query(self, query_bundle) -> Response:
        """Execute query with context enhancement."""

        query_str = str(query_bundle.query_str)

        # Get query context
        query_context = self.get_query_context(query_str)

        # Enhance query with context
        enhanced_query = self.enhance_query_with_context(query_str, query_context)

        # Store query execution context
        execution_context_id = self.adapter.store_query_execution(
            original_query=query_str,
            enhanced_query=enhanced_query,
            query_context=query_context,
            execution_config=self.context_config
        )

        # Execute base query
        start_time = time.time()
        response = self.base_engine.query(enhanced_query)
        execution_time = time.time() - start_time

        # Store query result with context
        result_context = {
            "original_query": query_str,
            "enhanced_query": enhanced_query,
            "response": str(response),
            "execution_time_ms": execution_time * 1000,
            "source_nodes_count": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0,
            "execution_context_id": execution_context_id
        }

        result_context_id = self.adapter.store_query_result_context(
            result_context,
            query_execution_id=execution_context_id
        )

        # Update query history
        self.query_history.append({
            "query": query_str,
            "execution_context_id": execution_context_id,
            "result_context_id": result_context_id,
            "timestamp": time.time()
        })

        # Enhance response with context metadata
        if hasattr(response, 'metadata'):
            response.metadata.update({
                "context_enhanced": True,
                "execution_context_id": execution_context_id,
                "result_context_id": result_context_id
            })

        return response

    def get_query_context(self, query: str) -> Dict[str, Any]:
        """Get relevant context for query enhancement."""

        # Get similar previous queries
        similar_queries = self.adapter.find_similar_queries(
            query=query,
            similarity_threshold=0.7,
            limit=5
        )

        # Get domain-specific context
        domain_context = self.adapter.get_domain_context(
            query=query,
            domain_categories=self.context_config.get("domains", [])
        )

        # Get temporal context (recent queries, trending topics)
        temporal_context = self.adapter.get_temporal_context(
            time_window_hours=self.context_config.get("temporal_window", 24)
        )

        return {
            "similar_queries": similar_queries,
            "domain_context": domain_context,
            "temporal_context": temporal_context,
            "query_history_count": len(self.query_history)
        }

    def enhance_query_with_context(self, original_query: str, context: Dict[str, Any]) -> str:
        """Enhance query using available context."""

        enhanced_parts = [original_query]

        # Add context from similar queries
        if context.get("similar_queries"):
            similar_contexts = [sq["context"] for sq in context["similar_queries"][:2]]
            if similar_contexts:
                enhanced_parts.append(f"Related context: {' '.join(similar_contexts)}")

        # Add domain context
        if context.get("domain_context"):
            domain_terms = context["domain_context"].get("key_terms", [])
            if domain_terms:
                enhanced_parts.append(f"Domain terms: {', '.join(domain_terms[:3])}")

        return " | ".join(enhanced_parts)

    def get_query_analytics(self) -> Dict[str, Any]:
        """Get analytics for query performance."""

        if not self.query_history:
            return {"total_queries": 0}

        # Analyze query patterns
        total_queries = len(self.query_history)
        recent_queries = [q for q in self.query_history if time.time() - q["timestamp"] < 3600]  # Last hour

        # Get performance metrics from stored contexts
        execution_times = []
        for query_record in self.query_history[-10:]:  # Last 10 queries
            try:
                result_context = self.adapter.retrieve_query_result_context(
                    query_record["result_context_id"]
                )
                execution_times.append(result_context["execution_time_ms"])
            except:
                continue

        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        return {
            "total_queries": total_queries,
            "recent_queries_count": len(recent_queries),
            "avg_execution_time_ms": avg_execution_time,
            "context_enhancement_rate": 1.0,  # All queries are enhanced
            "query_frequency_per_hour": len(recent_queries)
        }

# Create context-aware query engine
base_index = VectorStoreIndex.from_documents(documents)
base_query_engine = base_index.as_query_engine()

context_config = {
    "domains": ["artificial_intelligence", "machine_learning"],
    "temporal_window": 24,
    "enable_query_expansion": True
}

context_query_engine = ContextAwareQueryEngine(
    llamaindex_adapter,
    base_query_engine,
    context_config
)

# Query with context enhancement
response = context_query_engine.query("Explain neural networks")
print(f"Context-enhanced response: {response}")

# Get analytics
analytics = context_query_engine.get_query_analytics()
print(f"Query analytics: {analytics}")
```

## Vector Store Integration

### Enhanced Vector Store

```python
from llama_index.core.vector_stores import VectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
import numpy as np

class ContextAwareVectorStore(VectorStore):
    """Vector store with context store integration for enhanced performance."""

    def __init__(self, llamaindex_adapter: LlamaIndexAdapter, base_vector_store=None):
        self.adapter = llamaindex_adapter
        self.base_store = base_vector_store
        self.vector_cache = {}
        self.embedding_cache = {}

    def add(self, nodes, **kwargs) -> List[str]:
        """Add nodes with context tracking."""

        node_ids = []
        embeddings_generated = 0

        for node in nodes:
            # Generate or retrieve embedding
            if hasattr(node, 'embedding') and node.embedding is not None:
                embedding = node.embedding
            else:
                # Generate embedding and cache it
                embedding = self.generate_embedding(node.text)
                embeddings_generated += 1

            # Store vector with context
            vector_context = {
                "node_id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "added_at": time.time(),
                "vector_dimension": len(embedding)
            }

            vector_context_id = self.adapter.store_vector_context(vector_context)

            # Store in base vector store if available
            if self.base_store:
                self.base_store.add([node], **kwargs)

            # Cache vector locally
            self.vector_cache[node.node_id] = {
                "embedding": embedding,
                "context_id": vector_context_id,
                "metadata": node.metadata
            }

            node_ids.append(node.node_id)

        # Store batch operation context
        batch_context = {
            "operation": "add_vectors",
            "node_count": len(nodes),
            "embeddings_generated": embeddings_generated,
            "node_ids": node_ids,
            "timestamp": time.time()
        }

        self.adapter.store_vector_operation_context(batch_context)

        return node_ids

    def delete(self, ref_doc_id: str, **kwargs) -> None:
        """Delete vectors with context cleanup."""

        # Get vectors for document
        doc_vectors = self.adapter.get_document_vectors(ref_doc_id)

        # Delete from base store
        if self.base_store:
            self.base_store.delete(ref_doc_id, **kwargs)

        # Remove from local cache
        for vector_info in doc_vectors:
            node_id = vector_info.get("node_id")
            if node_id in self.vector_cache:
                del self.vector_cache[node_id]

        # Store deletion context
        deletion_context = {
            "operation": "delete_vectors",
            "ref_doc_id": ref_doc_id,
            "vectors_deleted": len(doc_vectors),
            "timestamp": time.time()
        }

        self.adapter.store_vector_operation_context(deletion_context)

        # Clean up vector contexts
        self.adapter.cleanup_document_vector_contexts(ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        """Execute vector query with context enhancement."""

        query_start_time = time.time()

        # Store query context
        query_context = {
            "query_str": query.query_str if hasattr(query, 'query_str') else "",
            "query_embedding": query.query_embedding.tolist() if hasattr(query, 'query_embedding') and query.query_embedding is not None else None,
            "similarity_top_k": query.similarity_top_k,
            "mode": query.mode.value if hasattr(query, 'mode') else "default",
            "timestamp": time.time()
        }

        query_context_id = self.adapter.store_vector_query_context(query_context)

        # Execute query
        if self.base_store:
            # Use base store for actual vector search
            result = self.base_store.query(query, **kwargs)
        else:
            # Simplified vector search implementation
            result = self.simple_vector_search(query)

        query_execution_time = (time.time() - query_start_time) * 1000

        # Enhance result with context information
        enhanced_nodes = []
        for node_with_score in result.nodes:
            node_id = node_with_score.node.node_id

            # Get vector context if available
            if node_id in self.vector_cache:
                vector_info = self.vector_cache[node_id]
                context_metadata = self.adapter.retrieve_vector_context(
                    vector_info["context_id"]
                )

                # Add context metadata to node
                enhanced_metadata = {
                    **node_with_score.node.metadata,
                    "vector_context_id": vector_info["context_id"],
                    "added_at": context_metadata.get("added_at"),
                    "vector_dimension": context_metadata.get("vector_dimension")
                }

                node_with_score.node.metadata = enhanced_metadata

            enhanced_nodes.append(node_with_score)

        # Store query result context
        result_context = {
            "query_context_id": query_context_id,
            "results_count": len(enhanced_nodes),
            "execution_time_ms": query_execution_time,
            "similarity_scores": [node.score for node in enhanced_nodes] if enhanced_nodes else [],
            "result_node_ids": [node.node.node_id for node in enhanced_nodes]
        }

        self.adapter.store_vector_query_result_context(result_context)

        # Update result with enhanced nodes
        result.nodes = enhanced_nodes

        return result

    def simple_vector_search(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Simple vector search implementation."""

        # This is a simplified implementation for demonstration
        # In practice, you would use proper vector similarity search

        if not query.query_embedding:
            # Return empty result if no embedding
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # Find most similar vectors from cache
        results = []
        query_embedding = np.array(query.query_embedding)

        for node_id, vector_info in self.vector_cache.items():
            stored_embedding = np.array(vector_info["embedding"])

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )

            results.append({
                "node_id": node_id,
                "similarity": float(similarity),
                "vector_info": vector_info
            })

        # Sort by similarity and take top k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:query.similarity_top_k]

        # Create NodeWithScore objects
        nodes_with_scores = []
        for result in top_results:
            # Create a simple node (in practice, you'd reconstruct the full node)
            from llama_index.core.schema import TextNode, NodeWithScore

            node = TextNode(
                text="[Retrieved from vector store]",  # Placeholder
                node_id=result["node_id"],
                metadata=result["vector_info"]["metadata"]
            )

            node_with_score = NodeWithScore(
                node=node,
                score=result["similarity"]
            )
            nodes_with_scores.append(node_with_score)

        return VectorStoreQueryResult(
            nodes=nodes_with_scores,
            similarities=[r["similarity"] for r in top_results],
            ids=[r["node_id"] for r in top_results]
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text with caching."""

        # Check cache first
        text_hash = hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        # Generate embedding (simplified - use actual embedding model)
        # This is a placeholder implementation
        import hashlib
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.md5(text_bytes)

        # Convert hash to pseudo-embedding (for demonstration)
        hash_int = int(hash_obj.hexdigest(), 16)
        embedding = [(hash_int >> i) & 1 for i in range(384)]  # 384-dim binary vector
        embedding = [float(x) for x in embedding]  # Convert to float

        # Cache embedding
        self.embedding_cache[text_hash] = embedding

        return embedding

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""

        return {
            "total_vectors": len(self.vector_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "total_operations": self.adapter.get_vector_operation_count(),
            "cache_hit_rate": self.adapter.get_embedding_cache_hit_rate(),
            "avg_query_time_ms": self.adapter.get_avg_vector_query_time()
        }

# Usage example
context_vector_store = ContextAwareVectorStore(llamaindex_adapter)

# Create index with context-aware vector store
from llama_index.core import StorageContext
storage_context = StorageContext.from_defaults(vector_store=context_vector_store)
context_index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# Query with enhanced vector store
query_engine = context_index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What are the applications of deep learning?")
print(f"Enhanced vector search response: {response}")

# Get vector store statistics
stats = context_vector_store.get_vector_store_stats()
print(f"Vector store stats: {stats}")
```

## Index Management

### Advanced Index Management

```python
class ContextAwareIndexManager:
    """Manage multiple indexes with context store integration."""

    def __init__(self, llamaindex_adapter: LlamaIndexAdapter):
        self.adapter = llamaindex_adapter
        self.indexes = {}
        self.index_metadata = {}

    def create_index(self, index_name: str, documents: List[Document],
                    index_config: Dict[str, Any]) -> str:
        """Create a new index with context tracking."""

        index_start_time = time.time()

        # Store index creation context
        index_context = {
            "index_name": index_name,
            "document_count": len(documents),
            "index_config": index_config,
            "created_at": index_start_time,
            "status": "creating"
        }

        index_context_id = self.adapter.store_index_creation_context(index_context)

        # Create index based on configuration
        index_type = index_config.get("type", "vector")

        if index_type == "vector":
            index = self.create_vector_index(documents, index_config)
        elif index_type == "list":
            index = self.create_list_index(documents, index_config)
        elif index_type == "tree":
            index = self.create_tree_index(documents, index_config)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Store index
        self.indexes[index_name] = index

        # Update index context
        index_creation_time = time.time() - index_start_time
        index_context.update({
            "status": "completed",
            "creation_time_seconds": index_creation_time,
            "index_size_bytes": self.estimate_index_size(index),
            "completed_at": time.time()
        })

        # Store final index metadata
        final_context_id = self.adapter.store_index_creation_context(index_context)

        self.index_metadata[index_name] = {
            "index_context_id": final_context_id,
            "type": index_type,
            "document_count": len(documents),
            "created_at": index_start_time,
            "last_queried": None,
            "query_count": 0
        }

        return index_name

    def create_vector_index(self, documents: List[Document], config: Dict[str, Any]) -> VectorStoreIndex:
        """Create vector index with context-aware vector store."""

        # Use context-aware vector store if specified
        if config.get("use_context_store", True):
            vector_store = ContextAwareVectorStore(self.adapter)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        else:
            return VectorStoreIndex.from_documents(documents)

    def create_list_index(self, documents: List[Document], config: Dict[str, Any]):
        """Create list index."""
        from llama_index.core import SummaryIndex
        return SummaryIndex.from_documents(documents)

    def create_tree_index(self, documents: List[Document], config: Dict[str, Any]):
        """Create tree index."""
        from llama_index.core import TreeIndex
        return TreeIndex.from_documents(documents)

    def query_index(self, index_name: str, query: str, query_config: Dict[str, Any] = None) -> Response:
        """Query an index with context tracking."""

        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} not found")

        index = self.indexes[index_name]
        query_config = query_config or {}

        # Store query context
        query_context = {
            "index_name": index_name,
            "query": query,
            "query_config": query_config,
            "timestamp": time.time()
        }

        query_context_id = self.adapter.store_index_query_context(query_context)

        # Execute query
        query_start_time = time.time()

        # Create query engine with configuration
        query_engine = index.as_query_engine(**query_config)
        response = query_engine.query(query)

        query_execution_time = (time.time() - query_start_time) * 1000

        # Store query result
        query_result_context = {
            "query_context_id": query_context_id,
            "response": str(response),
            "execution_time_ms": query_execution_time,
            "source_nodes_count": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
        }

        self.adapter.store_index_query_result_context(query_result_context)

        # Update index metadata
        self.index_metadata[index_name]["last_queried"] = time.time()
        self.index_metadata[index_name]["query_count"] += 1

        return response

    def update_index(self, index_name: str, new_documents: List[Document]) -> None:
        """Update an index with new documents."""

        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} not found")

        index = self.indexes[index_name]

        # Store update context
        update_context = {
            "index_name": index_name,
            "new_documents_count": len(new_documents),
            "update_type": "add_documents",
            "timestamp": time.time()
        }

        update_context_id = self.adapter.store_index_update_context(update_context)

        # Update index (implementation depends on index type)
        if hasattr(index, 'insert'):
            for doc in new_documents:
                index.insert(doc)
        elif hasattr(index, 'refresh'):
            # For indexes that need full refresh
            all_documents = self.get_index_documents(index_name) + new_documents
            index.refresh(all_documents)

        # Update metadata
        self.index_metadata[index_name]["document_count"] += len(new_documents)

        # Store update completion
        update_context["status"] = "completed"
        update_context["completed_at"] = time.time()
        self.adapter.store_index_update_context(update_context)

    def get_index_documents(self, index_name: str) -> List[Document]:
        """Get all documents in an index."""
        # This would need to be implemented based on how you track documents
        # For now, return empty list
        return []

    def estimate_index_size(self, index) -> int:
        """Estimate index size in bytes."""
        # Simplified estimation
        return 1000  # Placeholder

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index."""

        if index_name not in self.index_metadata:
            return {}

        metadata = self.index_metadata[index_name]

        # Get query statistics from adapter
        query_stats = self.adapter.get_index_query_statistics(index_name)

        return {
            **metadata,
            "avg_query_time_ms": query_stats.get("avg_execution_time_ms", 0),
            "total_queries": query_stats.get("total_queries", 0),
            "last_query_time": metadata.get("last_queried"),
            "queries_per_day": query_stats.get("queries_per_day", 0)
        }

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all managed indexes."""

        return [
            {
                "name": name,
                "stats": self.get_index_stats(name)
            }
            for name in self.indexes.keys()
        ]

# Usage example
index_manager = ContextAwareIndexManager(llamaindex_adapter)

# Create different types of indexes
vector_index_name = index_manager.create_index(
    "ai_vector_index",
    documents,
    {"type": "vector", "use_context_store": True}
)

list_index_name = index_manager.create_index(
    "ai_list_index",
    documents,
    {"type": "list"}
)

# Query indexes
vector_response = index_manager.query_index(
    vector_index_name,
    "What is artificial intelligence?",
    {"similarity_top_k": 3}
)

list_response = index_manager.query_index(
    list_index_name,
    "Summarize the key concepts",
    {"response_mode": "tree_summarize"}
)

# Get index statistics
indexes_list = index_manager.list_indexes()
for index_info in indexes_list:
    print(f"Index: {index_info['name']}")
    print(f"Stats: {index_info['stats']}")
    print()
```

## Retrieval Optimization

### Smart Retrieval with Context

```python
class OptimizedRetriever:
    """Optimized retriever with context-aware caching and ranking."""

    def __init__(self, llamaindex_adapter: LlamaIndexAdapter, base_retriever):
        self.adapter = llamaindex_adapter
        self.base_retriever = base_retriever
        self.retrieval_cache = {}
        self.ranking_models = {}

    def retrieve(self, query: str, retrieval_config: Dict[str, Any] = None) -> List[NodeWithScore]:
        """Retrieve nodes with context-aware optimization."""

        retrieval_config = retrieval_config or {}

        # Check cache first
        cache_key = self.get_cache_key(query, retrieval_config)
        if cache_key in self.retrieval_cache:
            cached_result = self.retrieval_cache[cache_key]
            if time.time() - cached_result["timestamp"] < retrieval_config.get("cache_ttl", 3600):
                # Return cached result with cache hit tracking
                self.adapter.log_retrieval_cache_hit(query, cache_key)
                return cached_result["nodes"]

        # Store retrieval context
        retrieval_context = {
            "query": query,
            "retrieval_config": retrieval_config,
            "cache_miss": cache_key not in self.retrieval_cache,
            "timestamp": time.time()
        }

        retrieval_context_id = self.adapter.store_retrieval_context(retrieval_context)

        # Execute base retrieval
        retrieval_start_time = time.time()
        base_nodes = self.base_retriever.retrieve(query)
        base_retrieval_time = (time.time() - retrieval_start_time) * 1000

        # Apply context-aware re-ranking
        reranked_nodes = self.rerank_with_context(query, base_nodes, retrieval_config)

        # Apply filtering based on context
        filtered_nodes = self.filter_with_context(query, reranked_nodes, retrieval_config)

        total_retrieval_time = (time.time() - retrieval_start_time) * 1000

        # Store retrieval result
        retrieval_result_context = {
            "retrieval_context_id": retrieval_context_id,
            "base_nodes_count": len(base_nodes),
            "final_nodes_count": len(filtered_nodes),
            "base_retrieval_time_ms": base_retrieval_time,
            "total_retrieval_time_ms": total_retrieval_time,
            "reranking_applied": True,
            "filtering_applied": True
        }

        self.adapter.store_retrieval_result_context(retrieval_result_context)

        # Cache result
        if retrieval_config.get("enable_caching", True):
            self.retrieval_cache[cache_key] = {
                "nodes": filtered_nodes,
                "timestamp": time.time(),
                "retrieval_context_id": retrieval_context_id
            }

        return filtered_nodes

    def rerank_with_context(self, query: str, nodes: List[NodeWithScore],
                          config: Dict[str, Any]) -> List[NodeWithScore]:
        """Re-rank nodes using context information."""

        # Get query context for re-ranking
        query_context = self.adapter.get_query_reranking_context(
            query=query,
            context_window=config.get("context_window", 10)
        )

        # Apply context-aware scoring
        reranked_nodes = []
        for node_with_score in nodes:
            # Get node context
            node_context = self.adapter.get_node_reranking_context(
                node_id=node_with_score.node.node_id
            )

            # Calculate context-aware score
            context_score = self.calculate_context_score(
                query,
                node_with_score.node,
                query_context,
                node_context
            )

            # Combine base score with context score
            alpha = config.get("context_weight", 0.3)
            combined_score = (1 - alpha) * node_with_score.score + alpha * context_score

            # Update node score
            node_with_score.score = combined_score
            reranked_nodes.append(node_with_score)

        # Sort by new scores
        reranked_nodes.sort(key=lambda x: x.score, reverse=True)

        return reranked_nodes

    def filter_with_context(self, query: str, nodes: List[NodeWithScore],
                          config: Dict[str, Any]) -> List[NodeWithScore]:
        """Filter nodes based on context criteria."""

        filtered_nodes = []
        min_score = config.get("min_score", 0.0)
        max_nodes = config.get("max_nodes", 10)

        # Get filtering context
        filtering_context = self.adapter.get_filtering_context(
            query=query,
            filter_criteria=config.get("filter_criteria", [])
        )

        for node_with_score in nodes:
            # Apply score threshold
            if node_with_score.score < min_score:
                continue

            # Apply context-based filters
            if self.passes_context_filters(node_with_score.node, filtering_context, config):
                filtered_nodes.append(node_with_score)

                # Stop at max nodes
                if len(filtered_nodes) >= max_nodes:
                    break

        return filtered_nodes

    def calculate_context_score(self, query: str, node, query_context: Dict,
                              node_context: Dict) -> float:
        """Calculate context-aware relevance score."""

        score = 0.0

        # Temporal relevance (recent nodes score higher)
        if node_context.get("created_at"):
            age_hours = (time.time() - node_context["created_at"]) / 3600
            temporal_score = max(0, 1 - age_hours / 168)  # Decay over 1 week
            score += 0.2 * temporal_score

        # Usage frequency (frequently retrieved nodes score higher)
        usage_count = node_context.get("retrieval_count", 0)
        usage_score = min(1.0, usage_count / 10)  # Normalize to 0-1
        score += 0.3 * usage_score

        # Query similarity (similar to previous queries)
        if query_context.get("similar_queries"):
            similar_query_score = len(query_context["similar_queries"]) / 5  # Max 5 similar queries
            score += 0.2 * min(1.0, similar_query_score)

        # Domain relevance
        if query_context.get("domain_terms") and node_context.get("domain_classification"):
            domain_overlap = self.calculate_domain_overlap(
                query_context["domain_terms"],
                node_context["domain_classification"]
            )
            score += 0.3 * domain_overlap

        return min(1.0, score)  # Ensure score is between 0 and 1

    def passes_context_filters(self, node, filtering_context: Dict, config: Dict) -> bool:
        """Check if node passes context-based filters."""

        # Time-based filtering
        if config.get("max_age_hours"):
            node_metadata = node.metadata or {}
            if "created_at" in node_metadata:
                age_hours = (time.time() - node_metadata["created_at"]) / 3600
                if age_hours > config["max_age_hours"]:
                    return False

        # Source filtering
        if config.get("allowed_sources"):
            node_source = node.metadata.get("source", "")
            if node_source not in config["allowed_sources"]:
                return False

        # Quality filtering (based on retrieval performance)
        if config.get("min_quality_score"):
            node_quality = filtering_context.get("node_quality_scores", {}).get(node.node_id, 0.5)
            if node_quality < config["min_quality_score"]:
                return False

        return True

    def calculate_domain_overlap(self, query_terms: List[str], node_domains: List[str]) -> float:
        """Calculate overlap between query domains and node domains."""

        if not query_terms or not node_domains:
            return 0.0

        query_set = set(term.lower() for term in query_terms)
        node_set = set(domain.lower() for domain in node_domains)

        overlap = len(query_set.intersection(node_set))
        union = len(query_set.union(node_set))

        return overlap / union if union > 0 else 0.0

    def get_cache_key(self, query: str, config: Dict[str, Any]) -> str:
        """Generate cache key for query and configuration."""

        import hashlib

        # Create deterministic string from query and config
        cache_string = f"{query}_{sorted(config.items())}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics."""

        return {
            "cache_size": len(self.retrieval_cache),
            "cache_hit_rate": self.adapter.get_retrieval_cache_hit_rate(),
            "avg_retrieval_time_ms": self.adapter.get_avg_retrieval_time(),
            "total_retrievals": self.adapter.get_total_retrieval_count(),
            "reranking_effectiveness": self.adapter.get_reranking_effectiveness_score()
        }

# Usage example
from llama_index.core.retrievers import VectorIndexRetriever

# Create base retriever
base_index = VectorStoreIndex.from_documents(documents)
base_retriever = VectorIndexRetriever(
    index=base_index,
    similarity_top_k=10
)

# Create optimized retriever
optimized_retriever = OptimizedRetriever(llamaindex_adapter, base_retriever)

# Retrieve with optimization
retrieval_config = {
    "enable_caching": True,
    "cache_ttl": 3600,
    "context_weight": 0.3,
    "min_score": 0.1,
    "max_nodes": 5,
    "max_age_hours": 168,  # 1 week
    "min_quality_score": 0.6
}

optimized_nodes = optimized_retriever.retrieve(
    "How do neural networks learn?",
    retrieval_config
)

print(f"Retrieved {len(optimized_nodes)} optimized nodes")

# Get retrieval statistics
stats = optimized_retriever.get_retrieval_stats()
print(f"Retrieval stats: {stats}")
```

## Multi-Index Systems

### Federated Index Management

```python
class FederatedIndexSystem:
    """Manage and query across multiple indexes with context coordination."""

    def __init__(self, llamaindex_adapter: LlamaIndexAdapter):
        self.adapter = llamaindex_adapter
        self.index_managers = {}
        self.federation_config = {}
        self.cross_index_cache = {}

    def register_index_manager(self, manager_id: str, index_manager: ContextAwareIndexManager,
                             manager_config: Dict[str, Any]) -> None:
        """Register an index manager with the federation."""

        self.index_managers[manager_id] = {
            "manager": index_manager,
            "config": manager_config,
            "registered_at": time.time(),
            "query_count": 0,
            "last_queried": None
        }

        # Store federation registration
        registration_context = {
            "manager_id": manager_id,
            "config": manager_config,
            "registered_at": time.time(),
            "available_indexes": list(index_manager.indexes.keys())
        }

        self.adapter.store_federation_registration(registration_context)

    def federated_query(self, query: str, federation_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute query across multiple index managers."""

        federation_config = federation_config or {}
        query_start_time = time.time()

        # Store federated query context
        fed_query_context = {
            "query": query,
            "federation_config": federation_config,
            "participating_managers": list(self.index_managers.keys()),
            "started_at": query_start_time
        }

        fed_query_context_id = self.adapter.store_federated_query_context(fed_query_context)

        # Determine which managers to query
        target_managers = self.select_target_managers(query, federation_config)

        # Execute queries in parallel (simplified - sequential for demo)
        manager_results = {}
        for manager_id in target_managers:
            manager_info = self.index_managers[manager_id]
            manager = manager_info["manager"]

            try:
                # Query all indexes in this manager
                manager_responses = {}
                for index_name in manager.indexes.keys():
                    response = manager.query_index(
                        index_name,
                        query,
                        federation_config.get("query_config", {})
                    )
                    manager_responses[index_name] = response

                manager_results[manager_id] = {
                    "status": "success",
                    "responses": manager_responses,
                    "execution_time_ms": (time.time() - query_start_time) * 1000
                }

                # Update manager stats
                manager_info["query_count"] += 1
                manager_info["last_queried"] = time.time()

            except Exception as e:
                manager_results[manager_id] = {
                    "status": "error",
                    "error": str(e),
                    "execution_time_ms": (time.time() - query_start_time) * 1000
                }

        # Aggregate and rank results
        aggregated_results = self.aggregate_federated_results(
            query,
            manager_results,
            federation_config
        )

        total_execution_time = (time.time() - query_start_time) * 1000

        # Store federated query results
        fed_result_context = {
            "fed_query_context_id": fed_query_context_id,
            "manager_results_count": len(manager_results),
            "successful_managers": len([r for r in manager_results.values() if r["status"] == "success"]),
            "total_responses": sum(len(r.get("responses", {})) for r in manager_results.values()),
            "total_execution_time_ms": total_execution_time,
            "aggregated_results_count": len(aggregated_results.get("ranked_results", []))
        }

        self.adapter.store_federated_query_result_context(fed_result_context)

        return {
            "query": query,
            "federation_results": aggregated_results,
            "manager_results": manager_results,
            "execution_summary": {
                "total_time_ms": total_execution_time,
                "managers_queried": len(target_managers),
                "successful_queries": len([r for r in manager_results.values() if r["status"] == "success"])
            }
        }

    def select_target_managers(self, query: str, config: Dict[str, Any]) -> List[str]:
        """Select which index managers to query based on query and configuration."""

        # Get query routing context
        routing_context = self.adapter.get_query_routing_context(
            query=query,
            available_managers=list(self.index_managers.keys())
        )

        # Simple routing strategy (can be enhanced with ML models)
        target_managers = []

        if config.get("query_all_managers", False):
            # Query all available managers
            target_managers = list(self.index_managers.keys())
        else:
            # Smart routing based on manager capabilities and past performance
            for manager_id, manager_info in self.index_managers.items():
                manager_config = manager_info["config"]

                # Check if manager handles this type of query
                if self.manager_handles_query_type(query, manager_config, routing_context):
                    target_managers.append(manager_id)

        # Ensure at least one manager is selected
        if not target_managers and self.index_managers:
            target_managers = [list(self.index_managers.keys())[0]]

        return target_managers

    def manager_handles_query_type(self, query: str, manager_config: Dict[str, Any],
                                 routing_context: Dict[str, Any]) -> bool:
        """Determine if a manager should handle a specific query type."""

        # Check domain compatibility
        query_domains = routing_context.get("query_domains", [])
        manager_domains = manager_config.get("domains", [])

        if manager_domains and query_domains:
            domain_overlap = set(query_domains).intersection(set(manager_domains))
            if not domain_overlap:
                return False

        # Check query type compatibility
        query_type = routing_context.get("query_type", "general")
        supported_types = manager_config.get("supported_query_types", ["general"])

        if query_type not in supported_types:
            return False

        # Check performance criteria
        manager_performance = routing_context.get("manager_performance", {}).get(manager_config.get("manager_id"))
        if manager_performance and manager_performance.get("avg_response_time_ms", 0) > 5000:  # 5 second threshold
            return False

        return True

    def aggregate_federated_results(self, query: str, manager_results: Dict[str, Any],
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple managers."""

        all_responses = []

        # Collect all successful responses
        for manager_id, result in manager_results.items():
            if result["status"] == "success":
                for index_name, response in result["responses"].items():
                    # Extract response information
                    response_info = {
                        "manager_id": manager_id,
                        "index_name": index_name,
                        "response_text": str(response),
                        "source_nodes": getattr(response, 'source_nodes', []),
                        "manager_execution_time": result["execution_time_ms"]
                    }
                    all_responses.append(response_info)

        # Rank responses using cross-manager scoring
        ranked_responses = self.rank_cross_manager_responses(query, all_responses, config)

        # Generate consensus response if requested
        consensus_response = None
        if config.get("generate_consensus", False):
            consensus_response = self.generate_consensus_response(ranked_responses)

        return {
            "ranked_results": ranked_responses,
            "consensus_response": consensus_response,
            "result_count": len(ranked_responses),
            "managers_contributed": len(set(r["manager_id"] for r in ranked_responses))
        }

    def rank_cross_manager_responses(self, query: str, responses: List[Dict[str, Any]],
                                   config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank responses across different managers."""

        # Get cross-manager ranking context
        ranking_context = self.adapter.get_cross_manager_ranking_context(
            query=query,
            manager_ids=[r["manager_id"] for r in responses]
        )

        # Score each response
        scored_responses = []
        for response in responses:
            # Calculate cross-manager score
            score = self.calculate_cross_manager_score(
                query,
                response,
                ranking_context,
                config
            )

            response["cross_manager_score"] = score
            scored_responses.append(response)

        # Sort by score
        scored_responses.sort(key=lambda x: x["cross_manager_score"], reverse=True)

        return scored_responses

    def calculate_cross_manager_score(self, query: str, response: Dict[str, Any],
                                    ranking_context: Dict[str, Any], config: Dict[str, Any]) -> float:
        """Calculate cross-manager relevance score."""

        score = 0.0

        # Manager reliability score
        manager_id = response["manager_id"]
        manager_stats = ranking_context.get("manager_performance", {}).get(manager_id, {})
        reliability_score = manager_stats.get("success_rate", 0.5)
        score += 0.3 * reliability_score

        # Response speed score (faster responses score higher)
        max_time = max((r["manager_execution_time"] for r in ranking_context.get("all_responses", [])), default=1000)
        speed_score = 1 - (response["manager_execution_time"] / max_time)
        score += 0.2 * speed_score

        # Content quality score (based on response length and structure)
        content_score = min(1.0, len(response["response_text"]) / 500)  # Normalize to 500 chars
        score += 0.3 * content_score

        # Source diversity score (responses with more sources score higher)
        source_count = len(response.get("source_nodes", []))
        source_score = min(1.0, source_count / 5)  # Normalize to 5 sources
        score += 0.2 * source_score

        return score

    def generate_consensus_response(self, ranked_responses: List[Dict[str, Any]]) -> str:
        """Generate consensus response from top-ranked responses."""

        if not ranked_responses:
            return "No responses available for consensus generation."

        # Take top 3 responses for consensus
        top_responses = ranked_responses[:3]

        # Simple consensus generation (can be enhanced with LLM)
        consensus_parts = []
        for i, response in enumerate(top_responses, 1):
            consensus_parts.append(f"Source {i} ({response['manager_id']}/{response['index_name']}): {response['response_text'][:200]}...")

        consensus = "Consensus from multiple sources:\n\n" + "\n\n".join(consensus_parts)

        return consensus

    def get_federation_stats(self) -> Dict[str, Any]:
        """Get statistics for the federated system."""

        total_queries = sum(info["query_count"] for info in self.index_managers.values())
        active_managers = len([info for info in self.index_managers.values() if info["last_queried"]])

        # Get performance stats from adapter
        federation_performance = self.adapter.get_federation_performance_stats()

        return {
            "total_managers": len(self.index_managers),
            "active_managers": active_managers,
            "total_federated_queries": total_queries,
            "avg_federation_time_ms": federation_performance.get("avg_federation_time_ms", 0),
            "consensus_generation_rate": federation_performance.get("consensus_rate", 0),
            "cross_manager_effectiveness": federation_performance.get("effectiveness_score", 0)
        }

# Usage example
# Create multiple index managers for different domains
ai_index_manager = ContextAwareIndexManager(llamaindex_adapter)
ml_index_manager = ContextAwareIndexManager(llamaindex_adapter)

# Create domain-specific indexes
ai_index_manager.create_index("ai_concepts", documents[:2], {"type": "vector"})
ml_index_manager.create_index("ml_algorithms", documents[1:], {"type": "vector"})

# Create federated system
federation = FederatedIndexSystem(llamaindex_adapter)

# Register managers
federation.register_index_manager(
    "ai_manager",
    ai_index_manager,
    {
        "domains": ["artificial_intelligence", "general_ai"],
        "supported_query_types": ["conceptual", "general"],
        "manager_id": "ai_manager"
    }
)

federation.register_index_manager(
    "ml_manager",
    ml_index_manager,
    {
        "domains": ["machine_learning", "algorithms"],
        "supported_query_types": ["technical", "algorithmic"],
        "manager_id": "ml_manager"
    }
)

# Execute federated query
fed_result = federation.federated_query(
    "Explain machine learning algorithms",
    {
        "query_all_managers": False,
        "generate_consensus": True,
        "query_config": {"similarity_top_k": 3}
    }
)

print(f"Federated query completed:")
print(f"Managers queried: {fed_result['execution_summary']['managers_queried']}")
print(f"Total time: {fed_result['execution_summary']['total_time_ms']:.2f}ms")
print(f"Results: {fed_result['federation_results']['result_count']}")

if fed_result['federation_results']['consensus_response']:
    print(f"Consensus: {fed_result['federation_results']['consensus_response'][:200]}...")

# Get federation statistics
fed_stats = federation.get_federation_stats()
print(f"Federation stats: {fed_stats}")
```

## Performance Optimization

### LlamaIndex Performance Tips

```python
# Optimal configuration for LlamaIndex workflows
optimized_store = ContextReferenceStore(
    cache_size=10000,             # Large cache for documents and embeddings
    use_compression=True,         # Compress document storage
    compression_algorithm="lz4",  # Fast compression for embeddings
    eviction_policy="LRU",        # Good for document access patterns
    use_disk_storage=True,        # Enable for large document collections
    memory_threshold_mb=500       # Higher threshold for document storage
)

# Optimized adapter configuration
llamaindex_adapter = LlamaIndexAdapter(
    context_store=optimized_store,
    enable_embedding_cache=True,
    enable_document_deduplication=True,
    batch_processing_size=50,
    enable_semantic_caching=True
)

# Performance monitoring for LlamaIndex
def monitor_llamaindex_performance(adapter: LlamaIndexAdapter):
    """Monitor LlamaIndex adapter performance."""

    perf_stats = adapter.get_performance_statistics()

    print(f"LlamaIndex Performance Metrics:")
    print(f"  Document indexing time: {perf_stats['avg_indexing_time_ms']:.2f}ms")
    print(f"  Query execution time: {perf_stats['avg_query_time_ms']:.2f}ms")
    print(f"  Embedding cache hit rate: {perf_stats['embedding_cache_hit_rate']:.2%}")
    print(f"  Document retrieval time: {perf_stats['avg_retrieval_time_ms']:.2f}ms")
    print(f"  Total documents indexed: {perf_stats['total_documents']}")

    # Performance recommendations
    if perf_stats['avg_indexing_time_ms'] > 5000:
        print("⚠️  Slow document indexing - consider batch processing")

    if perf_stats['embedding_cache_hit_rate'] < 0.6:
        print("⚠️  Low embedding cache hit rate - consider increasing cache size")

    if perf_stats['avg_query_time_ms'] > 2000:
        print("⚠️  Slow query execution - consider query optimization")
```

## Best Practices

### LlamaIndex Integration Best Practices

1. **Document Organization**

   ```python
   # Organize documents by domain and type
   def organize_documents_efficiently(documents):
       organized = {
           "technical": [],
           "conceptual": [],
           "reference": []
       }

       for doc in documents:
           doc_type = classify_document_type(doc)
           organized[doc_type].append(doc)

       return organized
   ```

2. **Efficient Indexing**

   ```python
   # Use batch processing for large document sets
   def batch_index_documents(documents, batch_size=50):
       for i in range(0, len(documents), batch_size):
           batch = documents[i:i+batch_size]
           process_document_batch(batch)
   ```

3. **Query Optimization**
   ```python
   # Optimize queries for better performance
   def optimize_query_config(query_type):
       if query_type == "semantic_search":
           return {"similarity_top_k": 5, "enable_reranking": True}
       elif query_type == "summarization":
           return {"response_mode": "tree_summarize", "max_tokens": 500}
       else:
           return {"similarity_top_k": 3}
   ```

## Troubleshooting

### Common LlamaIndex Integration Issues

#### 1. Slow Document Indexing

```python
# Problem: Large documents take too long to index
# Solution: Implement chunking and parallel processing

def optimize_document_indexing(documents):
    # Split large documents
    chunked_docs = []
    for doc in documents:
        if len(doc.text) > 10000:  # Large document
            chunks = split_document(doc, chunk_size=2000)
            chunked_docs.extend(chunks)
        else:
            chunked_docs.append(doc)

    return chunked_docs
```

#### 2. High Memory Usage with Embeddings

```python
# Problem: Embedding storage consuming too much memory
# Solution: Use context store for embedding management

def optimize_embedding_storage(adapter):
    # Enable embedding compression
    adapter.enable_embedding_compression(True)

    # Set embedding cache limits
    adapter.set_embedding_cache_limit(5000)

    # Use disk storage for embeddings
    adapter.enable_embedding_disk_storage(True)
```

#### 3. Slow Query Performance

```python
# Problem: Queries taking too long to execute
# Solution: Implement query optimization

def optimize_query_performance(query_engine, adapter):
    # Enable query result caching
    adapter.enable_query_caching(True, cache_ttl=3600)

    # Use optimized retrieval
    optimized_retriever = OptimizedRetriever(adapter, query_engine.retriever)
    query_engine.retriever = optimized_retriever

    # Enable parallel processing
    adapter.enable_parallel_processing(True, max_workers=4)
```

This comprehensive LlamaIndex integration guide provides everything needed to build sophisticated document-centric AI applications with Context Reference Store, from basic document management to advanced federated index systems with intelligent retrieval optimization.
