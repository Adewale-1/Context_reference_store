# Adapters API Reference

This document provides comprehensive API reference for all Context Reference Store adapters.

## Table of Contents

- [Base Adapter](#base-adapter)
- [LangChain Adapter](#langchain-adapter)
- [LangGraph Adapter](#langgraph-adapter)
- [LlamaIndex Adapter](#llamaindex-adapter)
- [Composio Adapter](#composio-adapter)
- [ADK Adapter](#adk-adapter)

## Base Adapter

### `BaseAdapter`

Base class for all framework adapters.

```python
class BaseAdapter:
    def __init__(self, context_store: ContextReferenceStore)
```

#### Methods

##### `store_context(context_data: Any, context_type: str = "default", metadata: Dict = None) -> str`

Store context data with optional metadata.

**Parameters:**
- `context_data`: Data to store
- `context_type`: Type identifier for the context
- `metadata`: Optional metadata dictionary

**Returns:** Context ID string

**Example:**
```python
context_id = adapter.store_context(
    {"message": "Hello world"},
    context_type="conversation",
    metadata={"timestamp": time.time()}
)
```

##### `retrieve_context(context_id: str) -> Any`

Retrieve context data by ID.

**Parameters:**
- `context_id`: Context identifier

**Returns:** Stored context data

**Raises:** `ValueError` if context not found

## LangChain Adapter

### `LangChainAdapter`

Adapter for LangChain framework integration.

```python
from context_store.adapters import LangChainAdapter

adapter = LangChainAdapter(context_store)
```

#### Methods

##### `store_messages(messages: List[BaseMessage], session_id: str, metadata: Dict = None) -> str`

Store LangChain messages with session context.

**Parameters:**
- `messages`: List of LangChain BaseMessage objects
- `session_id`: Session identifier
- `metadata`: Optional session metadata

**Returns:** Context ID for the stored messages

**Example:**
```python
from langchain.schema import HumanMessage, AIMessage

messages = [
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!")
]

context_id = adapter.store_messages(
    messages=messages,
    session_id="chat_session_1",
    metadata={"user_id": "user_123"}
)
```

##### `retrieve_messages(session_id: str) -> List[BaseMessage]`

Retrieve messages for a session.

**Parameters:**
- `session_id`: Session identifier

**Returns:** List of LangChain BaseMessage objects

##### `store_conversation_turn(session_id: str, input_message: BaseMessage, output_message: BaseMessage, metadata: Dict = None) -> str`

Store a single conversation turn.

**Parameters:**
- `session_id`: Session identifier
- `input_message`: User input message
- `output_message`: AI response message
- `metadata`: Optional turn metadata

**Returns:** Context ID for the conversation turn

##### `get_conversation_summary(session_id: str) -> str`

Get conversation summary for a session.

**Parameters:**
- `session_id`: Session identifier

**Returns:** Conversation summary string

##### `find_similar_conversations(query: str, limit: int = 5) -> List[Dict]`

Find conversations similar to the query.

**Parameters:**
- `query`: Search query
- `limit`: Maximum number of results

**Returns:** List of similar conversation metadata

## LangGraph Adapter

### `LangGraphAdapter`

Adapter for LangGraph stateful workflows.

```python
from context_store.adapters import LangGraphAdapter

adapter = LangGraphAdapter(context_store)
```

#### Methods

##### `store_node_state(graph_id: str, node_name: str, state: Dict, metadata: Dict = None) -> str`

Store state for a graph node.

**Parameters:**
- `graph_id`: Graph identifier
- `node_name`: Node name
- `state`: Node state data
- `metadata`: Optional metadata

**Returns:** Context ID for the stored state

##### `store_checkpoint(workflow_id: str, checkpoint_data: Dict) -> str`

Store workflow checkpoint.

**Parameters:**
- `workflow_id`: Workflow identifier  
- `checkpoint_data`: Checkpoint data

**Returns:** Checkpoint context ID

##### `retrieve_checkpoint(checkpoint_id: str) -> Dict`

Retrieve checkpoint data.

**Parameters:**
- `checkpoint_id`: Checkpoint identifier

**Returns:** Checkpoint data dictionary

##### `get_latest_checkpoint(graph_id: str, thread_id: str) -> Dict`

Get latest checkpoint for a graph thread.

**Parameters:**
- `graph_id`: Graph identifier
- `thread_id`: Thread identifier

**Returns:** Latest checkpoint data

##### `store_workflow_execution_context(execution_context: Dict) -> str`

Store workflow execution context.

**Parameters:**
- `execution_context`: Execution context data

**Returns:** Context ID

## LlamaIndex Adapter

### `LlamaIndexAdapter`

Adapter for LlamaIndex document processing and retrieval.

```python
from context_store.adapters import LlamaIndexAdapter

adapter = LlamaIndexAdapter(context_store)
```

#### Methods

##### `store_document_collection(documents: List[Document], collection_name: str, metadata: Dict = None) -> str`

Store a collection of documents.

**Parameters:**
- `documents`: List of LlamaIndex Document objects
- `collection_name`: Collection name
- `metadata`: Optional collection metadata

**Returns:** Collection context ID

##### `create_context_aware_index(collection_id: str, index_type: str = "vector", **kwargs) -> VectorStoreIndex`

Create context-aware index from document collection.

**Parameters:**
- `collection_id`: Document collection ID
- `index_type`: Type of index to create
- `**kwargs`: Additional index parameters

**Returns:** LlamaIndex VectorStoreIndex object

##### `store_query_execution(original_query: str, enhanced_query: str, query_context: Dict, execution_config: Dict) -> str`

Store query execution context.

**Parameters:**
- `original_query`: Original query string
- `enhanced_query`: Context-enhanced query
- `query_context`: Query context data
- `execution_config`: Execution configuration

**Returns:** Execution context ID

##### `get_document_vectors(ref_doc_id: str) -> List[Dict]`

Get vector information for a document.

**Parameters:**
- `ref_doc_id`: Document reference ID

**Returns:** List of vector information dictionaries

##### `store_vector_context(vector_context: Dict) -> str`

Store vector context information.

**Parameters:**
- `vector_context`: Vector context data

**Returns:** Vector context ID

## Composio Adapter

### `ComposioAdapter`

Adapter for Composio tool integration.

```python
from context_store.adapters import ComposioAdapter

adapter = ComposioAdapter(context_store)
```

#### Methods

##### `store_auth_context(user_id: str, auth_credentials: Dict, encryption_key: str) -> str`

Store encrypted authentication context.

**Parameters:**
- `user_id`: User identifier
- `auth_credentials`: Authentication credentials
- `encryption_key`: Encryption key for credentials

**Returns:** Authentication context ID

##### `execute_tool_with_context(app: str, action: str, params: Dict, auth_context_id: str, execution_metadata: Dict = None) -> Dict`

Execute tool with stored authentication context.

**Parameters:**
- `app`: App name
- `action`: Action name
- `params`: Action parameters
- `auth_context_id`: Authentication context ID
- `execution_metadata`: Optional execution metadata

**Returns:** Execution result dictionary

##### `store_execution_intent(execution_intent: Dict) -> str`

Store tool execution intent.

**Parameters:**
- `execution_intent`: Execution intent data

**Returns:** Intent context ID

##### `store_execution_result(execution_result: Dict) -> str`

Store tool execution result.

**Parameters:**
- `execution_result`: Execution result data

**Returns:** Result context ID

##### `encrypt_credentials(credentials: Dict, encryption_key: str) -> str`

Encrypt credentials for secure storage.

**Parameters:**
- `credentials`: Credentials to encrypt
- `encryption_key`: Encryption key

**Returns:** Encrypted credentials string

##### `decrypt_credentials(encrypted_credentials: str, encryption_key: str) -> Dict`

Decrypt stored credentials.

**Parameters:**
- `encrypted_credentials`: Encrypted credentials
- `encryption_key`: Decryption key

**Returns:** Decrypted credentials dictionary

## ADK Adapter

### `ADKAdapter`

Adapter for Agent Development Kit integration.

```python
from context_store.adapters import ADKAdapter

adapter = ADKAdapter(context_store)
```

#### Methods

##### `create_agent_store(agent_id: str, cache_size: int = 1000, use_compression: bool = True) -> ContextReferenceStore`

Create context store for an ADK agent.

**Parameters:**
- `agent_id`: Agent identifier
- `cache_size`: Cache size for the store
- `use_compression`: Enable compression

**Returns:** ContextReferenceStore instance

##### `store_agent_state(agent_id: str, state_data: Dict, metadata: Dict = None) -> str`

Store agent state.

**Parameters:**
- `agent_id`: Agent identifier
- `state_data`: Agent state data
- `metadata`: Optional metadata

**Returns:** State context ID

##### `store_workflow_context(workflow_id: str, context_data: Dict, metadata: Dict = None) -> str`

Store workflow execution context.

**Parameters:**
- `workflow_id`: Workflow identifier
- `context_data`: Workflow context data
- `metadata`: Optional metadata

**Returns:** Workflow context ID

##### `get_agent_coordination_context(workflow_id: str, agent_ids: List[str]) -> Dict`

Get coordination context for multiple agents.

**Parameters:**
- `workflow_id`: Workflow identifier
- `agent_ids`: List of agent identifiers

**Returns:** Coordination context dictionary

##### `store_multi_agent_interaction(interaction_data: Dict) -> str`

Store multi-agent interaction data.

**Parameters:**
- `interaction_data`: Interaction data

**Returns:** Interaction context ID

## Common Patterns

### Error Handling

All adapters follow consistent error handling patterns:

```python
try:
    result = adapter.store_context(data)
except ValueError as e:
    # Handle validation errors
    print(f"Validation error: {e}")
except Exception as e:
    # Handle other errors
    print(f"Unexpected error: {e}")
```

### Context Metadata

All adapters support optional metadata:

```python
metadata = {
    "timestamp": time.time(),
    "user_id": "user_123",
    "session_type": "interactive"
}

context_id = adapter.store_context(data, metadata=metadata)
```

### Performance Optimization

Enable caching and compression for better performance:

```python
# Most adapters support these optimizations
adapter.enable_caching(True)
adapter.enable_compression(True)
adapter.set_cache_size(5000)
```

For more detailed examples and advanced usage patterns, see the individual integration guides:

- [ADK Integration Guide](../integrations/adk.md)
- [LangChain Integration Guide](../integrations/langchain.md)
- [LangGraph Integration Guide](../integrations/langgraph.md)
- [LlamaIndex Integration Guide](../integrations/llamaindex.md)
- [Composio Integration Guide](../integrations/composio.md)
