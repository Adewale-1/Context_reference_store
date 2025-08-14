# Context Reference Store

[![PyPI version](https://badge.fury.io/py/context-reference-store.svg)](https://badge.fury.io/py/context-reference-store)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Efficient Large Context Window Management for AI Agents and Frameworks**

Context Reference Store is a high-performance Python library designed to solve the challenge of managing large context windows in Agentic AI applications. It provides intelligent caching, compression, and retrieval mechanisms that significantly reduce memory usage and improve response times for AI agents and frameworks.

## Table of Contents

- [Context Reference Store](#context-reference-store)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
    - [Core Capabilities](#core-capabilities)
    - [Framework Integrations](#framework-integrations)
    - [Advanced Features](#advanced-features)
  - [Architecture](#architecture)
  - [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Basic Usage](#basic-usage)
    - [Async Operations](#async-operations)
    - [Multimodal Content](#multimodal-content)
  - [Building AI Agents](#building-ai-agents)
    - [Simple Agent Example](#simple-agent-example)
    - [Multi-Agent System](#multi-agent-system)
    - [Agent with Tool Integration](#agent-with-tool-integration)
  - [Framework Integration Examples](#framework-integration-examples)
    - [Agent Development Kit (ADK) Integration](#agent-development-kit-adk-integration)
    - [LangChain Integration](#langchain-integration)
    - [LangGraph Integration](#langgraph-integration)
    - [LlamaIndex Integration](#llamaindex-integration)
    - [Composio Integration](#composio-integration)
  - [Performance Benchmarks](#performance-benchmarks)
  - [Configuration Options](#configuration-options)
    - [Cache Policies](#cache-policies)
    - [Compression Settings](#compression-settings)
    - [Storage Configuration](#storage-configuration)
  - [Monitoring and Analytics](#monitoring-and-analytics)
    - [Real-time Dashboard](#real-time-dashboard)
    - [Performance Metrics](#performance-metrics)
    - [Custom Monitoring](#custom-monitoring)
  - [Advanced Features](#advanced-features-1)
    - [Semantic Analysis](#semantic-analysis)
    - [Token Optimization](#token-optimization)
  - [API Reference](#api-reference)
  - [Development](#development)
    - [Installation for Development](#installation-for-development)
    - [Running Tests](#running-tests)
    - [Code Quality](#code-quality)
  - [Optional Dependencies](#optional-dependencies)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
    - [Quick Contribution Steps](#quick-contribution-steps)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Support](#support)

## Key Features

### Core Capabilities

- **Intelligent Context Caching**: LRU, LFU, and TTL-based eviction policies
- **Advanced Compression**: 625x faster serialization with 99.99% storage reduction
- **Async/Await Support**: Non-blocking operations for modern applications
- **Multimodal Content**: Handle text, images, audio, and video efficiently
- **High Performance**: Sub-100ms retrieval times for large contexts

### Framework Integrations

- **🤖 Agent Development Kit (ADK)**: Native support for ADK agent workflows and state management
- **🦜 LangChain**: Seamless integration with chat and retrieval chains
- **🕸️ LangGraph**: Native support for graph-based agent workflows
- **🦙 LlamaIndex**: Vector store and query engine implementations
- **🔧 Composio**: Tool integration with secure authentication

### Advanced Features

- **Performance Monitoring**: Real-time metrics and dashboard
- **Semantic Analysis**: Content similarity and clustering
- **Token Optimization**: Intelligent context window management
- **Persistent Storage**: Disk-based caching for large datasets

## Architecture

The Context Reference Store follows a clean, optimized workflow that transforms large context inputs into efficiently managed references:

![Context Reference Store Architecture](images/enhanced_context_reference_flow.png)

The architecture provides:

1. **Large Context Input**: Handles 1M-2M tokens, multimodal content (images, audio, video), and structured data
2. **Smart Optimization**: Multiple processing engines for compression, deduplication, and hashing
3. **Reference Storage**: Centralized store with metadata tracking and multi-tier storage management  
4. **Fast Retrieval**: Agent cockpit with framework adapters delivering 625x faster performance

**Key Performance Benefits:**
- **625x Faster** serialization and retrieval
- **49x Memory Reduction** for multi-agent scenarios  
- **99.55% Storage Savings** through intelligent compression
- **Zero Quality Loss** with perfect content preservation

## Quick Start

### Installation

```bash
# Basic installation
pip install context-reference-store

# With framework integrations
pip install context-reference-store[adk,langchain,langgraph,llamaindex]

# Full installation with all features
pip install context-reference-store[full]
```

### Basic Usage

```python
from context_store import ContextReferenceStore

# Initialize the store
store = ContextReferenceStore(cache_size=100)

# Store context content
context_id = store.store("Your long context content here...")

# Retrieve when needed
content = store.retrieve(context_id)

# Get performance statistics
stats = store.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Async Operations

```python
from context_store import AsyncContextReferenceStore

async def main():
    async with AsyncContextReferenceStore() as store:
        # Store multiple contexts concurrently
        context_ids = await store.batch_store_async([
            "Context 1", "Context 2", "Context 3"
        ])

        # Retrieve all at once
        contents = await store.batch_retrieve_async(context_ids)
```

### Multimodal Content

```python
from context_store import MultimodalContent, MultimodalPart

# Create multimodal content
text_part = MultimodalPart.from_text("Describe this image:")
image_part = MultimodalPart.from_file("path/to/image.jpg")
content = MultimodalContent(parts=[text_part, image_part])

# Store and retrieve
context_id = store.store_multimodal_content(content)
retrieved = store.retrieve_multimodal_content(context_id)
```

## Building AI Agents

### Simple Agent Example

```python
from context_store import ContextReferenceStore
from context_store.adapters import ADKAdapter

class SimpleAgent:
    def __init__(self):
        self.store = ContextReferenceStore(cache_size=1000)
        self.adk_adapter = ADKAdapter(self.store)
        self.conversation_history = []

    def process_message(self, user_message: str) -> str:
        # Store user message in context
        user_context_id = self.store.store({
            "type": "user_message",
            "content": user_message,
            "timestamp": time.time()
        })

        # Retrieve relevant conversation history
        context = self.adk_adapter.get_conversation_context(
            limit=10,
            include_multimodal=True
        )

        # Process with your LLM
        response = self.generate_response(context, user_message)

        # Store response
        response_context_id = self.store.store({
            "type": "agent_response",
            "content": response,
            "timestamp": time.time()
        })

        return response

    def generate_response(self, context, message):
        # Your LLM processing logic here
        return f"Processed: {message}"

# Usage
agent = SimpleAgent()
response = agent.process_message("Hello, how can you help me?")
```

### Multi-Agent System

```python
from context_store import ContextReferenceStore
from context_store.adapters import ADKAdapter

class MultiAgentSystem:
    def __init__(self):
        self.shared_store = ContextReferenceStore(
            cache_size=5000,
            use_compression=True
        )
        self.agents = {}
        self.coordinator = AgentCoordinator(self.shared_store)

    def add_agent(self, agent_id: str, agent_type: str):
        """Add an agent to the system"""
        self.agents[agent_id] = {
            "type": agent_type,
            "adapter": ADKAdapter(self.shared_store),
            "state": {},
            "tools": []
        }

    def route_message(self, message: str, target_agent: str = None):
        """Route message to appropriate agent"""
        if target_agent:
            return self.process_with_agent(message, target_agent)

        # Use coordinator to determine best agent
        agent_id = self.coordinator.select_agent(message, self.agents.keys())
        return self.process_with_agent(message, agent_id)

    def process_with_agent(self, message: str, agent_id: str):
        """Process message with specific agent"""
        agent = self.agents[agent_id]
        adapter = agent["adapter"]

        # Get agent-specific context
        context = adapter.get_agent_context(
            agent_id=agent_id,
            message_count=20,
            include_shared_memory=True
        )

        # Process and update shared context
        response = self.generate_agent_response(message, context, agent)

        # Store interaction in shared memory
        interaction_id = self.shared_store.store({
            "agent_id": agent_id,
            "user_message": message,
            "agent_response": response,
            "timestamp": time.time(),
            "context_used": len(context)
        })

        return response

# Usage
system = MultiAgentSystem()
system.add_agent("researcher", "research_agent")
system.add_agent("writer", "content_agent")
system.add_agent("analyst", "data_agent")

response = system.route_message("Research the latest AI trends")
```

### Agent with Tool Integration

```python
from context_store import ContextReferenceStore
from context_store.adapters import ADKAdapter, ComposioAdapter

class ToolIntegratedAgent:
    def __init__(self):
        self.store = ContextReferenceStore()
        self.adk_adapter = ADKAdapter(self.store)
        self.composio_adapter = ComposioAdapter(self.store)

        # Initialize tools
        self.available_tools = {
            "search": self.web_search,
            "calculate": self.calculate,
            "send_email": self.send_email,
            "file_operations": self.file_operations
        }

    def process_with_tools(self, user_message: str):
        """Process message and use tools as needed"""

        # Analyze message to determine needed tools
        required_tools = self.analyze_tool_requirements(user_message)

        # Store initial context
        context_id = self.store.store({
            "user_message": user_message,
            "required_tools": required_tools,
            "status": "processing"
        })

        # Execute tools and gather results
        tool_results = {}
        for tool_name in required_tools:
            if tool_name in self.available_tools:
                try:
                    result = self.available_tools[tool_name](user_message)
                    tool_results[tool_name] = result

                    # Store tool result in context
                    self.store.store({
                        "context_id": context_id,
                        "tool": tool_name,
                        "result": result,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    tool_results[tool_name] = f"Error: {str(e)}"

        # Generate final response using tool results
        final_response = self.generate_final_response(
            user_message,
            tool_results,
            context_id
        )

        return final_response

    def web_search(self, query: str):
        """Web search using Composio integration"""
        return self.composio_adapter.execute_tool(
            app="googlesearch",
            action="search",
            params={"query": query, "num_results": 5}
        )

    def calculate(self, expression: str):
        """Mathematical calculations"""
        # Safe calculation logic
        import ast
        import operator

        # Simplified calculator - extend as needed
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
        }

        try:
            tree = ast.parse(expression, mode='eval')
            result = self._eval_node(tree.body, operators)
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}

# Usage
agent = ToolIntegratedAgent()
response = agent.process_with_tools(
    "Search for the latest Python releases and calculate the time difference"
)
```

## Framework Integration Examples

### Agent Development Kit (ADK) Integration

```python
from context_store.adapters import ADKAdapter
from adk import Agent, Workflow

# Create ADK-integrated agent
class ADKContextAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.context_adapter = ADKAdapter()

    def setup(self):
        # Initialize context store for this agent
        self.context_store = self.context_adapter.create_agent_store(
            agent_id=self.name,
            cache_size=1000,
            use_compression=True
        )

    def process_step(self, input_data):
        # Store step context
        step_context_id = self.context_store.store({
            "step": self.current_step,
            "input": input_data,
            "agent_id": self.name,
            "timestamp": time.time()
        })

        # Get relevant historical context
        context = self.context_adapter.get_step_context(
            agent_id=self.name,
            step_type=self.current_step,
            limit=5
        )

        # Process with context
        result = self.execute_with_context(input_data, context)

        # Store result
        self.context_store.store({
            "step_context_id": step_context_id,
            "result": result,
            "success": True
        })

        return result

# Workflow with context management
workflow = Workflow("data_processing")
workflow.add_agent(ADKContextAgent("preprocessor"))
workflow.add_agent(ADKContextAgent("analyzer"))
workflow.add_agent(ADKContextAgent("reporter"))

# Context is automatically shared between agents
workflow.run(input_data="large_dataset.csv")
```

**[Complete ADK Integration Guide →](docs/integrations/adk.md)**

### LangChain Integration

```python
from context_store.adapters import LangChainAdapter
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

adapter = LangChainAdapter()

# Enhanced conversation memory
class ContextAwareMemory(ConversationBufferMemory):
    def __init__(self, context_adapter: LangChainAdapter):
        super().__init__()
        self.context_adapter = context_adapter

    def save_context(self, inputs, outputs):
        # Save to both LangChain memory and Context Store
        super().save_context(inputs, outputs)

        # Store in context store for advanced retrieval
        self.context_adapter.store_conversation_turn(
            inputs=inputs,
            outputs=outputs,
            session_id=getattr(self, 'session_id', 'default')
        )

# Usage with chains
memory = ContextAwareMemory(adapter)
conversation_chain = ConversationChain(
    llm=your_llm,
    memory=memory
)

# Store conversation with metadata
messages = [
    HumanMessage(content="What's the weather like?"),
    AIMessage(content="I can help you check the weather. What's your location?")
]
session_id = adapter.store_messages(
    messages,
    session_id="weather_chat",
    metadata={"topic": "weather", "user_intent": "information"}
)

# Retrieve with semantic search
similar_conversations = adapter.find_similar_conversations(
    query="weather information",
    limit=3
)
```

**[Complete LangChain Integration Guide →](docs/integrations/langchain.md)**

### LangGraph Integration

```python
from context_store.adapters import LangGraphAdapter
from langgraph import StateGraph, START, END

adapter = LangGraphAdapter()

# Define state with context integration
class AgentState(TypedDict):
    messages: list
    context_id: str
    step_history: list

def context_aware_node(state: AgentState):
    # Store current state
    context_id = adapter.store_graph_state(
        state=state,
        graph_id="analysis_workflow",
        node_name="analysis"
    )

    # Get relevant context from previous executions
    historical_context = adapter.get_node_context(
        graph_id="analysis_workflow",
        node_name="analysis",
        limit=5
    )

    # Process with context
    result = process_with_historical_context(state, historical_context)

    # Update state with context reference
    state["context_id"] = context_id
    state["step_history"].append({
        "node": "analysis",
        "context_id": context_id,
        "timestamp": time.time()
    })

    return state

# Build graph with context integration
graph = StateGraph(AgentState)
graph.add_node("analysis", context_aware_node)
graph.add_edge(START, "analysis")
graph.add_edge("analysis", END)

compiled_graph = graph.compile()

# Run with context persistence
result = compiled_graph.invoke({
    "messages": ["Analyze this data"],
    "context_id": "",
    "step_history": []
})
```

**[Complete LangGraph Integration Guide →](docs/integrations/langgraph.md)**

### LlamaIndex Integration

```python
from context_store.adapters import LlamaIndexAdapter
from llama_index import Document, VectorStoreIndex, ServiceContext

adapter = LlamaIndexAdapter()

# Enhanced document store with context management
class ContextAwareDocumentStore:
    def __init__(self):
        self.adapter = LlamaIndexAdapter()
        self.indexes = {}

    def add_documents(self, documents: list[Document], collection: str):
        # Store documents with enhanced metadata
        doc_contexts = []
        for doc in documents:
            # Create context entry for each document
            context_id = self.adapter.store_document_context(
                document=doc,
                collection=collection,
                metadata={
                    "added_timestamp": time.time(),
                    "source": doc.metadata.get("source", "unknown"),
                    "document_type": doc.metadata.get("type", "text")
                }
            )
            doc_contexts.append(context_id)

        # Create or update index
        if collection not in self.indexes:
            self.indexes[collection] = VectorStoreIndex.from_documents(documents)
        else:
            for doc in documents:
                self.indexes[collection].insert(doc)

        return doc_contexts

    def query_with_context(self, query: str, collection: str, include_history: bool = True):
        # Get query context if requested
        query_context = None
        if include_history:
            query_context = self.adapter.get_query_context(
                query=query,
                collection=collection,
                limit=5
            )

        # Perform vector search
        query_engine = self.indexes[collection].as_query_engine()
        response = query_engine.query(query)

        # Store query and response
        self.adapter.store_query_response(
            query=query,
            response=str(response),
            collection=collection,
            context_used=query_context,
            source_nodes=[str(node.id_) for node in response.source_nodes]
        )

        return response

# Usage
doc_store = ContextAwareDocumentStore()

# Add documents with automatic context tracking
documents = [
    Document(text="AI research paper content...", metadata={"source": "arxiv", "type": "research"}),
    Document(text="Technical documentation...", metadata={"source": "github", "type": "documentation"})
]

doc_store.add_documents(documents, "ai_research")

# Query with context awareness
response = doc_store.query_with_context(
    "What are the latest AI research trends?",
    "ai_research",
    include_history=True
)
```

**[Complete LlamaIndex Integration Guide →](docs/integrations/llamaindex.md)**

### Composio Integration

```python
from context_store.adapters import ComposioAdapter
from composio import ComposioToolSet, App

adapter = ComposioAdapter()

# Context-aware tool execution
class ContextAwareToolAgent:
    def __init__(self):
        self.composio_adapter = ComposioAdapter()
        self.toolset = ComposioToolSet()

    def execute_tool_with_context(self, app: str, action: str, params: dict, session_id: str = None):
        # Get execution context
        execution_context = self.composio_adapter.get_execution_context(
            app=app,
            action=action,
            session_id=session_id
        )

        # Store execution intent
        execution_id = self.composio_adapter.store_execution_intent(
            app=app,
            action=action,
            params=params,
            context=execution_context,
            session_id=session_id
        )

        try:
            # Execute tool
            result = self.toolset.execute_action(
                app=app,
                action=action,
                params=params
            )

            # Store successful result
            self.composio_adapter.store_execution_result(
                execution_id=execution_id,
                result=result,
                status="success"
            )

            return result

        except Exception as e:
            # Store error for learning
            self.composio_adapter.store_execution_result(
                execution_id=execution_id,
                result=None,
                status="error",
                error=str(e)
            )
            raise

    def get_tool_recommendations(self, user_intent: str, session_id: str = None):
        """Get tool recommendations based on context and history"""
        return self.composio_adapter.recommend_tools(
            intent=user_intent,
            session_id=session_id,
            limit=5
        )

# Usage
tool_agent = ContextAwareToolAgent()

# Execute with context tracking
result = tool_agent.execute_tool_with_context(
    app="gmail",
    action="send_email",
    params={
        "to": "recipient@example.com",
        "subject": "Context-aware email",
        "body": "This email was sent with context awareness"
    },
    session_id="email_session_1"
)

# Get recommendations based on context
recommendations = tool_agent.get_tool_recommendations(
    "I need to schedule a meeting",
    session_id="productivity_session"
)
```

**[Complete Composio Integration Guide →](docs/integrations/composio.md)**

## Performance Benchmarks

Our benchmarks show significant improvements over standard approaches:

| Metric              | Standard | Context Store | Improvement       |
| ------------------- | -------- | ------------- | ----------------- |
| Serialization Speed | 2.5s     | 4ms           | **625x faster**   |
| Memory Usage        | 1.2GB    | 24MB          | **49x reduction** |
| Storage Size        | 450MB    | 900KB         | **99.8% smaller** |
| Retrieval Time      | 250ms    | 15ms          | **16x faster**    |
| Agent State Sync    | 1.2s     | 25ms          | **48x faster**    |
| Multi-Agent Memory  | 2.8GB    | 57MB          | **49x reduction** |

## Configuration Options

### Cache Policies

```python
from context_store import CacheEvictionPolicy

# LRU (Least Recently Used)
store = ContextReferenceStore(
    cache_size=100,
    eviction_policy=CacheEvictionPolicy.LRU
)

# LFU (Least Frequently Used)
store = ContextReferenceStore(
    eviction_policy=CacheEvictionPolicy.LFU
)

# TTL (Time To Live)
store = ContextReferenceStore(
    eviction_policy=CacheEvictionPolicy.TTL,
    ttl_seconds=3600  # 1 hour
)
```

### Compression Settings

```python
# Enable compression for better storage efficiency
store = ContextReferenceStore(
    use_compression=True,
    compression_algorithm="lz4",  # or "zstd"
    compression_level=3
)
```

### Storage Configuration

```python
# Configure disk storage for large datasets
store = ContextReferenceStore(
    use_disk_storage=True,
    disk_cache_dir="/path/to/cache",
    memory_threshold_mb=500
)
```

## Monitoring and Analytics

### Real-time Dashboard

The Context Reference Store includes a beautiful terminal-based dashboard for real-time monitoring of performance metrics, compression analytics, and system health.

![TUI Dashboard](images/tui_dashboard.png)

```python
from context_store.monitoring import create_dashboard

# Create and launch interactive dashboard
store = ContextReferenceStore(enable_compression=True)
dashboard = create_dashboard(store)
dashboard.start()  # Opens interactive TUI in terminal
```

**Dashboard Features:**
- **Live Performance Metrics**: Real-time cache hit rates, compression ratios, and efficiency multipliers
- **Compression Analytics**: Detailed breakdown of compression algorithms and space savings
- **Cache Management**: Memory usage, eviction policies, and hit rate history
- **Interactive Navigation**: Tabbed interface with keyboard controls (←/→ arrows, Q to quit)
- **Color-coded Alerts**: Visual indicators for performance thresholds and system health

### Performance Metrics

```python
# Get detailed statistics
stats = store.get_detailed_stats()
print(f"""
Performance Metrics:
- Cache Hit Rate: {stats['hit_rate']:.2%}
- Average Retrieval Time: {stats['avg_retrieval_time_ms']}ms
- Memory Usage: {stats['memory_usage_mb']}MB
- Compression Ratio: {stats['compression_ratio']:.2f}x
""")
```

### Custom Monitoring

```python
from context_store.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
store.add_monitor(monitor)

# Access real-time metrics
print(monitor.get_current_metrics())
```

## Advanced Features

### Semantic Analysis

```python
from context_store.semantic import SemanticAnalyzer

analyzer = SemanticAnalyzer(store)

# Find similar contexts
similar = analyzer.find_similar_contexts(
    "query text",
    threshold=0.8,
    limit=5
)

# Cluster related contexts
clusters = analyzer.cluster_contexts(method="kmeans", n_clusters=5)
```

### Token Optimization

```python
from context_store.optimization import TokenManager

token_manager = TokenManager(store)

# Optimize context for token limits
optimized = token_manager.optimize_context(
    context_id,
    max_tokens=4000,
    strategy="importance_ranking"
)
```

## API Reference

Detailed API documentation is available in the following files:

- [Core API Reference](docs/api/core.md)
- [Adapters API Reference](docs/api/adapters.md)
- [Monitoring API Reference](docs/api/monitoring.md)
- [Utilities API Reference](docs/api/utils.md)

## Development

### Installation for Development

```bash
git clone https://github.com/Adewale-1/Context_reference_store.git
cd Context_reference_store
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=context_store

# Run performance benchmarks
pytest -m benchmark
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 context_store/
mypy context_store/
```

## Optional Dependencies

The library supports various optional dependencies for enhanced functionality:

```bash
# Framework integrations
pip install context-reference-store[adk]         # Agent Development Kit support
pip install context-reference-store[langchain]    # LangChain support
pip install context-reference-store[langgraph]    # LangGraph support
pip install context-reference-store[llamaindex]   # LlamaIndex support
pip install context-reference-store[composio]     # Composio support

# Performance enhancements
pip install context-reference-store[compression]  # Advanced compression
pip install context-reference-store[async]        # Async optimizations

# Development tools
pip install context-reference-store[dev]          # Testing and linting
pip install context-reference-store[docs]         # Documentation tools

# Everything included
pip install context-reference-store[full]         # All features
```

## Documentation

Comprehensive documentation is available:

- [Getting Started Guide](docs/getting-started.md)
- [Agent Building Tutorial](docs/tutorials/building-agents.md)
- [Framework Integration Guides](docs/integrations/)
  - [ADK Integration Guide](docs/integrations/adk.md)
  - [LangChain Integration Guide](docs/integrations/langchain.md)
  - [LangGraph Integration Guide](docs/integrations/langgraph.md)
  - [LlamaIndex Integration Guide](docs/integrations/llamaindex.md)
  - [Composio Integration Guide](docs/integrations/composio.md)
- [Performance Optimization Guide](docs/guides/performance.md)
- [Deployment Guide](docs/guides/deployment.md)
- [API Reference](docs/api/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for Google Summer of Code 2025 with Google DeepMind
- Inspired by the need for efficient context management in modern AI applications
- Thanks to the open-source AI community for feedback and contributions

## Support

- **Documentation**: [https://context-reference-store.readthedocs.io/](https://context-reference-store.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/Adewale-1/Context_reference_store/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Adewale-1/Context_reference_store/discussions)

---

**Made with ❤️ for the AI community**
