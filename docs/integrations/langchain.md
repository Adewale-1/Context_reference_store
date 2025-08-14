# LangChain Integration

This guide demonstrates how to integrate Context Reference Store with LangChain for building efficient conversational agents and chains.

## Table of Contents

- [LangChain Integration](#langchain-integration)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Basic LangChain Integration](#basic-langchain-integration)
  - [Memory Management](#memory-management)
  - [Chain Integration](#chain-integration)
  - [Agent Integration](#agent-integration)
  - [Retrieval Chains](#retrieval-chains)
  - [Multi-Session Management](#multi-session-management)
  - [Performance Optimization](#performance-optimization)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)

## Overview

LangChain integration with Context Reference Store provides:

- **Efficient Memory Management**: 49x memory reduction for conversation history
- **Scalable Session Storage**: Handle thousands of concurrent conversations
- **Advanced Retrieval**: Context-aware document retrieval with compression
- **Chain Optimization**: Faster chain execution with intelligent caching
- **Multi-Modal Support**: Handle text, images, and documents in conversations

## Installation

```bash
# Install with LangChain support
pip install context-reference-store[langchain]

# Or install specific LangChain components
pip install context-reference-store langchain langchain-core
```

## Basic LangChain Integration

### LangChain Adapter Setup

```python
from context_store.adapters import LangChainAdapter
from context_store import ContextReferenceStore
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import time

# Initialize context store and adapter
context_store = ContextReferenceStore(
    cache_size=2000,
    use_compression=True,
    eviction_policy="LRU"
)

langchain_adapter = LangChainAdapter(context_store)

# Store conversation messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Hello, how are you?"),
    AIMessage(content="I'm doing well, thank you! How can I help you today?")
]

# Store messages with session context
session_id = langchain_adapter.store_messages(
    messages=messages,
    session_id="user_123_session_1",
    metadata={
        "user_id": "user_123",
        "conversation_type": "general",
        "created_at": time.time()
    }
)

# Retrieve conversation history
retrieved_messages = langchain_adapter.retrieve_messages(session_id)
print(f"Retrieved {len(retrieved_messages)} messages")
```

## Memory Management

### Enhanced Conversation Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from typing import List, Dict, Any

class ContextAwareMemory(ConversationBufferMemory):
    """Enhanced LangChain memory with Context Reference Store."""

    def __init__(self, langchain_adapter: LangChainAdapter, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self.adapter = langchain_adapter
        self.session_id = session_id
        self.context_metadata = {
            "created_at": time.time(),
            "message_count": 0,
            "last_accessed": time.time()
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation context efficiently."""

        # Save to traditional LangChain memory
        super().save_context(inputs, outputs)

        # Extract messages for context store
        input_message = HumanMessage(content=inputs.get("input", ""))
        output_message = AIMessage(content=outputs.get("output", ""))

        # Store in context store with enhanced metadata
        turn_metadata = {
            **self.context_metadata,
            "turn_timestamp": time.time(),
            "input_length": len(str(inputs)),
            "output_length": len(str(outputs))
        }

        self.adapter.store_conversation_turn(
            session_id=self.session_id,
            input_message=input_message,
            output_message=output_message,
            metadata=turn_metadata
        )

        # Update metadata
        self.context_metadata["message_count"] += 2
        self.context_metadata["last_accessed"] = time.time()

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory with context store optimization."""

        # Update access time
        self.context_metadata["last_accessed"] = time.time()

        # Get memory from parent class
        memory_vars = super().load_memory_variables(inputs)

        # Enhance with context store data if needed
        if len(self.chat_memory.messages) > 100:  # For long conversations
            # Load recent context from store
            recent_context = self.adapter.get_recent_context(
                session_id=self.session_id,
                limit=20
            )
            memory_vars["recent_context_summary"] = recent_context

        return memory_vars

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get comprehensive conversation summary."""
        return {
            "session_id": self.session_id,
            "message_count": self.context_metadata["message_count"],
            "duration_minutes": (time.time() - self.context_metadata["created_at"]) / 60,
            "last_accessed": self.context_metadata["last_accessed"],
            "memory_size": len(self.chat_memory.messages)
        }

# Usage example
memory = ContextAwareMemory(
    langchain_adapter=langchain_adapter,
    session_id="enhanced_session_001",
    return_messages=True
)
```

## Chain Integration

### Context-Aware Conversation Chain

```python
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class ContextAwareConversationChain:
    """LangChain conversation chain with context store integration."""

    def __init__(self, llm, langchain_adapter: LangChainAdapter):
        self.llm = llm
        self.adapter = langchain_adapter
        self.sessions = {}

        # Enhanced prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["history", "input", "context_summary"],
            template="""You are a helpful AI assistant with conversation context.

Previous conversation:
{history}

Context summary:
{context_summary}

Current input: {input}
Assistant:"""
        )

    def create_session(self, session_id: str, system_message: str = None) -> str:
        """Create a new conversation session."""

        # Create enhanced memory
        memory = ContextAwareMemory(
            langchain_adapter=self.adapter,
            session_id=session_id,
            return_messages=True
        )

        # Create conversation chain
        chain = ConversationChain(
            llm=self.llm,
            memory=memory,
            prompt=self.prompt_template,
            verbose=False
        )

        self.sessions[session_id] = {
            "chain": chain,
            "memory": memory,
            "created_at": time.time()
        }

        # Add system message if provided
        if system_message:
            self.adapter.store_messages(
                messages=[SystemMessage(content=system_message)],
                session_id=session_id
            )

        return session_id

    def chat(self, session_id: str, message: str) -> str:
        """Process a chat message."""

        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]
        chain = session["chain"]
        memory = session["memory"]

        # Get context summary for enhanced prompt
        context_summary = self.adapter.get_conversation_summary(session_id)

        # Process message with enhanced context
        response = chain.predict(
            input=message,
            context_summary=context_summary
        )

        return response

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information."""

        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]
        memory = session["memory"]

        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "conversation_summary": memory.get_conversation_summary(),
            "message_count": len(memory.chat_memory.messages)
        }

# Usage example
llm = OpenAI(temperature=0.7)
conversation_chain = ContextAwareConversationChain(llm, langchain_adapter)

# Create session and chat
session_id = conversation_chain.create_session("user_456", "You are a helpful coding assistant.")
response = conversation_chain.chat(session_id, "How do I implement a binary search in Python?")
print(response)
```

### Sequential Chain with Context

```python
from langchain.chains import LLMChain, SequentialChain

class ContextAwareSequentialChain:
    """Sequential chain with context store integration."""

    def __init__(self, llm, langchain_adapter: LangChainAdapter):
        self.llm = llm
        self.adapter = langchain_adapter

        # Define chain steps
        self.analysis_prompt = PromptTemplate(
            input_variables=["input", "context"],
            template="Analyze the following input with context:\nContext: {context}\nInput: {input}\nAnalysis:"
        )

        self.response_prompt = PromptTemplate(
            input_variables=["analysis", "context"],
            template="Generate response based on analysis:\nContext: {context}\nAnalysis: {analysis}\nResponse:"
        )

        # Create individual chains
        self.analysis_chain = LLMChain(
            llm=llm,
            prompt=self.analysis_prompt,
            output_key="analysis"
        )

        self.response_chain = LLMChain(
            llm=llm,
            prompt=self.response_prompt,
            output_key="response"
        )

        # Create sequential chain
        self.sequential_chain = SequentialChain(
            chains=[self.analysis_chain, self.response_chain],
            input_variables=["input", "context"],
            output_variables=["analysis", "response"]
        )

    def process(self, session_id: str, input_text: str) -> Dict[str, str]:
        """Process input through sequential chain with context."""

        # Get relevant context
        context = self.adapter.get_relevant_context(
            session_id=session_id,
            query=input_text,
            limit=5
        )

        # Process through sequential chain
        result = self.sequential_chain({
            "input": input_text,
            "context": str(context)
        })

        # Store the interaction
        self.adapter.store_chain_interaction(
            session_id=session_id,
            input_text=input_text,
            analysis=result["analysis"],
            response=result["response"],
            context_used=context
        )

        return result

# Usage example
sequential_chain = ContextAwareSequentialChain(llm, langchain_adapter)
result = sequential_chain.process("session_789", "Explain machine learning concepts")
print(f"Analysis: {result['analysis']}")
print(f"Response: {result['response']}")
```

## Agent Integration

### LangChain Agent with Context Store

```python
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from typing import Optional

class ContextAwareLangChainAgent:
    """LangChain agent integrated with Context Reference Store."""

    def __init__(self, llm, langchain_adapter: LangChainAdapter):
        self.llm = llm
        self.adapter = langchain_adapter

        # Define tools
        self.tools = [
            Tool(
                name="Store Context",
                description="Store information in context for later retrieval",
                func=self.store_context_tool
            ),
            Tool(
                name="Retrieve Context",
                description="Retrieve relevant context based on query",
                func=self.retrieve_context_tool
            ),
            Tool(
                name="Conversation History",
                description="Get conversation history for current session",
                func=self.get_history_tool
            )
        ]

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        )

        self.current_session = None

    def store_context_tool(self, information: str) -> str:
        """Tool to store context information."""

        if not self.current_session:
            return "No active session. Please start a session first."

        context_id = self.adapter.store_context(
            session_id=self.current_session,
            context=information,
            context_type="user_provided"
        )

        return f"Information stored with ID: {context_id}"

    def retrieve_context_tool(self, query: str) -> str:
        """Tool to retrieve relevant context."""

        if not self.current_session:
            return "No active session. Please start a session first."

        relevant_context = self.adapter.search_context(
            session_id=self.current_session,
            query=query,
            limit=3
        )

        if relevant_context:
            return f"Relevant context found: {relevant_context}"
        else:
            return "No relevant context found."

    def get_history_tool(self, query: str = "") -> str:
        """Tool to get conversation history."""

        if not self.current_session:
            return "No active session."

        messages = self.adapter.retrieve_messages(self.current_session)

        if messages:
            history = []
            for msg in messages[-5:]:  # Last 5 messages
                history.append(f"{msg.__class__.__name__}: {msg.content}")
            return "\n".join(history)
        else:
            return "No conversation history found."

    def start_session(self, session_id: str) -> str:
        """Start a new agent session."""

        self.current_session = session_id

        # Initialize session context
        self.adapter.store_context(
            session_id=session_id,
            context={
                "session_started": time.time(),
                "agent_type": "conversational_react"
            },
            context_type="session_info"
        )

        return f"Started session: {session_id}"

    def chat(self, message: str) -> str:
        """Chat with the agent."""

        if not self.current_session:
            return "Please start a session first using start_session()."

        # Store user message
        self.adapter.store_messages(
            messages=[HumanMessage(content=message)],
            session_id=self.current_session
        )

        # Process with agent
        response = self.agent.run(message)

        # Store agent response
        self.adapter.store_messages(
            messages=[AIMessage(content=response)],
            session_id=self.current_session
        )

        return response

# Usage example
agent = ContextAwareLangChainAgent(llm, langchain_adapter)
session_id = agent.start_session("agent_session_001")

# Chat with context awareness
response1 = agent.chat("Remember that I like Python programming")
response2 = agent.chat("What do you know about my preferences?")
print(f"Response: {response2}")
```

## Retrieval Chains

### Context-Enhanced Retrieval Chain

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ContextEnhancedRetrievalChain:
    """Retrieval chain enhanced with context store."""

    def __init__(self, llm, langchain_adapter: LangChainAdapter):
        self.llm = llm
        self.adapter = langchain_adapter
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_stores = {}

    def create_knowledge_base(self, session_id: str, documents: List[str]) -> str:
        """Create knowledge base from documents."""

        # Split documents into chunks
        chunks = []
        for doc in documents:
            doc_chunks = self.text_splitter.split_text(doc)
            chunks.extend(doc_chunks)

        # Create vector store
        vector_store = FAISS.from_texts(chunks, self.embeddings)
        self.vector_stores[session_id] = vector_store

        # Store document metadata in context store
        kb_metadata = {
            "knowledge_base_id": session_id,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "created_at": time.time()
        }

        self.adapter.store_context(
            session_id=session_id,
            context=kb_metadata,
            context_type="knowledge_base"
        )

        return f"Knowledge base created with {len(chunks)} chunks"

    def query_with_context(self, session_id: str, query: str) -> str:
        """Query knowledge base with conversation context."""

        if session_id not in self.vector_stores:
            return "No knowledge base found for this session."

        vector_store = self.vector_stores[session_id]

        # Get conversation context
        conversation_context = self.adapter.get_conversation_summary(session_id)

        # Enhanced query with context
        enhanced_query = f"Context: {conversation_context}\nQuery: {query}"

        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )

        # Process query
        response = qa_chain.run(enhanced_query)

        # Store query and response
        self.adapter.store_qa_interaction(
            session_id=session_id,
            query=query,
            response=response,
            context_used=conversation_context
        )

        return response

# Usage example
retrieval_chain = ContextEnhancedRetrievalChain(llm, langchain_adapter)

# Create knowledge base
documents = [
    "Python is a high-level programming language...",
    "Machine learning is a subset of artificial intelligence...",
    "Natural language processing deals with text..."
]

retrieval_chain.create_knowledge_base("kb_session_001", documents)

# Query with context
response = retrieval_chain.query_with_context(
    "kb_session_001",
    "How does Python relate to machine learning?"
)
print(response)
```

## Multi-Session Management

### Session Manager for LangChain

```python
class LangChainSessionManager:
    """Manage multiple LangChain sessions with context store."""

    def __init__(self, langchain_adapter: LangChainAdapter):
        self.adapter = langchain_adapter
        self.active_sessions = {}
        self.session_configs = {}

    def create_session(self, user_id: str, session_config: Dict[str, Any]) -> str:
        """Create new session with configuration."""

        session_id = f"{user_id}_{int(time.time())}"

        # Store session configuration
        self.session_configs[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "config": session_config,
            "message_count": 0,
            "last_activity": time.time()
        }

        # Initialize in context store
        self.adapter.initialize_session(
            session_id=session_id,
            metadata=self.session_configs[session_id]
        )

        self.active_sessions[session_id] = True

        return session_id

    def get_session_memory(self, session_id: str) -> ContextAwareMemory:
        """Get memory instance for session."""

        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        return ContextAwareMemory(
            langchain_adapter=self.adapter,
            session_id=session_id,
            return_messages=True
        )

    def update_session_activity(self, session_id: str):
        """Update session last activity."""

        if session_id in self.session_configs:
            self.session_configs[session_id]["last_activity"] = time.time()
            self.session_configs[session_id]["message_count"] += 1

    def cleanup_inactive_sessions(self, inactive_hours: int = 24):
        """Clean up inactive sessions."""

        current_time = time.time()
        inactive_sessions = []

        for session_id, config in self.session_configs.items():
            last_activity = config["last_activity"]
            hours_inactive = (current_time - last_activity) / 3600

            if hours_inactive > inactive_hours:
                inactive_sessions.append(session_id)

        # Archive inactive sessions
        for session_id in inactive_sessions:
            self.adapter.archive_session(session_id)
            del self.session_configs[session_id]
            del self.active_sessions[session_id]

        return len(inactive_sessions)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions."""

        total_sessions = len(self.session_configs)
        total_messages = sum(config["message_count"] for config in self.session_configs.values())

        # Active sessions (last 1 hour)
        current_time = time.time()
        active_count = sum(
            1 for config in self.session_configs.values()
            if (current_time - config["last_activity"]) < 3600
        )

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_count,
            "total_messages": total_messages,
            "average_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0
        }

# Usage example
session_manager = LangChainSessionManager(langchain_adapter)

# Create multiple sessions
session1 = session_manager.create_session("user_123", {"type": "general"})
session2 = session_manager.create_session("user_456", {"type": "technical"})

# Get session-specific memory
memory1 = session_manager.get_session_memory(session1)
memory2 = session_manager.get_session_memory(session2)

# Create chains with session-specific memory
chain1 = ConversationChain(llm=llm, memory=memory1)
chain2 = ConversationChain(llm=llm, memory=memory2)
```

## Performance Optimization

### LangChain Performance Tips

```python
# Optimal configuration for LangChain
optimized_store = ContextReferenceStore(
    cache_size=5000,              # Large cache for frequent conversations
    use_compression=True,         # Enable compression for message history
    compression_algorithm="lz4",  # Fast compression for real-time chat
    eviction_policy="LRU",        # Good for conversation patterns
    use_disk_storage=True,        # Enable for long conversations
    memory_threshold_mb=300       # Reasonable threshold
)

# Efficient adapter configuration
langchain_adapter = LangChainAdapter(
    context_store=optimized_store,
    batch_size=10,                # Batch message storage
    enable_caching=True,          # Cache frequently accessed conversations
    compression_threshold=100     # Compress messages over 100 chars
)

# Performance monitoring
def monitor_langchain_performance(adapter: LangChainAdapter):
    """Monitor LangChain adapter performance."""

    stats = adapter.get_performance_stats()

    print(f"Message Storage Performance:")
    print(f"  Average store time: {stats['avg_store_time_ms']:.2f}ms")
    print(f"  Average retrieve time: {stats['avg_retrieve_time_ms']:.2f}ms")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Total conversations: {stats['total_conversations']}")

    # Performance recommendations
    if stats['avg_store_time_ms'] > 50:
        print("⚠️  Consider enabling compression for faster storage")

    if stats['cache_hit_rate'] < 0.8:
        print("⚠️  Consider increasing cache size")
```

## Best Practices

### LangChain Integration Best Practices

1. **Memory Management**

   ```python
   # Use context-aware memory for long conversations
   memory = ContextAwareMemory(
       langchain_adapter=adapter,
       session_id=session_id,
       max_token_limit=4000,  # Prevent token overflow
       return_messages=True
   )
   ```

2. **Session Organization**

   ```python
   # Organize sessions by user and conversation type
   session_id = f"{user_id}_{conversation_type}_{timestamp}"

   # Store session metadata
   adapter.store_session_metadata(session_id, {
       "user_id": user_id,
       "conversation_type": conversation_type,
       "created_at": time.time()
   })
   ```

3. **Error Handling**

   ```python
   def safe_chain_execution(chain, input_data, session_id):
       try:
           return chain.run(input_data)
       except Exception as e:
           # Log error with context
           adapter.log_error(session_id, str(e), input_data)
           return "I encountered an error. Please try again."
   ```

4. **Context Cleanup**
   ```python
   # Regular cleanup of old conversations
   def cleanup_old_conversations(adapter, days_old=30):
       cutoff_time = time.time() - (days_old * 24 * 3600)
       return adapter.cleanup_sessions_before(cutoff_time)
   ```

## Troubleshooting

### Common LangChain Integration Issues

#### 1. Memory Overflow in Long Conversations

```python
# Problem: Long conversations exceed memory limits
# Solution: Use sliding window memory

class SlidingWindowMemory(ContextAwareMemory):
    def __init__(self, window_size=20, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def load_memory_variables(self, inputs):
        memory_vars = super().load_memory_variables(inputs)

        # Keep only recent messages in active memory
        if len(self.chat_memory.messages) > self.window_size:
            recent_messages = self.chat_memory.messages[-self.window_size:]
            self.chat_memory.messages = recent_messages

        return memory_vars
```

#### 2. Slow Message Retrieval

```python
# Problem: Slow retrieval for large conversation histories
# Solution: Implement caching and indexing

def optimize_message_retrieval(adapter):
    # Enable message indexing
    adapter.enable_message_indexing(True)

    # Use compression for large messages
    adapter.set_compression_threshold(100)

    # Enable caching for frequently accessed conversations
    adapter.enable_conversation_caching(True)
```

#### 3. Context Loss Between Chain Calls

```python
# Problem: Context not persisting between chain executions
# Solution: Explicit context passing

class PersistentContextChain:
    def __init__(self, chain, adapter, session_id):
        self.chain = chain
        self.adapter = adapter
        self.session_id = session_id

    def run(self, input_data):
        # Load context before execution
        context = self.adapter.get_session_context(self.session_id)

        # Add context to input
        enhanced_input = f"Context: {context}\nInput: {input_data}"

        # Execute chain
        result = self.chain.run(enhanced_input)

        # Store result in context
        self.adapter.update_session_context(self.session_id, result)

        return result
```

This comprehensive LangChain integration guide provides everything needed to build efficient conversational applications with Context Reference Store, from basic memory management to advanced multi-session systems with performance optimization.
