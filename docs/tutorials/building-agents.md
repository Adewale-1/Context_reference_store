# Building AI Agents with Context Reference Store

This comprehensive tutorial will guide you through building intelligent AI agents that efficiently manage large contexts using Context Reference Store.

## Table of Contents

- [Building AI Agents with Context Reference Store](#building-ai-agents-with-context-reference-store)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Agent Architecture Fundamentals](#agent-architecture-fundamentals)
  - [Building Your First Agent](#building-your-first-agent)
  - [Advanced Agent Patterns](#advanced-agent-patterns)
  - [Multi-Agent Systems](#multi-agent-systems)
  - [Agent Memory Management](#agent-memory-management)
  - [Tool Integration](#tool-integration)
  - [State Management](#state-management)
  - [Production Deployment](#production-deployment)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)

## Introduction

AI agents benefit enormously from efficient context management. Context Reference Store enables agents to:

- Handle extremely large context windows efficiently
- Share context between multiple agents
- Persist agent state across sessions
- Optimize memory usage in production environments

## Agent Architecture Fundamentals

### Core Components

Every agent using Context Reference Store should have these components:

```python
from context_store import ContextReferenceStore
from typing import Dict, List, Any, Optional
import time
import uuid

class BaseAgent:
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.config = config or {}

        # Initialize context store
        self.context_store = ContextReferenceStore(
            cache_size=self.config.get('cache_size', 1000),
            use_compression=True,
            compression_algorithm="lz4"
        )

        # Agent state
        self.state = {
            "agent_id": agent_id,
            "created_at": time.time(),
            "message_count": 0,
            "active_sessions": {},
            "capabilities": []
        }

        # Store initial state
        self.state_context_id = self.context_store.store(self.state)

    def update_state(self, updates: Dict[str, Any]):
        """Update agent state and store in context"""
        self.state.update(updates)
        self.state["last_updated"] = time.time()
        self.state_context_id = self.context_store.store(self.state)
        return self.state_context_id

    def get_state(self) -> Dict[str, Any]:
        """Retrieve current agent state"""
        return self.context_store.retrieve(self.state_context_id)
```

### Context Management Patterns

#### 1. Session-Based Context

```python
class SessionAwareAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.sessions = {}

    def create_session(self, user_id: str) -> str:
        """Create a new conversation session"""
        session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"

        session_context = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": time.time(),
            "messages": [],
            "context_history": [],
            "metadata": {}
        }

        context_id = self.context_store.store(session_context)
        self.sessions[session_id] = context_id

        return session_id

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to the session context"""
        # Retrieve current session
        session_context_id = self.sessions.get(session_id)
        if not session_context_id:
            raise ValueError(f"Session {session_id} not found")

        session_context = self.context_store.retrieve(session_context_id)

        # Add new message
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        session_context["messages"].append(message)
        session_context["last_updated"] = time.time()

        # Store updated context
        new_context_id = self.context_store.store(session_context)
        self.sessions[session_id] = new_context_id

        return new_context_id

    def get_session_context(self, session_id: str, message_limit: int = None) -> Dict:
        """Get session context with optional message limiting"""
        session_context_id = self.sessions.get(session_id)
        if not session_context_id:
            return None

        context = self.context_store.retrieve(session_context_id)

        if message_limit and len(context["messages"]) > message_limit:
            # Return limited context for efficiency
            limited_context = context.copy()
            limited_context["messages"] = context["messages"][-message_limit:]
            return limited_context

        return context
```

#### 2. Hierarchical Context

```python
class HierarchicalAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.context_hierarchy = {
            "global": None,      # Global agent context
            "session": {},       # Session-level contexts
            "conversation": {},  # Conversation-level contexts
            "turn": {}          # Turn-level contexts
        }

    def create_global_context(self, data: Dict[str, Any]) -> str:
        """Create global agent context"""
        global_context = {
            "level": "global",
            "agent_id": self.agent_id,
            "data": data,
            "created_at": time.time()
        }

        context_id = self.context_store.store(global_context)
        self.context_hierarchy["global"] = context_id
        return context_id

    def create_session_context(self, session_id: str, data: Dict[str, Any]) -> str:
        """Create session-level context"""
        session_context = {
            "level": "session",
            "session_id": session_id,
            "parent_context": self.context_hierarchy["global"],
            "data": data,
            "created_at": time.time()
        }

        context_id = self.context_store.store(session_context)
        self.context_hierarchy["session"][session_id] = context_id
        return context_id

    def get_full_context(self, session_id: str) -> Dict[str, Any]:
        """Get full hierarchical context"""
        full_context = {
            "global": None,
            "session": None,
            "conversation": None
        }

        # Get global context
        if self.context_hierarchy["global"]:
            full_context["global"] = self.context_store.retrieve(
                self.context_hierarchy["global"]
            )

        # Get session context
        if session_id in self.context_hierarchy["session"]:
            full_context["session"] = self.context_store.retrieve(
                self.context_hierarchy["session"][session_id]
            )

        return full_context
```

## Building Your First Agent

Let's build a complete conversational agent:

```python
from context_store import ContextReferenceStore
from typing import Dict, List, Any, Optional
import time
import json

class ConversationalAgent:
    def __init__(self, name: str, system_prompt: str = None):
        self.name = name
        self.system_prompt = system_prompt or f"You are {name}, a helpful AI assistant."

        # Initialize context store
        self.context_store = ContextReferenceStore(
            cache_size=1000,
            use_compression=True,
            eviction_policy="LRU"
        )

        # Agent capabilities
        self.capabilities = [
            "conversation",
            "context_management",
            "memory_retention"
        ]

        # Active conversations
        self.conversations = {}

        # Performance tracking
        self.metrics = {
            "conversations_started": 0,
            "messages_processed": 0,
            "average_response_time": 0,
            "context_efficiency": 0
        }

    def start_conversation(self, user_id: str) -> str:
        """Start a new conversation with a user"""
        conversation_id = f"conv_{user_id}_{int(time.time())}"

        # Create conversation context
        conversation_context = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "agent_name": self.name,
            "started_at": time.time(),
            "system_prompt": self.system_prompt,
            "messages": [],
            "summary": "",
            "topics": [],
            "user_preferences": {},
            "context_references": []
        }

        # Store in context store
        context_id = self.context_store.store(conversation_context)
        self.conversations[conversation_id] = context_id

        # Update metrics
        self.metrics["conversations_started"] += 1

        return conversation_id

    def process_message(self, conversation_id: str, user_message: str) -> str:
        """Process a user message and generate response"""
        start_time = time.time()

        # Get conversation context
        context_id = self.conversations.get(conversation_id)
        if not context_id:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation_context = self.context_store.retrieve(context_id)

        # Add user message to context
        user_message_obj = {
            "role": "user",
            "content": user_message,
            "timestamp": time.time(),
            "processed": False
        }

        conversation_context["messages"].append(user_message_obj)

        # Generate response (this is where you'd integrate your LLM)
        response = self.generate_response(conversation_context, user_message)

        # Add assistant response to context
        assistant_message_obj = {
            "role": "assistant",
            "content": response,
            "timestamp": time.time(),
            "generation_time_ms": (time.time() - start_time) * 1000
        }

        conversation_context["messages"].append(assistant_message_obj)
        conversation_context["last_updated"] = time.time()

        # Update conversation summary if needed
        if len(conversation_context["messages"]) % 10 == 0:
            conversation_context["summary"] = self.generate_summary(
                conversation_context["messages"]
            )

        # Store updated context
        new_context_id = self.context_store.store(conversation_context)
        self.conversations[conversation_id] = new_context_id

        # Update metrics
        self.metrics["messages_processed"] += 1
        response_time = (time.time() - start_time) * 1000
        self.update_average_response_time(response_time)

        return response

    def generate_response(self, context: Dict, user_message: str) -> str:
        """Generate response using conversation context"""
        # This is a simplified example - integrate with your preferred LLM

        # Get recent conversation history
        recent_messages = context["messages"][-5:]  # Last 5 messages

        # Build prompt with context
        prompt = f"System: {context['system_prompt']}\n\n"

        # Add conversation history
        for msg in recent_messages:
            prompt += f"{msg['role'].title()}: {msg['content']}\n"

        # Add current message
        prompt += f"User: {user_message}\nAssistant:"

        # Simulate LLM response (replace with actual LLM call)
        response = f"I understand you said: '{user_message}'. This is a simulated response."

        return response

    def generate_summary(self, messages: List[Dict]) -> str:
        """Generate conversation summary"""
        # Simplified summary generation
        user_messages = [msg for msg in messages if msg["role"] == "user"]

        if len(user_messages) < 3:
            return "Brief conversation started"

        return f"Conversation with {len(user_messages)} user messages covering various topics"

    def get_conversation_stats(self, conversation_id: str) -> Dict:
        """Get statistics for a specific conversation"""
        context_id = self.conversations.get(conversation_id)
        if not context_id:
            return {}

        context = self.context_store.retrieve(context_id)

        return {
            "message_count": len(context["messages"]),
            "duration_minutes": (time.time() - context["started_at"]) / 60,
            "topics_discussed": len(context["topics"]),
            "context_size_bytes": len(json.dumps(context)),
            "last_activity": context.get("last_updated", context["started_at"])
        }

    def update_average_response_time(self, new_time: float):
        """Update running average of response times"""
        current_avg = self.metrics["average_response_time"]
        message_count = self.metrics["messages_processed"]

        if message_count == 1:
            self.metrics["average_response_time"] = new_time
        else:
            self.metrics["average_response_time"] = (
                (current_avg * (message_count - 1) + new_time) / message_count
            )

    def get_agent_metrics(self) -> Dict:
        """Get overall agent performance metrics"""
        cache_stats = self.context_store.get_cache_stats()

        return {
            **self.metrics,
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
            "active_conversations": len(self.conversations),
            "total_contexts_stored": cache_stats.get("total_contexts", 0)
        }

# Usage Example
def demo_conversational_agent():
    # Create agent
    agent = ConversationalAgent(
        name="ContextBot",
        system_prompt="You are ContextBot, an AI assistant that demonstrates efficient context management."
    )

    # Start conversation
    conversation_id = agent.start_conversation("user123")
    print(f"Started conversation: {conversation_id}")

    # Simulate conversation
    messages = [
        "Hello, I'm working on a Python project",
        "Can you help me understand context management?",
        "What are the benefits of using a context store?",
        "How does this compare to keeping everything in memory?"
    ]

    for msg in messages:
        response = agent.process_message(conversation_id, msg)
        print(f"\nUser: {msg}")
        print(f"Agent: {response}")

    # Get conversation statistics
    stats = agent.get_conversation_stats(conversation_id)
    print(f"\nConversation Stats: {stats}")

    # Get agent metrics
    metrics = agent.get_agent_metrics()
    print(f"Agent Metrics: {metrics}")

if __name__ == "__main__":
    demo_conversational_agent()
```

## Advanced Agent Patterns

### 1. Reactive Agent

```python
class ReactiveAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.reaction_patterns = {}
        self.event_history = []

    def register_reaction(self, trigger: str, reaction_func: callable):
        """Register a reaction to a specific trigger"""
        self.reaction_patterns[trigger] = reaction_func

    def process_event(self, event: Dict[str, Any]) -> List[str]:
        """Process an event and trigger appropriate reactions"""
        # Store event in context
        event_context_id = self.context_store.store({
            "type": "event",
            "data": event,
            "timestamp": time.time(),
            "agent_id": self.agent_id
        })

        self.event_history.append(event_context_id)

        # Check for triggers
        reactions = []
        for trigger, reaction_func in self.reaction_patterns.items():
            if self.matches_trigger(event, trigger):
                reaction_result = reaction_func(event, self.get_context())
                reactions.append(reaction_result)

        return reactions

    def matches_trigger(self, event: Dict, trigger: str) -> bool:
        """Check if event matches trigger pattern"""
        # Simplified pattern matching
        return trigger.lower() in str(event).lower()

    def get_context(self) -> Dict:
        """Get recent context for reactions"""
        if not self.event_history:
            return {}

        # Get last 5 events for context
        recent_events = []
        for event_id in self.event_history[-5:]:
            recent_events.append(self.context_store.retrieve(event_id))

        return {"recent_events": recent_events}
```

### 2. Learning Agent

```python
class LearningAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.knowledge_base = {}
        self.learning_patterns = []

    def learn_from_interaction(self, interaction: Dict[str, Any]):
        """Learn from user interactions"""
        # Store interaction
        interaction_id = self.context_store.store({
            "type": "learning_interaction",
            "data": interaction,
            "timestamp": time.time(),
            "processed": False
        })

        # Extract patterns
        patterns = self.extract_patterns(interaction)

        # Update knowledge base
        for pattern in patterns:
            pattern_id = self.context_store.store(pattern)
            self.learning_patterns.append(pattern_id)

        # Consolidate knowledge periodically
        if len(self.learning_patterns) % 50 == 0:
            self.consolidate_knowledge()

    def extract_patterns(self, interaction: Dict) -> List[Dict]:
        """Extract learning patterns from interaction"""
        patterns = []

        # Example: Extract user preference patterns
        if "user_feedback" in interaction:
            feedback = interaction["user_feedback"]
            if feedback.get("rating", 0) > 4:
                patterns.append({
                    "type": "positive_pattern",
                    "context": interaction["context"],
                    "action": interaction["action"],
                    "confidence": 0.8
                })

        return patterns

    def consolidate_knowledge(self):
        """Consolidate learning patterns into knowledge base"""
        # Retrieve all patterns
        all_patterns = []
        for pattern_id in self.learning_patterns:
            pattern = self.context_store.retrieve(pattern_id)
            all_patterns.append(pattern)

        # Group similar patterns
        consolidated = self.group_similar_patterns(all_patterns)

        # Update knowledge base
        for group_key, patterns in consolidated.items():
            knowledge_entry = {
                "pattern_type": group_key,
                "examples": patterns,
                "confidence": self.calculate_confidence(patterns),
                "last_updated": time.time()
            }

            knowledge_id = self.context_store.store(knowledge_entry)
            self.knowledge_base[group_key] = knowledge_id

    def group_similar_patterns(self, patterns: List[Dict]) -> Dict:
        """Group similar patterns together"""
        groups = {}

        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            if pattern_type not in groups:
                groups[pattern_type] = []
            groups[pattern_type].append(pattern)

        return groups

    def calculate_confidence(self, patterns: List[Dict]) -> float:
        """Calculate confidence score for pattern group"""
        if not patterns:
            return 0.0

        total_confidence = sum(p.get("confidence", 0.5) for p in patterns)
        return min(total_confidence / len(patterns), 1.0)
```

## Multi-Agent Systems

### Agent Coordinator

```python
class AgentCoordinator:
    def __init__(self, shared_store: ContextReferenceStore):
        self.shared_store = shared_store
        self.agents = {}
        self.message_queue = []
        self.coordination_context = None

    def register_agent(self, agent_id: str, agent_instance, capabilities: List[str]):
        """Register an agent with the coordinator"""
        agent_info = {
            "agent": agent_instance,
            "capabilities": capabilities,
            "status": "active",
            "last_activity": time.time(),
            "message_count": 0
        }

        self.agents[agent_id] = agent_info

        # Store agent info in shared context
        agent_context_id = self.shared_store.store(agent_info)

        # Update coordination context
        self.update_coordination_context()

    def route_message(self, message: Dict[str, Any]) -> str:
        """Route message to most appropriate agent"""
        required_capability = message.get("required_capability")

        if required_capability:
            # Find agents with required capability
            capable_agents = [
                agent_id for agent_id, info in self.agents.items()
                if required_capability in info["capabilities"] and info["status"] == "active"
            ]

            if capable_agents:
                # Select least busy agent
                selected_agent = min(
                    capable_agents,
                    key=lambda aid: self.agents[aid]["message_count"]
                )

                # Route message
                return self.send_to_agent(selected_agent, message)

        # Default routing logic
        return self.send_to_default_agent(message)

    def send_to_agent(self, agent_id: str, message: Dict[str, Any]) -> str:
        """Send message to specific agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent_info = self.agents[agent_id]
        agent = agent_info["agent"]

        # Update agent activity
        agent_info["message_count"] += 1
        agent_info["last_activity"] = time.time()

        # Process message with agent
        response = agent.process_message(message)

        # Store interaction in shared context
        interaction_context = {
            "agent_id": agent_id,
            "message": message,
            "response": response,
            "timestamp": time.time()
        }

        self.shared_store.store(interaction_context)

        return response

    def update_coordination_context(self):
        """Update shared coordination context"""
        coordination_data = {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a["status"] == "active"]),
            "capabilities_map": {
                agent_id: info["capabilities"]
                for agent_id, info in self.agents.items()
            },
            "last_updated": time.time()
        }

        self.coordination_context = self.shared_store.store(coordination_data)

# Example multi-agent system
def create_multi_agent_system():
    # Shared context store
    shared_store = ContextReferenceStore(cache_size=5000)

    # Create coordinator
    coordinator = AgentCoordinator(shared_store)

    # Create specialized agents
    research_agent = ConversationalAgent("ResearchBot", "You are a research specialist.")
    writing_agent = ConversationalAgent("WriterBot", "You are a writing specialist.")
    analysis_agent = ConversationalAgent("AnalystBot", "You are a data analysis specialist.")

    # Register agents
    coordinator.register_agent("researcher", research_agent, ["research", "information_gathering"])
    coordinator.register_agent("writer", writing_agent, ["writing", "content_creation"])
    coordinator.register_agent("analyst", analysis_agent, ["analysis", "data_processing"])

    return coordinator

# Usage
coordinator = create_multi_agent_system()

# Route messages to appropriate agents
research_response = coordinator.route_message({
    "content": "Research the latest AI trends",
    "required_capability": "research"
})

writing_response = coordinator.route_message({
    "content": "Write a summary of the research",
    "required_capability": "writing"
})
```

## Agent Memory Management

### Long-term Memory

```python
class MemoryManagedAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)

        # Memory hierarchies
        self.working_memory = {}      # Current session
        self.short_term_memory = {}   # Recent sessions
        self.long_term_memory = {}    # Persistent knowledge

        # Memory management policies
        self.working_memory_limit = config.get("working_memory_limit", 10)
        self.short_term_memory_limit = config.get("short_term_memory_limit", 100)

    def store_in_working_memory(self, key: str, value: Any) -> str:
        """Store in working memory (highest priority, fastest access)"""
        context_id = self.context_store.store(value)
        self.working_memory[key] = context_id

        # Manage working memory size
        if len(self.working_memory) > self.working_memory_limit:
            self.consolidate_working_memory()

        return context_id

    def store_in_short_term_memory(self, key: str, value: Any) -> str:
        """Store in short-term memory (session-based)"""
        context_id = self.context_store.store(value)
        self.short_term_memory[key] = context_id

        # Manage short-term memory size
        if len(self.short_term_memory) > self.short_term_memory_limit:
            self.consolidate_short_term_memory()

        return context_id

    def store_in_long_term_memory(self, key: str, value: Any) -> str:
        """Store in long-term memory (persistent)"""
        enriched_value = {
            "content": value,
            "stored_at": time.time(),
            "access_count": 0,
            "importance_score": self.calculate_importance(value)
        }

        context_id = self.context_store.store(enriched_value)
        self.long_term_memory[key] = context_id

        return context_id

    def recall_memory(self, key: str) -> Any:
        """Recall memory from any hierarchy"""
        # Check working memory first (fastest)
        if key in self.working_memory:
            context = self.context_store.retrieve(self.working_memory[key])
            return context

        # Check short-term memory
        if key in self.short_term_memory:
            context = self.context_store.retrieve(self.short_term_memory[key])
            # Promote to working memory if frequently accessed
            self.working_memory[key] = self.short_term_memory[key]
            return context

        # Check long-term memory
        if key in self.long_term_memory:
            context_id = self.long_term_memory[key]
            context = self.context_store.retrieve(context_id)

            # Update access count
            context["access_count"] += 1
            context["last_accessed"] = time.time()

            # Store updated context
            new_context_id = self.context_store.store(context)
            self.long_term_memory[key] = new_context_id

            return context["content"]

        return None

    def consolidate_working_memory(self):
        """Move least important items from working to short-term memory"""
        # Simple LRU consolidation
        if not self.working_memory:
            return

        # Find least recently used item
        oldest_key = min(
            self.working_memory.keys(),
            key=lambda k: self.get_last_access_time(self.working_memory[k])
        )

        # Move to short-term memory
        context_id = self.working_memory.pop(oldest_key)
        self.short_term_memory[oldest_key] = context_id

    def consolidate_short_term_memory(self):
        """Move items from short-term to long-term memory"""
        if not self.short_term_memory:
            return

        # Find least important item
        oldest_key = min(
            self.short_term_memory.keys(),
            key=lambda k: self.get_importance_score(self.short_term_memory[k])
        )

        # Move to long-term memory
        context_id = self.short_term_memory.pop(oldest_key)
        self.long_term_memory[oldest_key] = context_id

    def calculate_importance(self, value: Any) -> float:
        """Calculate importance score for memory item"""
        # Simplified importance calculation
        score = 0.5  # Default score

        if isinstance(value, dict):
            # Higher score for structured data
            score += 0.2

            # Score based on content size
            if len(str(value)) > 1000:
                score += 0.1

            # Score based on metadata
            if "important" in str(value).lower():
                score += 0.3

        return min(score, 1.0)

    def get_last_access_time(self, context_id: str) -> float:
        """Get last access time for context"""
        try:
            context = self.context_store.retrieve(context_id)
            return context.get("last_accessed", 0)
        except:
            return 0

    def get_importance_score(self, context_id: str) -> float:
        """Get importance score for context"""
        try:
            context = self.context_store.retrieve(context_id)
            return context.get("importance_score", 0.5)
        except:
            return 0.5
```

## Tool Integration

### Tool-Enabled Agent

```python
from context_store.adapters import ComposioAdapter

class ToolEnabledAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)

        # Tool integration
        self.composio_adapter = ComposioAdapter()
        self.available_tools = self.load_available_tools()
        self.tool_usage_history = []

    def load_available_tools(self) -> Dict[str, Any]:
        """Load available tools from Composio"""
        tools = {
            "web_search": {
                "app": "googlesearch",
                "action": "search",
                "description": "Search the web for information"
            },
            "send_email": {
                "app": "gmail",
                "action": "send_email",
                "description": "Send email messages"
            },
            "calendar_event": {
                "app": "googlecalendar",
                "action": "create_event",
                "description": "Create calendar events"
            },
            "file_upload": {
                "app": "googledrive",
                "action": "upload_file",
                "description": "Upload files to Google Drive"
            }
        }

        return tools

    def analyze_tool_needs(self, user_message: str) -> List[str]:
        """Analyze message to determine needed tools"""
        message_lower = user_message.lower()
        needed_tools = []

        # Simple keyword-based analysis (replace with more sophisticated NLP)
        if any(word in message_lower for word in ["search", "find", "look up"]):
            needed_tools.append("web_search")

        if any(word in message_lower for word in ["email", "send", "message"]):
            needed_tools.append("send_email")

        if any(word in message_lower for word in ["schedule", "calendar", "meeting"]):
            needed_tools.append("calendar_event")

        if any(word in message_lower for word in ["upload", "file", "document"]):
            needed_tools.append("file_upload")

        return needed_tools

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and store the result"""
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool {tool_name} not available")

        tool_config = self.available_tools[tool_name]

        # Store tool execution intent
        execution_context = {
            "tool_name": tool_name,
            "tool_config": tool_config,
            "params": params,
            "timestamp": time.time(),
            "status": "executing"
        }

        execution_id = self.context_store.store(execution_context)

        try:
            # Execute tool via Composio
            result = self.composio_adapter.execute_tool(
                app=tool_config["app"],
                action=tool_config["action"],
                params=params
            )

            # Store successful result
            execution_context.update({
                "result": result,
                "status": "success",
                "completed_at": time.time()
            })

            execution_id = self.context_store.store(execution_context)
            self.tool_usage_history.append(execution_id)

            return result

        except Exception as e:
            # Store error result
            execution_context.update({
                "error": str(e),
                "status": "error",
                "completed_at": time.time()
            })

            execution_id = self.context_store.store(execution_context)
            self.tool_usage_history.append(execution_id)

            raise

    def process_message_with_tools(self, user_message: str) -> str:
        """Process message and use tools as needed"""
        # Analyze tool needs
        needed_tools = self.analyze_tool_needs(user_message)

        # Store message context
        message_context = {
            "user_message": user_message,
            "needed_tools": needed_tools,
            "timestamp": time.time(),
            "tool_results": {}
        }

        message_id = self.context_store.store(message_context)

        # Execute needed tools
        tool_results = {}
        for tool_name in needed_tools:
            try:
                # Generate parameters for tool (simplified)
                params = self.generate_tool_params(tool_name, user_message)

                # Execute tool
                result = self.execute_tool(tool_name, params)
                tool_results[tool_name] = result

            except Exception as e:
                tool_results[tool_name] = {"error": str(e)}

        # Update message context with results
        message_context["tool_results"] = tool_results
        message_context["completed_at"] = time.time()

        self.context_store.store(message_context)

        # Generate response incorporating tool results
        response = self.generate_response_with_tools(user_message, tool_results)

        return response

    def generate_tool_params(self, tool_name: str, user_message: str) -> Dict[str, Any]:
        """Generate parameters for tool execution based on user message"""
        # Simplified parameter generation (enhance with NLP)

        if tool_name == "web_search":
            return {"query": user_message, "num_results": 5}

        elif tool_name == "send_email":
            # Extract email details from message (simplified)
            return {
                "to": "example@email.com",  # Extract from message
                "subject": "Message from AI Agent",
                "body": f"Regarding your request: {user_message}"
            }

        elif tool_name == "calendar_event":
            return {
                "title": "AI Agent Created Event",
                "description": user_message,
                "start_time": "2024-01-01T10:00:00Z",  # Extract from message
                "duration_minutes": 60
            }

        elif tool_name == "file_upload":
            return {
                "file_path": "/path/to/file",  # Extract from message
                "folder_name": "AI Agent Uploads"
            }

        return {}

    def generate_response_with_tools(self, user_message: str, tool_results: Dict) -> str:
        """Generate response incorporating tool results"""
        response_parts = [f"I've processed your request: '{user_message}'"]

        for tool_name, result in tool_results.items():
            if "error" in result:
                response_parts.append(f"❌ {tool_name} failed: {result['error']}")
            else:
                response_parts.append(f"✅ {tool_name} completed successfully")

                # Add specific result details
                if tool_name == "web_search" and "results" in result:
                    response_parts.append(f"Found {len(result['results'])} search results")
                elif tool_name == "send_email":
                    response_parts.append("Email sent successfully")
                elif tool_name == "calendar_event":
                    response_parts.append("Calendar event created")
                elif tool_name == "file_upload":
                    response_parts.append("File uploaded successfully")

        return "\n".join(response_parts)

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage"""
        if not self.tool_usage_history:
            return {"total_executions": 0}

        # Analyze tool usage history
        executions = []
        for execution_id in self.tool_usage_history:
            execution = self.context_store.retrieve(execution_id)
            executions.append(execution)

        # Calculate statistics
        total_executions = len(executions)
        successful_executions = len([e for e in executions if e["status"] == "success"])
        failed_executions = len([e for e in executions if e["status"] == "error"])

        tool_usage_count = {}
        for execution in executions:
            tool_name = execution["tool_name"]
            tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "tool_usage_count": tool_usage_count,
            "most_used_tool": max(tool_usage_count.items(), key=lambda x: x[1])[0] if tool_usage_count else None
        }

# Usage example
def demo_tool_enabled_agent():
    agent = ToolEnabledAgent("ToolBot", {"cache_size": 2000})

    # Process messages that require tools
    test_messages = [
        "Search for the latest Python tutorials",
        "Send an email to update the team about our progress",
        "Schedule a meeting for tomorrow at 2 PM",
        "Upload the project documentation to drive"
    ]

    for message in test_messages:
        print(f"\nUser: {message}")
        response = agent.process_message_with_tools(message)
        print(f"Agent: {response}")

    # Get tool usage statistics
    stats = agent.get_tool_usage_stats()
    print(f"\nTool Usage Stats: {stats}")

if __name__ == "__main__":
    demo_tool_enabled_agent()
```

## State Management

### Stateful Agent with Persistence

```python
class PersistentAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)

        # State management
        self.state_history = []
        self.checkpoint_interval = config.get("checkpoint_interval", 100)  # messages
        self.max_state_history = config.get("max_state_history", 1000)

        # Load existing state if available
        self.load_state()

    def save_state(self) -> str:
        """Save current agent state"""
        state_snapshot = {
            "agent_id": self.agent_id,
            "state": self.state.copy(),
            "timestamp": time.time(),
            "message_count": self.state.get("message_count", 0),
            "performance_metrics": self.get_performance_metrics(),
            "active_sessions": list(self.get_active_sessions()),
            "version": "1.0"
        }

        state_id = self.context_store.store(state_snapshot)
        self.state_history.append(state_id)

        # Manage state history size
        if len(self.state_history) > self.max_state_history:
            # Remove oldest states
            old_states = self.state_history[:-self.max_state_history]
            self.state_history = self.state_history[-self.max_state_history:]

            # Optionally archive old states instead of losing them
            self.archive_old_states(old_states)

        return state_id

    def load_state(self) -> bool:
        """Load the most recent state if available"""
        try:
            if self.state_history:
                latest_state_id = self.state_history[-1]
                state_snapshot = self.context_store.retrieve(latest_state_id)

                self.state = state_snapshot["state"]
                return True
        except Exception as e:
            print(f"Failed to load state: {e}")

        return False

    def create_checkpoint(self) -> str:
        """Create a checkpoint of current state"""
        checkpoint_data = {
            "type": "checkpoint",
            "agent_id": self.agent_id,
            "state": self.state.copy(),
            "context_store_stats": self.context_store.get_cache_stats(),
            "timestamp": time.time(),
            "checkpoint_reason": "periodic"
        }

        checkpoint_id = self.context_store.store(checkpoint_data)

        # Add to state history
        self.state_history.append(checkpoint_id)

        return checkpoint_id

    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from a specific checkpoint"""
        try:
            checkpoint_data = self.context_store.retrieve(checkpoint_id)

            if checkpoint_data["type"] == "checkpoint":
                self.state = checkpoint_data["state"]
                print(f"Restored from checkpoint at {checkpoint_data['timestamp']}")
                return True
        except Exception as e:
            print(f"Failed to restore from checkpoint: {e}")

        return False

    def process_message_with_state(self, session_id: str, user_message: str) -> str:
        """Process message with automatic state management"""
        # Update message count
        self.state["message_count"] = self.state.get("message_count", 0) + 1

        # Process message
        response = self.process_message(session_id, user_message)

        # Auto-checkpoint if needed
        if self.state["message_count"] % self.checkpoint_interval == 0:
            checkpoint_id = self.create_checkpoint()
            print(f"Auto-checkpoint created: {checkpoint_id}")

        # Save state
        self.save_state()

        return response

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        cache_stats = self.context_store.get_cache_stats()

        return {
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
            "total_contexts": cache_stats.get("total_contexts", 0),
            "memory_usage_mb": cache_stats.get("memory_usage_mb", 0),
            "messages_processed": self.state.get("message_count", 0),
            "uptime_seconds": time.time() - self.state.get("created_at", time.time())
        }

    def get_active_sessions(self) -> Dict[str, Any]:
        """Get information about active sessions"""
        # This would be implemented based on your session management
        return self.state.get("active_sessions", {})

    def archive_old_states(self, old_state_ids: List[str]):
        """Archive old states for potential future recovery"""
        archive_data = {
            "type": "state_archive",
            "agent_id": self.agent_id,
            "archived_states": old_state_ids,
            "archived_at": time.time(),
            "archive_reason": "state_history_cleanup"
        }

        archive_id = self.context_store.store(archive_data)
        print(f"Archived {len(old_state_ids)} old states: {archive_id}")

    def get_state_timeline(self) -> List[Dict]:
        """Get timeline of state changes"""
        timeline = []

        for state_id in self.state_history[-10:]:  # Last 10 states
            try:
                state_data = self.context_store.retrieve(state_id)
                timeline.append({
                    "state_id": state_id,
                    "timestamp": state_data.get("timestamp"),
                    "message_count": state_data.get("state", {}).get("message_count", 0),
                    "type": state_data.get("type", "state_snapshot")
                })
            except Exception as e:
                print(f"Failed to retrieve state {state_id}: {e}")

        return timeline

# Usage example
def demo_persistent_agent():
    # Create persistent agent
    agent = PersistentAgent("PersistentBot", {
        "cache_size": 1000,
        "checkpoint_interval": 5,  # Checkpoint every 5 messages for demo
        "max_state_history": 20
    })

    # Simulate multiple sessions
    session_id = "demo_session"

    messages = [
        "Hello, I'm starting a new conversation",
        "Can you remember our previous interactions?",
        "What's your current state?",
        "Process this important information",
        "Create a checkpoint now",  # This will trigger auto-checkpoint
        "Continue with more messages",
        "Test state persistence",
        "Final message in this session"
    ]

    for i, message in enumerate(messages, 1):
        print(f"\nMessage {i}: {message}")
        response = agent.process_message_with_state(session_id, message)
        print(f"Response: {response}")

        # Show performance metrics periodically
        if i % 3 == 0:
            metrics = agent.get_performance_metrics()
            print(f"Performance: {metrics}")

    # Show state timeline
    timeline = agent.get_state_timeline()
    print(f"\nState Timeline: {timeline}")

    # Demonstrate state restoration
    if len(agent.state_history) >= 2:
        previous_checkpoint = agent.state_history[-2]
        print(f"\nRestoring from previous checkpoint: {previous_checkpoint}")
        agent.restore_from_checkpoint(previous_checkpoint)

if __name__ == "__main__":
    demo_persistent_agent()
```

## Production Deployment

### Production-Ready Agent

```python
import logging
import asyncio
from typing import Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ProductionConfig:
    agent_id: str
    cache_size: int = 5000
    max_workers: int = 10
    enable_monitoring: bool = True
    log_level: str = "INFO"
    checkpoint_interval: int = 1000
    backup_enabled: bool = True
    health_check_interval: int = 300  # seconds

class ProductionAgent(BaseAgent):
    def __init__(self, config: ProductionConfig):
        super().__init__(config.agent_id, config.__dict__)

        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.is_healthy = True
        self.health_check_task = None

        # Setup logging
        self.setup_logging()

        # Setup monitoring
        if config.enable_monitoring:
            self.setup_monitoring()

        # Setup health checks
        self.start_health_checks()

        self.logger.info(f"Production agent {config.agent_id} initialized")

    def setup_logging(self):
        """Setup production logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'agent_{self.config.agent_id}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"Agent-{self.config.agent_id}")

    def setup_monitoring(self):
        """Setup production monitoring"""
        from context_store.monitoring import PerformanceMonitor

        self.monitor = PerformanceMonitor()
        self.context_store.add_monitor(self.monitor)

        # Setup metrics collection
        self.metrics_history = []
        self.alert_thresholds = {
            "memory_usage_mb": 1000,
            "cache_hit_rate": 0.7,
            "avg_response_time_ms": 1000
        }

    def start_health_checks(self):
        """Start background health checks"""
        if self.config.enable_monitoring:
            self.health_check_task = asyncio.create_task(self.health_check_loop())

    async def health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.perform_health_check()
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")

    async def perform_health_check(self):
        """Perform comprehensive health check"""
        health_status = {
            "timestamp": time.time(),
            "agent_id": self.config.agent_id,
            "status": "healthy",
            "checks": {}
        }

        try:
            # Check context store health
            cache_stats = self.context_store.get_cache_stats()
            health_status["checks"]["context_store"] = {
                "status": "healthy",
                "cache_hit_rate": cache_stats.get("hit_rate", 0),
                "memory_usage_mb": cache_stats.get("memory_usage_mb", 0)
            }

            # Check memory usage
            memory_usage = cache_stats.get("memory_usage_mb", 0)
            if memory_usage > self.alert_thresholds["memory_usage_mb"]:
                health_status["checks"]["memory"] = {
                    "status": "warning",
                    "message": f"High memory usage: {memory_usage}MB"
                }
                self.logger.warning(f"High memory usage: {memory_usage}MB")

            # Check cache performance
            hit_rate = cache_stats.get("hit_rate", 0)
            if hit_rate < self.alert_thresholds["cache_hit_rate"]:
                health_status["checks"]["cache_performance"] = {
                    "status": "warning",
                    "message": f"Low cache hit rate: {hit_rate:.2%}"
                }
                self.logger.warning(f"Low cache hit rate: {hit_rate:.2%}")

            # Store health check result
            health_check_id = self.context_store.store(health_status)

            self.is_healthy = True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            self.is_healthy = False

    async def process_message_async(self, session_id: str, user_message: str) -> str:
        """Async message processing for production"""
        start_time = time.time()

        try:
            # Log incoming message
            self.logger.info(f"Processing message for session {session_id}")

            # Process in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self.process_message_sync,
                session_id,
                user_message
            )

            # Log processing time
            processing_time = (time.time() - start_time) * 1000
            self.logger.info(f"Message processed in {processing_time:.2f}ms")

            # Check for performance alerts
            if processing_time > self.alert_thresholds["avg_response_time_ms"]:
                self.logger.warning(f"Slow response time: {processing_time:.2f}ms")

            return response

        except Exception as e:
            self.logger.error(f"Message processing failed: {e}")
            raise

    def process_message_sync(self, session_id: str, user_message: str) -> str:
        """Synchronous message processing"""
        # Implementation depends on your specific agent logic
        # This is a placeholder

        context_id = self.context_store.store({
            "session_id": session_id,
            "user_message": user_message,
            "timestamp": time.time()
        })

        # Generate response (replace with your logic)
        response = f"Processed message: {user_message[:50]}..."

        return response

    def get_production_metrics(self) -> Dict[str, Any]:
        """Get comprehensive production metrics"""
        base_metrics = self.get_agent_metrics()
        cache_stats = self.context_store.get_cache_stats()

        production_metrics = {
            **base_metrics,
            "health_status": "healthy" if self.is_healthy else "unhealthy",
            "uptime_seconds": time.time() - self.state.get("created_at", time.time()),
            "thread_pool_size": self.config.max_workers,
            "cache_stats": cache_stats,
            "log_level": self.config.log_level,
            "monitoring_enabled": self.config.enable_monitoring
        }

        return production_metrics

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating graceful shutdown")

        # Cancel health checks
        if self.health_check_task:
            self.health_check_task.cancel()

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        # Final state save
        self.save_state()

        self.logger.info("Shutdown complete")

# Production deployment example
async def deploy_production_agent():
    config = ProductionConfig(
        agent_id="prod-agent-001",
        cache_size=10000,
        max_workers=20,
        enable_monitoring=True,
        log_level="INFO",
        checkpoint_interval=1000,
        backup_enabled=True
    )

    agent = ProductionAgent(config)

    try:
        # Simulate production workload
        tasks = []
        for i in range(100):
            task = agent.process_message_async(
                session_id=f"session_{i % 10}",
                user_message=f"Production message {i}"
            )
            tasks.append(task)

        # Process messages concurrently
        responses = await asyncio.gather(*tasks)

        print(f"Processed {len(responses)} messages")

        # Get production metrics
        metrics = agent.get_production_metrics()
        print(f"Production metrics: {metrics}")

    finally:
        agent.shutdown()

if __name__ == "__main__":
    asyncio.run(deploy_production_agent())
```

## Best Practices

### 1. Context Size Management

```python
def optimize_context_size(agent, context_data):
    """Optimize context size for efficient storage"""

    # Remove unnecessary metadata
    if isinstance(context_data, dict):
        optimized = {k: v for k, v in context_data.items()
                    if not k.startswith('_temp_')}
    else:
        optimized = context_data

    # Compress large text fields
    if isinstance(optimized, dict) and 'content' in optimized:
        content = optimized['content']
        if isinstance(content, str) and len(content) > 1000:
            # Use context store's compression
            compressed_id = agent.context_store.store(content)
            optimized['content'] = {'type': 'reference', 'id': compressed_id}

    return optimized
```

### 2. Error Handling

```python
class RobustAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.error_count = 0
        self.max_errors = config.get("max_errors", 100)

    def safe_process_message(self, session_id: str, message: str) -> str:
        """Process message with comprehensive error handling"""
        try:
            return self.process_message(session_id, message)

        except MemoryError:
            # Handle memory issues
            self.handle_memory_error()
            return "I'm experiencing memory issues. Please try again."

        except KeyError as e:
            # Handle missing context
            self.logger.warning(f"Missing context: {e}")
            return "I couldn't find the relevant context. Starting fresh."

        except Exception as e:
            # Handle general errors
            self.error_count += 1
            self.logger.error(f"Error processing message: {e}")

            if self.error_count > self.max_errors:
                self.logger.critical("Too many errors, initiating restart")
                self.restart_agent()

            return "I encountered an error. Please try again."

    def handle_memory_error(self):
        """Handle memory pressure"""
        # Clear some cache
        cache_stats = self.context_store.get_cache_stats()
        if cache_stats.get("memory_usage_mb", 0) > 500:
            # Force garbage collection
            import gc
            gc.collect()

            # Reduce cache size temporarily
            self.context_store.cache_size = max(self.context_store.cache_size // 2, 100)
```

### 3. Performance Monitoring

```python
class MonitoredAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.performance_history = []

    def process_with_monitoring(self, session_id: str, message: str) -> str:
        """Process message with performance monitoring"""
        start_time = time.time()
        start_memory = self.get_memory_usage()

        try:
            response = self.process_message(session_id, message)

            # Record performance metrics
            end_time = time.time()
            end_memory = self.get_memory_usage()

            performance_data = {
                "timestamp": start_time,
                "processing_time_ms": (end_time - start_time) * 1000,
                "memory_delta_mb": end_memory - start_memory,
                "message_length": len(message),
                "response_length": len(response),
                "session_id": session_id
            }

            self.performance_history.append(performance_data)

            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

            return response

        except Exception as e:
            # Record error metrics
            error_data = {
                "timestamp": start_time,
                "error": str(e),
                "error_type": type(e).__name__,
                "session_id": session_id
            }

            self.performance_history.append(error_data)
            raise

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {}

        recent_data = [d for d in self.performance_history[-100:] if 'error' not in d]

        if not recent_data:
            return {"error": "No recent successful operations"}

        processing_times = [d["processing_time_ms"] for d in recent_data]
        memory_deltas = [d["memory_delta_mb"] for d in recent_data]

        return {
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "max_processing_time_ms": max(processing_times),
            "min_processing_time_ms": min(processing_times),
            "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "total_operations": len(recent_data),
            "error_rate": len([d for d in self.performance_history[-100:] if 'error' in d]) / 100
        }
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

```python
def diagnose_memory_usage(agent):
    """Diagnose and fix high memory usage"""
    cache_stats = agent.context_store.get_cache_stats()

    print(f"Current memory usage: {cache_stats.get('memory_usage_mb', 0)}MB")
    print(f"Cache size: {cache_stats.get('total_contexts', 0)} contexts")
    print(f"Hit rate: {cache_stats.get('hit_rate', 0):.2%}")

    # Solutions:
    # 1. Reduce cache size
    if cache_stats.get('memory_usage_mb', 0) > 500:
        agent.context_store.cache_size = agent.context_store.cache_size // 2
        print("Reduced cache size")

    # 2. Enable compression
    if not hasattr(agent.context_store, 'use_compression') or not agent.context_store.use_compression:
        agent.context_store.use_compression = True
        print("Enabled compression")

    # 3. Enable disk storage
    agent.context_store.use_disk_storage = True
    agent.context_store.memory_threshold_mb = 100
    print("Enabled disk storage")
```

#### 2. Slow Response Times

```python
def optimize_response_time(agent):
    """Optimize agent response times"""

    # 1. Increase cache size for better hit rates
    cache_stats = agent.context_store.get_cache_stats()
    if cache_stats.get('hit_rate', 0) < 0.8:
        agent.context_store.cache_size = min(agent.context_store.cache_size * 2, 5000)
        print("Increased cache size for better hit rates")

    # 2. Use faster compression
    agent.context_store.compression_algorithm = "lz4"
    print("Switched to faster compression")

    # 3. Implement context pooling
    agent.context_pool = {}
    print("Enabled context pooling")
```

#### 3. Context Loss

```python
def prevent_context_loss(agent):
    """Implement context persistence and recovery"""

    # 1. Enable automatic backup
    agent.backup_enabled = True
    agent.backup_interval = 1000  # messages

    # 2. Implement context recovery
    def recover_context(context_id):
        try:
            return agent.context_store.retrieve(context_id)
        except KeyError:
            # Try backup store
            return agent.backup_store.retrieve(context_id)

    agent.recover_context = recover_context

    # 3. Add context validation
    def validate_context(context_data):
        required_fields = ['timestamp', 'session_id']
        return all(field in context_data for field in required_fields)

    agent.validate_context = validate_context
```

### Performance Tuning Checklist

1. **Cache Configuration**

   - Set appropriate cache size based on available memory
   - Choose optimal eviction policy (LRU for most cases)
   - Monitor cache hit rates (target: >80%)

2. **Compression Settings**

   - Enable compression for large contexts
   - Use LZ4 for speed, ZSTD for better compression ratios
   - Monitor compression ratios

3. **Memory Management**

   - Set memory thresholds for disk storage
   - Implement regular garbage collection
   - Monitor memory usage patterns

4. **Context Lifecycle**

   - Implement proper context cleanup
   - Use hierarchical context management
   - Archive old contexts instead of deleting

5. **Error Handling**
   - Implement comprehensive error recovery
   - Add circuit breakers for external dependencies
   - Monitor error rates and patterns

This completes the comprehensive agent building tutorial. The examples provided cover everything from basic agents to production-ready systems with advanced features like tool integration, state management, and performance monitoring.
