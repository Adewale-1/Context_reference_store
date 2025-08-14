# Composio Integration

This guide demonstrates how to integrate Context Reference Store with Composio for building AI agents with powerful tool integration capabilities.

## Table of Contents

- [Composio Integration](#composio-integration)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Basic Integration](#basic-integration)
  - [Tool Management](#tool-management)
  - [Authentication Context](#authentication-context)
  - [Action Execution](#action-execution)
  - [Workflow Integration](#workflow-integration)
  - [Performance Optimization](#performance-optimization)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)

## Overview

Composio integration with Context Reference Store provides:

- **Secure Tool Execution**: Context-aware authentication and authorization
- **Action History**: Persistent storage of tool execution history
- **Performance Optimization**: Intelligent caching of tool responses
- **Workflow Context**: Enhanced tool coordination in multi-step workflows
- **Error Recovery**: Context-based error handling and retry mechanisms

## Installation

```bash
# Install with Composio support
pip install context-reference-store[composio]

# Or install specific components
pip install context-reference-store composio-core cryptography
```

## Basic Integration

### Composio Adapter Setup

```python
from context_store.adapters import ComposioAdapter
from context_store import ContextReferenceStore
from composio import ComposioToolSet, App
import time

# Initialize context store and adapter
context_store = ContextReferenceStore(
    cache_size=3000,
    use_compression=True,
    eviction_policy="LRU"
)

composio_adapter = ComposioAdapter(context_store)

# Initialize Composio toolset
toolset = ComposioToolSet()

# Store authentication context securely
auth_context_id = composio_adapter.store_auth_context(
    user_id="user_123",
    auth_credentials={
        "github_token": "encrypted_token_here",
        "google_oauth": "encrypted_oauth_here"
    },
    encryption_key="your_encryption_key"
)

# Execute tool with context
result = composio_adapter.execute_tool_with_context(
    app="github",
    action="create_issue",
    params={
        "title": "New feature request",
        "body": "Description of the feature",
        "labels": ["enhancement"]
    },
    auth_context_id=auth_context_id,
    execution_metadata={
        "user_id": "user_123",
        "session_id": "session_456"
    }
)

print(f"Tool execution result: {result}")
```

### Enhanced Tool Execution

```python
class ContextAwareComposioAgent:
    """Composio agent with context store integration."""

    def __init__(self, composio_adapter: ComposioAdapter):
        self.adapter = composio_adapter
        self.toolset = ComposioToolSet()
        self.execution_history = []
        self.auth_contexts = {}

    def setup_app_authentication(self, app_name: str, auth_data: dict, user_id: str) -> str:
        """Setup authentication for an app with context storage."""

        # Store encrypted authentication context
        auth_context_id = self.adapter.store_app_auth_context(
            app_name=app_name,
            user_id=user_id,
            auth_data=auth_data,
            metadata={
                "created_at": time.time(),
                "app_name": app_name,
                "user_id": user_id
            }
        )

        self.auth_contexts[f"{user_id}_{app_name}"] = auth_context_id

        return auth_context_id

    def execute_action(self, app_name: str, action_name: str, params: dict,
                      user_id: str, execution_context: dict = None) -> dict:
        """Execute Composio action with full context tracking."""

        execution_context = execution_context or {}
        execution_start_time = time.time()

        # Get authentication context
        auth_key = f"{user_id}_{app_name}"
        if auth_key not in self.auth_contexts:
            raise ValueError(f"No authentication context found for {app_name} and user {user_id}")

        auth_context_id = self.auth_contexts[auth_key]

        # Store execution intent
        execution_intent = {
            "app_name": app_name,
            "action_name": action_name,
            "params": params,
            "user_id": user_id,
            "auth_context_id": auth_context_id,
            "execution_context": execution_context,
            "started_at": execution_start_time,
            "status": "executing"
        }

        execution_intent_id = self.adapter.store_execution_intent(execution_intent)

        try:
            # Execute action via Composio
            execution_result = self.toolset.execute_action(
                app=app_name,
                action=action_name,
                params=params
            )

            execution_time = (time.time() - execution_start_time) * 1000

            # Store successful execution result
            execution_record = {
                "execution_intent_id": execution_intent_id,
                "result": execution_result,
                "status": "success",
                "execution_time_ms": execution_time,
                "completed_at": time.time()
            }

            execution_record_id = self.adapter.store_execution_result(execution_record)

            # Update execution history
            self.execution_history.append({
                "execution_intent_id": execution_intent_id,
                "execution_record_id": execution_record_id,
                "app_name": app_name,
                "action_name": action_name,
                "timestamp": time.time(),
                "status": "success"
            })

            return {
                "success": True,
                "result": execution_result,
                "execution_time_ms": execution_time,
                "execution_record_id": execution_record_id
            }

        except Exception as e:
            execution_time = (time.time() - execution_start_time) * 1000

            # Store error execution result
            error_record = {
                "execution_intent_id": execution_intent_id,
                "error": str(e),
                "status": "error",
                "execution_time_ms": execution_time,
                "completed_at": time.time()
            }

            error_record_id = self.adapter.store_execution_result(error_record)

            # Update execution history
            self.execution_history.append({
                "execution_intent_id": execution_intent_id,
                "execution_record_id": error_record_id,
                "app_name": app_name,
                "action_name": action_name,
                "timestamp": time.time(),
                "status": "error"
            })

            # Store error context for analysis
            self.adapter.store_error_context(
                app_name=app_name,
                action_name=action_name,
                error=str(e),
                params=params,
                execution_context=execution_context
            )

            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
                "error_record_id": error_record_id
            }

    def get_app_suggestions(self, user_query: str, user_id: str) -> list:
        """Get app suggestions based on user query and context."""

        # Get user's execution history for context
        user_history = self.adapter.get_user_execution_history(
            user_id=user_id,
            limit=50
        )

        # Get app usage patterns
        app_patterns = self.adapter.get_user_app_patterns(user_id)

        # Analyze query for app suggestions
        suggestions = self.adapter.analyze_query_for_apps(
            query=user_query,
            user_history=user_history,
            app_patterns=app_patterns
        )

        return suggestions

    def get_execution_analytics(self, user_id: str = None) -> dict:
        """Get execution analytics."""

        if user_id:
            # User-specific analytics
            user_executions = [e for e in self.execution_history if self.get_execution_user_id(e) == user_id]
            total_executions = len(user_executions)
            successful_executions = len([e for e in user_executions if e["status"] == "success"])
        else:
            # Global analytics
            total_executions = len(self.execution_history)
            successful_executions = len([e for e in self.execution_history if e["status"] == "success"])

        success_rate = successful_executions / total_executions if total_executions > 0 else 0

        # Get app usage distribution
        app_usage = {}
        for execution in self.execution_history:
            app_name = execution["app_name"]
            app_usage[app_name] = app_usage.get(app_name, 0) + 1

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": success_rate,
            "app_usage_distribution": app_usage,
            "most_used_app": max(app_usage.items(), key=lambda x: x[1])[0] if app_usage else None
        }

# Usage example
agent = ContextAwareComposioAgent(composio_adapter)

# Setup authentication for GitHub
github_auth_id = agent.setup_app_authentication(
    "github",
    {"token": "github_personal_access_token"},
    "user_123"
)

# Execute GitHub action
result = agent.execute_action(
    app_name="github",
    action_name="create_issue",
    params={
        "repo": "owner/repository",
        "title": "Bug report from agent",
        "body": "Automated bug report"
    },
    user_id="user_123",
    execution_context={"source": "automated_testing"}
)

print(f"GitHub action result: {result}")
```

## Tool Management

### Smart Tool Selection

```python
class ContextAwareToolManager:
    """Manage tools with context-aware selection and optimization."""

    def __init__(self, composio_adapter: ComposioAdapter):
        self.adapter = composio_adapter
        self.available_tools = {}
        self.tool_performance = {}
        self.user_preferences = {}

    def register_available_tools(self, tools_config: dict):
        """Register available tools with their configurations."""

        for app_name, tool_config in tools_config.items():
            self.available_tools[app_name] = tool_config

            # Store tool configuration in context
            self.adapter.store_tool_configuration(
                app_name=app_name,
                config=tool_config,
                registration_time=time.time()
            )

    def select_optimal_tool(self, user_intent: str, user_id: str,
                          context: dict = None) -> dict:
        """Select optimal tool based on intent and context."""

        context = context or {}

        # Get user's tool usage history
        usage_history = self.adapter.get_user_tool_history(user_id)

        # Get tool performance data
        performance_data = self.adapter.get_tool_performance_data()

        # Analyze user intent
        intent_analysis = self.adapter.analyze_user_intent(
            intent=user_intent,
            context=context,
            usage_history=usage_history
        )

        # Score available tools
        tool_scores = {}
        for app_name, tool_config in self.available_tools.items():
            score = self.calculate_tool_score(
                app_name=app_name,
                tool_config=tool_config,
                intent_analysis=intent_analysis,
                performance_data=performance_data.get(app_name, {}),
                user_history=usage_history
            )
            tool_scores[app_name] = score

        # Select best tool
        best_tool = max(tool_scores.items(), key=lambda x: x[1])

        # Store tool selection context
        selection_context = {
            "user_intent": user_intent,
            "user_id": user_id,
            "context": context,
            "tool_scores": tool_scores,
            "selected_tool": best_tool[0],
            "selection_score": best_tool[1],
            "timestamp": time.time()
        }

        self.adapter.store_tool_selection_context(selection_context)

        return {
            "selected_tool": best_tool[0],
            "confidence_score": best_tool[1],
            "alternative_tools": sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
        }

    def calculate_tool_score(self, app_name: str, tool_config: dict,
                           intent_analysis: dict, performance_data: dict,
                           user_history: list) -> float:
        """Calculate tool selection score."""

        score = 0.0

        # Intent match score (40% weight)
        intent_keywords = intent_analysis.get("keywords", [])
        tool_keywords = tool_config.get("keywords", [])
        intent_overlap = len(set(intent_keywords).intersection(set(tool_keywords)))
        intent_score = intent_overlap / max(len(intent_keywords), 1)
        score += 0.4 * intent_score

        # Performance score (30% weight)
        avg_execution_time = performance_data.get("avg_execution_time_ms", 1000)
        success_rate = performance_data.get("success_rate", 0.5)
        performance_score = success_rate * (1 - min(avg_execution_time / 5000, 1))  # Normalize to 5 second max
        score += 0.3 * performance_score

        # User preference score (20% weight)
        user_usage_count = len([h for h in user_history if h.get("app_name") == app_name])
        preference_score = min(user_usage_count / 10, 1)  # Normalize to 10 uses
        score += 0.2 * preference_score

        # Availability score (10% weight)
        availability_score = 1.0 if tool_config.get("available", True) else 0.0
        score += 0.1 * availability_score

        return score

    def update_tool_performance(self, app_name: str, execution_result: dict):
        """Update tool performance metrics."""

        if app_name not in self.tool_performance:
            self.tool_performance[app_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_execution_time_ms": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0
            }

        perf = self.tool_performance[app_name]
        perf["total_executions"] += 1

        if execution_result.get("success", False):
            perf["successful_executions"] += 1

        execution_time = execution_result.get("execution_time_ms", 0)
        perf["total_execution_time_ms"] += execution_time

        # Update derived metrics
        perf["success_rate"] = perf["successful_executions"] / perf["total_executions"]
        perf["avg_execution_time_ms"] = perf["total_execution_time_ms"] / perf["total_executions"]

        # Store updated performance data
        self.adapter.store_tool_performance_update(app_name, perf)

# Usage example
tool_manager = ContextAwareToolManager(composio_adapter)

# Register available tools
tools_config = {
    "github": {
        "keywords": ["code", "repository", "issue", "pull request"],
        "available": True,
        "categories": ["development", "collaboration"]
    },
    "gmail": {
        "keywords": ["email", "message", "communication"],
        "available": True,
        "categories": ["communication", "productivity"]
    },
    "slack": {
        "keywords": ["team", "chat", "notification"],
        "available": True,
        "categories": ["communication", "collaboration"]
    }
}

tool_manager.register_available_tools(tools_config)

# Select optimal tool for user intent
tool_selection = tool_manager.select_optimal_tool(
    user_intent="Send a notification about the new code changes",
    user_id="user_123",
    context={"project": "web_app", "urgency": "medium"}
)

print(f"Selected tool: {tool_selection['selected_tool']}")
print(f"Confidence: {tool_selection['confidence_score']:.2f}")
```

## Authentication Context

### Secure Authentication Management

```python
class SecureAuthManager:
    """Manage authentication contexts with encryption."""

    def __init__(self, composio_adapter: ComposioAdapter, encryption_key: str):
        self.adapter = composio_adapter
        self.encryption_key = encryption_key
        self.auth_cache = {}
        self.auth_expiry = {}

    def store_user_credentials(self, user_id: str, app_name: str,
                             credentials: dict, ttl_hours: int = 24) -> str:
        """Store encrypted user credentials."""

        # Encrypt credentials
        encrypted_creds = self.adapter.encrypt_credentials(
            credentials=credentials,
            encryption_key=self.encryption_key
        )

        # Store authentication context
        auth_context = {
            "user_id": user_id,
            "app_name": app_name,
            "encrypted_credentials": encrypted_creds,
            "created_at": time.time(),
            "expires_at": time.time() + (ttl_hours * 3600),
            "access_count": 0,
            "last_accessed": None
        }

        auth_context_id = self.adapter.store_auth_context(auth_context)

        # Cache for quick access
        cache_key = f"{user_id}_{app_name}"
        self.auth_cache[cache_key] = auth_context_id
        self.auth_expiry[cache_key] = auth_context["expires_at"]

        return auth_context_id

    def get_user_credentials(self, user_id: str, app_name: str) -> dict:
        """Retrieve and decrypt user credentials."""

        cache_key = f"{user_id}_{app_name}"

        # Check cache and expiry
        if cache_key in self.auth_cache:
            if time.time() < self.auth_expiry[cache_key]:
                auth_context_id = self.auth_cache[cache_key]
            else:
                # Expired, remove from cache
                del self.auth_cache[cache_key]
                del self.auth_expiry[cache_key]
                raise ValueError(f"Authentication expired for {app_name}")
        else:
            # Not in cache, retrieve from store
            auth_context_id = self.adapter.find_auth_context(user_id, app_name)
            if not auth_context_id:
                raise ValueError(f"No authentication found for {app_name}")

        # Retrieve auth context
        auth_context = self.adapter.retrieve_auth_context(auth_context_id)

        # Check expiry
        if time.time() > auth_context["expires_at"]:
            # Clean up expired auth
            self.adapter.cleanup_expired_auth(auth_context_id)
            raise ValueError(f"Authentication expired for {app_name}")

        # Decrypt credentials
        decrypted_creds = self.adapter.decrypt_credentials(
            encrypted_credentials=auth_context["encrypted_credentials"],
            encryption_key=self.encryption_key
        )

        # Update access tracking
        auth_context["access_count"] += 1
        auth_context["last_accessed"] = time.time()
        self.adapter.update_auth_context(auth_context_id, auth_context)

        return decrypted_creds

    def refresh_credentials(self, user_id: str, app_name: str,
                          new_credentials: dict) -> str:
        """Refresh user credentials."""

        # Remove old authentication
        cache_key = f"{user_id}_{app_name}"
        if cache_key in self.auth_cache:
            old_auth_id = self.auth_cache[cache_key]
            self.adapter.revoke_auth_context(old_auth_id)
            del self.auth_cache[cache_key]
            del self.auth_expiry[cache_key]

        # Store new credentials
        return self.store_user_credentials(user_id, app_name, new_credentials)

    def cleanup_expired_auth(self):
        """Clean up expired authentication contexts."""

        current_time = time.time()
        expired_keys = []

        for cache_key, expiry_time in self.auth_expiry.items():
            if current_time > expiry_time:
                expired_keys.append(cache_key)

        for key in expired_keys:
            if key in self.auth_cache:
                auth_context_id = self.auth_cache[key]
                self.adapter.cleanup_expired_auth(auth_context_id)
                del self.auth_cache[key]
                del self.auth_expiry[key]

        return len(expired_keys)

# Usage example
auth_manager = SecureAuthManager(composio_adapter, "your_encryption_key")

# Store user credentials securely
github_auth_id = auth_manager.store_user_credentials(
    user_id="user_123",
    app_name="github",
    credentials={"token": "github_personal_access_token"},
    ttl_hours=24
)

# Later, retrieve credentials for tool execution
try:
    github_creds = auth_manager.get_user_credentials("user_123", "github")
    print("GitHub credentials retrieved successfully")
except ValueError as e:
    print(f"Authentication error: {e}")
```

## Action Execution

### Advanced Action Execution with Context

```python
class AdvancedActionExecutor:
    """Execute actions with advanced context management."""

    def __init__(self, composio_adapter: ComposioAdapter):
        self.adapter = composio_adapter
        self.execution_queue = []
        self.retry_policies = {}

    def execute_with_retry(self, app_name: str, action_name: str, params: dict,
                          user_id: str, retry_config: dict = None) -> dict:
        """Execute action with intelligent retry logic."""

        retry_config = retry_config or {
            "max_retries": 3,
            "backoff_multiplier": 2,
            "initial_delay_seconds": 1
        }

        execution_context = {
            "app_name": app_name,
            "action_name": action_name,
            "params": params,
            "user_id": user_id,
            "retry_config": retry_config,
            "attempts": []
        }

        for attempt in range(retry_config["max_retries"] + 1):
            attempt_start_time = time.time()

            try:
                # Execute action
                result = self.adapter.execute_tool(
                    app=app_name,
                    action=action_name,
                    params=params,
                    user_id=user_id
                )

                # Record successful attempt
                attempt_record = {
                    "attempt_number": attempt + 1,
                    "status": "success",
                    "execution_time_ms": (time.time() - attempt_start_time) * 1000,
                    "result": result
                }
                execution_context["attempts"].append(attempt_record)

                # Store successful execution context
                self.adapter.store_retry_execution_context(execution_context)

                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1,
                    "total_time_ms": sum(a["execution_time_ms"] for a in execution_context["attempts"])
                }

            except Exception as e:
                # Record failed attempt
                attempt_record = {
                    "attempt_number": attempt + 1,
                    "status": "failed",
                    "execution_time_ms": (time.time() - attempt_start_time) * 1000,
                    "error": str(e)
                }
                execution_context["attempts"].append(attempt_record)

                # Check if we should retry
                if attempt < retry_config["max_retries"]:
                    if self.should_retry(e, attempt, retry_config):
                        # Calculate delay
                        delay = retry_config["initial_delay_seconds"] * (retry_config["backoff_multiplier"] ** attempt)

                        # Store retry context
                        self.adapter.store_retry_attempt(
                            execution_context=execution_context,
                            attempt_number=attempt + 1,
                            next_retry_delay=delay
                        )

                        # Wait before retry
                        time.sleep(delay)
                        continue
                    else:
                        # Error not retryable
                        break
                else:
                    # Max retries reached
                    break

        # All retries failed
        self.adapter.store_failed_execution_context(execution_context)

        return {
            "success": False,
            "error": execution_context["attempts"][-1]["error"],
            "attempts": len(execution_context["attempts"]),
            "total_time_ms": sum(a["execution_time_ms"] for a in execution_context["attempts"])
        }

    def should_retry(self, error: Exception, attempt: int, retry_config: dict) -> bool:
        """Determine if error is retryable."""

        error_str = str(error).lower()

        # Retryable errors
        retryable_errors = [
            "timeout",
            "rate limit",
            "temporary",
            "service unavailable",
            "connection error"
        ]

        # Non-retryable errors
        non_retryable_errors = [
            "authentication",
            "authorization",
            "invalid credentials",
            "forbidden",
            "not found"
        ]

        # Check non-retryable first
        for non_retryable in non_retryable_errors:
            if non_retryable in error_str:
                return False

        # Check retryable
        for retryable in retryable_errors:
            if retryable in error_str:
                return True

        # Default: retry for network-related errors on early attempts
        return attempt < 2

    def batch_execute(self, actions: list, user_id: str,
                     batch_config: dict = None) -> dict:
        """Execute multiple actions in batch with context coordination."""

        batch_config = batch_config or {
            "parallel": False,
            "fail_fast": False,
            "max_concurrent": 3
        }

        batch_context = {
            "batch_id": f"batch_{int(time.time())}",
            "user_id": user_id,
            "actions": actions,
            "batch_config": batch_config,
            "started_at": time.time(),
            "results": []
        }

        batch_context_id = self.adapter.store_batch_execution_context(batch_context)

        results = []

        if batch_config["parallel"]:
            # Parallel execution (simplified - would use threading in practice)
            for action in actions:
                try:
                    result = self.execute_with_retry(
                        app_name=action["app"],
                        action_name=action["action"],
                        params=action["params"],
                        user_id=user_id,
                        retry_config=action.get("retry_config")
                    )
                    results.append(result)
                except Exception as e:
                    error_result = {"success": False, "error": str(e)}
                    results.append(error_result)

                    if batch_config["fail_fast"]:
                        break
        else:
            # Sequential execution
            for action in actions:
                try:
                    result = self.execute_with_retry(
                        app_name=action["app"],
                        action_name=action["action"],
                        params=action["params"],
                        user_id=user_id,
                        retry_config=action.get("retry_config")
                    )
                    results.append(result)
                except Exception as e:
                    error_result = {"success": False, "error": str(e)}
                    results.append(error_result)

                    if batch_config["fail_fast"]:
                        break

        # Update batch context with results
        batch_context["results"] = results
        batch_context["completed_at"] = time.time()
        batch_context["total_time_ms"] = (batch_context["completed_at"] - batch_context["started_at"]) * 1000

        self.adapter.update_batch_execution_context(batch_context_id, batch_context)

        successful_actions = len([r for r in results if r.get("success", False)])

        return {
            "batch_id": batch_context["batch_id"],
            "total_actions": len(actions),
            "successful_actions": successful_actions,
            "failed_actions": len(actions) - successful_actions,
            "results": results,
            "total_time_ms": batch_context["total_time_ms"]
        }

# Usage example
executor = AdvancedActionExecutor(composio_adapter)

# Execute single action with retry
result = executor.execute_with_retry(
    app_name="github",
    action_name="create_issue",
    params={
        "repo": "owner/repo",
        "title": "Test issue",
        "body": "Test description"
    },
    user_id="user_123",
    retry_config={
        "max_retries": 3,
        "backoff_multiplier": 2,
        "initial_delay_seconds": 1
    }
)

print(f"Action result: {result}")

# Execute batch of actions
actions = [
    {
        "app": "github",
        "action": "create_issue",
        "params": {"repo": "owner/repo", "title": "Issue 1", "body": "Description 1"}
    },
    {
        "app": "slack",
        "action": "send_message",
        "params": {"channel": "#general", "text": "Hello from agent"}
    }
]

batch_result = executor.batch_execute(
    actions=actions,
    user_id="user_123",
    batch_config={"parallel": False, "fail_fast": True}
)

print(f"Batch result: {batch_result}")
```

## Workflow Integration

### Composio Workflow with Context

```python
class ComposioWorkflowManager:
    """Manage complex workflows with Composio tools."""

    def __init__(self, composio_adapter: ComposioAdapter):
        self.adapter = composio_adapter
        self.workflows = {}
        self.workflow_executions = {}

    def define_workflow(self, workflow_name: str, workflow_definition: dict) -> str:
        """Define a workflow with context management."""

        workflow_context = {
            "name": workflow_name,
            "definition": workflow_definition,
            "created_at": time.time(),
            "version": "1.0",
            "execution_count": 0
        }

        workflow_context_id = self.adapter.store_workflow_definition(workflow_context)
        self.workflows[workflow_name] = workflow_context_id

        return workflow_context_id

    def execute_workflow(self, workflow_name: str, input_data: dict,
                        user_id: str, execution_config: dict = None) -> dict:
        """Execute workflow with full context tracking."""

        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")

        execution_config = execution_config or {}
        execution_id = f"exec_{workflow_name}_{int(time.time())}"

        # Get workflow definition
        workflow_context_id = self.workflows[workflow_name]
        workflow_definition = self.adapter.retrieve_workflow_definition(workflow_context_id)

        # Initialize workflow execution context
        execution_context = {
            "execution_id": execution_id,
            "workflow_name": workflow_name,
            "workflow_context_id": workflow_context_id,
            "user_id": user_id,
            "input_data": input_data,
            "execution_config": execution_config,
            "started_at": time.time(),
            "steps": [],
            "current_step": 0,
            "status": "running",
            "output_data": {}
        }

        execution_context_id = self.adapter.store_workflow_execution_context(execution_context)
        self.workflow_executions[execution_id] = execution_context_id

        try:
            # Execute workflow steps
            for step_index, step_definition in enumerate(workflow_definition["definition"]["steps"]):
                step_result = self.execute_workflow_step(
                    execution_context,
                    step_definition,
                    step_index
                )

                execution_context["steps"].append(step_result)
                execution_context["current_step"] = step_index + 1

                # Update execution context
                self.adapter.update_workflow_execution_context(
                    execution_context_id,
                    execution_context
                )

                # Check for step failure
                if not step_result.get("success", False):
                    if execution_config.get("fail_fast", True):
                        break

            # Determine final status
            successful_steps = len([s for s in execution_context["steps"] if s.get("success", False)])
            total_steps = len(execution_context["steps"])

            if successful_steps == total_steps:
                execution_context["status"] = "completed"
            elif successful_steps == 0:
                execution_context["status"] = "failed"
            else:
                execution_context["status"] = "partial"

            execution_context["completed_at"] = time.time()
            execution_context["total_time_ms"] = (execution_context["completed_at"] - execution_context["started_at"]) * 1000

            # Final context update
            self.adapter.update_workflow_execution_context(
                execution_context_id,
                execution_context
            )

            return {
                "execution_id": execution_id,
                "status": execution_context["status"],
                "steps_completed": len(execution_context["steps"]),
                "steps_successful": successful_steps,
                "total_time_ms": execution_context["total_time_ms"],
                "output_data": execution_context["output_data"]
            }

        except Exception as e:
            # Handle workflow execution error
            execution_context["status"] = "error"
            execution_context["error"] = str(e)
            execution_context["completed_at"] = time.time()

            self.adapter.update_workflow_execution_context(
                execution_context_id,
                execution_context
            )

            return {
                "execution_id": execution_id,
                "status": "error",
                "error": str(e),
                "steps_completed": len(execution_context["steps"])
            }

    def execute_workflow_step(self, execution_context: dict,
                            step_definition: dict, step_index: int) -> dict:
        """Execute a single workflow step."""

        step_start_time = time.time()

        step_context = {
            "step_index": step_index,
            "step_name": step_definition.get("name", f"step_{step_index}"),
            "step_type": step_definition.get("type", "action"),
            "started_at": step_start_time
        }

        try:
            if step_definition["type"] == "action":
                # Execute Composio action
                result = self.adapter.execute_tool(
                    app=step_definition["app"],
                    action=step_definition["action"],
                    params=step_definition.get("params", {}),
                    user_id=execution_context["user_id"]
                )

                step_context.update({
                    "success": True,
                    "result": result,
                    "execution_time_ms": (time.time() - step_start_time) * 1000
                })

            elif step_definition["type"] == "condition":
                # Evaluate condition
                condition_result = self.evaluate_condition(
                    step_definition["condition"],
                    execution_context
                )

                step_context.update({
                    "success": True,
                    "condition_result": condition_result,
                    "execution_time_ms": (time.time() - step_start_time) * 1000
                })

            elif step_definition["type"] == "transform":
                # Transform data
                transform_result = self.transform_data(
                    step_definition["transform"],
                    execution_context
                )

                step_context.update({
                    "success": True,
                    "transform_result": transform_result,
                    "execution_time_ms": (time.time() - step_start_time) * 1000
                })

                # Update output data
                execution_context["output_data"].update(transform_result)

            return step_context

        except Exception as e:
            step_context.update({
                "success": False,
                "error": str(e),
                "execution_time_ms": (time.time() - step_start_time) * 1000
            })

            return step_context

    def evaluate_condition(self, condition: dict, context: dict) -> bool:
        """Evaluate workflow condition."""
        # Simplified condition evaluation
        return True

    def transform_data(self, transform: dict, context: dict) -> dict:
        """Transform workflow data."""
        # Simplified data transformation
        return {"transformed": True}

# Usage example
workflow_manager = ComposioWorkflowManager(composio_adapter)

# Define a workflow
workflow_definition = {
    "steps": [
        {
            "name": "create_github_issue",
            "type": "action",
            "app": "github",
            "action": "create_issue",
            "params": {
                "repo": "owner/repo",
                "title": "Workflow created issue",
                "body": "This issue was created by a workflow"
            }
        },
        {
            "name": "notify_team",
            "type": "action",
            "app": "slack",
            "action": "send_message",
            "params": {
                "channel": "#dev-team",
                "text": "New GitHub issue created by workflow"
            }
        }
    ]
}

workflow_id = workflow_manager.define_workflow("issue_creation_workflow", workflow_definition)

# Execute workflow
workflow_result = workflow_manager.execute_workflow(
    workflow_name="issue_creation_workflow",
    input_data={"priority": "high"},
    user_id="user_123",
    execution_config={"fail_fast": True}
)

print(f"Workflow execution result: {workflow_result}")
```

## Performance Optimization

### Composio Performance Tips

```python
# Optimal configuration for Composio integration
optimized_store = ContextReferenceStore(
    cache_size=4000,              # Cache for tool executions
    use_compression=True,         # Compress tool responses
    compression_algorithm="lz4",  # Fast compression for API responses
    eviction_policy="LFU",        # Keep frequently used tools in cache
    use_disk_storage=True,        # Store execution history on disk
    memory_threshold_mb=300       # Reasonable threshold for tools
)

# Optimized adapter configuration
composio_adapter = ComposioAdapter(
    context_store=optimized_store,
    enable_response_caching=True,
    cache_ttl_seconds=3600,
    enable_auth_caching=True,
    batch_execution_size=10
)

# Performance monitoring
def monitor_composio_performance(adapter: ComposioAdapter):
    """Monitor Composio adapter performance."""

    stats = adapter.get_tool_performance_stats()

    print(f"Composio Performance Metrics:")
    print(f"  Average tool execution: {stats['avg_execution_time_ms']:.2f}ms")
    print(f"  Tool success rate: {stats['success_rate']:.2%}")
    print(f"  Auth cache hit rate: {stats['auth_cache_hit_rate']:.2%}")
    print(f"  Response cache hit rate: {stats['response_cache_hit_rate']:.2%}")
    print(f"  Total tool executions: {stats['total_executions']}")

    # Performance recommendations
    if stats['avg_execution_time_ms'] > 3000:
        print("⚠️  Slow tool execution - consider response caching")

    if stats['success_rate'] < 0.9:
        print("⚠️  Low success rate - check authentication and error handling")
```

## Best Practices

### Composio Integration Best Practices

1. **Secure Authentication**

   ```python
   # Always encrypt stored credentials
   auth_manager = SecureAuthManager(adapter, encryption_key)

   # Set appropriate TTL for credentials
   auth_manager.store_user_credentials(
       user_id, app_name, credentials, ttl_hours=24
   )
   ```

2. **Error Handling**

   ```python
   # Implement proper retry logic
   def safe_tool_execution(app, action, params, user_id):
       try:
           return executor.execute_with_retry(app, action, params, user_id)
       except Exception as e:
           adapter.log_tool_error(app, action, str(e))
           return {"success": False, "error": str(e)}
   ```

3. **Performance Optimization**

   ```python
   # Cache frequently used tool responses
   adapter.enable_response_caching(True, ttl=3600)

   # Use batch execution for multiple tools
   batch_result = executor.batch_execute(actions, user_id)
   ```

## Troubleshooting

### Common Composio Integration Issues

#### 1. Authentication Failures

```python
# Problem: Authentication tokens expire or become invalid
# Solution: Implement automatic token refresh

def handle_auth_failure(app_name, user_id, error):
    if "authentication" in str(error).lower():
        # Trigger credential refresh
        adapter.trigger_credential_refresh(user_id, app_name)
        return True
    return False
```

#### 2. Rate Limiting

```python
# Problem: API rate limits causing failures
# Solution: Implement intelligent rate limiting

def handle_rate_limit(app_name, error):
    if "rate limit" in str(error).lower():
        # Extract retry-after header if available
        retry_after = extract_retry_after(error)
        adapter.set_rate_limit_backoff(app_name, retry_after)
        return True
    return False
```

#### 3. Tool Response Caching Issues

```python
# Problem: Stale cached responses
# Solution: Implement cache invalidation

def invalidate_tool_cache(app_name, action_name, conditions):
    cache_keys = adapter.get_cache_keys(app_name, action_name)
    for key in cache_keys:
        if meets_invalidation_conditions(key, conditions):
            adapter.invalidate_cache_entry(key)
```

This comprehensive Composio integration guide provides everything needed to build sophisticated tool-integrated AI applications with Context Reference Store, from basic tool execution to complex workflow management with secure authentication and performance optimization.
