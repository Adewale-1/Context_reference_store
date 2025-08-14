# LangGraph Integration

This guide demonstrates how to integrate Context Reference Store with LangGraph for building stateful, graph-based agent workflows.

## Table of Contents

- [LangGraph Integration](#langgraph-integration)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Basic LangGraph Integration](#basic-langgraph-integration)
  - [State Management](#state-management)
  - [Graph Workflows](#graph-workflows)
  - [Conditional Routing](#conditional-routing)
  - [Parallel Processing](#parallel-processing)
  - [Checkpointing and Recovery](#checkpointing-and-recovery)
  - [Multi-Agent Graphs](#multi-agent-graphs)
  - [Performance Optimization](#performance-optimization)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)

## Overview

LangGraph integration with Context Reference Store provides:

- **Stateful Workflows**: Efficient state management across graph nodes
- **Graph Checkpointing**: Automatic checkpoint creation and recovery
- **Parallel Execution**: Optimized context sharing for parallel nodes
- **Conditional Routing**: Context-aware decision making in graph flows
- **Workflow Memory**: Persistent workflow execution history

## Installation

```bash
# Install with LangGraph support
pip install context-reference-store[langgraph]

# Or install specific components
pip install context-reference-store langgraph
```

## Basic LangGraph Integration

### LangGraph Adapter Setup

```python
from context_store.adapters import LangGraphAdapter
from context_store import ContextReferenceStore
from langgraph import StateGraph, START, END
from typing import TypedDict, Annotated, List
import operator
import time

# Initialize context store and adapter
context_store = ContextReferenceStore(
    cache_size=3000,
    use_compression=True,
    eviction_policy="LRU"
)

langgraph_adapter = LangGraphAdapter(context_store)

# Define state structure
class WorkflowState(TypedDict):
    messages: Annotated[List[str], operator.add]
    context_id: str
    current_step: str
    step_history: List[str]
    shared_data: dict

# Create context-aware node functions
def context_aware_node(state: WorkflowState):
    """Example node with context integration."""

    # Store current state in context
    state_context_id = langgraph_adapter.store_node_state(
        graph_id="example_workflow",
        node_name="context_aware_node",
        state=state,
        metadata={
            "timestamp": time.time(),
            "step_count": len(state["step_history"])
        }
    )

    # Get historical context for this node
    historical_context = langgraph_adapter.get_node_history(
        graph_id="example_workflow",
        node_name="context_aware_node",
        limit=5
    )

    # Process with historical awareness
    new_message = f"Processed step {len(state['step_history'])} with {len(historical_context)} historical executions"

    # Update state
    state["messages"].append(new_message)
    state["context_id"] = state_context_id
    state["current_step"] = "context_aware_node"
    state["step_history"].append(f"context_aware_node_{time.time()}")

    return state

# Basic workflow setup
workflow = StateGraph(WorkflowState)
workflow.add_node("process", context_aware_node)
workflow.add_edge(START, "process")
workflow.add_edge("process", END)

# Compile with context integration
app = workflow.compile()

# Execute workflow
initial_state = {
    "messages": [],
    "context_id": "",
    "current_step": "start",
    "step_history": [],
    "shared_data": {}
}

result = app.invoke(initial_state)
print(f"Workflow completed: {result}")
```

### Enhanced State Management

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

class ContextAwareCheckpointer:
    """Enhanced checkpointer with context store integration."""

    def __init__(self, langgraph_adapter: LangGraphAdapter):
        self.adapter = langgraph_adapter
        self.memory_saver = MemorySaver()

    def save_checkpoint(self, graph_id: str, thread_id: str, state: dict) -> str:
        """Save checkpoint with context store."""

        # Save to LangGraph memory
        checkpoint_id = self.memory_saver.put(
            {"configurable": {"thread_id": thread_id}},
            state
        )

        # Enhanced save to context store
        context_checkpoint_id = self.adapter.store_checkpoint(
            graph_id=graph_id,
            thread_id=thread_id,
            state=state,
            checkpoint_metadata={
                "langgraph_checkpoint_id": checkpoint_id,
                "created_at": time.time(),
                "state_size": len(str(state))
            }
        )

        return context_checkpoint_id

    def load_checkpoint(self, graph_id: str, thread_id: str, checkpoint_id: str = None) -> dict:
        """Load checkpoint from context store."""

        if checkpoint_id:
            # Load specific checkpoint
            checkpoint_data = self.adapter.retrieve_checkpoint(checkpoint_id)
            return checkpoint_data["state"]
        else:
            # Load latest checkpoint
            latest_checkpoint = self.adapter.get_latest_checkpoint(graph_id, thread_id)
            return latest_checkpoint["state"] if latest_checkpoint else {}

    def list_checkpoints(self, graph_id: str, thread_id: str) -> List[dict]:
        """List all checkpoints for a thread."""

        return self.adapter.list_thread_checkpoints(graph_id, thread_id)

# Usage with enhanced checkpointing
checkpointer = ContextAwareCheckpointer(langgraph_adapter)
```

## State Management

### Advanced State Handling

```python
class StatefulWorkflowManager:
    """Manage stateful workflows with context store."""

    def __init__(self, langgraph_adapter: LangGraphAdapter):
        self.adapter = langgraph_adapter
        self.active_workflows = {}

    def create_workflow_state(self, workflow_id: str, initial_state: dict) -> str:
        """Create and track workflow state."""

        # Store initial state
        state_id = self.adapter.store_workflow_state(
            workflow_id=workflow_id,
            state=initial_state,
            metadata={
                "created_at": time.time(),
                "status": "initialized",
                "version": 1
            }
        )

        self.active_workflows[workflow_id] = {
            "state_id": state_id,
            "created_at": time.time(),
            "current_version": 1
        }

        return state_id

    def update_workflow_state(self, workflow_id: str, state_updates: dict) -> str:
        """Update workflow state with versioning."""

        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Get current state
        current_state_id = self.active_workflows[workflow_id]["state_id"]
        current_state = self.adapter.retrieve_workflow_state(current_state_id)

        # Apply updates
        updated_state = {**current_state, **state_updates}

        # Store new version
        new_version = self.active_workflows[workflow_id]["current_version"] + 1
        new_state_id = self.adapter.store_workflow_state(
            workflow_id=workflow_id,
            state=updated_state,
            metadata={
                "updated_at": time.time(),
                "status": "updated",
                "version": new_version,
                "previous_version": current_state_id
            }
        )

        # Update tracking
        self.active_workflows[workflow_id]["state_id"] = new_state_id
        self.active_workflows[workflow_id]["current_version"] = new_version

        return new_state_id

    def get_state_history(self, workflow_id: str) -> List[dict]:
        """Get complete state history for workflow."""

        return self.adapter.get_workflow_state_history(workflow_id)

    def rollback_state(self, workflow_id: str, target_version: int) -> str:
        """Rollback workflow state to specific version."""

        state_history = self.get_state_history(workflow_id)

        target_state = None
        for state_record in state_history:
            if state_record["metadata"]["version"] == target_version:
                target_state = state_record
                break

        if not target_state:
            raise ValueError(f"Version {target_version} not found")

        # Create new state as rollback
        rollback_state_id = self.adapter.store_workflow_state(
            workflow_id=workflow_id,
            state=target_state["state"],
            metadata={
                "updated_at": time.time(),
                "status": "rollback",
                "version": self.active_workflows[workflow_id]["current_version"] + 1,
                "rollback_to_version": target_version
            }
        )

        # Update tracking
        self.active_workflows[workflow_id]["state_id"] = rollback_state_id
        self.active_workflows[workflow_id]["current_version"] += 1

        return rollback_state_id

# Enhanced workflow state
class EnhancedWorkflowState(TypedDict):
    messages: Annotated[List[str], operator.add]
    context_id: str
    workflow_id: str
    current_node: str
    node_history: List[dict]
    shared_context: dict
    error_count: int
    execution_time: float
```

## Graph Workflows

### Complex Workflow with Context

```python
def create_analysis_workflow():
    """Create a complex analysis workflow with context integration."""

    # Define enhanced state
    class AnalysisState(TypedDict):
        input_data: str
        preprocessed_data: str
        analysis_results: dict
        final_report: str
        context_id: str
        node_contexts: dict
        workflow_metadata: dict

    def preprocess_node(state: AnalysisState):
        """Preprocess input data with context."""

        # Get preprocessing context
        preprocessing_context = langgraph_adapter.get_node_context(
            graph_id="analysis_workflow",
            node_name="preprocess",
            context_type="preprocessing_patterns"
        )

        # Apply preprocessing (simplified)
        preprocessed = f"Preprocessed: {state['input_data']}"

        # Store preprocessing context
        context_id = langgraph_adapter.store_node_execution(
            graph_id="analysis_workflow",
            node_name="preprocess",
            input_data=state["input_data"],
            output_data=preprocessed,
            context_used=preprocessing_context,
            metadata={"preprocessing_method": "standard"}
        )

        state["preprocessed_data"] = preprocessed
        state["node_contexts"]["preprocess"] = context_id

        return state

    def analyze_node(state: AnalysisState):
        """Analyze preprocessed data with historical context."""

        # Get analysis patterns from history
        analysis_patterns = langgraph_adapter.get_analysis_patterns(
            graph_id="analysis_workflow",
            data_type=type(state["preprocessed_data"]).__name__,
            limit=10
        )

        # Perform analysis
        analysis_results = {
            "data_length": len(state["preprocessed_data"]),
            "patterns_used": len(analysis_patterns),
            "analysis_type": "comprehensive",
            "confidence": 0.85
        }

        # Store analysis context
        context_id = langgraph_adapter.store_node_execution(
            graph_id="analysis_workflow",
            node_name="analyze",
            input_data=state["preprocessed_data"],
            output_data=analysis_results,
            context_used=analysis_patterns,
            metadata={"analysis_method": "pattern_based"}
        )

        state["analysis_results"] = analysis_results
        state["node_contexts"]["analyze"] = context_id

        return state

    def report_node(state: AnalysisState):
        """Generate report with full workflow context."""

        # Get complete workflow context
        workflow_context = langgraph_adapter.get_workflow_context(
            graph_id="analysis_workflow",
            include_node_contexts=True,
            include_previous_executions=True
        )

        # Generate comprehensive report
        report = f"""
Analysis Report:
- Input: {state['input_data']}
- Preprocessed: {state['preprocessed_data']}
- Results: {state['analysis_results']}
- Workflow Context: {len(workflow_context)} previous executions
- Node Contexts: {list(state['node_contexts'].keys())}
"""

        # Store final execution context
        context_id = langgraph_adapter.store_workflow_completion(
            graph_id="analysis_workflow",
            final_state=state,
            report=report,
            execution_metadata={
                "total_nodes": len(state["node_contexts"]),
                "workflow_duration": time.time() - state["workflow_metadata"]["start_time"]
            }
        )

        state["final_report"] = report
        state["context_id"] = context_id

        return state

    # Build the workflow graph
    workflow = StateGraph(AnalysisState)

    # Add nodes
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("report", report_node)

    # Add edges
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)

    return workflow.compile()

# Execute analysis workflow
analysis_app = create_analysis_workflow()

# Run with context tracking
initial_state = {
    "input_data": "Sample data for analysis",
    "preprocessed_data": "",
    "analysis_results": {},
    "final_report": "",
    "context_id": "",
    "node_contexts": {},
    "workflow_metadata": {"start_time": time.time()}
}

result = analysis_app.invoke(initial_state)
print(f"Analysis completed: {result['final_report']}")
```

## Conditional Routing

### Context-Aware Routing

```python
def create_conditional_workflow():
    """Create workflow with context-aware conditional routing."""

    class ConditionalState(TypedDict):
        input_type: str
        processing_path: str
        results: dict
        context_decisions: List[dict]
        routing_history: List[str]

    def classify_input(state: ConditionalState):
        """Classify input with historical context."""

        # Get classification history
        classification_history = langgraph_adapter.get_classification_history(
            graph_id="conditional_workflow",
            input_pattern=state["input_type"]
        )

        # Make classification decision
        if "text" in state["input_type"].lower():
            classification = "text_processing"
        elif "data" in state["input_type"].lower():
            classification = "data_processing"
        else:
            classification = "general_processing"

        # Store decision context
        decision_context = {
            "classification": classification,
            "confidence": 0.9,
            "historical_patterns": len(classification_history),
            "decision_timestamp": time.time()
        }

        langgraph_adapter.store_routing_decision(
            graph_id="conditional_workflow",
            node_name="classify",
            input_state=state,
            decision=classification,
            context=decision_context
        )

        state["processing_path"] = classification
        state["context_decisions"].append(decision_context)
        state["routing_history"].append(f"classify->{classification}")

        return state

    def text_processing_node(state: ConditionalState):
        """Process text data."""

        text_context = langgraph_adapter.get_processing_context(
            graph_id="conditional_workflow",
            processing_type="text_processing"
        )

        results = {
            "processing_type": "text",
            "word_count": len(state["input_type"].split()),
            "context_patterns": len(text_context)
        }

        state["results"] = results
        state["routing_history"].append("text_processing")

        return state

    def data_processing_node(state: ConditionalState):
        """Process data."""

        data_context = langgraph_adapter.get_processing_context(
            graph_id="conditional_workflow",
            processing_type="data_processing"
        )

        results = {
            "processing_type": "data",
            "data_length": len(state["input_type"]),
            "context_patterns": len(data_context)
        }

        state["results"] = results
        state["routing_history"].append("data_processing")

        return state

    def general_processing_node(state: ConditionalState):
        """General processing."""

        results = {
            "processing_type": "general",
            "input_length": len(state["input_type"])
        }

        state["results"] = results
        state["routing_history"].append("general_processing")

        return state

    def route_to_processor(state: ConditionalState):
        """Route to appropriate processor based on classification."""

        # Store routing decision
        langgraph_adapter.store_routing_execution(
            graph_id="conditional_workflow",
            current_state=state,
            routing_decision=state["processing_path"]
        )

        return state["processing_path"]

    # Build conditional workflow
    workflow = StateGraph(ConditionalState)

    # Add nodes
    workflow.add_node("classify", classify_input)
    workflow.add_node("text_processing", text_processing_node)
    workflow.add_node("data_processing", data_processing_node)
    workflow.add_node("general_processing", general_processing_node)

    # Add conditional edges
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_to_processor,
        {
            "text_processing": "text_processing",
            "data_processing": "data_processing",
            "general_processing": "general_processing"
        }
    )

    # All processing nodes go to END
    workflow.add_edge("text_processing", END)
    workflow.add_edge("data_processing", END)
    workflow.add_edge("general_processing", END)

    return workflow.compile()

# Test conditional routing
conditional_app = create_conditional_workflow()

test_inputs = [
    "This is a text input for processing",
    "data_analysis_2024.csv",
    "general_input_file"
]

for input_data in test_inputs:
    initial_state = {
        "input_type": input_data,
        "processing_path": "",
        "results": {},
        "context_decisions": [],
        "routing_history": []
    }

    result = conditional_app.invoke(initial_state)
    print(f"Input: {input_data}")
    print(f"Route: {' -> '.join(result['routing_history'])}")
    print(f"Results: {result['results']}")
    print()
```

## Parallel Processing

### Parallel Node Execution with Context

```python
def create_parallel_workflow():
    """Create workflow with parallel processing and shared context."""

    class ParallelState(TypedDict):
        input_data: str
        branch_results: dict
        shared_context: dict
        parallel_metadata: dict
        final_output: str

    def initialize_parallel(state: ParallelState):
        """Initialize shared context for parallel execution."""

        # Create shared context for parallel branches
        shared_context_id = langgraph_adapter.store_shared_context(
            graph_id="parallel_workflow",
            context_data={
                "input_data": state["input_data"],
                "execution_start": time.time(),
                "parallel_branches": ["branch_a", "branch_b", "branch_c"]
            },
            context_type="parallel_initialization"
        )

        state["shared_context"] = {"context_id": shared_context_id}
        state["parallel_metadata"] = {
            "initialized_at": time.time(),
            "expected_branches": 3
        }

        return state

    def branch_a_processing(state: ParallelState):
        """Process data in branch A."""

        # Get shared context
        shared_context = langgraph_adapter.retrieve_shared_context(
            state["shared_context"]["context_id"]
        )

        # Get branch-specific context
        branch_context = langgraph_adapter.get_branch_context(
            graph_id="parallel_workflow",
            branch_name="branch_a"
        )

        # Process data
        result = {
            "branch": "A",
            "processing_type": "statistical_analysis",
            "result_value": len(state["input_data"]) * 2,
            "context_influence": len(branch_context),
            "shared_context_used": bool(shared_context)
        }

        # Store branch execution
        langgraph_adapter.store_branch_execution(
            graph_id="parallel_workflow",
            branch_name="branch_a",
            input_state=state,
            branch_result=result,
            shared_context_id=state["shared_context"]["context_id"]
        )

        state["branch_results"]["branch_a"] = result

        return state

    def branch_b_processing(state: ParallelState):
        """Process data in branch B."""

        # Get shared context
        shared_context = langgraph_adapter.retrieve_shared_context(
            state["shared_context"]["context_id"]
        )

        # Get branch-specific context
        branch_context = langgraph_adapter.get_branch_context(
            graph_id="parallel_workflow",
            branch_name="branch_b"
        )

        # Process data
        result = {
            "branch": "B",
            "processing_type": "semantic_analysis",
            "result_value": len(state["input_data"].split()),
            "context_influence": len(branch_context),
            "shared_context_used": bool(shared_context)
        }

        # Store branch execution
        langgraph_adapter.store_branch_execution(
            graph_id="parallel_workflow",
            branch_name="branch_b",
            input_state=state,
            branch_result=result,
            shared_context_id=state["shared_context"]["context_id"]
        )

        state["branch_results"]["branch_b"] = result

        return state

    def branch_c_processing(state: ParallelState):
        """Process data in branch C."""

        # Get shared context
        shared_context = langgraph_adapter.retrieve_shared_context(
            state["shared_context"]["context_id"]
        )

        # Get branch-specific context
        branch_context = langgraph_adapter.get_branch_context(
            graph_id="parallel_workflow",
            branch_name="branch_c"
        )

        # Process data
        result = {
            "branch": "C",
            "processing_type": "pattern_matching",
            "result_value": state["input_data"].count('a'),
            "context_influence": len(branch_context),
            "shared_context_used": bool(shared_context)
        }

        # Store branch execution
        langgraph_adapter.store_branch_execution(
            graph_id="parallel_workflow",
            branch_name="branch_c",
            input_state=state,
            branch_result=result,
            shared_context_id=state["shared_context"]["context_id"]
        )

        state["branch_results"]["branch_c"] = result

        return state

    def merge_results(state: ParallelState):
        """Merge results from parallel branches."""

        # Get all branch execution contexts
        branch_contexts = langgraph_adapter.get_parallel_execution_context(
            graph_id="parallel_workflow",
            shared_context_id=state["shared_context"]["context_id"]
        )

        # Merge results
        merged_result = {
            "total_branches": len(state["branch_results"]),
            "combined_value": sum(r["result_value"] for r in state["branch_results"].values()),
            "processing_types": [r["processing_type"] for r in state["branch_results"].values()],
            "execution_duration": time.time() - state["parallel_metadata"]["initialized_at"]
        }

        # Store merge execution
        merge_context_id = langgraph_adapter.store_merge_execution(
            graph_id="parallel_workflow",
            branch_results=state["branch_results"],
            merged_result=merged_result,
            branch_contexts=branch_contexts
        )

        state["final_output"] = f"Parallel processing completed: {merged_result}"
        state["shared_context"]["merge_context_id"] = merge_context_id

        return state

    # Build parallel workflow
    workflow = StateGraph(ParallelState)

    # Add nodes
    workflow.add_node("initialize", initialize_parallel)
    workflow.add_node("branch_a", branch_a_processing)
    workflow.add_node("branch_b", branch_b_processing)
    workflow.add_node("branch_c", branch_c_processing)
    workflow.add_node("merge", merge_results)

    # Sequential start
    workflow.add_edge(START, "initialize")

    # Parallel branches
    workflow.add_edge("initialize", "branch_a")
    workflow.add_edge("initialize", "branch_b")
    workflow.add_edge("initialize", "branch_c")

    # Merge after all branches
    workflow.add_edge("branch_a", "merge")
    workflow.add_edge("branch_b", "merge")
    workflow.add_edge("branch_c", "merge")

    workflow.add_edge("merge", END)

    return workflow.compile()

# Test parallel processing
parallel_app = create_parallel_workflow()

initial_state = {
    "input_data": "This is sample data for parallel processing analysis",
    "branch_results": {},
    "shared_context": {},
    "parallel_metadata": {},
    "final_output": ""
}

result = parallel_app.invoke(initial_state)
print(f"Parallel workflow result: {result['final_output']}")
print(f"Branch results: {result['branch_results']}")
```

## Checkpointing and Recovery

### Advanced Checkpointing

```python
class AdvancedWorkflowCheckpointer:
    """Advanced checkpointing with context store integration."""

    def __init__(self, langgraph_adapter: LangGraphAdapter):
        self.adapter = langgraph_adapter
        self.checkpoint_strategies = {
            "every_node": self.checkpoint_every_node,
            "on_error": self.checkpoint_on_error,
            "time_based": self.checkpoint_time_based,
            "state_change": self.checkpoint_on_state_change
        }

    def create_checkpoint(self, workflow_id: str, state: dict, checkpoint_type: str = "manual") -> str:
        """Create workflow checkpoint."""

        checkpoint_data = {
            "workflow_id": workflow_id,
            "state": state,
            "checkpoint_type": checkpoint_type,
            "created_at": time.time(),
            "state_hash": hash(str(state)),
            "metadata": {
                "node_count": len(state.get("node_history", [])),
                "execution_time": state.get("execution_time", 0)
            }
        }

        return self.adapter.store_checkpoint(
            workflow_id=workflow_id,
            checkpoint_data=checkpoint_data
        )

    def recover_from_checkpoint(self, workflow_id: str, checkpoint_id: str = None) -> dict:
        """Recover workflow from checkpoint."""

        if checkpoint_id:
            checkpoint_data = self.adapter.retrieve_checkpoint(checkpoint_id)
        else:
            # Get latest checkpoint
            checkpoint_data = self.adapter.get_latest_checkpoint(workflow_id)

        if not checkpoint_data:
            raise ValueError(f"No checkpoint found for workflow {workflow_id}")

        # Log recovery
        self.adapter.log_recovery_event(
            workflow_id=workflow_id,
            checkpoint_id=checkpoint_data.get("id"),
            recovery_timestamp=time.time()
        )

        return checkpoint_data["state"]

    def checkpoint_every_node(self, workflow_id: str, state: dict, node_name: str) -> str:
        """Create checkpoint after every node execution."""

        return self.create_checkpoint(
            workflow_id=workflow_id,
            state=state,
            checkpoint_type=f"node_completion_{node_name}"
        )

    def checkpoint_on_error(self, workflow_id: str, state: dict, error: Exception) -> str:
        """Create checkpoint when error occurs."""

        error_checkpoint_data = {
            **state,
            "error_info": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_timestamp": time.time()
            }
        }

        return self.create_checkpoint(
            workflow_id=workflow_id,
            state=error_checkpoint_data,
            checkpoint_type="error_recovery"
        )

    def get_checkpoint_timeline(self, workflow_id: str) -> List[dict]:
        """Get complete checkpoint timeline for workflow."""

        return self.adapter.get_checkpoint_timeline(workflow_id)

# Workflow with automatic checkpointing
def create_checkpointed_workflow():
    """Create workflow with automatic checkpointing."""

    checkpointer = AdvancedWorkflowCheckpointer(langgraph_adapter)

    class CheckpointedState(TypedDict):
        data: str
        step_count: int
        checkpoints: List[str]
        errors: List[dict]
        execution_metadata: dict

    def step_with_checkpoint(state: CheckpointedState, step_name: str):
        """Execute step with automatic checkpointing."""

        try:
            # Simulate processing
            processed_data = f"{state['data']} -> processed by {step_name}"
            state["data"] = processed_data
            state["step_count"] += 1

            # Create checkpoint after successful execution
            checkpoint_id = checkpointer.checkpoint_every_node(
                workflow_id="checkpointed_workflow",
                state=state,
                node_name=step_name
            )

            state["checkpoints"].append(checkpoint_id)

        except Exception as e:
            # Create error checkpoint
            error_checkpoint_id = checkpointer.checkpoint_on_error(
                workflow_id="checkpointed_workflow",
                state=state,
                error=e
            )

            state["errors"].append({
                "step": step_name,
                "error": str(e),
                "checkpoint_id": error_checkpoint_id
            })

            # Re-raise error for handling
            raise

        return state

    def step_1(state: CheckpointedState):
        return step_with_checkpoint(state, "step_1")

    def step_2(state: CheckpointedState):
        return step_with_checkpoint(state, "step_2")

    def step_3(state: CheckpointedState):
        return step_with_checkpoint(state, "step_3")

    # Build workflow
    workflow = StateGraph(CheckpointedState)
    workflow.add_node("step_1", step_1)
    workflow.add_node("step_2", step_2)
    workflow.add_node("step_3", step_3)

    workflow.add_edge(START, "step_1")
    workflow.add_edge("step_1", "step_2")
    workflow.add_edge("step_2", "step_3")
    workflow.add_edge("step_3", END)

    return workflow.compile(), checkpointer

# Test checkpointed workflow
checkpointed_app, checkpointer = create_checkpointed_workflow()

initial_state = {
    "data": "initial_data",
    "step_count": 0,
    "checkpoints": [],
    "errors": [],
    "execution_metadata": {"start_time": time.time()}
}

try:
    result = checkpointed_app.invoke(initial_state)
    print(f"Workflow completed successfully")
    print(f"Checkpoints created: {len(result['checkpoints'])}")

    # Show checkpoint timeline
    timeline = checkpointer.get_checkpoint_timeline("checkpointed_workflow")
    print(f"Checkpoint timeline: {len(timeline)} entries")

except Exception as e:
    print(f"Workflow failed: {e}")

    # Recover from last checkpoint
    recovered_state = checkpointer.recover_from_checkpoint("checkpointed_workflow")
    print(f"Recovered state: {recovered_state}")
```

## Multi-Agent Graphs

### Collaborative Agent Workflows

```python
def create_multi_agent_graph():
    """Create graph with multiple collaborative agents."""

    class MultiAgentState(TypedDict):
        task_description: str
        agent_assignments: dict
        agent_results: dict
        collaboration_history: List[dict]
        final_synthesis: str

    def task_coordinator(state: MultiAgentState):
        """Coordinate task distribution among agents."""

        # Get agent capability context
        agent_capabilities = langgraph_adapter.get_agent_capabilities(
            graph_id="multi_agent_graph"
        )

        # Assign tasks based on capabilities and context
        task_assignments = {
            "research_agent": "Research relevant information",
            "analysis_agent": "Analyze gathered data",
            "synthesis_agent": "Synthesize final output"
        }

        # Store coordination context
        coordination_context_id = langgraph_adapter.store_coordination_event(
            graph_id="multi_agent_graph",
            coordinator="task_coordinator",
            assignments=task_assignments,
            agent_capabilities=agent_capabilities
        )

        state["agent_assignments"] = task_assignments
        state["collaboration_history"].append({
            "event": "task_coordination",
            "context_id": coordination_context_id,
            "timestamp": time.time()
        })

        return state

    def research_agent(state: MultiAgentState):
        """Research agent with context awareness."""

        # Get research context from previous executions
        research_context = langgraph_adapter.get_agent_context(
            graph_id="multi_agent_graph",
            agent_name="research_agent",
            task_type="research"
        )

        # Perform research (simplified)
        research_result = {
            "agent": "research_agent",
            "findings": f"Research findings for: {state['task_description']}",
            "sources": ["source1", "source2", "source3"],
            "confidence": 0.8,
            "context_patterns_used": len(research_context)
        }

        # Store agent execution
        agent_context_id = langgraph_adapter.store_agent_execution(
            graph_id="multi_agent_graph",
            agent_name="research_agent",
            task_input=state["task_description"],
            agent_output=research_result,
            collaboration_context=state["collaboration_history"]
        )

        state["agent_results"]["research_agent"] = research_result
        state["collaboration_history"].append({
            "event": "research_completion",
            "agent": "research_agent",
            "context_id": agent_context_id,
            "timestamp": time.time()
        })

        return state

    def analysis_agent(state: MultiAgentState):
        """Analysis agent that builds on research results."""

        # Get analysis context and research results
        analysis_context = langgraph_adapter.get_agent_context(
            graph_id="multi_agent_graph",
            agent_name="analysis_agent",
            task_type="analysis"
        )

        # Access research results from state
        research_data = state["agent_results"].get("research_agent", {})

        # Perform analysis
        analysis_result = {
            "agent": "analysis_agent",
            "analysis": f"Analysis of research findings: {research_data.get('findings', 'No research data')}",
            "insights": ["insight1", "insight2"],
            "recommendations": ["rec1", "rec2"],
            "confidence": 0.85,
            "context_patterns_used": len(analysis_context),
            "research_integration": bool(research_data)
        }

        # Store agent execution with collaboration context
        agent_context_id = langgraph_adapter.store_agent_execution(
            graph_id="multi_agent_graph",
            agent_name="analysis_agent",
            task_input=research_data,
            agent_output=analysis_result,
            collaboration_context=state["collaboration_history"],
            predecessor_agents=["research_agent"]
        )

        state["agent_results"]["analysis_agent"] = analysis_result
        state["collaboration_history"].append({
            "event": "analysis_completion",
            "agent": "analysis_agent",
            "context_id": agent_context_id,
            "timestamp": time.time()
        })

        return state

    def synthesis_agent(state: MultiAgentState):
        """Synthesis agent that combines all results."""

        # Get synthesis context
        synthesis_context = langgraph_adapter.get_agent_context(
            graph_id="multi_agent_graph",
            agent_name="synthesis_agent",
            task_type="synthesis"
        )

        # Get all agent results
        research_data = state["agent_results"].get("research_agent", {})
        analysis_data = state["agent_results"].get("analysis_agent", {})

        # Synthesize final output
        synthesis_result = {
            "agent": "synthesis_agent",
            "synthesis": f"Final synthesis combining research and analysis",
            "research_integration": research_data.get("findings", ""),
            "analysis_integration": analysis_data.get("analysis", ""),
            "recommendations": analysis_data.get("recommendations", []),
            "confidence": 0.9,
            "collaboration_quality": len(state["collaboration_history"])
        }

        # Store final synthesis with complete collaboration context
        final_context_id = langgraph_adapter.store_multi_agent_completion(
            graph_id="multi_agent_graph",
            participating_agents=list(state["agent_results"].keys()),
            collaboration_history=state["collaboration_history"],
            final_synthesis=synthesis_result,
            task_description=state["task_description"]
        )

        state["agent_results"]["synthesis_agent"] = synthesis_result
        state["final_synthesis"] = synthesis_result["synthesis"]
        state["collaboration_history"].append({
            "event": "synthesis_completion",
            "agent": "synthesis_agent",
            "context_id": final_context_id,
            "timestamp": time.time()
        })

        return state

    # Build multi-agent workflow
    workflow = StateGraph(MultiAgentState)

    # Add agent nodes
    workflow.add_node("coordinator", task_coordinator)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("analysis_agent", analysis_agent)
    workflow.add_node("synthesis_agent", synthesis_agent)

    # Sequential workflow with coordination
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "research_agent")
    workflow.add_edge("research_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "synthesis_agent")
    workflow.add_edge("synthesis_agent", END)

    return workflow.compile()

# Test multi-agent collaboration
multi_agent_app = create_multi_agent_graph()

initial_state = {
    "task_description": "Analyze the impact of AI on software development",
    "agent_assignments": {},
    "agent_results": {},
    "collaboration_history": [],
    "final_synthesis": ""
}

result = multi_agent_app.invoke(initial_state)
print(f"Multi-agent task completed")
print(f"Final synthesis: {result['final_synthesis']}")
print(f"Participating agents: {list(result['agent_results'].keys())}")
print(f"Collaboration events: {len(result['collaboration_history'])}")
```

## Performance Optimization

### LangGraph Performance Tips

```python
# Optimal configuration for LangGraph workflows
optimized_store = ContextReferenceStore(
    cache_size=8000,              # Large cache for workflow states
    use_compression=True,         # Compress workflow states
    compression_algorithm="lz4",  # Fast compression for state transitions
    eviction_policy="LRU",        # Good for workflow patterns
    use_disk_storage=True,        # Enable for complex workflows
    memory_threshold_mb=400       # Higher threshold for workflows
)

# Optimized adapter configuration
langgraph_adapter = LangGraphAdapter(
    context_store=optimized_store,
    enable_state_compression=True,
    checkpoint_batch_size=5,
    context_sharing_optimization=True
)

# Performance monitoring for workflows
def monitor_workflow_performance(adapter: LangGraphAdapter, workflow_id: str):
    """Monitor workflow performance metrics."""

    workflow_stats = adapter.get_workflow_performance(workflow_id)

    print(f"Workflow Performance ({workflow_id}):")
    print(f"  Average node execution: {workflow_stats['avg_node_time_ms']:.2f}ms")
    print(f"  State transition time: {workflow_stats['avg_state_transition_ms']:.2f}ms")
    print(f"  Checkpoint creation time: {workflow_stats['avg_checkpoint_time_ms']:.2f}ms")
    print(f"  Context retrieval time: {workflow_stats['avg_context_retrieval_ms']:.2f}ms")
    print(f"  Total workflow executions: {workflow_stats['execution_count']}")

    # Performance alerts
    if workflow_stats['avg_node_time_ms'] > 1000:
        print("⚠️  Slow node execution - consider optimizing node logic")

    if workflow_stats['avg_state_transition_ms'] > 100:
        print("⚠️  Slow state transitions - consider state compression")
```

## Best Practices

### LangGraph Integration Best Practices

1. **State Design**

   ```python
   # Design efficient state structures
   class OptimalState(TypedDict):
       # Keep essential data in state
       current_data: str

       # Use context references for large data
       large_data_context_id: str

       # Include workflow metadata
       workflow_metadata: dict
   ```

2. **Context Management**

   ```python
   # Store node-specific contexts
   def efficient_node(state):
       # Get only necessary context
       relevant_context = adapter.get_node_context(
           graph_id="workflow",
           node_name="current_node",
           limit=5  # Limit context size
       )

       # Process efficiently
       result = process_with_context(state, relevant_context)

       # Store compact result
       return {"processed": True, "result_ref": adapter.store(result)}
   ```

3. **Error Handling**

   ```python
   def robust_node_execution(state):
       try:
           return process_node(state)
       except Exception as e:
           # Store error context for recovery
           error_context_id = adapter.store_error_context(
               node_name="current_node",
               error=e,
               state=state
           )

           # Return recovery state
           return {"error": True, "error_context_id": error_context_id}
   ```

## Troubleshooting

### Common LangGraph Integration Issues

#### 1. Large State Serialization

```python
# Problem: Large states slow down workflow execution
# Solution: Use context references for large data

def optimize_large_state(state):
    # Store large data separately
    if len(str(state.get("large_data", ""))) > 10000:
        large_data_id = adapter.store(state["large_data"])
        state["large_data"] = {"ref": large_data_id}

    return state
```

#### 2. Context Memory Leaks

```python
# Problem: Context accumulation causes memory issues
# Solution: Implement context cleanup

def cleanup_old_contexts(adapter, workflow_id, hours_old=24):
    cutoff_time = time.time() - (hours_old * 3600)
    return adapter.cleanup_workflow_contexts(workflow_id, cutoff_time)
```

#### 3. Slow Checkpoint Recovery

```python
# Problem: Slow recovery from checkpoints
# Solution: Optimize checkpoint structure

def create_optimized_checkpoint(state):
    # Create minimal checkpoint
    essential_state = {
        key: value for key, value in state.items()
        if key in ["current_step", "essential_data", "workflow_id"]
    }

    # Store full state separately if needed
    if len(str(state)) > 50000:  # Large state
        full_state_id = adapter.store(state)
        essential_state["full_state_ref"] = full_state_id

    return essential_state
```

This comprehensive LangGraph integration guide provides everything needed to build sophisticated stateful workflows with Context Reference Store, from basic state management to complex multi-agent collaboration systems with advanced checkpointing and recovery capabilities.
