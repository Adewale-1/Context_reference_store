# Agent Development Kit (ADK) Integration

This guide shows how to integrate Context Reference Store with the Agent Development Kit (ADK) for building high-performance agent applications.

## Table of Contents

- [Agent Development Kit (ADK) Integration](#agent-development-kit-adk-integration)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Basic ADK Integration](#basic-adk-integration)
  - [Agent Workflow Integration](#agent-workflow-integration)
  - [Multi-Agent ADK Systems](#multi-agent-adk-systems)
  - [State Management with ADK](#state-management-with-adk)
  - [Tool Integration](#tool-integration)
  - [Advanced Patterns](#advanced-patterns)
  - [Performance Optimization](#performance-optimization)
  - [Production Deployment](#production-deployment)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)

## Overview

The Agent Development Kit (ADK) is Google's framework for building intelligent agents. Context Reference Store provides native support for ADK, enabling efficient memory management and context sharing across agent workflows.

### Key Benefits

- **Efficient State Management**: 49x memory reduction for agent state
- **Fast Context Switching**: Sub-100ms context retrieval for agent workflows
- **Seamless Integration**: Native ADK adapter with minimal code changes
- **Scalable Architecture**: Support for multi-agent systems and complex workflows

## Installation

Install Context Reference Store with ADK support:

```bash
# Basic installation with ADK support
pip install context-reference-store[adk]

# Full installation with all frameworks
pip install context-reference-store[full]
```

## Basic ADK Integration

### Setting Up the ADK Adapter

```python
from context_store.adapters import ADKAdapter
from context_store import ContextReferenceStore
import adk

# Initialize context store and adapter
context_store = ContextReferenceStore(
    cache_size=2000,
    use_compression=True,
    eviction_policy="LRU"
)

adk_adapter = ADKAdapter(context_store)

# Create context-aware ADK agent
class ContextAwareAgent(adk.Agent):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)

        # Initialize context management
        self.context_adapter = adk_adapter
        self.agent_context_id = None

        # Agent-specific context store
        self.agent_store = self.context_adapter.create_agent_store(
            agent_id=name,
            cache_size=1000
        )

    def setup(self):
        """Initialize agent with context support"""
        super().setup()

        # Create initial agent context
        initial_context = {
            "agent_name": self.name,
            "created_at": time.time(),
            "capabilities": self.get_capabilities(),
            "state": "initialized",
            "step_history": [],
            "performance_metrics": {}
        }

        self.agent_context_id = self.agent_store.store(initial_context)
        print(f"Agent {self.name} initialized with context {self.agent_context_id}")

    def process_step(self, input_data):
        """Process step with context management"""
        step_start_time = time.time()

        # Get current agent context
        agent_context = self.agent_store.retrieve(self.agent_context_id)

        # Create step context
        step_context = {
            "step_id": len(agent_context["step_history"]) + 1,
            "input_data": input_data,
            "started_at": step_start_time,
            "agent_name": self.name,
            "previous_steps": len(agent_context["step_history"])
        }

        step_context_id = self.agent_store.store(step_context)

        # Process the step
        try:
            result = self.execute_step(input_data, agent_context)

            # Update step context with result
            step_context.update({
                "result": result,
                "completed_at": time.time(),
                "status": "success",
                "processing_time_ms": (time.time() - step_start_time) * 1000
            })

        except Exception as e:
            # Handle errors
            step_context.update({
                "error": str(e),
                "completed_at": time.time(),
                "status": "error",
                "processing_time_ms": (time.time() - step_start_time) * 1000
            })
            raise

        finally:
            # Update step context
            self.agent_store.store(step_context)

            # Update agent context
            agent_context["step_history"].append(step_context_id)
            agent_context["last_updated"] = time.time()
            agent_context["total_steps"] = len(agent_context["step_history"])

            self.agent_context_id = self.agent_store.store(agent_context)

        return result

    def execute_step(self, input_data, agent_context):
        """Override this method in your specific agent"""
        # Default implementation
        return f"Processed {input_data} with {len(agent_context['step_history'])} previous steps"

    def get_capabilities(self):
        """Return agent capabilities"""
        return ["context_management", "step_processing"]

    def get_context_summary(self):
        """Get summary of agent context"""
        agent_context = self.agent_store.retrieve(self.agent_context_id)

        return {
            "agent_name": agent_context["agent_name"],
            "total_steps": len(agent_context["step_history"]),
            "current_state": agent_context.get("state", "unknown"),
            "uptime_seconds": time.time() - agent_context["created_at"],
            "last_activity": agent_context.get("last_updated", agent_context["created_at"])
        }

# Usage example
def basic_adk_example():
    # Create context-aware agent
    agent = ContextAwareAgent("ProcessingAgent")
    agent.setup()

    # Process multiple steps
    test_inputs = [
        "First input data",
        "Second input data",
        "Third input data"
    ]

    for i, input_data in enumerate(test_inputs, 1):
        print(f"\n--- Step {i} ---")
        result = agent.process_step(input_data)
        print(f"Result: {result}")

        # Show context summary
        summary = agent.get_context_summary()
        print(f"Context Summary: {summary}")

if __name__ == "__main__":
    basic_adk_example()
```

## Agent Workflow Integration

### Workflow with Context Sharing

```python
import adk
from context_store.adapters import ADKAdapter
import time

class ContextAwareWorkflow(adk.Workflow):
    def __init__(self, name: str, shared_context_store=None):
        super().__init__(name)

        # Shared context store for all agents in workflow
        if shared_context_store is None:
            shared_context_store = ContextReferenceStore(
                cache_size=5000,
                use_compression=True
            )

        self.shared_store = shared_context_store
        self.adk_adapter = ADKAdapter(shared_context_store)

        # Workflow-level context
        self.workflow_context_id = None
        self.agent_contexts = {}

    def setup(self):
        """Initialize workflow with shared context"""
        super().setup()

        # Create workflow context
        workflow_context = {
            "workflow_name": self.name,
            "created_at": time.time(),
            "agents": [],
            "execution_history": [],
            "shared_data": {},
            "status": "initialized"
        }

        self.workflow_context_id = self.shared_store.store(workflow_context)
        print(f"Workflow {self.name} initialized with shared context")

    def add_context_agent(self, agent_name: str, agent_class, agent_config: dict = None):
        """Add agent with context integration to workflow"""

        # Create agent with shared context
        agent = agent_class(agent_name, agent_config or {})
        agent.shared_store = self.shared_store
        agent.workflow_context_id = self.workflow_context_id

        # Initialize agent
        agent.setup()

        # Store agent context reference
        self.agent_contexts[agent_name] = agent.agent_context_id

        # Update workflow context
        workflow_context = self.shared_store.retrieve(self.workflow_context_id)
        workflow_context["agents"].append({
            "name": agent_name,
            "class": agent_class.__name__,
            "context_id": agent.agent_context_id,
            "added_at": time.time()
        })

        self.workflow_context_id = self.shared_store.store(workflow_context)

        return agent

    def execute_workflow(self, initial_data):
        """Execute workflow with context sharing"""
        execution_start = time.time()

        # Create execution context
        execution_context = {
            "execution_id": f"exec_{int(time.time())}",
            "initial_data": initial_data,
            "started_at": execution_start,
            "workflow_name": self.name,
            "steps": [],
            "shared_data": {}
        }

        execution_id = self.shared_store.store(execution_context)

        try:
            # Execute workflow steps
            result = self.run_workflow_steps(initial_data, execution_context)

            # Update execution context
            execution_context.update({
                "result": result,
                "completed_at": time.time(),
                "status": "success",
                "total_time_ms": (time.time() - execution_start) * 1000
            })

        except Exception as e:
            execution_context.update({
                "error": str(e),
                "completed_at": time.time(),
                "status": "error",
                "total_time_ms": (time.time() - execution_start) * 1000
            })
            raise

        finally:
            # Store final execution context
            self.shared_store.store(execution_context)

            # Update workflow context
            workflow_context = self.shared_store.retrieve(self.workflow_context_id)
            workflow_context["execution_history"].append(execution_id)
            workflow_context["last_execution"] = time.time()

            self.workflow_context_id = self.shared_store.store(workflow_context)

        return result

    def run_workflow_steps(self, data, execution_context):
        """Override this method to define workflow steps"""
        # Default implementation
        return f"Processed {data} through workflow {self.name}"

    def share_data(self, key: str, value, scope: str = "workflow"):
        """Share data between agents in workflow"""
        shared_data_context = {
            "key": key,
            "value": value,
            "scope": scope,
            "shared_at": time.time(),
            "workflow_name": self.name
        }

        shared_data_id = self.shared_store.store(shared_data_context)

        # Update workflow context
        workflow_context = self.shared_store.retrieve(self.workflow_context_id)
        workflow_context["shared_data"][key] = shared_data_id

        self.workflow_context_id = self.shared_store.store(workflow_context)

        return shared_data_id

    def get_shared_data(self, key: str):
        """Retrieve shared data"""
        workflow_context = self.shared_store.retrieve(self.workflow_context_id)

        if key in workflow_context["shared_data"]:
            shared_data_id = workflow_context["shared_data"][key]
            shared_data_context = self.shared_store.retrieve(shared_data_id)
            return shared_data_context["value"]

        return None

    def get_workflow_summary(self):
        """Get workflow execution summary"""
        workflow_context = self.shared_store.retrieve(self.workflow_context_id)

        return {
            "workflow_name": workflow_context["workflow_name"],
            "total_agents": len(workflow_context["agents"]),
            "total_executions": len(workflow_context["execution_history"]),
            "shared_data_keys": list(workflow_context["shared_data"].keys()),
            "uptime_seconds": time.time() - workflow_context["created_at"],
            "status": workflow_context.get("status", "unknown")
        }

# Specialized agents for workflow
class DataPreprocessorAgent(ContextAwareAgent):
    def execute_step(self, input_data, agent_context):
        """Preprocess data step"""

        # Simulate data preprocessing
        preprocessed_data = {
            "original": input_data,
            "cleaned": str(input_data).strip().lower(),
            "processed_at": time.time(),
            "agent": self.name
        }

        return preprocessed_data

class DataAnalyzerAgent(ContextAwareAgent):
    def execute_step(self, input_data, agent_context):
        """Analyze preprocessed data"""

        # Get data from preprocessing (if available)
        if isinstance(input_data, dict) and "cleaned" in input_data:
            data_to_analyze = input_data["cleaned"]
        else:
            data_to_analyze = str(input_data)

        # Simulate analysis
        analysis_result = {
            "input": data_to_analyze,
            "length": len(data_to_analyze),
            "word_count": len(data_to_analyze.split()),
            "analysis_type": "basic_text_analysis",
            "analyzed_at": time.time(),
            "agent": self.name
        }

        return analysis_result

class ReportGeneratorAgent(ContextAwareAgent):
    def execute_step(self, input_data, agent_context):
        """Generate report from analysis"""

        # Extract analysis data
        if isinstance(input_data, dict) and "word_count" in input_data:
            word_count = input_data["word_count"]
            length = input_data["length"]
        else:
            word_count = 0
            length = 0

        # Generate report
        report = {
            "report_type": "analysis_summary",
            "word_count": word_count,
            "character_count": length,
            "generated_at": time.time(),
            "agent": self.name,
            "summary": f"Analyzed text with {word_count} words and {length} characters"
        }

        return report

# Example workflow implementation
class DataProcessingWorkflow(ContextAwareWorkflow):
    def setup(self):
        """Setup data processing workflow"""
        super().setup()

        # Add agents to workflow
        self.preprocessor = self.add_context_agent(
            "preprocessor",
            DataPreprocessorAgent
        )

        self.analyzer = self.add_context_agent(
            "analyzer",
            DataAnalyzerAgent
        )

        self.reporter = self.add_context_agent(
            "reporter",
            ReportGeneratorAgent
        )

    def run_workflow_steps(self, data, execution_context):
        """Execute the data processing pipeline"""

        print(f"Starting data processing workflow with: {data}")

        # Step 1: Preprocess data
        preprocessed = self.preprocessor.process_step(data)
        execution_context["steps"].append({
            "step": "preprocess",
            "agent": "preprocessor",
            "completed_at": time.time()
        })

        # Share preprocessed data
        self.share_data("preprocessed_data", preprocessed)

        # Step 2: Analyze data
        analysis = self.analyzer.process_step(preprocessed)
        execution_context["steps"].append({
            "step": "analyze",
            "agent": "analyzer",
            "completed_at": time.time()
        })

        # Share analysis results
        self.share_data("analysis_results", analysis)

        # Step 3: Generate report
        report = self.reporter.process_step(analysis)
        execution_context["steps"].append({
            "step": "report",
            "agent": "reporter",
            "completed_at": time.time()
        })

        # Share final report
        self.share_data("final_report", report)

        return report

# Usage example
def workflow_integration_example():
    # Create and setup workflow
    workflow = DataProcessingWorkflow("DataPipeline")
    workflow.setup()

    # Execute workflow with different inputs
    test_inputs = [
        "This is a sample text for processing",
        "Another piece of data to analyze",
        "Final test input for the workflow"
    ]

    for i, input_data in enumerate(test_inputs, 1):
        print(f"\n=== Workflow Execution {i} ===")

        result = workflow.execute_workflow(input_data)
        print(f"Final Result: {result}")

        # Show workflow summary
        summary = workflow.get_workflow_summary()
        print(f"Workflow Summary: {summary}")

        # Show shared data
        shared_keys = ["preprocessed_data", "analysis_results", "final_report"]
        for key in shared_keys:
            shared_data = workflow.get_shared_data(key)
            if shared_data:
                print(f"Shared {key}: {shared_data}")

if __name__ == "__main__":
    workflow_integration_example()
```

## Multi-Agent ADK Systems

### Coordinated Multi-Agent System

```python
from context_store.adapters import ADKAdapter
import adk
import asyncio
from typing import Dict, List, Any
import uuid

class ADKMultiAgentCoordinator:
    def __init__(self, shared_context_store=None):
        if shared_context_store is None:
            shared_context_store = ContextReferenceStore(
                cache_size=10000,
                use_compression=True,
                use_disk_storage=True
            )

        self.shared_store = shared_context_store
        self.adk_adapter = ADKAdapter(shared_context_store)

        # System-level management
        self.system_context_id = None
        self.registered_agents = {}
        self.active_workflows = {}
        self.message_queue = asyncio.Queue()

        # Coordination policies
        self.load_balancing_enabled = True
        self.auto_scaling_enabled = True
        self.fault_tolerance_enabled = True

    def initialize_system(self):
        """Initialize multi-agent system"""
        system_context = {
            "system_id": str(uuid.uuid4()),
            "created_at": time.time(),
            "agents": {},
            "workflows": {},
            "message_history": [],
            "performance_metrics": {
                "total_messages": 0,
                "total_agents": 0,
                "total_workflows": 0,
                "system_uptime": 0
            },
            "configuration": {
                "load_balancing": self.load_balancing_enabled,
                "auto_scaling": self.auto_scaling_enabled,
                "fault_tolerance": self.fault_tolerance_enabled
            }
        }

        self.system_context_id = self.shared_store.store(system_context)
        print(f"Multi-agent system initialized: {system_context['system_id']}")

    def register_agent(self, agent: ContextAwareAgent, capabilities: List[str], max_concurrent: int = 5):
        """Register agent with the coordination system"""

        agent_info = {
            "agent": agent,
            "capabilities": capabilities,
            "max_concurrent": max_concurrent,
            "current_load": 0,
            "total_processed": 0,
            "status": "ready",
            "registered_at": time.time(),
            "last_activity": time.time()
        }

        self.registered_agents[agent.name] = agent_info

        # Update system context
        system_context = self.shared_store.retrieve(self.system_context_id)
        system_context["agents"][agent.name] = {
            "capabilities": capabilities,
            "max_concurrent": max_concurrent,
            "status": "ready",
            "context_id": agent.agent_context_id
        }
        system_context["performance_metrics"]["total_agents"] = len(self.registered_agents)

        self.system_context_id = self.shared_store.store(system_context)

        print(f"Agent {agent.name} registered with capabilities: {capabilities}")

    def register_workflow(self, workflow: ContextAwareWorkflow):
        """Register workflow with the system"""

        workflow_info = {
            "workflow": workflow,
            "registered_at": time.time(),
            "execution_count": 0,
            "status": "ready"
        }

        self.active_workflows[workflow.name] = workflow_info

        # Update system context
        system_context = self.shared_store.retrieve(self.system_context_id)
        system_context["workflows"][workflow.name] = {
            "status": "ready",
            "context_id": workflow.workflow_context_id,
            "execution_count": 0
        }
        system_context["performance_metrics"]["total_workflows"] = len(self.active_workflows)

        self.system_context_id = self.shared_store.store(system_context)

        print(f"Workflow {workflow.name} registered")

    async def route_message(self, message: Dict[str, Any]) -> Any:
        """Route message to appropriate agent or workflow"""

        # Store message in system context
        message_context = {
            "message_id": str(uuid.uuid4()),
            "content": message,
            "received_at": time.time(),
            "routing_status": "pending",
            "assigned_to": None
        }

        message_id = self.shared_store.store(message_context)

        try:
            # Determine routing target
            if "workflow" in message:
                # Route to workflow
                result = await self.route_to_workflow(message, message_id)
            else:
                # Route to agent
                result = await self.route_to_agent(message, message_id)

            # Update message context
            message_context.update({
                "result": result,
                "completed_at": time.time(),
                "routing_status": "completed"
            })

        except Exception as e:
            message_context.update({
                "error": str(e),
                "completed_at": time.time(),
                "routing_status": "failed"
            })
            raise

        finally:
            self.shared_store.store(message_context)
            self.update_system_metrics()

        return result

    async def route_to_agent(self, message: Dict[str, Any], message_id: str) -> Any:
        """Route message to best available agent"""

        required_capability = message.get("capability", "general")

        # Find capable agents
        capable_agents = [
            name for name, info in self.registered_agents.items()
            if required_capability in info["capabilities"] and info["status"] == "ready"
        ]

        if not capable_agents:
            raise ValueError(f"No agents available for capability: {required_capability}")

        # Select best agent (load balancing)
        if self.load_balancing_enabled:
            selected_agent_name = min(
                capable_agents,
                key=lambda name: self.registered_agents[name]["current_load"]
            )
        else:
            selected_agent_name = capable_agents[0]

        # Execute with selected agent
        agent_info = self.registered_agents[selected_agent_name]
        agent = agent_info["agent"]

        # Update load
        agent_info["current_load"] += 1
        agent_info["last_activity"] = time.time()

        try:
            # Process message
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                agent.process_step,
                message
            )

            agent_info["total_processed"] += 1

            # Update message context
            message_context = self.shared_store.retrieve(message_id)
            message_context["assigned_to"] = selected_agent_name
            message_context["processing_time_ms"] = (time.time() - message_context["received_at"]) * 1000
            self.shared_store.store(message_context)

            return result

        finally:
            agent_info["current_load"] -= 1

    async def route_to_workflow(self, message: Dict[str, Any], message_id: str) -> Any:
        """Route message to specified workflow"""

        workflow_name = message["workflow"]

        if workflow_name not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_name} not found")

        workflow_info = self.active_workflows[workflow_name]
        workflow = workflow_info["workflow"]

        # Execute workflow
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            workflow.execute_workflow,
            message.get("data", {})
        )

        workflow_info["execution_count"] += 1
        workflow_info["last_execution"] = time.time()

        return result

    def update_system_metrics(self):
        """Update system performance metrics"""
        system_context = self.shared_store.retrieve(self.system_context_id)

        # Update metrics
        metrics = system_context["performance_metrics"]
        metrics["total_messages"] += 1
        metrics["system_uptime"] = time.time() - system_context["created_at"]

        # Agent metrics
        total_processed = sum(
            info["total_processed"] for info in self.registered_agents.values()
        )
        metrics["total_messages_processed"] = total_processed

        # Workflow metrics
        total_workflow_executions = sum(
            info["execution_count"] for info in self.active_workflows.values()
        )
        metrics["total_workflow_executions"] = total_workflow_executions

        self.system_context_id = self.shared_store.store(system_context)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        system_context = self.shared_store.retrieve(self.system_context_id)

        # Agent status
        agent_status = {}
        for name, info in self.registered_agents.items():
            agent_status[name] = {
                "status": info["status"],
                "current_load": info["current_load"],
                "total_processed": info["total_processed"],
                "capabilities": info["capabilities"]
            }

        # Workflow status
        workflow_status = {}
        for name, info in self.active_workflows.items():
            workflow_status[name] = {
                "status": info["status"],
                "execution_count": info["execution_count"]
            }

        return {
            "system_id": system_context["system_id"],
            "uptime_seconds": time.time() - system_context["created_at"],
            "agents": agent_status,
            "workflows": workflow_status,
            "performance_metrics": system_context["performance_metrics"],
            "configuration": system_context["configuration"]
        }

# Example multi-agent system
async def multi_agent_system_example():
    # Initialize coordinator
    coordinator = ADKMultiAgentCoordinator()
    coordinator.initialize_system()

    # Create specialized agents
    research_agent = ContextAwareAgent("ResearchAgent")
    research_agent.setup()

    analysis_agent = ContextAwareAgent("AnalysisAgent")
    analysis_agent.setup()

    reporting_agent = ContextAwareAgent("ReportingAgent")
    reporting_agent.setup()

    # Register agents with capabilities
    coordinator.register_agent(research_agent, ["research", "data_gathering"], max_concurrent=3)
    coordinator.register_agent(analysis_agent, ["analysis", "data_processing"], max_concurrent=2)
    coordinator.register_agent(reporting_agent, ["reporting", "documentation"], max_concurrent=1)

    # Create and register workflow
    workflow = DataProcessingWorkflow("AnalysisWorkflow")
    workflow.setup()
    coordinator.register_workflow(workflow)

    # Process messages through the system
    test_messages = [
        {"capability": "research", "data": "Research latest AI trends"},
        {"capability": "analysis", "data": "Analyze market data"},
        {"workflow": "AnalysisWorkflow", "data": "Process customer feedback"},
        {"capability": "reporting", "data": "Generate monthly report"},
        {"capability": "research", "data": "Research competitor analysis"}
    ]

    # Process messages concurrently
    tasks = []
    for message in test_messages:
        task = coordinator.route_message(message)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Show results
    for i, result in enumerate(results):
        print(f"Message {i+1} result: {result}")

    # Show system status
    status = coordinator.get_system_status()
    print(f"\nSystem Status: {status}")

if __name__ == "__main__":
    asyncio.run(multi_agent_system_example())
```

## State Management with ADK

### Advanced State Management

```python
from context_store.adapters import ADKAdapter
import adk
import pickle
import json

class StatefulADKAgent(ContextAwareAgent):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)

        # State management configuration
        self.state_config = {
            "checkpoint_interval": config.get("checkpoint_interval", 100),
            "max_state_history": config.get("max_state_history", 1000),
            "state_compression": config.get("state_compression", True),
            "auto_backup": config.get("auto_backup", True)
        }

        # State tracking
        self.state_history = []
        self.checkpoints = []
        self.current_step_count = 0

    def setup(self):
        """Setup with enhanced state management"""
        super().setup()

        # Initialize state management
        self.initialize_state_management()

    def initialize_state_management(self):
        """Initialize comprehensive state management"""

        # Create state management context
        state_mgmt_context = {
            "agent_name": self.name,
            "state_config": self.state_config,
            "initialized_at": time.time(),
            "checkpoints": [],
            "state_history": [],
            "recovery_points": []
        }

        self.state_mgmt_context_id = self.agent_store.store(state_mgmt_context)

        # Create initial checkpoint
        self.create_checkpoint("initialization")

    def process_step(self, input_data):
        """Process step with comprehensive state management"""
        self.current_step_count += 1

        # Create pre-execution state snapshot
        pre_state = self.capture_state_snapshot("pre_execution")

        try:
            # Execute step
            result = super().process_step(input_data)

            # Create post-execution state snapshot
            post_state = self.capture_state_snapshot("post_execution", result)

            # Auto-checkpoint if needed
            if self.current_step_count % self.state_config["checkpoint_interval"] == 0:
                self.create_checkpoint(f"auto_checkpoint_step_{self.current_step_count}")

            return result

        except Exception as e:
            # Create error state snapshot
            error_state = self.capture_state_snapshot("error", error=str(e))

            # Create recovery checkpoint
            self.create_recovery_point(pre_state, str(e))

            raise

    def capture_state_snapshot(self, snapshot_type: str, result=None, error=None):
        """Capture comprehensive state snapshot"""

        # Get current agent context
        agent_context = self.agent_store.retrieve(self.agent_context_id)

        # Create snapshot
        snapshot = {
            "snapshot_id": f"{self.name}_{snapshot_type}_{int(time.time())}",
            "snapshot_type": snapshot_type,
            "agent_name": self.name,
            "timestamp": time.time(),
            "step_count": self.current_step_count,
            "agent_context": agent_context.copy(),
            "result": result,
            "error": error,
            "memory_usage": self.get_memory_usage(),
            "performance_metrics": self.get_current_performance_metrics()
        }

        # Store snapshot
        snapshot_id = self.agent_store.store(snapshot)

        # Add to state history
        self.state_history.append(snapshot_id)

        # Manage state history size
        if len(self.state_history) > self.state_config["max_state_history"]:
            # Archive old states
            old_states = self.state_history[:-self.state_config["max_state_history"]]
            self.archive_states(old_states)
            self.state_history = self.state_history[-self.state_config["max_state_history"]:]

        return snapshot_id

    def create_checkpoint(self, checkpoint_name: str):
        """Create a named checkpoint"""

        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "agent_name": self.name,
            "created_at": time.time(),
            "step_count": self.current_step_count,
            "agent_context_id": self.agent_context_id,
            "state_snapshot_id": self.capture_state_snapshot(f"checkpoint_{checkpoint_name}"),
            "checkpoint_type": "manual" if not checkpoint_name.startswith("auto_") else "automatic"
        }

        checkpoint_id = self.agent_store.store(checkpoint_data)
        self.checkpoints.append(checkpoint_id)

        # Update state management context
        state_mgmt_context = self.agent_store.retrieve(self.state_mgmt_context_id)
        state_mgmt_context["checkpoints"].append(checkpoint_id)
        self.state_mgmt_context_id = self.agent_store.store(state_mgmt_context)

        print(f"Checkpoint '{checkpoint_name}' created for agent {self.name}")

        return checkpoint_id

    def restore_from_checkpoint(self, checkpoint_name: str = None, checkpoint_id: str = None):
        """Restore agent state from checkpoint"""

        target_checkpoint = None

        if checkpoint_id:
            # Restore from specific checkpoint ID
            target_checkpoint = self.agent_store.retrieve(checkpoint_id)
        elif checkpoint_name:
            # Find checkpoint by name
            for checkpoint_id in self.checkpoints:
                checkpoint = self.agent_store.retrieve(checkpoint_id)
                if checkpoint["checkpoint_name"] == checkpoint_name:
                    target_checkpoint = checkpoint
                    break
        else:
            # Restore from latest checkpoint
            if self.checkpoints:
                latest_checkpoint_id = self.checkpoints[-1]
                target_checkpoint = self.agent_store.retrieve(latest_checkpoint_id)

        if not target_checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_name or checkpoint_id}")

        # Restore agent context
        restored_context_id = target_checkpoint["agent_context_id"]
        restored_context = self.agent_store.retrieve(restored_context_id)

        # Update current agent context
        self.agent_context_id = self.agent_store.store(restored_context)
        self.current_step_count = target_checkpoint["step_count"]

        print(f"Agent {self.name} restored from checkpoint '{target_checkpoint['checkpoint_name']}'")

        return target_checkpoint

    def create_recovery_point(self, pre_state_id: str, error_message: str):
        """Create recovery point for error handling"""

        recovery_point = {
            "agent_name": self.name,
            "created_at": time.time(),
            "pre_state_id": pre_state_id,
            "error_message": error_message,
            "step_count": self.current_step_count,
            "recovery_strategy": "restore_pre_state"
        }

        recovery_id = self.agent_store.store(recovery_point)

        # Update state management context
        state_mgmt_context = self.agent_store.retrieve(self.state_mgmt_context_id)
        state_mgmt_context["recovery_points"].append(recovery_id)
        self.state_mgmt_context_id = self.agent_store.store(state_mgmt_context)

        return recovery_id

    def get_memory_usage(self):
        """Get current memory usage metrics"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent()
        }

    def get_current_performance_metrics(self):
        """Get current performance metrics"""
        cache_stats = self.agent_store.get_cache_stats()

        return {
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
            "total_contexts": cache_stats.get("total_contexts", 0),
            "memory_usage_mb": cache_stats.get("memory_usage_mb", 0)
        }

    def archive_states(self, state_ids: List[str]):
        """Archive old state snapshots"""

        archive_data = {
            "agent_name": self.name,
            "archived_at": time.time(),
            "archived_states": state_ids,
            "archive_reason": "state_history_cleanup"
        }

        archive_id = self.agent_store.store(archive_data)
        print(f"Archived {len(state_ids)} old states for agent {self.name}")

        return archive_id

    def get_state_timeline(self, limit: int = 10):
        """Get timeline of recent state changes"""

        timeline = []
        recent_states = self.state_history[-limit:]

        for state_id in recent_states:
            try:
                state = self.agent_store.retrieve(state_id)
                timeline.append({
                    "timestamp": state["timestamp"],
                    "snapshot_type": state["snapshot_type"],
                    "step_count": state["step_count"],
                    "had_error": state.get("error") is not None,
                    "memory_usage_mb": state.get("memory_usage", {}).get("rss_mb", 0)
                })
            except Exception as e:
                print(f"Failed to retrieve state {state_id}: {e}")

        return timeline

    def get_checkpoint_summary(self):
        """Get summary of all checkpoints"""

        checkpoint_summary = []

        for checkpoint_id in self.checkpoints:
            try:
                checkpoint = self.agent_store.retrieve(checkpoint_id)
                checkpoint_summary.append({
                    "name": checkpoint["checkpoint_name"],
                    "created_at": checkpoint["created_at"],
                    "step_count": checkpoint["step_count"],
                    "type": checkpoint["checkpoint_type"]
                })
            except Exception as e:
                print(f"Failed to retrieve checkpoint {checkpoint_id}: {e}")

        return checkpoint_summary

# Example stateful agent usage
def stateful_agent_example():
    # Create stateful agent
    agent = StatefulADKAgent("StatefulAgent", {
        "checkpoint_interval": 5,  # Checkpoint every 5 steps
        "max_state_history": 20,
        "state_compression": True
    })

    agent.setup()

    # Process multiple steps
    test_inputs = [
        "Step 1: Initialize data",
        "Step 2: Process first batch",
        "Step 3: Validate results",
        "Step 4: Generate intermediate output",
        "Step 5: Checkpoint reached",  # Auto-checkpoint
        "Step 6: Continue processing",
        "Step 7: Handle edge case",
        "Step 8: Optimize performance"
    ]

    for input_data in test_inputs:
        print(f"\nProcessing: {input_data}")

        try:
            result = agent.process_step(input_data)
            print(f"Result: {result}")

        except Exception as e:
            print(f"Error: {e}")

            # Show recovery options
            checkpoints = agent.get_checkpoint_summary()
            print(f"Available checkpoints: {checkpoints}")

    # Manual checkpoint
    manual_checkpoint = agent.create_checkpoint("before_final_step")

    # Show state timeline
    timeline = agent.get_state_timeline()
    print(f"\nState Timeline: {timeline}")

    # Show checkpoint summary
    checkpoint_summary = agent.get_checkpoint_summary()
    print(f"Checkpoint Summary: {checkpoint_summary}")

    # Demonstrate state restoration
    print("\nDemonstrating state restoration...")
    agent.restore_from_checkpoint("before_final_step")
    print("State restored successfully")

if __name__ == "__main__":
    stateful_agent_example()
```

## Tool Integration

### ADK with Tool Support

```python
from context_store.adapters import ADKAdapter, ComposioAdapter
import adk

class ToolEnabledADKAgent(StatefulADKAgent):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)

        # Tool integration
        self.composio_adapter = ComposioAdapter(self.agent_store)
        self.available_tools = {}
        self.tool_usage_history = []

        # Tool configuration
        self.tool_config = {
            "max_concurrent_tools": config.get("max_concurrent_tools", 3),
            "tool_timeout_seconds": config.get("tool_timeout_seconds", 30),
            "auto_tool_selection": config.get("auto_tool_selection", True)
        }

    def setup(self):
        """Setup with tool integration"""
        super().setup()

        # Initialize available tools
        self.initialize_tools()

    def initialize_tools(self):
        """Initialize available tools"""

        # Define available tools
        self.available_tools = {
            "web_search": {
                "app": "googlesearch",
                "action": "search",
                "description": "Search the web for information",
                "required_params": ["query"],
                "optional_params": ["num_results"]
            },
            "send_email": {
                "app": "gmail",
                "action": "send_email",
                "description": "Send email messages",
                "required_params": ["to", "subject", "body"],
                "optional_params": ["cc", "bcc"]
            },
            "create_calendar_event": {
                "app": "googlecalendar",
                "action": "create_event",
                "description": "Create calendar events",
                "required_params": ["title", "start_time"],
                "optional_params": ["description", "duration_minutes"]
            },
            "upload_file": {
                "app": "googledrive",
                "action": "upload_file",
                "description": "Upload files to Google Drive",
                "required_params": ["file_path"],
                "optional_params": ["folder_name"]
            }
        }

        # Store tool configuration
        tool_config_context = {
            "agent_name": self.name,
            "available_tools": self.available_tools,
            "tool_config": self.tool_config,
            "initialized_at": time.time()
        }

        self.tool_config_id = self.agent_store.store(tool_config_context)

    def analyze_tool_requirements(self, input_data) -> List[Dict[str, Any]]:
        """Analyze input to determine required tools"""

        if isinstance(input_data, str):
            text = input_data.lower()
        else:
            text = str(input_data).lower()

        required_tools = []

        # Simple keyword-based analysis (enhance with NLP)
        if any(keyword in text for keyword in ["search", "find", "look up", "google"]):
            required_tools.append({
                "tool": "web_search",
                "confidence": 0.8,
                "params": {"query": input_data}
            })

        if any(keyword in text for keyword in ["email", "send", "notify", "message"]):
            required_tools.append({
                "tool": "send_email",
                "confidence": 0.7,
                "params": {
                    "to": "recipient@example.com",  # Would extract from context
                    "subject": f"Regarding: {input_data}",
                    "body": f"This is regarding your request: {input_data}"
                }
            })

        if any(keyword in text for keyword in ["schedule", "calendar", "meeting", "appointment"]):
            required_tools.append({
                "tool": "create_calendar_event",
                "confidence": 0.9,
                "params": {
                    "title": f"Event: {input_data}",
                    "start_time": "2024-01-01T10:00:00Z",  # Would extract from context
                    "description": input_data
                }
            })

        if any(keyword in text for keyword in ["upload", "file", "document", "save"]):
            required_tools.append({
                "tool": "upload_file",
                "confidence": 0.6,
                "params": {
                    "file_path": "/path/to/file",  # Would extract from context
                    "folder_name": "Agent Uploads"
                }
            })

        return required_tools

    def execute_step(self, input_data, agent_context):
        """Execute step with tool integration"""

        # Analyze tool requirements
        required_tools = self.analyze_tool_requirements(input_data)

        # Execute tools if needed
        tool_results = {}
        if required_tools:
            tool_results = self.execute_tools(required_tools, input_data)

        # Generate response incorporating tool results
        if tool_results:
            response = self.generate_response_with_tools(input_data, tool_results)
        else:
            response = f"Processed: {input_data} (no tools required)"

        return {
            "response": response,
            "tools_used": list(tool_results.keys()),
            "tool_results": tool_results
        }

    def execute_tools(self, required_tools: List[Dict], input_data) -> Dict[str, Any]:
        """Execute required tools"""

        tool_results = {}

        for tool_req in required_tools:
            tool_name = tool_req["tool"]
            tool_params = tool_req["params"]

            if tool_name not in self.available_tools:
                tool_results[tool_name] = {"error": f"Tool {tool_name} not available"}
                continue

            try:
                # Execute tool
                result = self.execute_single_tool(tool_name, tool_params)
                tool_results[tool_name] = result

                # Record tool usage
                self.record_tool_usage(tool_name, tool_params, result, True)

            except Exception as e:
                error_result = {"error": str(e)}
                tool_results[tool_name] = error_result

                # Record failed tool usage
                self.record_tool_usage(tool_name, tool_params, error_result, False)

        return tool_results

    def execute_single_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool"""

        tool_config = self.available_tools[tool_name]

        # Store execution context
        execution_context = {
            "tool_name": tool_name,
            "tool_config": tool_config,
            "params": params,
            "agent_name": self.name,
            "started_at": time.time(),
            "status": "executing"
        }

        execution_id = self.agent_store.store(execution_context)

        try:
            # Execute via Composio adapter
            result = self.composio_adapter.execute_tool(
                app=tool_config["app"],
                action=tool_config["action"],
                params=params
            )

            # Update execution context
            execution_context.update({
                "result": result,
                "completed_at": time.time(),
                "status": "success",
                "execution_time_ms": (time.time() - execution_context["started_at"]) * 1000
            })

            self.agent_store.store(execution_context)

            return result

        except Exception as e:
            # Update execution context with error
            execution_context.update({
                "error": str(e),
                "completed_at": time.time(),
                "status": "error",
                "execution_time_ms": (time.time() - execution_context["started_at"]) * 1000
            })

            self.agent_store.store(execution_context)

            raise

    def record_tool_usage(self, tool_name: str, params: Dict, result: Dict, success: bool):
        """Record tool usage for analytics"""

        usage_record = {
            "tool_name": tool_name,
            "params": params,
            "result": result,
            "success": success,
            "agent_name": self.name,
            "timestamp": time.time(),
            "step_count": self.current_step_count
        }

        usage_id = self.agent_store.store(usage_record)
        self.tool_usage_history.append(usage_id)

        # Keep limited history
        if len(self.tool_usage_history) > 1000:
            old_usage = self.tool_usage_history[:-1000]
            self.archive_tool_usage(old_usage)
            self.tool_usage_history = self.tool_usage_history[-1000:]

    def generate_response_with_tools(self, input_data: str, tool_results: Dict[str, Any]) -> str:
        """Generate response incorporating tool results"""

        response_parts = [f"Processed your request: '{input_data}'"]

        for tool_name, result in tool_results.items():
            if "error" in result:
                response_parts.append(f"❌ {tool_name}: {result['error']}")
            else:
                response_parts.append(f"✅ {tool_name}: Executed successfully")

                # Add tool-specific response details
                if tool_name == "web_search" and "results" in result:
                    response_parts.append(f"   Found {len(result['results'])} search results")
                elif tool_name == "send_email":
                    response_parts.append("   Email sent successfully")
                elif tool_name == "create_calendar_event":
                    response_parts.append("   Calendar event created")
                elif tool_name == "upload_file":
                    response_parts.append("   File uploaded to drive")

        return "\n".join(response_parts)

    def get_tool_analytics(self) -> Dict[str, Any]:
        """Get tool usage analytics"""

        if not self.tool_usage_history:
            return {"total_tool_executions": 0}

        # Analyze tool usage
        tool_executions = []
        for usage_id in self.tool_usage_history:
            try:
                usage = self.agent_store.retrieve(usage_id)
                tool_executions.append(usage)
            except Exception as e:
                print(f"Failed to retrieve tool usage {usage_id}: {e}")

        if not tool_executions:
            return {"total_tool_executions": 0}

        # Calculate analytics
        total_executions = len(tool_executions)
        successful_executions = len([e for e in tool_executions if e["success"]])

        tool_usage_count = {}
        tool_success_rate = {}

        for execution in tool_executions:
            tool_name = execution["tool_name"]

            # Count usage
            tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1

            # Calculate success rate
            if tool_name not in tool_success_rate:
                tool_success_rate[tool_name] = {"total": 0, "successful": 0}

            tool_success_rate[tool_name]["total"] += 1
            if execution["success"]:
                tool_success_rate[tool_name]["successful"] += 1

        # Calculate success rates
        for tool_name in tool_success_rate:
            stats = tool_success_rate[tool_name]
            stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0

        return {
            "total_tool_executions": total_executions,
            "successful_executions": successful_executions,
            "overall_success_rate": successful_executions / total_executions,
            "tool_usage_count": tool_usage_count,
            "tool_success_rates": tool_success_rate,
            "most_used_tool": max(tool_usage_count.items(), key=lambda x: x[1])[0] if tool_usage_count else None
        }

    def archive_tool_usage(self, usage_ids: List[str]):
        """Archive old tool usage records"""

        archive_data = {
            "agent_name": self.name,
            "archived_at": time.time(),
            "archived_usage_records": usage_ids,
            "archive_reason": "tool_usage_history_cleanup"
        }

        archive_id = self.agent_store.store(archive_data)
        print(f"Archived {len(usage_ids)} tool usage records for agent {self.name}")

# Example tool-enabled agent usage
def tool_enabled_agent_example():
    # Create tool-enabled agent
    agent = ToolEnabledADKAgent("ToolAgent", {
        "checkpoint_interval": 3,
        "max_concurrent_tools": 2,
        "auto_tool_selection": True
    })

    agent.setup()

    # Test inputs that require different tools
    test_inputs = [
        "Search for the latest Python tutorials online",
        "Send an email to the team about our progress",
        "Schedule a meeting for tomorrow at 2 PM",
        "Upload the project documentation to drive",
        "Find information about machine learning trends",
        "Create a calendar event for the quarterly review"
    ]

    for input_data in test_inputs:
        print(f"\n{'='*50}")
        print(f"Processing: {input_data}")
        print(f"{'='*50}")

        try:
            result = agent.process_step(input_data)
            print(f"Result: {result['response']}")
            print(f"Tools Used: {result['tools_used']}")

        except Exception as e:
            print(f"Error: {e}")

    # Show tool analytics
    analytics = agent.get_tool_analytics()
    print(f"\n{'='*50}")
    print("Tool Usage Analytics:")
    print(f"{'='*50}")
    print(f"Total Executions: {analytics['total_tool_executions']}")
    print(f"Success Rate: {analytics.get('overall_success_rate', 0):.2%}")
    print(f"Tool Usage Count: {analytics['tool_usage_count']}")
    print(f"Most Used Tool: {analytics.get('most_used_tool', 'None')}")

    # Show state timeline
    timeline = agent.get_state_timeline()
    print(f"\nState Timeline: {timeline}")

if __name__ == "__main__":
    tool_enabled_agent_example()
```

## Advanced Patterns

### Hierarchical Agent Systems

```python
class HierarchicalADKSystem:
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.shared_store = ContextReferenceStore(
            cache_size=20000,
            use_compression=True,
            use_disk_storage=True
        )

        # System hierarchy
        self.supervisor_agents = {}  # High-level coordination
        self.worker_agents = {}      # Task execution
        self.specialist_agents = {}  # Specialized tasks

        # System context
        self.system_context_id = None
        self.hierarchy_map = {}

    def initialize_system(self):
        """Initialize hierarchical system"""

        system_context = {
            "system_name": self.system_name,
            "created_at": time.time(),
            "hierarchy_levels": {
                "supervisor": [],
                "worker": [],
                "specialist": []
            },
            "communication_patterns": {},
            "performance_metrics": {}
        }

        self.system_context_id = self.shared_store.store(system_context)

    def add_supervisor_agent(self, agent: ToolEnabledADKAgent, supervised_agents: List[str]):
        """Add supervisor agent to the hierarchy"""

        # Configure agent as supervisor
        agent.role = "supervisor"
        agent.supervised_agents = supervised_agents
        agent.hierarchy_level = 1

        self.supervisor_agents[agent.name] = agent

        # Update hierarchy map
        self.hierarchy_map[agent.name] = {
            "level": 1,
            "role": "supervisor",
            "supervises": supervised_agents,
            "agent": agent
        }

        self.update_system_hierarchy()

    def add_worker_agent(self, agent: ToolEnabledADKAgent, supervisor: str):
        """Add worker agent under supervisor"""

        agent.role = "worker"
        agent.supervisor = supervisor
        agent.hierarchy_level = 2

        self.worker_agents[agent.name] = agent

        self.hierarchy_map[agent.name] = {
            "level": 2,
            "role": "worker",
            "supervisor": supervisor,
            "agent": agent
        }

        self.update_system_hierarchy()

    def add_specialist_agent(self, agent: ToolEnabledADKAgent, specialization: str):
        """Add specialist agent"""

        agent.role = "specialist"
        agent.specialization = specialization
        agent.hierarchy_level = 3

        self.specialist_agents[agent.name] = agent

        self.hierarchy_map[agent.name] = {
            "level": 3,
            "role": "specialist",
            "specialization": specialization,
            "agent": agent
        }

        self.update_system_hierarchy()

    def route_hierarchical_message(self, message: Dict[str, Any]) -> Any:
        """Route message through hierarchy"""

        message_type = message.get("type", "task")
        target_level = message.get("target_level", "auto")

        if target_level == "auto":
            # Auto-determine target level based on message
            target_level = self.determine_target_level(message)

        if target_level == "supervisor":
            return self.route_to_supervisor(message)
        elif target_level == "worker":
            return self.route_to_worker(message)
        elif target_level == "specialist":
            return self.route_to_specialist(message)
        else:
            raise ValueError(f"Unknown target level: {target_level}")

    def determine_target_level(self, message: Dict[str, Any]) -> str:
        """Determine appropriate hierarchy level for message"""

        content = str(message.get("content", "")).lower()

        # Coordination keywords -> supervisor
        if any(word in content for word in ["coordinate", "plan", "organize", "manage"]):
            return "supervisor"

        # Specialized keywords -> specialist
        if any(word in content for word in ["analyze", "research", "calculate", "design"]):
            return "specialist"

        # Default to worker
        return "worker"

    def route_to_supervisor(self, message: Dict[str, Any]) -> Any:
        """Route message to appropriate supervisor"""

        if not self.supervisor_agents:
            raise ValueError("No supervisor agents available")

        # Select supervisor (simple round-robin for now)
        supervisor_name = list(self.supervisor_agents.keys())[0]
        supervisor = self.supervisor_agents[supervisor_name]

        return supervisor.process_step(message)

    def route_to_worker(self, message: Dict[str, Any]) -> Any:
        """Route message to available worker"""

        if not self.worker_agents:
            raise ValueError("No worker agents available")

        # Find least busy worker
        worker_name = min(
            self.worker_agents.keys(),
            key=lambda name: self.get_agent_load(name)
        )

        worker = self.worker_agents[worker_name]
        return worker.process_step(message)

    def route_to_specialist(self, message: Dict[str, Any]) -> Any:
        """Route message to appropriate specialist"""

        required_specialization = message.get("specialization")

        if required_specialization:
            # Find specialist with required specialization
            for agent_name, agent in self.specialist_agents.items():
                if agent.specialization == required_specialization:
                    return agent.process_step(message)

            raise ValueError(f"No specialist found for: {required_specialization}")

        # Default to first available specialist
        if self.specialist_agents:
            specialist = list(self.specialist_agents.values())[0]
            return specialist.process_step(message)

        raise ValueError("No specialist agents available")

    def get_agent_load(self, agent_name: str) -> int:
        """Get current load for agent"""
        # Simplified load calculation
        return 0  # Would implement actual load tracking

    def update_system_hierarchy(self):
        """Update system hierarchy context"""

        system_context = self.shared_store.retrieve(self.system_context_id)

        # Update hierarchy levels
        system_context["hierarchy_levels"] = {
            "supervisor": list(self.supervisor_agents.keys()),
            "worker": list(self.worker_agents.keys()),
            "specialist": list(self.specialist_agents.keys())
        }

        # Update hierarchy map
        system_context["hierarchy_map"] = {
            name: {
                "level": info["level"],
                "role": info["role"],
                "context_id": info["agent"].agent_context_id
            }
            for name, info in self.hierarchy_map.items()
        }

        self.system_context_id = self.shared_store.store(system_context)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        status = {
            "system_name": self.system_name,
            "total_agents": len(self.hierarchy_map),
            "supervisor_count": len(self.supervisor_agents),
            "worker_count": len(self.worker_agents),
            "specialist_count": len(self.specialist_agents),
            "hierarchy_map": self.hierarchy_map
        }

        return status

# Example hierarchical system
def hierarchical_system_example():
    # Create hierarchical system
    system = HierarchicalADKSystem("ProductionSystem")
    system.initialize_system()

    # Create agents for different levels
    # Supervisor agent
    supervisor = ToolEnabledADKAgent("ProductionSupervisor")
    supervisor.setup()

    # Worker agents
    worker1 = ToolEnabledADKAgent("Worker1")
    worker1.setup()

    worker2 = ToolEnabledADKAgent("Worker2")
    worker2.setup()

    # Specialist agents
    analyst = ToolEnabledADKAgent("DataAnalyst")
    analyst.setup()

    researcher = ToolEnabledADKAgent("Researcher")
    researcher.setup()

    # Add agents to hierarchy
    system.add_supervisor_agent(supervisor, ["Worker1", "Worker2"])
    system.add_worker_agent(worker1, "ProductionSupervisor")
    system.add_worker_agent(worker2, "ProductionSupervisor")
    system.add_specialist_agent(analyst, "data_analysis")
    system.add_specialist_agent(researcher, "research")

    # Test hierarchical routing
    test_messages = [
        {"content": "Coordinate the daily tasks", "type": "coordination"},
        {"content": "Process this batch of data", "type": "task"},
        {"content": "Analyze sales trends", "specialization": "data_analysis"},
        {"content": "Research market competitors", "specialization": "research"},
        {"content": "Plan the weekly schedule", "type": "planning"}
    ]

    for message in test_messages:
        print(f"\nRouting message: {message}")
        try:
            result = system.route_hierarchical_message(message)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

    # Show system status
    status = system.get_system_status()
    print(f"\nSystem Status: {status}")

if __name__ == "__main__":
    hierarchical_system_example()
```

## Performance Optimization

### Optimized ADK Configuration

```python
def optimize_adk_performance():
    """Configure Context Reference Store for optimal ADK performance"""

    # High-performance configuration
    optimized_store = ContextReferenceStore(
        cache_size=10000,           # Large cache for frequent access
        use_compression=True,       # Enable compression for memory efficiency
        compression_algorithm="lz4", # Fast compression for ADK workflows
        eviction_policy="LRU",      # Good for agent workflows
        use_disk_storage=True,      # Enable disk storage for large contexts
        memory_threshold_mb=200,    # Reasonable memory threshold
        disk_cache_dir="./adk_cache" # Dedicated cache directory
    )

    return optimized_store

def benchmark_adk_performance():
    """Benchmark ADK integration performance"""

    import time

    # Create optimized agent
    store = optimize_adk_performance()
    adapter = ADKAdapter(store)

    agent = ToolEnabledADKAgent("BenchmarkAgent", {
        "checkpoint_interval": 100,
        "max_concurrent_tools": 5
    })
    agent.context_store = store
    agent.setup()

    # Benchmark metrics
    start_time = time.time()
    processing_times = []
    memory_usage = []

    # Process test workload
    for i in range(100):
        step_start = time.time()

        result = agent.process_step(f"Benchmark step {i}")

        step_time = (time.time() - step_start) * 1000
        processing_times.append(step_time)

        # Get memory usage
        cache_stats = store.get_cache_stats()
        memory_usage.append(cache_stats.get("memory_usage_mb", 0))

    total_time = time.time() - start_time

    # Calculate performance metrics
    avg_processing_time = sum(processing_times) / len(processing_times)
    max_processing_time = max(processing_times)
    avg_memory_usage = sum(memory_usage) / len(memory_usage)

    final_cache_stats = store.get_cache_stats()

    print(f"""
ADK Performance Benchmark Results:
===================================
Total Time: {total_time:.2f} seconds
Average Processing Time: {avg_processing_time:.2f} ms
Max Processing Time: {max_processing_time:.2f} ms
Average Memory Usage: {avg_memory_usage:.2f} MB
Cache Hit Rate: {final_cache_stats.get('hit_rate', 0):.2%}
Total Contexts: {final_cache_stats.get('total_contexts', 0)}
""")

if __name__ == "__main__":
    benchmark_adk_performance()
```

## Production Deployment

### Production ADK Configuration

```python
import logging
import asyncio
from dataclasses import dataclass

@dataclass
class ProductionADKConfig:
    system_name: str
    cache_size: int = 20000
    max_agents: int = 50
    checkpoint_interval: int = 1000
    monitoring_enabled: bool = True
    backup_enabled: bool = True
    log_level: str = "INFO"
    health_check_interval: int = 300

class ProductionADKSystem:
    def __init__(self, config: ProductionADKConfig):
        self.config = config

        # Initialize production context store
        self.context_store = ContextReferenceStore(
            cache_size=config.cache_size,
            use_compression=True,
            compression_algorithm="lz4",
            use_disk_storage=True,
            memory_threshold_mb=500,
            eviction_policy="LRU"
        )

        # System components
        self.adk_adapter = ADKAdapter(self.context_store)
        self.coordinator = ADKMultiAgentCoordinator(self.context_store)
        self.health_monitor = None

        # Setup logging
        self.setup_logging()

        # Initialize system
        self.initialize_production_system()

    def setup_logging(self):
        """Setup production logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'adk_system_{self.config.system_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"ADKSystem-{self.config.system_name}")

    def initialize_production_system(self):
        """Initialize production system"""

        # Initialize coordinator
        self.coordinator.initialize_system()

        # Setup monitoring
        if self.config.monitoring_enabled:
            self.setup_monitoring()

        # Setup health checks
        self.start_health_monitoring()

        self.logger.info(f"Production ADK system {self.config.system_name} initialized")

    def setup_monitoring(self):
        """Setup production monitoring"""
        from context_store.monitoring import PerformanceMonitor

        self.performance_monitor = PerformanceMonitor()
        self.context_store.add_monitor(self.performance_monitor)

    def start_health_monitoring(self):
        """Start health monitoring"""
        if self.config.monitoring_enabled:
            self.health_monitor = asyncio.create_task(self.health_monitoring_loop())

    async def health_monitoring_loop(self):
        """Background health monitoring"""
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
            "system_name": self.config.system_name,
            "status": "healthy",
            "checks": {}
        }

        try:
            # Check context store health
            cache_stats = self.context_store.get_cache_stats()
            health_status["checks"]["context_store"] = {
                "status": "healthy",
                "cache_hit_rate": cache_stats.get("hit_rate", 0),
                "memory_usage_mb": cache_stats.get("memory_usage_mb", 0),
                "total_contexts": cache_stats.get("total_contexts", 0)
            }

            # Check coordinator health
            coordinator_status = self.coordinator.get_system_status()
            health_status["checks"]["coordinator"] = {
                "status": "healthy",
                "active_agents": len(coordinator_status["agents"]),
                "active_workflows": len(coordinator_status["workflows"])
            }

            # Performance checks
            if self.config.monitoring_enabled:
                performance_metrics = self.performance_monitor.get_current_metrics()
                health_status["checks"]["performance"] = performance_metrics

            self.logger.info(f"Health check completed: {health_status['status']}")

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")

    def deploy_agent(self, agent_class, agent_name: str, agent_config: dict, capabilities: List[str]):
        """Deploy agent to production system"""

        try:
            # Create agent
            agent = agent_class(agent_name, agent_config)
            agent.context_store = self.context_store
            agent.setup()

            # Register with coordinator
            self.coordinator.register_agent(agent, capabilities)

            self.logger.info(f"Agent {agent_name} deployed successfully")

            return agent

        except Exception as e:
            self.logger.error(f"Failed to deploy agent {agent_name}: {e}")
            raise

    def deploy_workflow(self, workflow_class, workflow_name: str, workflow_config: dict):
        """Deploy workflow to production system"""

        try:
            # Create workflow
            workflow = workflow_class(workflow_name, self.context_store)
            workflow.setup()

            # Register with coordinator
            self.coordinator.register_workflow(workflow)

            self.logger.info(f"Workflow {workflow_name} deployed successfully")

            return workflow

        except Exception as e:
            self.logger.error(f"Failed to deploy workflow {workflow_name}: {e}")
            raise

    async def process_production_message(self, message: Dict[str, Any]) -> Any:
        """Process message in production environment"""

        start_time = time.time()

        try:
            self.logger.info(f"Processing production message: {message.get('id', 'unknown')}")

            # Route through coordinator
            result = await self.coordinator.route_message(message)

            processing_time = (time.time() - start_time) * 1000
            self.logger.info(f"Message processed in {processing_time:.2f}ms")

            return result

        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            raise

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating graceful shutdown")

        # Cancel health monitoring
        if self.health_monitor:
            self.health_monitor.cancel()

        # Shutdown coordinator
        # (Implementation depends on coordinator shutdown logic)

        self.logger.info("Production ADK system shutdown complete")

# Example production deployment
async def production_deployment_example():

    config = ProductionADKConfig(
        system_name="production-adk-001",
        cache_size=50000,
        max_agents=100,
        monitoring_enabled=True,
        backup_enabled=True,
        log_level="INFO"
    )

    # Create production system
    production_system = ProductionADKSystem(config)

    try:
        # Deploy agents
        research_agent = production_system.deploy_agent(
            ToolEnabledADKAgent,
            "ProductionResearcher",
            {"checkpoint_interval": 500},
            ["research", "data_gathering"]
        )

        analysis_agent = production_system.deploy_agent(
            ToolEnabledADKAgent,
            "ProductionAnalyst",
            {"checkpoint_interval": 500},
            ["analysis", "data_processing"]
        )

        # Deploy workflow
        workflow = production_system.deploy_workflow(
            DataProcessingWorkflow,
            "ProductionDataPipeline",
            {}
        )

        # Process production workload
        production_messages = [
            {"id": "msg_001", "capability": "research", "data": "Market analysis"},
            {"id": "msg_002", "capability": "analysis", "data": "Financial data"},
            {"id": "msg_003", "workflow": "ProductionDataPipeline", "data": "Customer data"},
        ]

        # Process messages concurrently
        tasks = []
        for message in production_messages:
            task = production_system.process_production_message(message)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        print(f"Processed {len(results)} production messages")

    finally:
        production_system.shutdown()

if __name__ == "__main__":
    asyncio.run(production_deployment_example())
```

## Best Practices

### ADK Integration Best Practices

1. **Context Store Configuration**

   ```python
   # Optimal configuration for ADK
   store = ContextReferenceStore(
       cache_size=5000,              # Adjust based on memory
       use_compression=True,         # Always enable for ADK
       compression_algorithm="lz4",  # Fast compression for workflows
       eviction_policy="LRU",        # Good for agent patterns
       use_disk_storage=True         # For large workflows
   )
   ```

2. **Agent Lifecycle Management**

   ```python
   def proper_agent_lifecycle(agent):
       try:
           agent.setup()
           # Agent operations
           return agent.process_workflow()
       finally:
           agent.cleanup()  # Always cleanup
   ```

3. **Error Handling and Recovery**

   ```python
   def robust_agent_execution(agent, input_data):
       checkpoint_id = agent.create_checkpoint("before_execution")

       try:
           return agent.process_step(input_data)
       except Exception as e:
           # Restore from checkpoint
           agent.restore_from_checkpoint(checkpoint_id)
           # Implement fallback logic
           return agent.fallback_processing(input_data)
   ```

4. **Performance Monitoring**

   ```python
   def monitor_agent_performance(agent):
       stats = agent.get_performance_metrics()

       if stats["avg_processing_time_ms"] > 1000:
           # Optimize agent or scale horizontally
           pass

       if stats["memory_usage_mb"] > 500:
           # Enable disk storage or reduce cache
           agent.enable_disk_storage()
   ```

## Troubleshooting

### Common ADK Integration Issues

#### 1. High Memory Usage in Workflows

```python
def diagnose_workflow_memory():
    """Diagnose workflow memory usage"""

    # Check cache stats
    cache_stats = workflow.shared_store.get_cache_stats()

    if cache_stats.get("memory_usage_mb", 0) > 1000:
        # Solutions:
        # 1. Enable compression
        workflow.shared_store.use_compression = True

        # 2. Reduce cache size
        workflow.shared_store.cache_size = workflow.shared_store.cache_size // 2

        # 3. Enable disk storage
        workflow.shared_store.use_disk_storage = True
        workflow.shared_store.memory_threshold_mb = 200
```

#### 2. Slow Agent Communication

```python
def optimize_agent_communication():
    """Optimize communication between agents"""

    # Use shared context store
    shared_store = ContextReferenceStore(cache_size=10000)

    # Configure agents to use shared store
    for agent in agents:
        agent.shared_store = shared_store

    # Use context references instead of full data
    def share_data_efficiently(agent, data):
        context_id = agent.shared_store.store(data)
        return {"type": "context_reference", "id": context_id}
```

#### 3. Checkpoint Corruption

```python
def handle_checkpoint_corruption(agent):
    """Handle corrupted checkpoints"""

    try:
        agent.restore_from_checkpoint("latest")
    except Exception as e:
        # Try previous checkpoints
        checkpoints = agent.get_checkpoint_summary()

        for checkpoint in reversed(checkpoints):
            try:
                agent.restore_from_checkpoint(checkpoint["name"])
                break
            except Exception:
                continue
        else:
            # Create new checkpoint from current state
            agent.create_checkpoint("recovery_checkpoint")
```

This comprehensive ADK integration guide provides everything needed to build sophisticated agent systems with Context Reference Store, from basic integration to production deployment with advanced features like hierarchical coordination, tool integration, and comprehensive state management.
