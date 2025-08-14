"""
Enhanced Composio adapter for Context Reference Store.

This module provides comprehensive integration between the Context Reference Store and Composio,
enabling:
- Faster tool execution result caching and state management
- Memory reduction for large tool execution histories and data
- Advanced authentication state management with secure caching
- Tool execution optimization with intelligent result reuse
- Trigger and webhook state persistence
- Production-ready performance monitoring and analytics

Composio provides access to 3000+ tools with normalized APIs, authentication handling,
and execution management. This adapter enhances these capabilities with massive
performance improvements through the Context Reference Store.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Sequence
import json
import time
import uuid
import hashlib
import base64
import secrets
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from context_store.core.context_reference_store import ContextReferenceStore
from context_store.core.large_context_state import LargeContextState

# Encryption support
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

try:
    # Core Composio imports
    from composio import ComposioToolSet, App, Action
    from composio.client import Composio
    from composio.client.collections import TriggerEventData
    from composio.tools import ComposioToolSet as BaseComposioToolSet
    from composio.tools.base import Tool

    try:
        from composio.client.enums import AppType, ActionType, TriggerType
        from composio.client.collections import ActionModel, TriggerModel

        COMPOSIO_ENUMS_AVAILABLE = True
    except ImportError:
        COMPOSIO_ENUMS_AVAILABLE = False

    try:
        from composio.tools.env.factory import WorkspaceFactory
        from composio.tools.env.base import WorkspaceType

        COMPOSIO_WORKSPACE_AVAILABLE = True
    except ImportError:
        COMPOSIO_WORKSPACE_AVAILABLE = False

    COMPOSIO_AVAILABLE = True
except ImportError:
    COMPOSIO_AVAILABLE = False
    COMPOSIO_ENUMS_AVAILABLE = False
    COMPOSIO_WORKSPACE_AVAILABLE = False

    # Create mock classes for when Composio is not available
    class ComposioToolSet:
        pass

    class Composio:
        pass

    class App:
        pass

    class Action:
        pass

    class Tool:
        pass

    class TriggerEventData:
        pass


class ExecutionStatus(Enum):
    """Status of tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"
    TIMEOUT = "timeout"


@dataclass
class ToolExecutionMetrics:
    """Metrics for individual tool executions."""

    execution_id: str
    tool_name: str
    action_name: str
    app_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    status: ExecutionStatus = ExecutionStatus.PENDING
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    cache_hit: bool = False
    authentication_time: float = 0.0
    execution_time: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.end_time is None:
            self.end_time = datetime.now()
        if self.duration_seconds == 0.0 and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class AuthenticationState:
    """Authentication state for Composio apps."""

    app_name: str
    auth_type: str
    credentials: Dict[str, Any]
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    is_valid: bool = True
    refresh_token: Optional[str] = None
    access_token: Optional[str] = None


@dataclass
class TriggerState:
    """State management for Composio triggers."""

    trigger_id: str
    app_name: str
    trigger_name: str
    config: Dict[str, Any]
    webhook_url: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    event_count: int = 0


class ComposioContextAdapter:
    """
    Enhanced adapter for integrating Context Reference Store with Composio applications.

    This adapter provides:
    - Faster tool execution result caching and state management
    - Memory reduction for large tool execution histories
    - Advanced authentication state management with secure credential encryption
    - Tool execution optimization with intelligent result reuse and caching
    - Trigger and webhook state management with event history
    - Comprehensive performance monitoring and analytics for tool usage
    - Workspace state management and environment persistence

    Security Features:
    - Automatic credential encryption using Fernet (AES 128) for stored authentication states
    - Environment variable support for encryption keys (COMPOSIO_ENCRYPTION_KEY, COMPOSIO_ENCRYPTION_SALT)
    - Secure fallback handling when encryption is unavailable
    - Legacy credential format support for smooth migrations

    Note:
    - Install cryptography for encryption support: pip install cryptography
    - Set COMPOSIO_ENCRYPTION_KEY and COMPOSIO_ENCRYPTION_SALT environment variables in production
    """

    def __init__(
        self,
        context_store: Optional[ContextReferenceStore] = None,
        cache_size: int = 200,
        enable_tool_caching: bool = True,
        enable_auth_caching: bool = True,
        enable_trigger_management: bool = True,
        enable_workspace_support: bool = True,
        enable_performance_monitoring: bool = True,
        cache_expiry_hours: int = 24,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the enhanced Composio adapter.

        Args:
            context_store: Optional pre-configured context store
            cache_size: Maximum number of contexts to keep in memory
            enable_tool_caching: Whether to enable tool execution result caching
            enable_auth_caching: Whether to enable authentication state caching
            enable_trigger_management: Whether to enable trigger state management
            enable_workspace_support: Whether to enable workspace environment support
            enable_performance_monitoring: Whether to enable comprehensive monitoring
            cache_expiry_hours: Hours before cached results expire
            api_key: Composio API key
            base_url: Optional custom Composio base URL
        """
        if not COMPOSIO_AVAILABLE:
            raise ImportError(
                "Composio is required for ComposioContextAdapter. "
                "Install with: pip install composio-core"
            )

        self.context_store = context_store or ContextReferenceStore(
            cache_size=cache_size,
            enable_compression=True,
            use_disk_storage=True,
            large_binary_threshold=1024 * 10,  # 10KB threshold
        )
        # Configuration
        self.enable_tool_caching = enable_tool_caching
        self.enable_auth_caching = enable_auth_caching
        self.enable_trigger_management = enable_trigger_management
        self.enable_workspace_support = (
            enable_workspace_support and COMPOSIO_WORKSPACE_AVAILABLE
        )
        self.enable_performance_monitoring = enable_performance_monitoring
        self.cache_expiry_hours = cache_expiry_hours
        # Core state management
        self.state = LargeContextState(context_store=self.context_store)
        self.composio_client = (
            Composio(api_key=api_key, base_url=base_url) if api_key else None
        )
        self.toolset = ComposioToolSet(api_key=api_key) if api_key else None

        # Advanced feature tracking
        self._tool_cache = {}
        self._auth_states = {}
        self._trigger_states = {}
        self._workspace_states = {}
        self._execution_metrics = {}
        self._active_executions = {}

        # Performance tracking
        self._performance_stats = {
            "total_executions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_execution_time": 0.0,
            "total_auth_operations": 0,
            "total_trigger_operations": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
        }

        # Initialize encryption key for credentials
        self._encryption_key = self._initialize_encryption_key()

    def _initialize_encryption_key(self) -> Optional[Fernet]:
        """
        Initialize encryption key for credential security.

        Returns:
            Fernet encryption instance or None if encryption not available
        """
        if not ENCRYPTION_AVAILABLE:
            return None

        try:
            # In production, this should be loaded from environment variables
            # or a secure key management system
            password = os.environ.get(
                "COMPOSIO_ENCRYPTION_KEY", "default-key-change-in-production"
            )
            password_bytes = password.encode()

            # Generate a salt (in production, store this securely)
            salt = os.environ.get(
                "COMPOSIO_ENCRYPTION_SALT", "default-salt-change-in-production"
            ).encode()

            # Derive encryption key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            return Fernet(key)
        except Exception as e:
            print(f"Warning: Failed to initialize encryption: {e}")
            return None

    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        """
        Encrypt credentials for secure storage.

        Args:
            credentials: Credentials dictionary to encrypt

        Returns:
            Encrypted credentials as base64 string

        Raises:
            ValueError: If encryption is not available or fails
        """
        if not self._encryption_key:
            raise ValueError(
                "Encryption not available. Install cryptography: pip install cryptography"
            )

        try:
            # Convert credentials to JSON string
            credentials_json = json.dumps(credentials, sort_keys=True)
            credentials_bytes = credentials_json.encode("utf-8")

            # Encrypt the credentials
            encrypted_bytes = self._encryption_key.encrypt(credentials_bytes)

            # Return as base64 string for storage
            return base64.urlsafe_b64encode(encrypted_bytes).decode("utf-8")

        except Exception as e:
            raise ValueError(f"Failed to encrypt credentials: {e}")

    def _decrypt_credentials(self, encrypted_credentials: str) -> Dict[str, Any]:
        """
        Decrypt stored credentials.

        Args:
            encrypted_credentials: Encrypted credentials as base64 string

        Returns:
            Decrypted credentials dictionary

        Raises:
            ValueError: If decryption fails
        """
        if not self._encryption_key:
            raise ValueError("Encryption not available for decryption")

        try:
            # Decode from base64
            encrypted_bytes = base64.urlsafe_b64decode(
                encrypted_credentials.encode("utf-8")
            )

            # Decrypt the credentials
            decrypted_bytes = self._encryption_key.decrypt(encrypted_bytes)

            # Convert back to dictionary
            credentials_json = decrypted_bytes.decode("utf-8")
            return json.loads(credentials_json)

        except Exception as e:
            raise ValueError(f"Failed to decrypt credentials: {e}")

    def _generate_cache_key(
        self, tool_name: str, action_name: str, inputs: Dict[str, Any]
    ) -> str:
        """Generate a unique cache key for tool execution results."""
        # Create a stable hash of the inputs
        inputs_str = json.dumps(inputs, sort_keys=True)
        inputs_hash = hashlib.md5(inputs_str.encode()).hexdigest()
        return f"tool_cache_{tool_name}_{action_name}_{inputs_hash}"

    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid based on expiry time."""
        if not cached_data.get("cached_at"):
            return False

        cached_at = datetime.fromisoformat(cached_data["cached_at"])
        expiry_time = cached_at + timedelta(hours=self.cache_expiry_hours)
        return datetime.now() < expiry_time

    def store_tool_execution_result(
        self,
        tool_name: str,
        action_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_metrics: Optional[ToolExecutionMetrics] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store tool execution results with intelligent caching.

        Args:
            tool_name: Name of the tool executed
            action_name: Name of the action performed
            inputs: Input parameters used for execution
            outputs: Results from the tool execution
            execution_metrics: Optional execution metrics
            metadata: Additional metadata about the execution

        Returns:
            Reference ID for the stored execution result
        """
        cache_key = self._generate_cache_key(tool_name, action_name, inputs)
        # Prepare execution data
        execution_data = {
            "tool_name": tool_name,
            "action_name": action_name,
            "inputs": inputs,
            "outputs": outputs,
            "cached_at": datetime.now().isoformat(),
            "execution_id": str(uuid.uuid4()),
            "metadata": metadata or {},
        }

        # Add execution metrics if provided
        if execution_metrics:
            execution_data["metrics"] = {
                "execution_id": execution_metrics.execution_id,
                "duration_seconds": execution_metrics.duration_seconds,
                "status": execution_metrics.status.value,
                "input_size_bytes": execution_metrics.input_size_bytes,
                "output_size_bytes": execution_metrics.output_size_bytes,
                "authentication_time": execution_metrics.authentication_time,
                "execution_time": execution_metrics.execution_time,
                "error_message": execution_metrics.error_message,
            }

        # Store in context store
        context_metadata = {
            "content_type": "composio/tool_execution",
            "tool_name": tool_name,
            "action_name": action_name,
            "app_name": metadata.get("app_name", "unknown") if metadata else "unknown",
            "execution_timestamp": datetime.now().isoformat(),
        }

        ref_id = self.state.add_large_context(
            execution_data,
            metadata=context_metadata,
            key=cache_key,
        )
        # Update performance stats
        if self.enable_performance_monitoring:
            self._performance_stats["total_executions"] += 1
            if (
                execution_metrics
                and execution_metrics.status == ExecutionStatus.SUCCESS
            ):
                self._performance_stats["successful_executions"] += 1
            elif (
                execution_metrics and execution_metrics.status == ExecutionStatus.FAILED
            ):
                self._performance_stats["failed_executions"] += 1

        return ref_id

    def get_cached_tool_result(
        self, tool_name: str, action_name: str, inputs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached tool execution results if available and valid.

        Args:
            tool_name: Name of the tool
            action_name: Name of the action
            inputs: Input parameters

        Returns:
            Cached execution results or None if not found/expired
        """
        if not self.enable_tool_caching:
            return None

        cache_key = self._generate_cache_key(tool_name, action_name, inputs)

        try:
            cached_data = self.state.get_context(cache_key)

            if self._is_cache_valid(cached_data):
                # Update performance stats
                if self.enable_performance_monitoring:
                    self._performance_stats["cache_hits"] += 1

                return cached_data
            else:
                # Remove expired cache entry
                if cache_key in self.state:
                    del self.state[cache_key]
                return None

        except KeyError:
            # Update performance stats
            if self.enable_performance_monitoring:
                self._performance_stats["cache_misses"] += 1
            return None

    def execute_tool_with_caching(
        self,
        tool_name: str,
        action_name: str,
        inputs: Dict[str, Any],
        entity_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a Composio tool with intelligent caching.

        Args:
            tool_name: Name of the tool to execute
            action_name: Name of the action to perform
            inputs: Input parameters for the tool
            entity_id: Optional entity ID for authentication context
            force_refresh: Whether to bypass cache and force fresh execution

        Returns:
            Tool execution results
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_result = self.get_cached_tool_result(tool_name, action_name, inputs)
            if cached_result:
                return {
                    "result": cached_result["outputs"],
                    "cached": True,
                    "execution_id": cached_result["execution_id"],
                    "cached_at": cached_result["cached_at"],
                }
        # Create execution metrics
        execution_id = str(uuid.uuid4())
        metrics = ToolExecutionMetrics(
            execution_id=execution_id,
            tool_name=tool_name,
            action_name=action_name,
            app_name=tool_name.split("_")[0] if "_" in tool_name else tool_name,
            start_time=datetime.now(),
            input_size_bytes=len(json.dumps(inputs).encode()),
        )

        # Track active execution
        self._active_executions[execution_id] = metrics

        try:
            # Execute the tool using Composio
            if not self.toolset:
                raise ValueError(
                    "Composio toolset not initialized. Please provide API key."
                )

            start_time = time.time()
            # Get the tool action
            app = App(name=tool_name)
            action = Action(name=action_name)
            # Execute with entity context if provided
            if entity_id:
                result = self.toolset.execute_action(
                    action=action, params=inputs, entity_id=entity_id
                )
            else:
                result = self.toolset.execute_action(action=action, params=inputs)

            execution_time = time.time() - start_time
            # Update metrics
            metrics.end_time = datetime.now()
            metrics.execution_time = execution_time
            metrics.status = ExecutionStatus.SUCCESS

            # Process result
            if hasattr(result, "output"):
                outputs = result.output
            elif isinstance(result, dict):
                outputs = result
            else:
                outputs = {"result": str(result)}

            metrics.output_size_bytes = len(json.dumps(outputs).encode())

            # Store result in cache
            self.store_tool_execution_result(
                tool_name=tool_name,
                action_name=action_name,
                inputs=inputs,
                outputs=outputs,
                execution_metrics=metrics,
                metadata={
                    "app_name": metrics.app_name,
                    "entity_id": entity_id,
                    "execution_type": "fresh",
                },
            )
            # Update performance stats
            if self.enable_performance_monitoring:
                self._performance_stats["total_execution_time"] += execution_time
                self._update_average_execution_time()

            return {
                "result": outputs,
                "cached": False,
                "execution_id": execution_id,
                "duration": execution_time,
                "status": "success",
            }

        except Exception as e:
            # Update metrics for failed execution
            metrics.end_time = datetime.now()
            metrics.status = ExecutionStatus.FAILED
            metrics.error_message = str(e)
            # Store failed execution for analysis
            self.store_tool_execution_result(
                tool_name=tool_name,
                action_name=action_name,
                inputs=inputs,
                outputs={"error": str(e)},
                execution_metrics=metrics,
                metadata={
                    "app_name": metrics.app_name,
                    "entity_id": entity_id,
                    "execution_type": "failed",
                },
            )
            return {
                "result": None,
                "error": str(e),
                "cached": False,
                "execution_id": execution_id,
                "status": "failed",
            }

        finally:
            # Remove from active executions
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]
            # Store metrics
            if self.enable_performance_monitoring:
                self._execution_metrics[execution_id] = metrics

    def _update_average_execution_time(self):
        """
        Update the average execution time statistic.
        """
        total_executions = self._performance_stats["total_executions"]
        if total_executions > 0:
            self._performance_stats["average_execution_time"] = (
                self._performance_stats["total_execution_time"] / total_executions
            )

    def store_authentication_state(
        self,
        app_name: str,
        auth_type: str,
        credentials: Dict[str, Any],
        entity_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> str:
        """
        Store authentication state securely with caching.

        Args:
            app_name: Name of the app being authenticated
            auth_type: Type of authentication (oauth, api_key, etc.)
            credentials: Authentication credentials (will be encrypted)
            entity_id: Optional entity ID for scoped authentication
            expires_at: Optional expiration time for credentials

        Returns:
            Reference ID for stored authentication state
        """
        if not self.enable_auth_caching:
            raise ValueError("Authentication caching is disabled")

        # Encrypt credentials for secure storage
        try:
            encrypted_credentials = self._encrypt_credentials(credentials)
        except ValueError as e:
            # Fall back to plaintext if encryption fails (with warning)
            print(
                f"Warning: Storing credentials in plaintext due to encryption failure: {e}"
            )
            encrypted_credentials = credentials

        # Create authentication state
        auth_state = AuthenticationState(
            app_name=app_name,
            auth_type=auth_type,
            credentials=credentials,  # Keep original for in-memory state
            expires_at=expires_at,
        )

        # Prepare storage data with encrypted credentials
        auth_data = {
            "app_name": app_name,
            "auth_type": auth_type,
            "encrypted_credentials": encrypted_credentials,
            "is_encrypted": isinstance(encrypted_credentials, str),
            "expires_at": expires_at.isoformat() if expires_at else None,
            "created_at": auth_state.created_at.isoformat(),
            "last_used": auth_state.last_used.isoformat(),
            "is_valid": auth_state.is_valid,
            "entity_id": entity_id,
        }

        # Create storage key
        auth_key = f"auth_{app_name}_{entity_id}" if entity_id else f"auth_{app_name}"

        context_metadata = {
            "content_type": "composio/authentication",
            "app_name": app_name,
            "auth_type": auth_type,
            "entity_id": entity_id,
            "created_timestamp": auth_state.created_at.isoformat(),
        }

        ref_id = self.state.add_large_context(
            auth_data,
            metadata=context_metadata,
            key=auth_key,
        )
        # Cache in memory for quick access
        self._auth_states[auth_key] = auth_state
        # Update performance stats
        if self.enable_performance_monitoring:
            self._performance_stats["total_auth_operations"] += 1

        return ref_id

    def get_authentication_state(
        self,
        app_name: str,
        entity_id: Optional[str] = None,
    ) -> Optional[AuthenticationState]:
        """
        Retrieve authentication state for an app.

        Args:
            app_name: Name of the app
            entity_id: Optional entity ID for scoped authentication

        Returns:
            Authentication state or None if not found/expired
        """
        if not self.enable_auth_caching:
            return None

        auth_key = f"auth_{app_name}_{entity_id}" if entity_id else f"auth_{app_name}"

        # Check memory cache first
        if auth_key in self._auth_states:
            auth_state = self._auth_states[auth_key]
            if auth_state.expires_at and datetime.now() > auth_state.expires_at:
                # Expired, remove from cache
                del self._auth_states[auth_key]
                if auth_key in self.state:
                    del self.state[auth_key]
                return None
            return auth_state
        # Check persistent storage
        try:
            auth_data = self.state.get_context(auth_key)

            # Decrypt credentials if they were encrypted
            credentials = auth_data.get("credentials")  # Legacy fallback
            if (
                auth_data.get("is_encrypted", False)
                and "encrypted_credentials" in auth_data
            ):
                try:
                    credentials = self._decrypt_credentials(
                        auth_data["encrypted_credentials"]
                    )
                except ValueError as e:
                    print(f"Warning: Failed to decrypt credentials: {e}")
                    # Fall back to encrypted data (will likely fail on use)
                    credentials = auth_data.get("encrypted_credentials", {})
            elif "encrypted_credentials" in auth_data:
                # Handle legacy data that might not have is_encrypted flag
                credentials = auth_data["encrypted_credentials"]

            # Reconstruct authentication state
            auth_state = AuthenticationState(
                app_name=auth_data["app_name"],
                auth_type=auth_data["auth_type"],
                credentials=credentials,
                expires_at=(
                    datetime.fromisoformat(auth_data["expires_at"])
                    if auth_data.get("expires_at")
                    else None
                ),
                created_at=datetime.fromisoformat(auth_data["created_at"]),
                last_used=datetime.fromisoformat(auth_data["last_used"]),
                is_valid=auth_data.get("is_valid", True),
            )

            # Check if expired
            if auth_state.expires_at and datetime.now() > auth_state.expires_at:
                # Clean up expired auth
                del self.state[auth_key]
                return None

            # Cache in memory
            self._auth_states[auth_key] = auth_state
            return auth_state

        except KeyError:
            return None

    def store_trigger_state(
        self,
        trigger_id: str,
        app_name: str,
        trigger_name: str,
        config: Dict[str, Any],
        webhook_url: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        """
        Store trigger configuration and state.

        Args:
            trigger_id: Unique trigger identifier
            app_name: Name of the app
            trigger_name: Name of the trigger
            config: Trigger configuration
            webhook_url: Optional webhook URL
            entity_id: Optional entity ID

        Returns:
            Reference ID for stored trigger state
        """
        if not self.enable_trigger_management:
            raise ValueError("Trigger management is disabled")

        # Create trigger state
        trigger_state = TriggerState(
            trigger_id=trigger_id,
            app_name=app_name,
            trigger_name=trigger_name,
            config=config,
            webhook_url=webhook_url,
        )
        trigger_data = {
            "trigger_id": trigger_id,
            "app_name": app_name,
            "trigger_name": trigger_name,
            "config": config,
            "webhook_url": webhook_url,
            "is_active": trigger_state.is_active,
            "created_at": trigger_state.created_at.isoformat(),
            "last_triggered": (
                trigger_state.last_triggered.isoformat()
                if trigger_state.last_triggered
                else None
            ),
            "event_count": trigger_state.event_count,
            "entity_id": entity_id,
        }
        context_metadata = {
            "content_type": "composio/trigger",
            "app_name": app_name,
            "trigger_name": trigger_name,
            "trigger_id": trigger_id,
            "entity_id": entity_id,
            "created_timestamp": trigger_state.created_at.isoformat(),
        }

        ref_id = self.state.add_large_context(
            trigger_data,
            metadata=context_metadata,
            key=f"trigger_{trigger_id}",
        )

        # Cache in memory
        self._trigger_states[trigger_id] = trigger_state
        # Update performance stats
        if self.enable_performance_monitoring:
            self._performance_stats["total_trigger_operations"] += 1

        return ref_id

    def handle_trigger_event(
        self,
        trigger_id: str,
        event_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Handle and store trigger event data.

        Args:
            trigger_id: Trigger that generated the event
            event_data: Event payload data
            metadata: Optional event metadata

        Returns:
            Reference ID for stored event
        """
        event_id = str(uuid.uuid4())

        # Prepare event storage
        event_storage = {
            "event_id": event_id,
            "trigger_id": trigger_id,
            "event_data": event_data,
            "received_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Update trigger state
        if trigger_id in self._trigger_states:
            trigger_state = self._trigger_states[trigger_id]
            trigger_state.last_triggered = datetime.now()
            trigger_state.event_count += 1
            # Update stored trigger state
            self.store_trigger_state(
                trigger_id=trigger_state.trigger_id,
                app_name=trigger_state.app_name,
                trigger_name=trigger_state.trigger_name,
                config=trigger_state.config,
                webhook_url=trigger_state.webhook_url,
            )

        context_metadata = {
            "content_type": "composio/trigger_event",
            "trigger_id": trigger_id,
            "event_id": event_id,
            "received_timestamp": datetime.now().isoformat(),
        }

        ref_id = self.state.add_large_context(
            event_storage,
            metadata=context_metadata,
            key=f"event_{event_id}",
        )

        return ref_id

    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        app_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve execution history with optional filtering.

        Args:
            tool_name: Optional tool name filter
            app_name: Optional app name filter
            limit: Maximum number of results

        Returns:
            List of execution history records
        """
        history = []

        # Get all execution contexts
        all_refs = self.state.list_context_references()
        execution_refs = [ref for ref in all_refs if ref.startswith("tool_cache_")]

        for ref in execution_refs[:limit]:
            try:
                exec_data = self.state.get_context(ref)
                # Apply filters
                if tool_name and exec_data.get("tool_name") != tool_name:
                    continue
                if (
                    app_name
                    and exec_data.get("metadata", {}).get("app_name") != app_name
                ):
                    continue

                history.append(exec_data)

            except KeyError:
                continue

        # Sort by cached_at timestamp
        history.sort(key=lambda x: x.get("cached_at", ""), reverse=True)
        return history[:limit]

    def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics for the adapter.

        Returns:
            Dictionary containing performance metrics and analytics
        """
        context_stats = self.context_store.get_cache_stats()

        # Calculate cache hit rate
        total_cache_attempts = (
            self._performance_stats["cache_hits"]
            + self._performance_stats["cache_misses"]
        )
        cache_hit_rate = (
            (self._performance_stats["cache_hits"] / total_cache_attempts * 100)
            if total_cache_attempts > 0
            else 0
        )

        return {
            "context_store_stats": context_stats,
            "composio_performance": self._performance_stats,
            "cache_analytics": {
                "cache_hit_rate": cache_hit_rate,
                "total_cache_attempts": total_cache_attempts,
                "cache_hits": self._performance_stats["cache_hits"],
                "cache_misses": self._performance_stats["cache_misses"],
            },
            "feature_usage": {
                "tool_caching_enabled": self.enable_tool_caching,
                "auth_caching_enabled": self.enable_auth_caching,
                "trigger_management_enabled": self.enable_trigger_management,
                "workspace_support_enabled": self.enable_workspace_support,
                "performance_monitoring_enabled": self.enable_performance_monitoring,
            },
            "active_components": {
                "active_executions": len(self._active_executions),
                "cached_auth_states": len(self._auth_states),
                "managed_triggers": len(self._trigger_states),
                "workspace_environments": len(self._workspace_states),
            },
            "configuration": {
                "cache_expiry_hours": self.cache_expiry_hours,
                "cache_size": getattr(self.context_store, "cache_size", "unknown"),
            },
            "recent_executions": {
                exec_id: {
                    "tool_name": metrics.tool_name,
                    "action_name": metrics.action_name,
                    "duration": metrics.duration_seconds,
                    "status": metrics.status.value,
                }
                for exec_id, metrics in list(self._execution_metrics.items())[
                    -10:
                ]  # Last 10
            },
        }

    def cleanup_expired_data(self, max_age_hours: int = 72):
        """
        Clean up expired execution results, auth states, and other cached data.

        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        # Clean up expired tool execution cache
        expired_refs = []
        all_refs = self.state.list_context_references()

        for ref in all_refs:
            try:
                if ref.startswith("tool_cache_"):
                    data = self.state.get_context(ref)
                    cached_at = datetime.fromisoformat(data.get("cached_at", ""))
                    if cached_at < cutoff_time:
                        expired_refs.append(ref)
            except (KeyError, ValueError):
                continue

        # Remove expired references
        for ref in expired_refs:
            if ref in self.state:
                del self.state[ref]

        # Clean up expired auth states
        expired_auth_keys = []
        for auth_key, auth_state in self._auth_states.items():
            if auth_state.expires_at and auth_state.expires_at < datetime.now():
                expired_auth_keys.append(auth_key)

        for auth_key in expired_auth_keys:
            del self._auth_states[auth_key]
            if auth_key in self.state:
                del self.state[auth_key]

        # Clean up old execution metrics
        old_metrics = [
            exec_id
            for exec_id, metrics in self._execution_metrics.items()
            if metrics.start_time < cutoff_time
        ]

        for exec_id in old_metrics:
            del self._execution_metrics[exec_id]

    def create_enhanced_toolset(
        self, entity_id: Optional[str] = None
    ) -> "EnhancedComposioToolSet":
        """
        Create an enhanced Composio toolset with Context Reference Store optimization.

        Args:
            entity_id: Optional entity ID for authentication context

        Returns:
            Enhanced toolset instance
        """
        return EnhancedComposioToolSet(adapter=self, entity_id=entity_id)

    def get_available_tools(
        self, app_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of available tools, optionally filtered by app.

        Args:
            app_name: Optional app name filter

        Returns:
            List of available tools with metadata
        """
        if not self.composio_client:
            return []

        try:
            # Get tools from Composio
            if app_name:
                app = App(name=app_name)
                actions = app.get_actions()
            else:
                actions = self.composio_client.actions.list()

            tools = []
            for action in actions:
                tool_info = {
                    "name": action.name,
                    "app_name": getattr(action, "app_name", app_name),
                    "description": getattr(action, "description", ""),
                    "parameters": getattr(action, "parameters", {}),
                    "enabled": getattr(action, "enabled", True),
                }
                tools.append(tool_info)

            return tools

        except Exception as e:
            print(f"Error fetching available tools: {e}")
            return []


class EnhancedComposioToolSet:
    """
    Enhanced Composio toolset with Context Reference Store optimization.

    Provides intelligent caching, performance monitoring, and state management
    for Composio tool executions.
    """

    def __init__(
        self, adapter: ComposioContextAdapter, entity_id: Optional[str] = None
    ):
        """
        Initialize the enhanced toolset.

        Args:
            adapter: ComposioContextAdapter instance
            entity_id: Optional entity ID for authentication context
        """
        self.adapter = adapter
        self.entity_id = entity_id
        self._execution_history = []

    def execute(
        self,
        tool_name: str,
        action_name: str,
        inputs: Dict[str, Any],
        use_cache: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool with enhanced caching and monitoring.

        Args:
            tool_name: Name of the tool to execute
            action_name: Name of the action to perform
            inputs: Input parameters
            use_cache: Whether to use cached results
            timeout: Optional execution timeout

        Returns:
            Tool execution results
        """
        result = self.adapter.execute_tool_with_caching(
            tool_name=tool_name,
            action_name=action_name,
            inputs=inputs,
            entity_id=self.entity_id,
            force_refresh=not use_cache,
        )
        # Track in execution history
        self._execution_history.append(
            {
                "tool_name": tool_name,
                "action_name": action_name,
                "inputs": inputs,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "cached": result.get("cached", False),
            }
        )

        return result

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of tool executions for this toolset.

        Returns:
            Execution summary statistics
        """
        total_executions = len(self._execution_history)
        cached_executions = sum(
            1 for exec in self._execution_history if exec.get("cached")
        )

        return {
            "total_executions": total_executions,
            "cached_executions": cached_executions,
            "cache_hit_rate": (
                (cached_executions / total_executions * 100)
                if total_executions > 0
                else 0
            ),
            "recent_executions": self._execution_history[-10:],  # Last 10
        }

    def clear_history(self):
        """Clear the execution history for this toolset."""
        self._execution_history.clear()
