
"""
Minimal State interface for Context Reference Store.

This provides a compatible interface with ADK's State class while being
framework-agnostic.
"""

from typing import Any, Dict, Optional


class BaseState:
    """A state dict that maintains the current value and pending-commit delta."""

    APP_PREFIX = "app:"
    USER_PREFIX = "user:"
    TEMP_PREFIX = "temp:"

    def __init__(
        self,
        value: Optional[Dict[str, Any]] = None,
        delta: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            value: The current value of the state dict.
            delta: The delta change to the current value that hasn't been committed.
        """
        self._value = value or {}
        self._delta = delta or {}

    def __getitem__(self, key: str) -> Any:
        """Returns the value of the state dict for the given key."""
        if key in self._delta:
            return self._delta[key]
        return self._value[key]

    def __setitem__(self, key: str, value: Any):
        """Sets the value of the state dict for the given key."""
        self._value[key] = value
        self._delta[key] = value

    def __contains__(self, key: str) -> bool:
        """Whether the state dict contains the given key."""
        return key in self._value or key in self._delta

    def has_delta(self) -> bool:
        """Whether the state has pending delta."""
        return bool(self._delta)

    def get(self, key: str, default: Any = None) -> Any:
        """Returns the value of the state dict for the given key."""
        if key not in self:
            return default
        return self[key]

    def update(self, delta: Dict[str, Any]):
        """Updates the state dict with the given delta."""
        self._value.update(delta)
        self._delta.update(delta)

    def to_dict(self) -> Dict[str, Any]:
        """Returns the state dict."""
        result = {}
        result.update(self._value)
        result.update(self._delta)
        return result

    def clear_delta(self):
        """Clears the pending delta."""
        self._delta.clear()

    def commit_delta(self):
        """Commits the pending delta to the main value."""
        self._value.update(self._delta)
        self._delta.clear()

    def keys(self):
        """Returns all keys in the state."""
        all_keys = set(self._value.keys())
        all_keys.update(self._delta.keys())
        return all_keys

    def items(self):
        """Returns all key-value pairs in the state."""
        result = self.to_dict()
        return result.items()

    def values(self):
        """Returns all values in the state."""
        result = self.to_dict()
        return result.values()
