"""State management for the graph system.

This module provides:
1. NodeStatus: An enumeration of node execution statuses
2. NodeStateProtocol: Protocol defining the interface for state objects
3. NodeState: A lightweight state container for passing data between nodes
"""

from typing import Dict, Any, Set, Protocol, runtime_checkable, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

@runtime_checkable
class NodeStateProtocol(Protocol):
    """Protocol defining the interface for NodeState."""
    data: Dict[str, Any]
    results: Dict[str, Dict[str, Any]]
    errors: Dict[str, str]

class NodeStatus(str, Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TERMINAL = "terminal"

class NodeState(BaseModel):
    """
    Lightweight state container for passing data between nodes.
    
    Attributes:
        data: Global data shared across workflow
        results: Node-specific results (structured outputs)
        errors: Error messages by node ID
        status: Node execution status
        parallel_tasks: Currently running parallel tasks
        created_at: Time of state creation
        updated_at: Time of last state modification
        start_time: Time when current node started processing
    """
    data: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    status: Dict[str, NodeStatus] = Field(default_factory=dict)
    parallel_tasks: Set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    start_time: Optional[datetime] = Field(default=None)

    def mark_status(self, node_id: str, status: NodeStatus) -> None:
        """Mark a node's execution status."""
        self.status[node_id] = status
        self._update_timestamp()

    def add_parallel_task(self, node_id: str) -> None:
        """Add a node to the set of parallel tasks."""
        self.parallel_tasks.add(node_id)
        self._update_timestamp()

    def remove_parallel_task(self, node_id: str) -> None:
        """Remove a node from the set of parallel tasks."""
        self.parallel_tasks.discard(node_id)
        self._update_timestamp()

    def get_result(self, node_id: str, key: str) -> Any:
        """Get a specific result value for a node."""
        return self.results.get(node_id, {}).get(key)

    def set_result(self, node_id: str, key: str, value: Any) -> None:
        """Set a specific result value for a node."""
        if node_id not in self.results:
            self.results[node_id] = {}
        self.results[node_id][key] = value
        self._update_timestamp()

    def add_error(self, node_id: str, error: str) -> None:
        """Add an error message for a node."""
        self.errors[node_id] = error
        self._update_timestamp()

    def _update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        object.__setattr__(self, "updated_at", datetime.utcnow()) 