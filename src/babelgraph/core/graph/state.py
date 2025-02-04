"""State management for the graph system.

This module provides:
1. NodeStatus: An enumeration of node execution statuses
2. NodeStateProtocol: Protocol defining the interface for state objects
3. NodeState: A lightweight state container for passing data between nodes
4. ParallelTask: Rich state tracking for parallel execution
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
    WAITING = "waiting"  # New status for parallel coordination

class ParallelTaskStatus(str, Enum):
    """Status tracking for parallel tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    BLOCKED = "blocked"  # Waiting on dependencies

class ParallelTask(BaseModel):
    """Rich state tracking for parallel tasks."""
    id: str
    status: ParallelTaskStatus = Field(default=ParallelTaskStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: Set[str] = Field(default_factory=set)
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class NodeState(BaseModel):
    """
    Lightweight state container for passing data between nodes.
    
    Attributes:
        data: Global data shared across workflow
        results: Node-specific results (structured outputs)
        errors: Error messages by node ID
        status: Node execution status
        parallel_tasks: Rich state tracking for parallel execution
        created_at: Time of state creation
        updated_at: Time of last state modification
        start_time: Time when current node started processing
    """
    data: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    status: Dict[str, NodeStatus] = Field(default_factory=dict)
    parallel_tasks: Dict[str, ParallelTask] = Field(default_factory=dict)  # Enhanced parallel tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    start_time: Optional[datetime] = Field(default=None)

    def mark_status(self, node_id: str, status: NodeStatus) -> None:
        """Mark a node's execution status."""
        self.status[node_id] = status
        self._update_timestamp()

    def add_parallel_task(self, task_id: str, dependencies: Optional[Set[str]] = None) -> None:
        """Add a parallel task with optional dependencies."""
        self.parallel_tasks[task_id] = ParallelTask(
            id=task_id,
            dependencies=dependencies or set()
        )
        self._update_timestamp()

    def start_parallel_task(self, task_id: str) -> None:
        """Mark a parallel task as running."""
        if task_id in self.parallel_tasks:
            task = self.parallel_tasks[task_id]
            task.status = ParallelTaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self._update_timestamp()

    def complete_parallel_task(self, task_id: str, results: Optional[Dict[str, Any]] = None) -> None:
        """Mark a parallel task as completed with optional results."""
        if task_id in self.parallel_tasks:
            task = self.parallel_tasks[task_id]
            task.status = ParallelTaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            if results:
                task.results.update(results)
            self._update_timestamp()

    def fail_parallel_task(self, task_id: str, error: str) -> None:
        """Mark a parallel task as failed with error message."""
        if task_id in self.parallel_tasks:
            task = self.parallel_tasks[task_id]
            task.status = ParallelTaskStatus.ERROR
            task.error = error
            task.completed_at = datetime.utcnow()
            self._update_timestamp()

    def get_ready_tasks(self) -> Set[str]:
        """Get tasks that are ready to run (dependencies satisfied)."""
        ready = set()
        for task_id, task in self.parallel_tasks.items():
            if task.status == ParallelTaskStatus.PENDING:
                if all(
                    self.parallel_tasks[dep].status == ParallelTaskStatus.COMPLETED
                    for dep in task.dependencies
                ):
                    ready.add(task_id)
        return ready

    def get_running_tasks(self) -> Set[str]:
        """Get currently running tasks."""
        return {
            task_id for task_id, task in self.parallel_tasks.items()
            if task.status == ParallelTaskStatus.RUNNING
        }

    def get_blocked_tasks(self) -> Set[str]:
        """Get tasks blocked on dependencies."""
        return {
            task_id for task_id, task in self.parallel_tasks.items()
            if task.status == ParallelTaskStatus.PENDING
            and any(
                self.parallel_tasks[dep].status != ParallelTaskStatus.COMPLETED
                for dep in task.dependencies
            )
        }

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