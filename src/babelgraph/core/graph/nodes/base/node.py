"""Base node class for the graph system.

This module defines the Node abstraction for the Graph framework. A Node represents
an individual unit of work (e.g., an LLM call, a tool invocation, a context injection)
that can be executed within a larger workflow. Nodes are validated via Pydantic and
support asynchronous execution for flexible orchestration.

Typical Usage:
    - Create a subclass of Node
    - Override the 'process' method to implement custom logic
    - Use 'next_nodes' to specify transitions to other nodes in the graph
"""

import abc
from typing import Tuple, Dict, Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING, Callable, Set
from functools import wraps
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from babelgraph.core.logging import get_logger, LogComponent
from babelgraph.core.graph.state import NodeStateProtocol, NodeStatus, ParallelTaskStatus
import json
from babelgraph.core.logging import Colors
from mirascope.core import BaseMessageParam

if TYPE_CHECKING:
    from babelgraph.core.graph.state import NodeState

# Get logger for node operations
logger = get_logger(LogComponent.NODES)

class ParallelExecutionPattern(str, Enum):
    """Parallel execution patterns."""
    NONE = "none"           # Not parallel
    CONCURRENT = "concurrent"  # Run all in parallel
    PIPELINE = "pipeline"      # Parallel pipeline stages
    MAP = "map"               # Map over inputs
    JOIN = "join"             # Join parallel flows

class ParallelConfig(BaseModel):
    """Configuration for parallel execution."""
    pattern: ParallelExecutionPattern = Field(default=ParallelExecutionPattern.NONE)
    dependencies: Set[str] = Field(default_factory=set)
    max_concurrent: Optional[int] = None
    timeout: Optional[float] = None
    retry_count: Optional[int] = None

def parallel_node(
    pattern: ParallelExecutionPattern,
    dependencies: Optional[Set[str]] = None,
    max_concurrent: Optional[int] = None,
    timeout: Optional[float] = None,
    retry_count: Optional[int] = None
):
    """Decorator to configure parallel execution."""
    def decorator(cls):
        cls.parallel_config = ParallelConfig(
            pattern=pattern,
            dependencies=dependencies or set(),
            max_concurrent=max_concurrent,
            timeout=timeout,
            retry_count=retry_count
        )
        return cls
    return decorator

def terminal_node(cls):
    """Decorator to mark a node as terminal.
    
    Example:
        @terminal_node
        class EndNode(Node):
            async def process(self, state: NodeState) -> Optional[str]:
                # Process and display results
                return None
    """
    cls.is_terminal = True
    cls.next_nodes = {}
    return cls

def state_handler(func: Callable):
    """Decorator to handle state updates and error handling.
    
    This decorator provides flexible state management:
    1. Basic success/error states
    2. Custom state transitions
    3. Conditional state handling
    
    Example (Basic):
        @state_handler
        async def process(self, state):
            # Basic success/error
            return "success"
    
    Example (Custom States):
        @state_handler
        async def process(self, state):
            result = await self.analyze()
            # Return any state that matches your next_nodes
            return "high_confidence" if result.score > 0.8 else "low_confidence"
            
    Example (Conditional):
        @state_handler
        async def process(self, state):
            if self.is_validation_node:
                return "valid" if validate() else "invalid"
            return "default"
    """
    @wraps(func)
    async def wrapper(self, state: "NodeState") -> Optional[str]:
        try:
            # Mark node as running
            state.mark_status(self.id, "running")
            
            # Execute the process function
            result = await func(self, state)
            
            # Mark as completed if we got a valid transition
            if result in self.next_nodes or result is None and self.is_terminal:
                state.mark_status(self.id, "completed")
            else:
                state.mark_status(self.id, "error")
                state.add_error(self.id, f"Invalid transition state: {result}")
                return "error"
            
            return result
        except Exception as e:
            # Handle errors
            state.mark_status(self.id, "error")
            state.add_error(self.id, str(e))
            logger.error(f"Error in node {self.id}: {e}")
            return "error"
    return wrapper

class Node(BaseModel):
    """
    Abstract base node for graph operations.
    
    Attributes:
        id: Unique node identifier
        next_nodes: Mapping of conditions to next node IDs
        metadata: Optional node metadata
        parallel_config: Configuration for parallel execution
        input_map: Mapping of parameter names to state references
        is_terminal: Whether the node is terminal
    """
    id: str = Field(..., description="Unique identifier for this node")
    next_nodes: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {"default": None, "error": None}
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parallel_config: ParallelConfig = Field(
        default_factory=lambda: ParallelConfig()
    )
    input_map: Dict[str, str] = Field(default_factory=dict)
    is_terminal: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def validate_node(self) -> 'Node':
        """Validate node configuration."""
        if not self.id:
            raise ValueError("Node must have an ID")
        return self

    def get_next_node(self, condition: str = "default") -> Optional[str]:
        """Get next node ID for given condition."""
        return self.next_nodes.get(condition)

    async def process(self, state: "NodeState") -> Optional[str]:
        """Process node logic. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process()")

    def validate(self) -> bool:
        """
        Validate node configuration.

        Override in subclasses if additional checks are required.

        Returns:
            True if the node is considered valid.
        """
        return True

    def is_ready(self, state: "NodeState") -> bool:
        """Check if node is ready to execute based on parallel config."""
        if self.parallel_config.pattern == ParallelExecutionPattern.NONE:
            return True
            
        # Check dependencies
        for dep in self.parallel_config.dependencies:
            if dep not in state.parallel_tasks:
                return False
            task = state.parallel_tasks[dep]
            if task.status != ParallelTaskStatus.COMPLETED:
                return False
                
        # Check concurrent limit
        if (self.parallel_config.max_concurrent and 
            len(state.get_running_tasks()) >= self.parallel_config.max_concurrent):
            return False
            
        return True

    @staticmethod
    def _extract_node_reference(ref: str) -> Tuple[str, str]:
        """
        Extract the node ID and the nested key path from a string like 'node.my_node_id.some.deep.key'.
        
        Args:
            ref: The reference string, beginning with 'node.'.

        Returns:
            Tuple of (node_id, nested_key).
        
        Raises:
            ValueError if the format does not match the pattern 'node.<node_id>.<key>'.
        """
        # Must start with node. 
        if not ref.startswith("node."):
            raise ValueError(f"Invalid node reference: {ref}")
        parts = ref.split('.', 2)
        if len(parts) < 3:
            # We need at least: 'node', node_id, some_key
            raise ValueError(f"Reference must be at least 'node.<node_id>.<key>': {ref}")
        node_id = parts[1]  # The segment after 'node.'
        nested_key = parts[2]  # The remainder after node.<node_id>.
        return node_id, nested_key

    @staticmethod
    def _extract_data_reference(ref: str) -> str:
        """
        Extract the nested key path from a string like 'data.some.deep.key'.

        Args:
            ref: The reference string, beginning with 'data.'.

        Returns:
            The nested key (e.g., 'some.deep.key').

        Raises:
            ValueError if the format does not match 'data.<key>'.
        """
        if not ref.startswith("data."):
            raise ValueError(f"Invalid data reference: {ref}")
        parts = ref.split('.', 1)
        if len(parts) < 2:
            raise ValueError(f"Reference must be at least 'data.<key>': {ref}")
        return parts[1]

    def _get_nested_value(self, data: Dict[str, Any], dotted_key: str) -> Any:
        """
        Retrieve a nested value from a dictionary using a dotted key path.

        Args:
            data: The dictionary to search.
            dotted_key: The dotted key path, e.g., 'user.profile.name'.

        Returns:
            The value found at the specified key path.

        Raises:
            ValueError: If the key path does not exist in the data.
        """
        current = data
        for part in dotted_key.split('.'):
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Key '{part}' not found while traversing '{dotted_key}'")
            current = current[part]
        return current

    def _prepare_input_data(self, state: NodeStateProtocol) -> Dict[str, Any]:
        """
        Prepare input data for the node based on input_map references and NodeState.

        This method interprets references like:
            'node.<node_id>.<path>' - retrieves from state.results[node_id], nested by <path>
            'data.<path>'           - retrieves from state.data, nested by <path>

        Returns:
            A dictionary of resolved parameters and their values.
        """
        input_data = {}
        logger.debug(f"Node {self.id}: Preparing input data")
        logger.debug(f"Node {self.id}: Current state results: {state.results}")
        logger.debug(f"Node {self.id}: Current state data: {state.data}")
        logger.debug(f"Node {self.id}: Input map: {self.input_map}")

        for param_name, ref in self.input_map.items():
            logger.debug(f"Node {self.id}: Processing parameter '{param_name}' with reference '{ref}'")

            if ref.startswith("node."):
                # Node result reference
                node_id, nested_key = self._extract_node_reference(ref)
                if node_id not in state.results:
                    raise ValueError(
                        f"Node ID '{node_id}' not found in results for param '{param_name}'. "
                        f"Available results: {list(state.results.keys())}"
                    )
                value = self._get_nested_value(state.results[node_id], nested_key)
                input_data[param_name] = value
                logger.debug(f"Node {self.id}: Retrieved from node '{node_id}' with dotted_key '{nested_key}': {value}")

            elif ref.startswith("data."):
                # State data reference
                data_path = self._extract_data_reference(ref)
                value = self._get_nested_value(state.data, data_path)
                input_data[param_name] = value
                logger.debug(f"Node {self.id}: Retrieved from state data path '{data_path}': {value}")

            else:
                # If there's no prefix, we can either default to data or raise an error.
                # Here, we choose to raise an error for clarity.
                error_msg = (
                    f"Invalid reference '{ref}' for param '{param_name}'. "
                    "Must start with 'node.' or 'data.'."
                )
                logger.error(f"Node {self.id}: {error_msg}")
                raise ValueError(error_msg)

        logger.debug(f"Node {self.id}: Final prepared input data: {input_data}")
        return input_data

    # Helper methods for state management
    def update_state(self, state: "NodeState", key: str, value: Any) -> None:
        """Update state with a key-value pair."""
        state.data[key] = value
        
    def get_state(self, state: "NodeState", key: str, default: Any = None) -> Any:
        """Get value from state."""
        return state.data.get(key, default)
        
    def _log_node_result(self, state: "NodeState", result: Any, suppress: bool = False) -> None:
        """Log node results at INFO level.
        
        Args:
            state: Current node state
            result: Result to log
            suppress: Whether to suppress logging (useful for streaming)
        """
        if suppress:
            return
            
        try:
            # Format result based on type
            if hasattr(result, 'model_dump_json'):
                formatted = result.model_dump_json(indent=2)
            elif isinstance(result, dict):
                formatted = json.dumps(result, indent=2)
            else:
                formatted = str(result)
            
            # Log the result with node context
            logger.info(
                f"\n{Colors.BOLD}Node {self.id} Output:{Colors.RESET}\n"
                f"{Colors.INFO}{formatted}{Colors.RESET}\n"
                f"{Colors.DIM}{'─' * 50}{Colors.RESET}"
            )
        except Exception as e:
            logger.error(f"Error logging node {self.id} result: {str(e)}")

    def set_result(self, state: "NodeState", key: str, value: Any, suppress_logging: bool = False) -> None:
        """Store a result in the state and optionally log it.
        
        Args:
            state: The node state to update
            key: Key for the result
            value: Value to store
            suppress_logging: Whether to suppress result logging (useful for streaming)
        """
        if self.id not in state.results:
            state.results[self.id] = {}
        state.results[self.id][key] = value
        
        # Log the result if it's the primary output and logging isn't suppressed
        if key == 'response' and not suppress_logging:
            self._log_node_result(state, value)

    # Helper methods for node configuration
    def add_next_node(self, condition: str, node_id: Optional[str]) -> None:
        """Add a transition to another node."""
        self.next_nodes[condition] = node_id
        
    def remove_next_node(self, condition: str) -> None:
        """Remove a transition."""
        self.next_nodes.pop(condition, None)
        
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    def set_input(self, state: "NodeState", value: Any, key: str = "input") -> None:
        """
        Set input value in state with optional key name.
        
        Args:
            state: The node state to update
            value: The input value to set
            key: Optional key name, defaults to "input"
        """
        state.data[f"{self.id}_{key}"] = value
        
    def get_input(self, state: "NodeState", key: str = "input", default: Any = None) -> Any:
        """
        Get input value from state with optional key name.
        
        Args:
            state: The node state to read from
            key: Optional key name, defaults to "input"
            default: Default value if not found
            
        Returns:
            The input value or default if not found
        """
        return state.data.get(f"{self.id}_{key}", default)

    def set_message_input(
        self, 
        state: "NodeState", 
        content: str, 
        role: str = "user"
    ) -> None:
        """
        Set a message input in state using Mirascope's BaseMessageParam structure.
        
        Args:
            state: The node state to update
            content: The message content
            role: Message role (user/assistant/system)
        """
        message = BaseMessageParam(
            role=role,
            content=content
        ).model_dump()
        
        self.set_input(state, message, key="message")
        
    def get_message_input(self, state: "NodeState", default: Optional[Dict] = None) -> Optional[Dict]:
        """
        Get message input from state.
        
        Args:
            state: The node state to read from
            default: Default value if not found
            
        Returns:
            The message dict or default if not found
        """
        return self.get_input(state, key="message", default=default)

    def clear_input(self, state: "NodeState", key: str = "input") -> None:
        """
        Clear input value from state.
        
        Args:
            state: The node state to update
            key: Optional key name, defaults to "input"
        """
        state.data.pop(f"{self.id}_{key}", None)
        
    def has_input(self, state: "NodeState", key: str = "input") -> bool:
        """
        Check if input exists in state.
        
        Args:
            state: The node state to check
            key: Optional key name, defaults to "input"
            
        Returns:
            True if input exists, False otherwise
        """
        return f"{self.id}_{key}" in state.data 