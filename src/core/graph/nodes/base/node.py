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
from typing import Tuple, Dict, Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator
from alchemist.ai.base.logging import get_logger, LogComponent
from alchemist.ai.graph.state import NodeStateProtocol

if TYPE_CHECKING:
    from alchemist.ai.graph.state import NodeState

# Get logger for node operations
logger = get_logger(LogComponent.NODES)

class Node(BaseModel):
    """
    Abstract base node for graph operations.
    
    Attributes:
        id: Unique node identifier
        next_nodes: Mapping of conditions to next node IDs
        metadata: Optional node metadata
        parallel: Whether node can run in parallel
        input_map: Mapping of parameter names to state references
    """
    id: str = Field(..., description="Unique identifier for this node")
    next_nodes: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {"default": None, "error": None}
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parallel: bool = Field(default=False)
    input_map: Dict[str, str] = Field(default_factory=dict)

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