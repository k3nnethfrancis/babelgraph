"""Graph Base Classes

This module defines the core graph system for orchestrating agent workflows.
The graph provides a lightweight way to:
1. Connect nodes (agents, tools, etc.) in a directed graph
2. Route messages and state between nodes
3. Execute workflows asynchronously
4. Support structured outputs via Pydantic models

Example:
    ```python
    # Create nodes
    analyzer = AnalyzerNode(id="analyzer")
    decision = DecisionNode(id="decision") 
    action = ActionNode(id="action")

    # Build graph
    graph = Graph()
    graph.add_node(analyzer)
    graph.add_node(decision)
    graph.add_node(action)
    
    # Connect nodes
    graph.add_edge("analyzer", "success", "decision")
    graph.add_edge("decision", "execute", "action")
    
    # Run graph
    state = await graph.run(entry_point="analyzer")
    ```
"""

from typing import Dict, Any, Optional, Union, List, Set
import logging
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field, PrivateAttr

from babelgraph.core.logging import (
    LogComponent,
    BabelLoggingConfig,
    log_verbose,
    VerbosityLevel,
    get_logger,
    configure_logging,
    LogLevel
)
from babelgraph.core.graph.state import NodeState, NodeStatus
from babelgraph.core.graph.nodes.base.node import Node

# Get logger for graph component
logger = logging.getLogger(LogComponent.GRAPH.value)

class Graph(BaseModel):
    """A directed graph for orchestrating agent workflows.
    
    The graph manages:
    - Node registration and connections
    - State passing between nodes  
    - Async execution of workflows
    - Structured outputs via Pydantic
    
    Attributes:
        nodes: Dictionary mapping node IDs to Node instances
        entry_points: Dictionary mapping entry point names to starting node IDs
        logging_config: Controls logging verbosity
    """
    nodes: Dict[str, Node] = Field(default_factory=dict)
    entry_points: Dict[str, str] = Field(default_factory=dict) 
    logging_config: BabelLoggingConfig = Field(
        default_factory=BabelLoggingConfig
    )
    _logger: logging.Logger = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._logger = get_logger(LogComponent.GRAPH)

    def add_node(self, node: Node) -> None:
        """Register a node with the graph.

        Args:
            node: Node instance to add

        Raises:
            ValueError: If node has no ID or fails validation
        """
        if not node.id:
            raise ValueError("Node must have an id set")
        
        if not node.validate():
            raise ValueError(f"Node {node.id} failed validation")

        self.nodes[node.id] = node
        self._logger.info(f"Added node: {node.id} of type {type(node).__name__}")

    def add_edge(self, from_node_id: str, transition_key: str, to_node_id: str) -> None:
        """Add a directed edge between nodes.

        Args:
            from_node_id: Source node ID
            transition_key: Key identifying the transition (e.g. "success", "error")
            to_node_id: Target node ID

        Raises:
            ValueError: If either node ID is not found
        """
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node not found: {from_node_id}")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node not found: {to_node_id}")

        self.nodes[from_node_id].next_nodes[transition_key] = to_node_id
        self._logger.info(
            f"Added edge: {from_node_id} --[{transition_key}]--> {to_node_id}"
        )

    def set_entry_point(self, node_id: str, name: str = "default") -> None:
        """Set an entry point for the graph.

        Args:
            node_id: ID of the starting node
            name: Optional name for the entry point (defaults to "default")

        Raises:
            ValueError: If node_id is not found
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")
        self.entry_points[name] = node_id
        self._logger.info(f"Set entry point '{name}' to node: {node_id}")

    def chain(self, nodes: List[Node], transition_key: str = "next") -> None:
        """Connect a sequence of nodes in order.

        Args:
            nodes: List of nodes to chain together
            transition_key: Key to use for transitions (defaults to "next")
        """
        for node in nodes:
            self.add_node(node)
            
        for i in range(len(nodes) - 1):
            self.add_edge(nodes[i].id, transition_key, nodes[i + 1].id)

        # Set first node as default entry if no entry points exist
        if not self.entry_points:
            self.set_entry_point(nodes[0].id)

    def compose(self, other: "Graph", namespace: str) -> None:
        """Compose another graph into this one under a namespace.
        
        Args:
            other: The graph to compose into this one
            namespace: Prefix for node IDs to avoid collisions
            
        Example:
            main_graph.compose(subgraph, "agent1")
            # Node "start" becomes "agent1.start"
        """
        # Copy and namespace all nodes
        for node_id, node in other.nodes.items():
            new_id = f"{namespace}.{node_id}"
            node_copy = node.copy()
            node_copy.id = new_id
            
            # Update next_nodes references, treating values starting with '/' as absolute references
            node_copy.next_nodes = {
                k: (v.lstrip('/') if v.startswith('/') else f"{namespace}.{v}") if v else None
                for k, v in node.next_nodes.items()
            }
            
            self.nodes[new_id] = node_copy
            
        # Update entry points
        for name, node_id in other.entry_points.items():
            self.entry_points[f"{namespace}.{name}"] = f"{namespace}.{node_id}"
            
        self._logger.info(f"Composed graph under namespace: {namespace}")

    async def run(
        self,
        entry_point: Optional[str] = None,
        state: Optional[NodeState] = None,
        max_iterations: int = 10
    ) -> NodeState:
        """Run the graph from a specified entry point.

        Args:
            entry_point: Name of entry point (defaults to "default")
            state: Optional initial state
            max_iterations: Maximum number of node transitions to execute before aborting

        Returns:
            Final NodeState after execution

        Raises:
            ValueError: If entry point is not found
        """
        # Initialize or use provided state
        if state is None:
            state = NodeState()

        # Use default entry point if none specified
        entry_point = entry_point or "default"
        if entry_point not in self.entry_points:
            raise ValueError(f"Entry point not found: {entry_point}")

        current_node_id = self.entry_points[entry_point]
        self._logger.info(f"Starting graph execution at node: {current_node_id}")

        iteration_count = 0
        while current_node_id:
            iteration_count += 1
            if iteration_count > max_iterations:
                self._logger.error("Maximum loop iterations exceeded, aborting execution.")
                break
            node = self.nodes[current_node_id]

            # Update node status
            state.mark_status(current_node_id, NodeStatus.RUNNING)
            state.start_time = datetime.now()

            try:
                # Process node
                transition_key = await self._process_node(node, state)
                state.mark_status(current_node_id, NodeStatus.COMPLETED)

                # Get next node if transition exists
                current_node_id = (
                    node.next_nodes.get(transition_key)
                    if transition_key else None
                )

                if current_node_id:
                    self._logger.info(
                        f"Transitioning {node.id} --[{transition_key}]--> {current_node_id}"
                    )
                else:
                    self._logger.info(f"Reached terminal node: {node.id}")

            except Exception as e:
                state.mark_status(current_node_id, NodeStatus.ERROR)
                state.add_error(current_node_id, str(e))
                self._logger.error(f"Error in node {node.id}: {e}")
                raise

        return state

    async def _process_node(self, node: Node, state: NodeState) -> Optional[str]:
        """Process a single node.

        Args:
            node: Node to process
            state: Current state

        Returns:
            Optional transition key for next node
        """
        return await node.process(state)

    def validate(self) -> List[str]:
        """Validate the graph configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # Check for empty graph
        if not self.nodes:
            errors.append("Graph has no nodes")
            return errors

        # Validate nodes and edges
        for node_id, node in self.nodes.items():
            if not node.validate():
                errors.append(f"Node {node_id} failed validation")

            for key, next_id in node.next_nodes.items():
                if next_id and next_id not in self.nodes:
                    errors.append(f"Node {node_id} references unknown node: {next_id}")

        # Validate entry points
        for name, node_id in self.entry_points.items():
            if node_id not in self.nodes:
                errors.append(f"Entry point '{name}' references unknown node: {node_id}")

        return errors

async def test_graph() -> None:
    """
    Test the graph framework functionality with sample local nodes.

    This function:
    1. Defines simple nodes that perform arithmetic.
    2. Demonstrates serial and parallel node execution.
    3. Shows how to attach context suppliers to NodeState.
    """
    import asyncio
    from babelgraph.core.logging import configure_logging, LogLevel, LogComponent

    # Configure logging for tests
    configure_logging(
        default_level=LogLevel.INFO,
        component_levels={
            LogComponent.GRAPH: LogLevel.INFO,
            LogComponent.NODES: LogLevel.INFO
        }
    )

    # Local test Node definitions (example only)
    class AddNode(Node):
        async def process(self, state: NodeState) -> Optional[str]:
            a = state.get_data("a") or 0
            b = state.get_data("b") or 0
            state.set_result(self.id, "sum", a + b)
            return self.get_next_node()

    class MultiplyNode(Node):
        async def process(self, state: NodeState) -> Optional[str]:
            a = state.get_data("a") or 1
            b = state.get_data("b") or 1
            state.set_result(self.id, "product", a * b)
            return self.get_next_node()

    class SlowNode(Node):
        parallel = True

        async def process(self, state: NodeState) -> Optional[str]:
            await asyncio.sleep(1)
            state.set_result(self.id, "done", True)
            return self.get_next_node()

    # Build a sample graph
    graph = Graph()

    add_node = AddNode(id="add")
    mult_node = MultiplyNode(id="multiply")
    slow_node1 = SlowNode(id="slow1")
    slow_node2 = SlowNode(id="slow2")

    graph.add_node(add_node)
    graph.add_node(mult_node)
    graph.add_node(slow_node1)
    graph.add_node(slow_node2)

    graph.add_edge("add", "default", "multiply")
    graph.add_edge("multiply", "default", "slow1")
    graph.add_edge("slow1", "default", "slow2")

    graph.set_entry_point("add", "main")

    # Validate and run
    errors = graph.validate()
    assert not errors, f"Graph validation failed: {errors}"

    from datetime import datetime

    state = NodeState()
    state.set_data("a", 5)
    state.set_data("b", 3)

    # Example of adding a context supplier
    async def time_supplier(**kwargs) -> str:
        return datetime.now().isoformat()

    # This code references 'context' usage; adapt if you
    # have a separate context mechanism:
    # state.context.add_supplier("time", time_supplier)

    start_time = datetime.now()
    final_state = await graph.run("main", state)
    end_time = datetime.now()

    # Verify results
    assert final_state.results["add"]["sum"] == 8
    assert final_state.results["multiply"]["product"] == 15
    assert final_state.results["slow1"]["done"] is True
    assert final_state.results["slow2"]["done"] is True

    # Parallel execution check
    duration = (end_time - start_time).total_seconds()
    assert duration < 3.0, "Parallel execution took too long."

    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_graph()) 