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

from typing import Dict, Any, Optional, Union, List, Set, Tuple
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

    def set_entry_point(self, node_id: str) -> None:
        """Set the entry point for the graph.
        
        Args:
            node_id: The ID of the node to use as entry point
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in graph")
        self.entry_points["default"] = node_id
        logger.info(f"Set entry point to node: {node_id}")

    def chain(self, nodes: List[Node], transition_key: str = "success") -> None:
        """Chain nodes together in sequence.
        
        Example:
            graph.chain([analyzer, decision, end])
            
        This will automatically:
        1. Add all nodes to the graph
        2. Connect them in sequence using the transition_key
        3. Set the first node as entry point if none exists
        """
        for node in nodes:
            self.add_node(node)
            
        for i in range(len(nodes) - 1):
            self.add_edge(nodes[i].id, transition_key, nodes[i + 1].id)

        # Set first node as default entry if no entry points exist
        if not self.entry_points:
            self.set_entry_point(nodes[0].id)
            
    def branch(self, node: Node, conditions: Dict[str, Node]) -> None:
        """Create conditional branches from a node.
        
        Example:
            graph.branch(analyzer, {
                "positive": positive_handler,
                "negative": negative_handler,
                "neutral": neutral_handler
            })
        """
        self.add_node(node)
        for condition, target_node in conditions.items():
            self.add_node(target_node)
            self.add_edge(node.id, condition, target_node.id)
            
    def merge(self, nodes: List[Node], target: Node, condition: str = "success") -> None:
        """Merge multiple nodes into a single target node.
        
        Example:
            graph.merge([pos_handler, neg_handler], final_node)
        """
        self.add_node(target)
        for node in nodes:
            self.add_node(node)
            self.add_edge(node.id, condition, target.id)
            
    def create_workflow(self, nodes: Dict[str, Node], flows: List[Tuple[str, str, str]]) -> None:
        """Create a workflow from a dictionary of nodes and flow definitions.
        
        Example:
            graph.create_workflow(
                nodes={
                    "analyze": analyzer,
                    "decide": decision,
                    "end": end_node
                },
                flows=[
                    ("analyze", "success", "decide"),
                    ("analyze", "error", "end"),
                    ("decide", "success", "end")
                ]
            )
        """
        # Add all nodes with their dictionary keys as IDs
        for node_id, node in nodes.items():
            if not node.id:  # Only set ID if not already set
                node.id = node_id
            self.add_node(node)
            
        # Create flows
        for source, condition, target in flows:
            self.add_edge(source, condition, target)
            
        # Set first node as entry point if none exists
        if not self.entry_points and flows:
            self.set_entry_point(flows[0][0])  # First source node

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

    async def run(self, entry_point: Optional[str] = None, state: Optional[NodeState] = None) -> NodeState:
        """Run the graph from the entry point.
        
        Args:
            entry_point: Optional override for the graph's entry point
            state: Optional initial state, creates new state if None
            
        Returns:
            The final state after graph execution
        """
        # Use provided entry point or graph's default
        current_node_id = entry_point or self.entry_points["default"]
        if not current_node_id:
            raise ValueError("No entry point specified")
        if current_node_id not in self.nodes:
            raise ValueError(f"Entry point not found: {current_node_id}")
            
        # Initialize or use provided state
        state = state or NodeState()
        state.start_time = datetime.now()
        
        try:
            while current_node_id:
                current_node = self.nodes[current_node_id]
                logger.debug(f"Processing node: {current_node_id}")
                
                # Process node
                next_condition = await current_node.process(state)
                logger.debug(f"Node {current_node_id} returned condition: {next_condition}")
                
                # Get next node based on condition
                current_node_id = current_node.get_next_node(next_condition)
                logger.debug(f"Next node: {current_node_id}")
                
            return state
            
        except Exception as e:
            logger.error(f"Error in graph execution: {e}")
            state.errors["graph"] = str(e)
            raise

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

    graph.set_entry_point("add")

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
    final_state = await graph.run("add", state)
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