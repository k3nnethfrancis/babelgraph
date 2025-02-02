"""Tests for the base graph functionality."""
from typing import Dict, Any, Optional
import pytest
import asyncio

from alchemist.ai.graph.base import Graph, Node
from alchemist.ai.graph.config import GraphConfig
from alchemist.ai.graph.state import NodeState
from pydantic import Field


class SimpleTestNode(Node):
    """A simple test node that just returns a value."""
    id: str = Field(description="Node ID")
    delay: float = Field(default=0.0, description="Delay in seconds before processing")
    processed: bool = Field(default=False, description="Whether the node has been processed")
    next_nodes: Dict[str, Optional[str]] = Field(default_factory=lambda: {"default": None, "error": None}, description="Next node mapping")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")
    parallel: bool = Field(default=False, description="Whether the node can be processed in parallel")

    async def process(self, state: NodeState) -> Optional[str]:
        """Process the node and return next node ID."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.processed = True
        state.results[self.id] = {"processed": True}
        return self.get_next_node()


class ErrorNode(Node):
    """A node that raises an error during processing."""
    id: str = Field(description="Node ID")
    next_nodes: Dict[str, Optional[str]] = Field(default_factory=lambda: {"default": None, "error": None}, description="Next node mapping")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")
    parallel: bool = Field(default=False, description="Whether the node can be processed in parallel")

    async def process(self, state: NodeState) -> Optional[str]:
        """Process the node and raise an error."""
        raise ValueError("Test error")


@pytest.fixture
def simple_graph() -> Graph:
    """Create a simple graph for testing."""
    return Graph()


class TestGraphInitialization:
    """Test graph initialization."""

    def test_graph_init(self, simple_graph: Graph):
        """Test basic graph initialization."""
        assert isinstance(simple_graph, Graph)
        assert not simple_graph.nodes
        assert not simple_graph.entry_points

    def test_graph_with_config(self):
        """Test graph initialization with config."""
        config = GraphConfig(max_parallel=4, timeout=60, retry_count=3)
        graph = Graph(config=config)
        assert graph.config.max_parallel == 4
        assert graph.config.timeout == 60
        assert graph.config.retry_count == 3


class TestNodeManagement:
    """Test node management functionality."""

    def test_add_node(self, simple_graph: Graph):
        """Test adding a node to the graph."""
        node = SimpleTestNode(id="test")
        simple_graph.add_node(node)
        assert "test" in simple_graph.nodes
        assert simple_graph.nodes["test"] == node

    def test_add_duplicate_node(self, simple_graph: Graph):
        """Test adding a duplicate node."""
        node1 = SimpleTestNode(id="test")
        node2 = SimpleTestNode(id="test")
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)  # Should overwrite node1
        assert simple_graph.nodes["test"] == node2

    def test_node_access(self, simple_graph: Graph):
        """Test accessing nodes from the graph."""
        node = SimpleTestNode(id="test")
        simple_graph.add_node(node)
        assert simple_graph.nodes["test"] == node

    def test_nonexistent_node_access(self, simple_graph: Graph):
        """Test accessing a nonexistent node."""
        with pytest.raises(KeyError):
            _ = simple_graph.nodes["nonexistent"]


class TestEdgeManagement:
    """Test edge management functionality."""

    def test_add_edge(self, simple_graph: Graph):
        """Test adding an edge between nodes."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, "default", node2.id)
        assert node1.next_nodes["default"] == node2.id

    def test_add_edge_with_condition(self, simple_graph: Graph):
        """Test adding an edge with a condition."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, "success", node2.id)
        assert node1.next_nodes["success"] == node2.id

    def test_add_invalid_edge(self, simple_graph: Graph):
        """Test adding an edge with invalid nodes."""
        node = SimpleTestNode(id="node1")
        simple_graph.add_node(node)
        with pytest.raises(ValueError):
            simple_graph.add_edge(node.id, "default", "nonexistent")


class TestEntryPoints:
    """Test entry point management."""

    def test_add_entry_point(self, simple_graph: Graph):
        """Test adding an entry point."""
        node = SimpleTestNode(id="start")
        simple_graph.add_node(node)
        simple_graph.add_entry_point("main", node.id)
        assert simple_graph.entry_points["main"] == node.id

    def test_add_invalid_entry_point(self, simple_graph: Graph):
        """Test adding an invalid entry point."""
        with pytest.raises(ValueError):
            simple_graph.add_entry_point("main", "nonexistent")

    def test_entry_point_access(self, simple_graph: Graph):
        """Test accessing entry points."""
        node = SimpleTestNode(id="start")
        simple_graph.add_node(node)
        simple_graph.add_entry_point("main", node.id)
        assert simple_graph.entry_points["main"] == node.id


class TestGraphExecution:
    """Test graph execution functionality."""

    async def test_basic_execution(self, simple_graph: Graph):
        """Test basic graph execution."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, "default", node2.id)
        simple_graph.add_entry_point("start", node1.id)

        state = await simple_graph.run("start")
        assert node1.processed and node2.processed
        assert state.results["node1"]["processed"]
        assert state.results["node2"]["processed"]

    async def test_parallel_execution(self, simple_graph: Graph):
        """Test parallel node execution."""
        nodes = [SimpleTestNode(id=f"node{i}", parallel=True) for i in range(3)]
        for node in nodes:
            simple_graph.add_node(node)

        # Connect nodes in parallel
        simple_graph.add_entry_point("start", nodes[0].id)
        simple_graph.add_edge(nodes[0].id, "default", nodes[1].id)
        simple_graph.add_edge(nodes[1].id, "default", nodes[2].id)

        state = await simple_graph.run("start")
        assert all(state.results[node.id]["processed"] for node in nodes)

    async def test_error_handling(self, simple_graph: Graph):
        """Test error handling during execution."""
        node1 = SimpleTestNode(id="node1")
        error_node = ErrorNode(id="error_node")
        node3 = SimpleTestNode(id="node3")

        simple_graph.add_node(node1)
        simple_graph.add_node(error_node)
        simple_graph.add_node(node3)

        simple_graph.add_edge(node1.id, "default", error_node.id)
        simple_graph.add_edge(error_node.id, "default", node3.id)
        simple_graph.add_entry_point("start", node1.id)

        state = await simple_graph.run("start")
        assert node1.processed
        assert error_node.id in state.errors
        assert state.errors[error_node.id] == "Test error"


class TestGraphValidation:
    """Test graph validation functionality."""

    def test_validate_graph(self, simple_graph: Graph):
        """Test validation of a valid graph."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")
        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_edge(node1.id, "default", node2.id)
        simple_graph.add_entry_point("start", node1.id)
        errors = simple_graph.validate()
        assert not errors

    def test_validate_empty_graph(self, simple_graph: Graph):
        """Test validation of empty graph."""
        errors = simple_graph.validate()
        assert errors  # Just check that there are errors, don't check specific messages

    def test_validate_disconnected_graph(self, simple_graph: Graph):
        """Test validation of disconnected graph."""
        node1 = SimpleTestNode(id="node1")
        node2 = SimpleTestNode(id="node2")  # Disconnected node

        simple_graph.add_node(node1)
        simple_graph.add_node(node2)
        simple_graph.add_entry_point("start", node1.id)

        errors = simple_graph.validate()
        assert errors  # Just check that there are errors, don't check specific messages 