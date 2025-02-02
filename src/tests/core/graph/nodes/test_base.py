"""Tests for base Node functionality.

This module tests the core Node class functionality including:
- Node initialization and validation
- Input mapping and data access
- State management
- Parallel execution support
"""

import pytest
from typing import Dict, Any, Optional
import asyncio
from pydantic import Field

from alchemist.ai.graph.nodes.base import Node
from alchemist.ai.graph.state import NodeState


class SimpleNode(Node):
    """Simple test node that stores input values."""
    
    id: str = Field(description="Node identifier")
    input_map: Dict[str, str] = Field(default_factory=dict)
    next_nodes: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {"default": None, "error": None}
    )
    
    async def process(self, state: NodeState) -> Optional[str]:
        value = state.get_data(self.input_map.get("value", "default_value"))
        state.results[self.id] = {"processed_value": value}
        return self.next_nodes.get("default")


class ParallelTestNode(Node):
    """Test node that supports parallel execution."""
    
    id: str = Field(description="Node identifier")
    parallel: bool = Field(default=True, description="Whether node supports parallel execution")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    next_nodes: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {"default": None, "error": None}
    )
    
    async def process(self, state: NodeState) -> Optional[str]:
        delay = self.metadata.get("delay", 0.1)
        await asyncio.sleep(delay)
        state.results[self.id] = {"completed": True}
        return self.next_nodes.get("default")


@pytest.fixture
def simple_node() -> SimpleNode:
    """Fixture providing a basic test node."""
    return SimpleNode(id="test")


@pytest.fixture
def parallel_node() -> ParallelTestNode:
    """Fixture providing a parallel test node."""
    return ParallelTestNode(id="parallel", metadata={"delay": 0.1})


class TestNodeInitialization:
    """Test suite for node initialization."""

    def test_node_init(self, simple_node: SimpleNode):
        """Test basic node initialization."""
        assert simple_node.id == "test"
        assert "default" in simple_node.next_nodes
        assert "error" in simple_node.next_nodes

    def test_node_without_id(self):
        """Test node initialization without ID."""
        with pytest.raises(ValueError):
            SimpleNode()

    def test_node_with_input_map(self):
        """Test node initialization with input mapping."""
        node = SimpleNode(
            id="test",
            input_map={"value": "source.data.value"}
        )
        assert node.input_map["value"] == "source.data.value"


class TestNodeValidation:
    """Test suite for node validation."""

    def test_validate_node(self, simple_node: SimpleNode):
        """Test node validation."""
        assert simple_node.model_validate(simple_node.model_dump()) == simple_node

    def test_validate_next_nodes(self, simple_node: SimpleNode):
        """Test next_nodes validation."""
        simple_node.next_nodes["default"] = "next_node"
        assert "next_node" == simple_node.next_nodes.get("default")
        assert None == simple_node.next_nodes.get("nonexistent")


class TestNodeProcessing:
    """Test suite for node processing."""

    async def test_basic_processing(self, simple_node: SimpleNode):
        """Test basic node processing."""
        state = NodeState()
        state.data["default_value"] = 42
        
        next_node = await simple_node.process(state)
        assert next_node is None  # Default next_node is None
        assert state.results[simple_node.id]["processed_value"] == 42

    async def test_input_mapping(self):
        """Test input mapping during processing."""
        node = SimpleNode(
            id="test",
            input_map={"value": "source.value"}
        )
        
        state = NodeState()
        state.data["source.value"] = 123
        
        await node.process(state)
        assert state.results[node.id]["processed_value"] == 123

    async def test_missing_input(self, simple_node: SimpleNode):
        """Test processing with missing input."""
        state = NodeState()
        await simple_node.process(state)
        assert state.results[simple_node.id]["processed_value"] is None


class TestParallelExecution:
    """Test suite for parallel execution support."""

    async def test_parallel_flag(self, parallel_node: ParallelTestNode):
        """Test parallel execution flag."""
        assert parallel_node.parallel is True

    async def test_parallel_processing(self, parallel_node: ParallelTestNode):
        """Test parallel node processing."""
        state = NodeState()
        await parallel_node.process(state)
        assert state.results[parallel_node.id]["completed"] is True

    async def test_multiple_parallel_nodes(self):
        """Test multiple parallel nodes execution."""
        nodes = [
            ParallelTestNode(id=f"p{i}", metadata={"delay": 0.1})
            for i in range(3)
        ]
        
        state = NodeState()
        tasks = [node.process(state) for node in nodes]
        await asyncio.gather(*tasks)
        
        assert all(
            state.results[node.id]["completed"] is True
            for node in nodes
        )


class TestMetadataHandling:
    """Test suite for node metadata handling."""

    def test_metadata_storage(self, simple_node: SimpleNode):
        """Test metadata storage."""
        simple_node.metadata["test_key"] = "test_value"
        assert simple_node.metadata["test_key"] == "test_value"

    def test_metadata_in_processing(self):
        """Test metadata usage during processing."""
        node = ParallelTestNode(
            id="test",
            metadata={"delay": 0.2}
        )
        assert node.metadata["delay"] == 0.2 