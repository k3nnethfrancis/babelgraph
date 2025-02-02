"""Tests for ActionNode functionality.

This module tests the ActionNode class which handles:
- Tool execution
- Input/output mapping
- Error handling
- State management
"""

import pytest
from typing import Dict, Any, Optional
from pydantic import BaseModel

from alchemist.ai.graph.nodes import ActionNode
from alchemist.ai.graph.state import NodeState


class MockTool:
    """Mock tool for testing action node functionality."""
    
    def __init__(self):
        """Initialize the mock tool."""
        self.called = False
    
    def __call__(self, value: str = "default") -> Dict[str, Any]:
        """Simulate tool execution."""
        self.called = True
        return {"result": value}


class FailingTool:
    """Tool that raises an error during execution."""
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Simulate tool failure."""
        raise ValueError("Tool execution failed")


@pytest.fixture
def mock_tool() -> MockTool:
    """Fixture providing a mock tool."""
    return MockTool()


@pytest.fixture
def failing_tool() -> FailingTool:
    """Fixture providing a failing tool."""
    return FailingTool()


@pytest.fixture
def action_node(mock_tool: MockTool) -> ActionNode:
    """Fixture providing a configured action node."""
    return ActionNode(
        id="test_action",
        tool=mock_tool,
        input_map={"value": "data.input.value"},
        next_nodes={"default": "next_node", "error": "error_node"}
    )


class TestActionNodeInitialization:
    """Test suite for action node initialization."""

    def test_action_node_init(self, action_node: ActionNode):
        """Test basic action node initialization."""
        assert action_node.id == "test_action"
        assert callable(action_node.tool)
        assert action_node.input_map["value"] == "data.input.value"

    def test_action_node_with_invalid_tool(self):
        """Test action node initialization with invalid tool."""
        class InvalidTool:
            def __init__(self):
                pass  # Not callable
        
        with pytest.raises(ValueError):
            ActionNode(id="test", tool=InvalidTool())


class TestActionNodeProcessing:
    """Test suite for action node processing."""

    @pytest.mark.asyncio
    async def test_basic_processing(self, action_node: ActionNode):
        """Test basic action node processing."""
        state = NodeState()
        state.data["input"] = {"value": "test_input"}
        
        next_node = await action_node.process(state)
        assert next_node == "next_node"
        assert state.results[action_node.id]["result"] == {"result": "test_input"}
        assert "timing" in state.results[action_node.id]

    @pytest.mark.asyncio
    async def test_tool_failure(self, failing_tool: FailingTool):
        """Test handling of tool execution failure."""
        node = ActionNode(
            id="failing_action",
            tool=failing_tool,
            input_map={"value": "data.input.value"},
            next_nodes={"default": "next_node", "error": "error_node"}
        )
        
        state = NodeState()
        state.data["input"] = {"value": "test_input"}
        
        next_node = await node.process(state)
        assert next_node == "error_node"
        assert "failing_action" in state.errors
        assert "Tool execution failed" in state.errors["failing_action"]


class TestStateManagement:
    """Test suite for state management."""

    @pytest.mark.asyncio
    async def test_required_state(self, mock_tool: MockTool):
        """Test required state validation."""
        node = ActionNode(
            id="test",
            tool=mock_tool,
            required_state=["required_key"],
            input_map={"value": "data.input.value"},
            next_nodes={"default": "next", "error": "error"}
        )
        
        state = NodeState()
        next_node = await node.process(state)
        assert next_node == "error"
        assert "test" in state.errors
        assert "Missing required state keys" in state.errors["test"]
        
        # Add required state and input data
        state.data["required_key"] = "value"
        state.data["input"] = {"value": "test_input"}
        next_node = await node.process(state)
        assert next_node == "next"
        assert "test" in state.results

    @pytest.mark.asyncio
    async def test_preserve_state(self, mock_tool: MockTool):
        """Test state preservation."""
        node = ActionNode(
            id="test",
            tool=mock_tool,
            preserve_state=["keep_this"],
            input_map={"value": "data.input.value"},
            next_nodes={"default": "next"}
        )
        
        state = NodeState()
        state.results["keep_this"] = "preserved"
        state.results["remove_this"] = "temporary"
        state.data["input"] = {"value": "test_input"}
        
        await node.process(state)
        assert "keep_this" in state.results
        assert "remove_this" not in state.results
        assert node.id in state.results 