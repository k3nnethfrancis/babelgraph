"""Tests for graph state management.

This module tests the state management functionality including:
- NodeState initialization and validation
- Data access and storage
- Result management
- Error handling
- State persistence
"""

import pytest
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import Field

from alchemist.ai.graph.state import NodeState, StateManager, NodeStatus


@pytest.fixture
def state() -> NodeState:
    """Fixture providing a basic node state."""
    return NodeState()


@pytest.fixture
def state_manager() -> StateManager:
    """Fixture providing a state manager."""
    return StateManager()


class TestNodeState:
    """Test suite for NodeState functionality."""

    def test_state_init(self, state: NodeState):
        """Test basic state initialization."""
        assert isinstance(state.data, dict)
        assert isinstance(state.results, dict)
        assert isinstance(state.errors, dict)
        assert isinstance(state.status, dict)

    def test_data_access(self, state: NodeState):
        """Test data access methods."""
        state.data["test.key"] = "test_value"
        assert state.data.get("test.key") == "test_value"
        assert state.data.get("nonexistent") is None

    def test_nested_data_access(self, state: NodeState):
        """Test nested data access."""
        state.data["parent.child.key"] = "nested_value"
        assert state.data.get("parent.child.key") == "nested_value"
        
        # Create nested structure
        state.data["parent"] = {"child": {"key2": "value2"}}
        assert state.data["parent"]["child"]["key2"] == "value2"

    def test_result_management(self, state: NodeState):
        """Test result management."""
        state.results["node1"] = {"key": "value"}
        assert state.results["node1"]["key"] == "value"

    def test_error_handling(self, state: NodeState):
        """Test error handling."""
        error = ValueError("Test error")
        state.errors["node1"] = error
        assert isinstance(state.errors["node1"], ValueError)
        assert str(state.errors["node1"]) == "Test error"

    def test_status_tracking(self, state: NodeState):
        """Test node status tracking."""
        state.status["node1"] = NodeStatus.RUNNING
        assert state.status["node1"] == NodeStatus.RUNNING

    def test_data_validation(self, state: NodeState):
        """Test data validation."""
        # Test that nested paths are properly handled
        state.data["valid.key"] = "valid_value"
        assert state.data["valid.key"] == "valid_value"
        
        # Test that nested data access works
        state.data["parent"] = {"child": {"key": "nested_value"}}
        assert state.get_nested_data("parent.child.key") == "nested_value"
        
        # Test that invalid nested paths raise ValueError
        with pytest.raises(ValueError):
            state.get_nested_data("invalid.path")
        
        # Test that non-dict values can't be traversed
        state.data["leaf"] = "value"
        with pytest.raises(ValueError):
            state.get_nested_data("leaf.invalid")

    def test_state_serialization(self, state: NodeState):
        """Test state serialization."""
        state.data["test.key"] = "test_value"
        state.results["node1"] = {"key": "value"}
        
        serialized = state.model_dump()
        assert isinstance(serialized, dict)
        assert "data" in serialized
        assert "results" in serialized


class TestStateManager:
    """Test suite for StateManager functionality."""

    def test_manager_init(self, state_manager: StateManager):
        """Test state manager initialization."""
        assert isinstance(state_manager.states, dict)

    def test_create_state(self, state_manager: StateManager):
        """Test state creation."""
        state = state_manager.create_state()
        assert isinstance(state, NodeState)

    def test_persist_state(self, state_manager: StateManager):
        """Test state persistence."""
        state = NodeState()
        state.data["test.key"] = "test_value"
        
        key = "test_state"
        state_manager.states[key] = state
        assert key in state_manager.states

    def test_retrieve_state(self, state_manager: StateManager):
        """Test state retrieval."""
        original_state = NodeState()
        original_state.data["test.key"] = "test_value"
        
        key = "test_state"
        state_manager.states[key] = original_state
        
        retrieved_state = state_manager.states.get(key)
        assert retrieved_state is not None
        assert retrieved_state.data["test.key"] == "test_value"

    def test_retrieve_nonexistent_state(self, state_manager: StateManager):
        """Test retrieval of nonexistent state."""
        assert state_manager.states.get("nonexistent") is None

    def test_clear_states(self, state_manager: StateManager):
        """Test clearing all states."""
        state = NodeState()
        state_manager.states["test_state"] = state
        
        state_manager.states.clear()
        assert len(state_manager.states) == 0


class TestStateIntegration:
    """Test suite for state integration scenarios."""

    def test_complex_state_management(self, state_manager: StateManager):
        """Test complex state management scenario."""
        # Create and configure state
        state = state_manager.create_state()
        state.data["user.name"] = "Test User"
        state.data["user.preferences.theme"] = "dark"
        state.results["node1"] = {"processed": True}
        state.status["node1"] = NodeStatus.COMPLETED
        
        # Persist and retrieve
        state_manager.states["test_session"] = state
        retrieved = state_manager.states["test_session"]
        
        # Verify all data is preserved
        assert retrieved.data["user.name"] == "Test User"
        assert retrieved.data["user.preferences.theme"] == "dark"
        assert retrieved.results["node1"]["processed"] is True
        assert retrieved.status["node1"] == NodeStatus.COMPLETED

    def test_state_isolation(self, state_manager: StateManager):
        """Test state isolation between different sessions."""
        state1 = state_manager.create_state()
        state2 = state_manager.create_state()
        
        state1.data["key"] = "value1"
        state2.data["key"] = "value2"
        
        assert state1.data["key"] == "value1"
        assert state2.data["key"] == "value2"
