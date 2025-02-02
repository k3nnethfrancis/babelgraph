"""Tests for AgentNode functionality.

This module tests the AgentNode class which handles:
- LLM integration via Mirascope
- Prompt templating
- System prompt configuration
- Conversation state management
"""

import pytest
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from mirascope.core import BaseMessageParam, Messages
from mirascope.core.base import TextPart
from mirascope.core import prompt_template
from alchemist.ai.graph.nodes import AgentNode
from alchemist.ai.graph.state import NodeState
from alchemist.ai.prompts.base import PersonaConfig
from mirascope.core.base.messages import Messages
from mirascope.core.base.prompt import prompt_template
from mirascope.core.base.dynamic_config import BaseDynamicConfig


@prompt_template()
def mock_prompt(text: str) -> list[BaseMessageParam]:
    """Mock prompt template for testing."""
    return [Messages.User(content=[
        TextPart(type="text", text=f"Summarize this text: {text}")
    ])]


class TestAgentNodeInitialization:
    """Test cases for agent node initialization."""

    def test_agent_node_init(self):
        """Test agent node initialization."""
        agent_node = AgentNode(
            id="test_agent",
            next_nodes={"default": "next_node", "error": "error_node"},
            prompt_template=mock_prompt,
            runtime_config={"model": "gpt-4o-mini"}
        )
        assert agent_node.id == "test_agent"
        assert agent_node.prompt_template == mock_prompt
        assert agent_node.runtime_config == {"model": "gpt-4o-mini"}

    def test_agent_node_without_prompt(self):
        """Test agent node initialization without prompt."""
        with pytest.raises(ValueError) as exc_info:
            AgentNode(
                id="test_agent",
                next_nodes={"default": "next_node", "error": "error_node"},
                input_map={"text": "data.text"},
                runtime_config={"model": "gpt-4o-mini"}
            )
        assert "AgentNode test_agent requires either prompt or prompt_template" in str(exc_info.value)

    def test_agent_node_with_raw_prompt(self):
        """Test agent node initialization with raw prompt."""
        agent_node = AgentNode(
            id="test_agent",
            next_nodes={"default": "next_node", "error": "error_node"},
            input_map={"text": "data.text"},
            prompt="Summarize this text: {text}",
            runtime_config={"model": "gpt-4o-mini"}
        )
        assert agent_node.id == "test_agent"
        assert agent_node.prompt == "Summarize this text: {text}"
        assert agent_node.runtime_config == {"model": "gpt-4o-mini"}


class TestAgentNodeProcessing:
    """Test cases for AgentNode processing."""

    @pytest.fixture
    def agent_node(self) -> AgentNode:
        """Create an agent node for testing."""
        node = AgentNode(
            id="test_agent",
            next_nodes={"default": "next_node", "error": "error_node"},
            prompt_template=mock_prompt,
            input_map={"text": "data.text"},
        )
        
        class MockAgent:
            async def _step(self, state) -> str:
                return "Test response"
        
        node.agent = MockAgent()
        return node

    @pytest.mark.asyncio
    async def test_basic_processing(self, agent_node: AgentNode):
        """Test basic agent node processing."""
        state = NodeState()
        state.data["text"] = "Test input"
        next_id = await agent_node.process(state)
        assert next_id == "next_node"
        assert state.results[agent_node.id]["response"] == "Test response"

    @pytest.mark.asyncio
    async def test_missing_input(self, agent_node: AgentNode):
        """Test agent node processing with missing input."""
        state = NodeState()
        next_id = await agent_node.process(state)
        assert next_id == "error_node"
        assert "Error formatting prompt: Key 'text' not found while traversing 'text'" in state.errors[agent_node.id]


class TestPromptFormatting:
    """Test cases for prompt formatting."""

    def test_template_formatting(self):
        """Test prompt template formatting."""
        agent_node = AgentNode(
            id="test_agent",
            next_nodes={"default": "next_node", "error": "error_node"},
            input_map={"text": "data.text"},
            prompt_template=mock_prompt,
            runtime_config={"model": "gpt-4o-mini"}
        )
        state = NodeState()
        state.data["text"] = "Test input"
        formatted = agent_node._format_prompt(state)
        assert formatted == [Messages.User(content=[
            TextPart(type="text", text="Summarize this text: Test input")
        ])]

    def test_raw_prompt_formatting(self):
        """Test raw prompt formatting."""
        agent_node = AgentNode(
            id="test_agent",
            next_nodes={"default": "next_node", "error": "error_node"},
            input_map={"text": "data.text"},
            prompt="Summarize this text: {text}",
            runtime_config={"model": "gpt-4o-mini"}
        )
        state = NodeState()
        state.data["text"] = "Test input"
        formatted = agent_node._format_prompt(state)
        assert formatted == "Summarize this text: Test input" 