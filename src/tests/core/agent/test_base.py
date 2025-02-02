"""Tests for BaseAgent functionality.

This module contains tests for the BaseAgent class, which handles:
- LLM interactions via Mirascope
- Tool execution
- Conversation history management
- Persona configuration
"""

import pytest
from typing import Dict, Any, List
from pydantic import BaseModel

from mirascope.core import BaseMessageParam, BaseTool, BaseDynamicConfig
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.base.logging import AlchemistLoggingConfig, VerbosityLevel


class MockTool(BaseTool):
    """Mock tool for testing agent functionality."""
    
    def __init__(self):
        self.called = False
        self.args = {}
    
    @classmethod
    def _name(cls) -> str:
        return "mock_tool"
        
    async def call(self) -> Dict[str, Any]:
        """Simulate tool execution."""
        self.called = True
        return {"result": "mock_result"}


@pytest.fixture
def base_agent() -> BaseAgent:
    """Fixture providing a basic agent with mock tool."""
    return BaseAgent(
        tools=[MockTool],
        persona=PersonaConfig(
            id="test-assistant-v1",
            name="Test Assistant",
            nickname="Testy",
            bio="A test assistant for unit testing",
            personality={
                "traits": {
                    "neuroticism": 0.2,
                    "extraversion": 0.5,
                    "openness": 0.7,
                    "agreeableness": 0.8,
                    "conscientiousness": 0.9
                },
                "stats": {
                    "intelligence": 0.8,
                    "wisdom": 0.7,
                    "charisma": 0.6
                }
            },
            lore=["Created for testing purposes", "Helps validate agent functionality"],
            style={
                "all": ["Uses clear, professional language", "Stays focused on testing"],
                "chat": ["Responds concisely", "Maintains test context"]
            }
        ),
        logging_config=AlchemistLoggingConfig(
            level=VerbosityLevel.DEBUG
        )
    )


class TestAgentInitialization:
    """Test suite for agent initialization."""

    def test_agent_init(self, base_agent: BaseAgent):
        """Test basic agent initialization."""
        assert base_agent.tools == [MockTool]
        assert base_agent.history == []
        assert base_agent.persona.name == "Test Assistant"
        assert base_agent.logging_config.level == VerbosityLevel.DEBUG

    def test_agent_with_history(self, test_persona: PersonaConfig):
        """Test agent initialization with existing history."""
        history = [
            BaseMessageParam(role="user", content="Hello"),
            BaseMessageParam(role="assistant", content="Hi there")
        ]
        agent = BaseAgent(
            persona=test_persona,
            history=history
        )
        assert len(agent.history) == 2
        assert agent.history[0].content == "Hello"


class TestMessageHandling:
    """Test suite for message handling."""

    async def test_basic_message(self, base_agent: BaseAgent):
        """Test basic message without tool use."""
        response = await base_agent._step("Hello")
        assert isinstance(response, str)
        assert len(base_agent.history) == 2  # User message + assistant response

    async def test_history_management(self, base_agent: BaseAgent):
        """Test conversation history management."""
        await base_agent._step("First message")
        await base_agent._step("Second message")
        
        assert len(base_agent.history) == 4  # 2 user messages + 2 assistant responses
        assert base_agent.history[0].role == "user"
        assert base_agent.history[0].content == "First message"


class TestToolExecution:
    """Test suite for tool execution."""

    async def test_tool_call(self, base_agent: BaseAgent):
        """Test agent executing a tool."""
        # This test requires mocking the LLM to return a tool call
        # For now, we can test that the tool is properly registered
        assert MockTool in base_agent.tools

    async def test_tool_result_in_history(self, base_agent: BaseAgent):
        """Test that tool results are added to history."""
        # This would require mocking the LLM to return a tool call
        # Then verify the tool result is in history
        pass


class TestLoggingBehavior:
    """Test suite for agent logging."""

    async def test_verbose_logging(self, test_persona: PersonaConfig):
        """Test verbose logging configuration."""
        agent = BaseAgent(
            persona=test_persona,
            logging_config=AlchemistLoggingConfig(
                level=VerbosityLevel.DEBUG,
                show_llm_messages=True,
                show_tool_calls=True
            )
        )
        assert agent.logging_config.show_llm_messages
        assert agent.logging_config.show_tool_calls

    async def test_minimal_logging(self, test_persona: PersonaConfig):
        """Test minimal logging configuration."""
        agent = BaseAgent(
            persona=test_persona,
            logging_config=AlchemistLoggingConfig(
                level=VerbosityLevel.INFO,
                show_llm_messages=False,
                show_tool_calls=False
            )
        )
        assert not agent.logging_config.show_llm_messages
        assert not agent.logging_config.show_tool_calls 