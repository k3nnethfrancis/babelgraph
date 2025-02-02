"""Shared test fixtures for base component tests.

This module provides fixtures that can be shared across test files for the base
components (agent and runtime).
"""

import pytest
from typing import Dict, Any, List
from pydantic import BaseModel

from mirascope.core import BaseMessageParam, BaseTool
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.runtime import BaseRuntime
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.base.logging import AlchemistLoggingConfig, VerbosityLevel


class MockTool(BaseTool):
    """Mock tool for testing."""
    
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
def mock_tool() -> MockTool:
    """Fixture providing a mock tool."""
    return MockTool()


@pytest.fixture
def test_persona() -> PersonaConfig:
    """Fixture providing a test persona configuration."""
    return PersonaConfig(
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
                "charisma": 0.6,
                "authenticity": 1.0,
                "adaptability": 0.8,
                "reliability": 0.9
            }
        },
        lore=[
            "Created for testing purposes",
            "Focuses on providing accurate test results",
            "Maintains clear testing boundaries"
        ],
        style={
            "all": [
                "Uses clear, professional language",
                "Maintains appropriate boundaries",
                "Stays focused on testing"
            ],
            "chat": [
                "Responds concisely and clearly",
                "Uses appropriate formatting",
                "Maintains professional tone"
            ]
        }
    )


@pytest.fixture
def test_logging_config() -> AlchemistLoggingConfig:
    """Fixture providing a test logging configuration."""
    return AlchemistLoggingConfig(
        level=VerbosityLevel.DEBUG,
        show_llm_messages=True,
        show_tool_calls=True
    )


@pytest.fixture
def base_agent(test_persona: PersonaConfig, test_logging_config: AlchemistLoggingConfig) -> BaseAgent:
    """Fixture providing a configured base agent."""
    return BaseAgent(
        tools=[MockTool],
        persona=test_persona,
        logging_config=test_logging_config
    )


@pytest.fixture
def chat_runtime(base_agent: BaseAgent) -> BaseRuntime:
    """Fixture providing a configured chat runtime."""
    return BaseRuntime(
        agent=base_agent,
        config={
            "mode": "cli",
            "max_history": 100,
            "timeout": 30
        }
    )


@pytest.fixture
def sample_conversation() -> List[BaseMessageParam]:
    """Fixture providing a sample conversation history."""
    return [
        BaseMessageParam(role="user", content="Hello"),
        BaseMessageParam(role="assistant", content="Hi there! How can I help you?"),
        BaseMessageParam(role="user", content="What's the weather?"),
        BaseMessageParam(role="assistant", content="I don't have access to real-time weather data.")
    ] 