"""Tests for ChatRuntime functionality.

This module contains tests for the ChatRuntime class, which handles:
- Message processing
- Runtime initialization and shutdown
- Agent integration
- Error handling
"""

import pytest
from typing import Dict, Any, Optional
from pydantic import BaseModel

from alchemist.ai.base.runtime import BaseRuntime
from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.prompts.base import PersonaConfig
from alchemist.ai.base.logging import AlchemistLoggingConfig, VerbosityLevel


class TestRuntime(BaseRuntime):
    """Concrete implementation of BaseRuntime for testing."""
    
    VALID_MODES = {"cli", "api", "web"}
    
    def __init__(self, agent: BaseAgent, config: Dict[str, Any]):
        if "mode" in config and config["mode"] not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {config['mode']}. Must be one of {self.VALID_MODES}")
        self.agent = agent
        self.config = config
        self.is_running = False

    def _create_agent(self):
        """Create agent instance."""
        return self.agent

    async def start(self) -> None:
        """Start the runtime."""
        self.is_running = True

    async def stop(self) -> None:
        """Stop the runtime."""
        self.is_running = False

    async def process_message(self, message: str) -> str:
        """Process a single message."""
        if message is None:
            return "Something went wrong processing your message."
        if not message:
            return "Unable to generate a response for an empty message."
        response = await self.agent._step(message)
        return response


@pytest.fixture
def chat_runtime(base_agent: BaseAgent) -> BaseRuntime:
    """Fixture providing a configured chat runtime."""
    return TestRuntime(
        agent=base_agent,
        config={
            "mode": "cli",
            "max_history": 100,
            "timeout": 30
        }
    )


class TestRuntimeInitialization:
    """Test suite for runtime initialization."""

    def test_runtime_init(self, chat_runtime: BaseRuntime):
        """Test basic runtime initialization."""
        assert chat_runtime.agent is not None
        assert chat_runtime.config == {"mode": "cli", "max_history": 100, "timeout": 30}

    def test_runtime_with_custom_config(self, base_agent: BaseAgent):
        """Test runtime initialization with custom config."""
        config = {
            "mode": "cli",
            "max_history": 100,
            "timeout": 30
        }
        runtime = TestRuntime(agent=base_agent, config=config)
        assert runtime.config == config


class TestMessageProcessing:
    """Test suite for message processing."""

    async def test_basic_message(self, chat_runtime: BaseRuntime):
        """Test basic message processing."""
        response = await chat_runtime.process_message("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    async def test_empty_message(self, chat_runtime: BaseRuntime):
        """Test handling of empty messages."""
        response = await chat_runtime.process_message("")
        assert isinstance(response, str)
        assert "unable to generate" in response.lower()

    async def test_error_handling(self, chat_runtime: BaseRuntime):
        """Test error handling during message processing."""
        # Simulate an error by passing None
        response = await chat_runtime.process_message(None)
        assert "something went wrong" in response.lower()


class TestRuntimeLifecycle:
    """Test suite for runtime lifecycle."""

    async def test_start_runtime(self, chat_runtime: BaseRuntime):
        """Test runtime startup."""
        await chat_runtime.start()
        assert isinstance(chat_runtime, TestRuntime)
        assert chat_runtime.is_running

    async def test_stop_runtime(self, chat_runtime: BaseRuntime):
        """Test runtime shutdown."""
        await chat_runtime.start()
        await chat_runtime.stop()
        assert isinstance(chat_runtime, TestRuntime)
        assert not chat_runtime.is_running


class TestRuntimeConfiguration:
    """Test suite for runtime configuration."""

    def test_default_config(self, chat_runtime: BaseRuntime):
        """Test default runtime configuration."""
        assert "mode" in chat_runtime.config
        assert chat_runtime.config["mode"] == "cli"

    def test_config_validation(self, base_agent: BaseAgent):
        """Test runtime configuration validation."""
        # Test with invalid config
        with pytest.raises(ValueError):
            TestRuntime(
                agent=base_agent,
                config={"mode": "invalid_mode"}
            )


class TestRuntimeIntegration:
    """Test suite for runtime integration with agent."""

    async def test_agent_interaction(self, chat_runtime: BaseRuntime):
        """Test runtime interaction with agent."""
        response = await chat_runtime.process_message("Hello")
        assert isinstance(response, str)
        assert len(chat_runtime.agent.history) > 0

    async def test_persona_integration(self, chat_runtime: BaseRuntime):
        """Test runtime integration with agent persona."""
        assert chat_runtime.agent.persona.name == "Test Assistant"
        response = await chat_runtime.process_message("Who are you?")
        assert isinstance(response, str)
        # The response should reflect the persona
        # But since it's an LLM call, we can't assert exact content 