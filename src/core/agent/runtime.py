"""Base Runtime Module for Agent Execution Environments"""

from typing import Dict, Any, Optional, Union, Callable, AsyncGenerator
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
import logging
import re
import asyncio

from babelgraph.core.logging import LogComponent
from babelgraph.core.agent import BaseAgent

# Get logger for runtime component
logger = logging.getLogger(LogComponent.RUNTIME.value)

class Session(BaseModel):
    """Tracks runtime session data."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: str
    start_time: datetime = Field(default_factory=datetime.now)
    agent_config: Dict[str, Any] = Field(default_factory=dict)

class RuntimeConfig(BaseModel):
    """Configuration for runtime environments.
    
    Attributes:
        provider: The LLM provider to use (default: "openai")
        model: The model to use (default: "gpt-4o-mini")
        system_prompt: System prompt configuration as string or Pydantic model
        tools: List of available tools
        platform_config: Additional platform-specific configuration
        response_model: Optional Pydantic model for structured output
        json_mode: Whether to enforce JSON output format
        stream: Whether to use streaming mode
        output_parser: Optional custom parser for processing responses
    """
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    system_prompt: Union[str, BaseModel]
    tools: list = Field(default_factory=list)
    platform_config: Dict[str, Any] = Field(default_factory=dict)
    response_model: Optional[type[BaseModel]] = None
    json_mode: bool = False
    stream: bool = False
    output_parser: Optional[Callable] = None

class BaseRuntime(ABC):
    """Abstract base for all runtime environments."""
    
    def __init__(self, agent: BaseAgent, config: Optional[RuntimeConfig] = None) -> None:
        """Initialize runtime with an agent instance.
        
        Args:
            agent: Instance of BaseAgent or its subclasses
            config: Optional runtime configuration
        """
        self.agent = agent
        self.config = config or RuntimeConfig(
            system_prompt=agent.system_prompt,
            tools=agent.tools
        )
        self.current_session = None

    @abstractmethod
    def _create_agent(self):
        """Create appropriate agent instance."""
        pass

    def _start_session(self, platform: str) -> None:
        """Start a new session."""
        self.current_session = Session(
            platform=platform,
            agent_config=self.config.model_dump()
        )
        
    @abstractmethod
    async def start(self) -> None:
        """Start the runtime."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the runtime."""
        pass

class BaseChatRuntime(BaseRuntime):
    """Base class for chat-based runtime environments."""
    
    def _create_agent(self):
        """Create agent instance."""
        return BaseAgent(
            system_prompt=self.config.system_prompt,
            tools=self.config.tools,
            response_model=self.config.response_model,
            json_mode=self.config.json_mode,
            stream=self.config.stream,
            output_parser=self.config.output_parser
        )
    
    async def process_message(self, message: str) -> Union[AsyncGenerator[str, None], str, BaseModel, Any]:
        """Process a single message and return the response.
        
        Args:
            message: The user's message to process
            
        Returns:
            Union[AsyncGenerator[str, None], str, BaseModel, Any]:
                If streaming is enabled, returns an async generator yielding chunks
                Otherwise returns the complete response
        """
        if not self.current_session:
            self._start_session(platform="chat")
            
        try:
            if self.config.stream:
                async def stream_response():
                    chunks = []
                    async for chunk, tool in self.agent._stream_step(message):
                        if tool:
                            logger.info(f"[Calling Tool '{tool._name()}' with args {tool.args}]")
                        elif chunk:
                            chunks.append(chunk)
                            yield chunk
                    logger.debug(f"Streamed response: {''.join(chunks)}")
                return stream_response()
            else:
                response = await self.agent._step(message)
                logger.debug(f"Agent response: {response}")
                return response
        except Exception as e:
            logger.exception("Error processing message")
            raise

class LocalRuntime(BaseChatRuntime):
    """Runtime for local console chat interactions."""
    
    async def start(self) -> None:
        """Start a local chat session."""
        self._start_session("local")
        logger.info("Starting chat session. Type 'exit' or 'quit' to stop.")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                print("\nAssistant: ", end="", flush=True)
                if self.config.stream:
                    async for chunk in await self.process_message(user_input):
                        print(chunk, end="", flush=True)
                    print()
                else:
                    response = await self.process_message(user_input)
                    print(response)
                
            except KeyboardInterrupt:
                logger.info("Chat session interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n[Error] {str(e)}")
        
        await self.stop()
        logger.info("Chat session ended. Goodbye! ✨")
    
    async def stop(self) -> None:
        """Stop the local runtime."""
        pass

__all__ = ["RuntimeConfig", "BaseRuntime", "BaseChatRuntime", "LocalRuntime"]
