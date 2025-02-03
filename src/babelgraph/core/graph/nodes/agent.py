"""
Agent Node Implementation

This module provides the AgentNode class for executing LLM agent steps within a graph.
Supports:
- Message passing to agent
- Streaming responses
- Structured outputs (required)
- Tool execution via agent
- Runtime injection for platform-specific behavior
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Union, AsyncGenerator, Type
from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from functools import partial

from babelgraph.core.agent.base import BaseAgent
from babelgraph.core.runtime import BaseRuntime
from babelgraph.core.logging import get_logger, LogComponent, Colors
from babelgraph.core.graph.nodes.base.node import Node
from babelgraph.core.graph.state import NodeState, NodeStatus

logger = get_logger(LogComponent.NODES)

class AgentNode(Node):
    """
    Node for executing LLM agent steps.
    
    Features:
        - Message passing to agent
        - Streaming support
        - Optional structured outputs
        - State management
        - Response timing and logging
        - Runtime injection for platform-specific behavior
    """
    
    agent: BaseAgent = Field(
        ...,  # Required
        description="Agent instance to use for LLM calls"
    )
    runtime: Optional[BaseRuntime] = Field(
        default=None,
        description="Optional runtime for platform-specific behavior"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    response_model: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Optional Pydantic model for response validation"
    )
    maintain_history: bool = Field(
        default=True,
        description="Whether to maintain conversation history across executions"
    )

    @model_validator(mode='after')
    def validate_agent_config(self) -> "AgentNode":
        """Validate agent configuration."""
        if not self.agent:
            raise ValueError(f"AgentNode {self.id} requires an agent")
        
        # If response model is set, configure agent
        if self.response_model:
            self.agent.response_model = self.response_model
            
        return self

    async def process(self, state: "NodeState") -> Optional[str]:
        """Process the node using the agent."""
        try:
            # Get message input
            message = self.get_message_input(state)
            if not message:
                raise ValueError("No message input provided")

            logger.debug(f"Processing message in {self.id}: {message['content']}")

            try:
                # Process with agent - pass only the content
                if self.stream:
                    # Stream response with callback
                    response = ""
                    async for chunk, tool in self.agent._stream_step(message['content']):
                        if tool:
                            # Handle tool execution if needed
                            logger.info(f"Tool execution: {tool._name()}")
                            result = tool.call()
                            logger.info(f"Tool result: {result}")
                        elif chunk:
                            # Stream the chunk if callback exists
                            if "on_stream" in self.metadata:
                                await self.metadata["on_stream"](chunk, state, self.id)
                            response += chunk
                    
                    # Store final response with logging suppressed (since we streamed it)
                    self.set_result(state, "response", response, suppress_logging=True)
                else:
                    # Process normally and store result
                    response = await self.agent._step(message['content'])
                    
                    # Log raw response for debugging
                    logger.debug(f"Raw agent response: {response}")
                    
                    # Validate response if model specified
                    if self.response_model:
                        try:
                            validated = self.response_model.model_validate(response)
                            response = validated
                        except Exception as validation_error:
                            logger.error(f"Response validation error: {validation_error}")
                            if hasattr(validation_error, 'errors'):
                                for error in validation_error.errors():
                                    logger.error(f"- {error}")
                            raise
                    
                    # Store result
                    self.set_result(state, "response", response)
                
                # Clear history if not maintaining
                if not self.maintain_history:
                    self.agent.clear_history()
                
                return "success"
                
            except Exception as agent_error:
                # Log detailed error info
                logger.error(f"Agent error in {self.id}:")
                logger.error(f"Error type: {type(agent_error)}")
                logger.error(f"Error message: {str(agent_error)}")
                
                # Handle RetryError specially
                if hasattr(agent_error, 'last_attempt'):
                    try:
                        last_error = agent_error.last_attempt.exception()
                        if last_error:
                            logger.error(f"Original error type: {type(last_error)}")
                            logger.error(f"Original error message: {str(last_error)}")
                    except Exception as e:
                        logger.error(f"Error extracting original error: {e}")
                
                raise
            
        except Exception as e:
            logger.error(f"Error in agent node {self.id}: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"

async def test_agent_node():
    """Test the AgentNode with streaming and structured output."""
    from babelgraph.core.agent.base import BaseAgent
    from babelgraph.core.runtime import BaseRuntime
    from babelgraph.core.graph.state import NodeState
    from pydantic import BaseModel
    
    # Define a test response model
    class TestResponse(BaseModel):
        message: str
        confidence: float
    
    # Create test agent and node
    agent = BaseAgent()
    
    # Test with direct agent execution
    node1 = AgentNode(
        id="direct_agent",
        agent=agent,
        response_model=TestResponse,
        message_key="user_message",
        stream=True
    )
    
    # Test with runtime
    runtime = BaseRuntime(agent=agent)
    node2 = AgentNode(
        id="runtime_agent",
        agent=agent,
        runtime=runtime,
        response_model=TestResponse,
        message_key="user_message",
        stream=True
    )
    
    # Create test state
    state = NodeState()
    state.data["user_message"] = "Tell me about AI."
    
    # Run tests
    print("\nTesting direct agent execution:")
    next_id = await node1.process(state)
    print(f"Next node: {next_id}")
    print(f"Results: {state.results}")
    
    print("\nTesting runtime execution:")
    next_id = await node2.process(state)
    print(f"Next node: {next_id}")
    print(f"Results: {state.results}")

if __name__ == "__main__":
    asyncio.run(test_agent_node()) 