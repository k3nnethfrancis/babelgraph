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
from typing import Optional, Dict, Any, Union, AsyncGenerator, List
from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from functools import partial
import json

from babelgraph.core.agent.base import BaseAgent
from babelgraph.core.runtime import BaseRuntime
from babelgraph.core.logging import get_logger, LogComponent, Colors
from babelgraph.core.graph.nodes.base.node import Node
from babelgraph.core.graph.state import NodeState, NodeStatus
from mirascope.core import BaseMessageParam

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

    @model_validator(mode='after')
    def validate_agent_config(self) -> "AgentNode":
        """Validate agent configuration."""
        if not self.agent:
            raise ValueError(f"AgentNode {self.id} requires an agent")
        return self

    def _serialize_history(self, history: List[BaseMessageParam]) -> List[dict]:
        """Serialize BaseMessageParam history to JSON-compatible dict."""
        return [
            {
                "role": msg.role,
                "content": msg.content
            } if isinstance(msg, BaseMessageParam) else msg
            for msg in history
        ]

    async def process(self, state: "NodeState") -> Optional[str]:
        """Process the node using the agent."""
        try:
            # Get message input
            message = self.get_message_input(state)
            if not message:
                raise ValueError("No message input provided")

            # Create proper BaseMessageParam
            message_param = BaseMessageParam(
                role=message.get('role', 'user'),
                content=message['content']
            )

            # Log with proper serialization
            logger.debug(
                f"Agent {self.id} processing message:\n"
                f"Role: {message_param.role}\n"
                f"Content: {message_param.content}\n"
                f"Current History: {json.dumps(self._serialize_history(self.agent.history), indent=2)}"
            )

            # Process with agent - pass BaseMessageParam
            if self.stream:
                response = await self.agent._stream(
                    message_param,  # Pass full message param
                    callback=partial(
                        self._handle_stream_callback,
                        state=state,
                        node_id=self.id
                    )
                )
            else:
                response = await self.agent._step(message_param)  # Pass full message param
            
            # Log updated history with proper serialization
            logger.debug(
                f"Agent {self.id} updated history after response:\n"
                f"{json.dumps(self._serialize_history(self.agent.history), indent=2)}"
            )
            
            # Store result and return success
            self.set_result(state, "response", response)
            return "success"
            
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