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
from typing import Optional, Dict, Any, Union, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

from alchemist.ai.base.agent import BaseAgent
from alchemist.ai.base.runtime import BaseRuntime
from alchemist.ai.base.logging import get_logger, LogComponent, Colors
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.state import NodeState, NodeStatus

logger = get_logger(LogComponent.NODES)

class AgentNode(Node):
    """
    Node for executing LLM agent steps with structured outputs.
    
    Features:
        - Message passing to agent
        - Streaming support
        - Required structured outputs
        - State management
        - Response timing and logging
        - Runtime injection for platform-specific behavior
        
    The node can operate in two modes:
    1. Direct agent execution (when no runtime is provided)
    2. Runtime-based execution (when a runtime is injected)
    """
    
    agent: BaseAgent = Field(
        ...,  # Required
        description="Agent instance to use for LLM calls"
    )
    runtime: Optional[BaseRuntime] = Field(
        default=None,
        description="Optional runtime for platform-specific behavior"
    )
    response_model: type[BaseModel] = Field(
        ...,  # Required
        description="Pydantic model for structured output validation"
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
        if not self.response_model:
            raise ValueError(f"AgentNode {self.id} requires a response_model")
        return self

    async def process(self, state: NodeState) -> Optional[str]:
        """Process the node based on its type."""
        try:
            # Get runtime from metadata or use direct agent
            runtime = self.metadata.get("runtime")
            if not runtime:
                # Use direct agent execution
                message = state.data.get("message", "")
                if not message:
                    raise ValueError(f"No message found in state.data[message]")
                
                # Configure agent
                self.agent.response_model = self.response_model
                
                # Process with agent
                if self.stream:
                    chunks = []
                    async for chunk, tool in self.agent._stream_step(message):
                        if tool:
                            logger.info(f"[Calling Tool '{tool._name()}' with args {tool.args}]")
                            if hasattr(tool, 'call'):
                                result = await tool.call() if asyncio.iscoroutinefunction(tool.call) else tool.call()
                                logger.info(f"Tool result: {result}")
                        else:
                            chunks.append(chunk)
                            # Emit streaming event
                            if "on_stream" in self.metadata:
                                await self.metadata["on_stream"](chunk, state, self.id)
                    response = "".join(chunks)
                else:
                    response = await self.agent._step(message)
                
                # Store result
                state.results[self.id] = {"response": response}
                return "default"
            
            # Process based on node type using runtime
            if "validate" in self.id:
                # Get story content to validate
                story_node_id = self.id.replace("validate_", "")
                story_content = state.results[story_node_id]["response"]["content"]
                agent_id = f"agent{self.metadata['agent_id']}"
                previous_parts = runtime.stories[agent_id]
                
                # Validate using runtime
                response = await runtime.validate_story_part(story_content, previous_parts)
                
                # Store result
                state.results[self.id] = {"response": response}
                
                # Return next node based on validation
                return "valid" if response.is_valid else "invalid"
                
            elif "reflection" in self.id:
                # Use runtime for reflection
                response = await runtime.synthesize_stories()
                state.results[self.id] = {"response": response}
                return "meets_criteria" if response.meets_criteria else "needs_improvement"
                
            elif "final_synthesis" in self.id:
                # Final synthesis using runtime
                response = await runtime.synthesize_stories()
                state.results[self.id] = {"response": response}
                return "default"
                
            else:
                # Story generation node
                part = self.metadata["part"]
                themes = self.metadata["themes"]
                agent_id = f"agent{self.metadata['agent_id']}"
                
                # Generate story part using runtime
                response = await runtime.generate_story_part(agent_id, part, themes)
                state.results[self.id] = {"response": response}
                return "default"
                
        except Exception as e:
            state.add_error(self.id, str(e))
            return "error"

async def test_agent_node():
    """Test the AgentNode with streaming and structured output."""
    from alchemist.ai.base.agent import BaseAgent
    from alchemist.ai.base.runtime import BaseRuntime
    from alchemist.ai.graph.state import NodeState
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