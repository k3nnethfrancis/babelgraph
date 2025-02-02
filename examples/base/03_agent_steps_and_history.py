"""
Agent Steps and History Example

This example demonstrates:
    1. How to perform a single interaction using BaseAgent's _step() method.
    2. How to inspect the conversation history maintained by the agent.
    3. Defining a system prompt (instead of a persona) to style agent responses.

Key Concepts:
    - _step(): Processes a single user query, updating the conversation history.
    - Conversation History: Maintained by the agent to store previous inputs and outputs.
    - System Prompt: A basic instruction passed to the agent to guide its responses.
    
Usage:
    Run this module to see an example of a single-step interaction and then view the agent's history.

To Execute:
    python examples/base/03_agent_steps_and_history.py
"""

import asyncio
from pydantic import BaseModel, Field
from babelgraph.core.agent import BaseAgent
from babelgraph.core.logging import configure_logging, LogLevel

# Define a basic system prompt configuration.
class BasicAssistantConfig(BaseModel):
    """
    Basic system prompt configuration for the agent.
    
    Instructs the agent to provide helpful and concise responses.
    """
    instruction: str = Field(
        default="You are a helpful and concise assistant.",
        description="Instruction for the system prompt"
    )

BASIC_PROMPT = BasicAssistantConfig()

async def main():
    """
    Demonstrate a single interaction and show conversation history.

    Configures logging, creates an agent with a basic system prompt, 
    sends a query, prints the response, and then outputs the conversation history.
    """
    configure_logging(default_level=LogLevel.INFO)

    # Create an agent with the basic system prompt.
    agent = BaseAgent(system_prompt=BASIC_PROMPT)

    question = "What are three interesting facts about quantum computing?"
    print(f"\nAsking: {question}")

    response = await agent._step(question)
    print(f"\nResponse: {response}")

    # Print the conversation history for review.
    print("\nConversation History:")
    print(agent.history)


if __name__ == "__main__":
    asyncio.run(main()) 