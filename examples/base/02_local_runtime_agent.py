"""
Local Runtime Agent Example

This module demonstrates how to use the LocalRuntime for a command-line chat 
experience. The runtime configuration now includes a basic system prompt configuration, 
which replaces the previous persona model.

Key Concepts:
    - RuntimeConfig: Holds configuration for the runtime (provider, model, system_prompt, tools).
    - LocalRuntime: Provides a command-line interface that repeatedly calls the agent's 
      _step() method to process user inputs.
    - System Prompt: A basic configuration (via a Pydantic model) that instructs the assistant's behavior.
    
Usage:
    Run this module to start the CLI chat session.

To Execute:
    python examples/base/02_local_runtime_agent.py
"""

import asyncio
import logging
from pydantic import BaseModel, Field
from babelgraph.core.runtime import RuntimeConfig, LocalRuntime
from babelgraph.core.agent import BaseAgent
from babelgraph.tools.image_generator import ImageGenerationTool
from babelgraph.core.logging import configure_logging, LogLevel, LogComponent
from dotenv import load_dotenv

# Optionally configure logging for a cleaner output
# logger = logging.getLogger(__name__)

# Define a basic system prompt configuration replacing the previous PersonaConfig.
class BasicAssistantConfig(BaseModel):
    """
    Basic system prompt configuration for the assistant.
    
    This instructs the assistant to be helpful and concise.
    """
    instruction: str = Field(
        default="You are a helpful and concise assistant.",
        description="Instruction for the assistant system prompt"
    )

ASSISTANT_PROMPT = BasicAssistantConfig()

async def run_with_runtime():
    """
    Run the chat session using LocalRuntime.

    Creates a BaseAgent with the basic system prompt and tools, then wraps it in a LocalRuntime
    for interactive chat functionality.
    """
    # First create the agent with our system prompt and tools
    agent = BaseAgent(
        system_prompt=ASSISTANT_PROMPT,
        tools=[ImageGenerationTool]
    )

    # Create runtime configuration
    config = RuntimeConfig(
        provider="openpipe",
        model="openpipe:ken0-llama31-8B-instruct",
        system_prompt=ASSISTANT_PROMPT,
        tools=[ImageGenerationTool],
        platform_config={
            "prompt_prefix": "You: ",
            "response_prefix": "Assistant: "
        }
    )

    # Initialize and start the local runtime with our agent instance
    runtime = LocalRuntime(agent=agent, config=config)
    print("\nChat using runtime (Ctrl+C to exit)")
    print("Try asking for calculations or image generation!")
    print("-----------------------------------")
    
    await runtime.start()


async def main():
    """
    Entry point for the demonstration.

    Loads environment variables and starts the LocalRuntime configured for chat.
    """
    load_dotenv()  # Load environment variables if needed
    await run_with_runtime()


if __name__ == "__main__":
    asyncio.run(main())