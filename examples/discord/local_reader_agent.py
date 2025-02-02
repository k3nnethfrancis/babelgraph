"""Local Discord reader example.

This example demonstrates using the Discord toolkit to read channel messages
through an interactive chat interface using our new system prompt mechanism.
"""

import asyncio
import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field

from babelgraph.core.runtime import RuntimeConfig, LocalRuntime
from babelgraph.core.agent import BaseAgent
from babelgraph.tools import DiscordTools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a basic system prompt configuration for the local Discord reader.
class LocalDiscordReaderPrompt(BaseModel):
    """
    Basic system prompt configuration for the local Discord reader.

    Instructs the agent to summarize or report on Discord channel messages.
    """
    instruction: str = Field(
        default="You are a helpful assistant that can read and summarize Discord channel messages.",
        description="Local Discord reader prompt"
    )

READER_PROMPT = LocalDiscordReaderPrompt()

async def run_chat():
    """Run Discord reader chat with LocalRuntime."""
    print("\nStarting Discord Reader Chat...")
    
    # Load channel data from config
    config_path = Path(__file__).parent.parent.parent / "config" / "channels.json"
    with open(config_path) as f:
        data = json.load(f)
        channels = data["channels"]
        categories = data["categories"]
        print(f"\nFound {len(channels)} channels in {len(categories)} categories")
        print("\nAvailable channels:")
        for name in channels:
            print(f"  #{name}")
    
    # Initialize Discord toolkit and get its tools
    toolkit = DiscordTools(
        channels=channels,
        categories=categories
    )
    
    # Configure runtime
    config = RuntimeConfig(
        provider="openpipe",
        model="openpipe:ken0-llama31-8B-instruct",
        system_prompt=READER_PROMPT,
        tools=toolkit.create_tools(),  # Create the actual tools from the toolkit
        platform_config={
            "prompt_prefix": "You: ",
            "response_prefix": "Assistant: "
        }
    )
    
    # Print help message
    print("\nChat with me! I can read messages from Discord channels.")
    print('Type "exit", "quit", or send an empty message to end the chat.')
    print("\nExample commands:")
    print('- "What are the latest messages in #ai-news?"')
    print('- "Show me the last 5 messages from #general"')
    print('- "Summarize recent updates from #resources"\n')
    
    # Create an agent instance using the provided system prompt and tools.
    agent = BaseAgent(
        system_prompt=config.system_prompt,
        tools=config.tools
    )
    
    # Initialize and start runtime (this will handle the chat loop)
    runtime = LocalRuntime(agent=agent, config=config)
    await runtime.start()

if __name__ == "__main__":
    asyncio.run(run_chat()) 