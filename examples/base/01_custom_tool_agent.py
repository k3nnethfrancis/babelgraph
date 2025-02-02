"""
Custom Tool Agent Example

This example demonstrates how to create an agent that uses a custom tool.
In this instance, we create a "WeatherTool" that the agent can call to 
retrieve (mock) weather information. This helps to illustrate how the 
agent processes user inputs, routes tool calls, and manages the conversation
history.

Key Concepts:
    - BaseAgent: The core agent that manages conversation state and LLM calls.
    - BaseTool: The abstraction for tools, allowing integration of domain-specific functionality.
    - Tool usage: When the agent detects a request that requires a tool, it delegates the action.

Usage:
    Run this module to see the agent in action. The agent will prompt for a city 
    name and then use the WeatherTool to return mock weather data.

To Execute:
    python examples/base/01_custom_tool_agent.py
"""

import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# Import the required modules from the Alchemist framework
from mirascope.core import BaseTool
from babelgraph.core.agent import BaseAgent
from babelgraph.core.logging import configure_logging, LogLevel, LogComponent

# Uncomment and configure if using lilypad tracing (disabled for clarity)
# import lilypad
# lilypad.configure()

class WeatherTool(BaseTool):
    """
    Tool for looking up weather information.

    This tool simulates a weather lookup. In a real implementation, it would call an 
    external API. It demonstrates how the agent integrates tool usage in the conversation.
    """
    city: str = Field(..., description="The city to look up weather for")

    @classmethod
    def _name(cls) -> str:
        """Provide the tool's name for LLM function calling."""
        return "weather"

    @classmethod
    def _description(cls) -> str:
        """Provide a description of the tool for LLM guidance."""
        return "Look up current weather for a city"

    def call(self) -> Dict[str, Any]:
        """
        Execute the weather lookup.

        Returns:
            dict: A dictionary containing weather information.
        """
        # In a real scenario, you would make an HTTP request to a weather API.
        return {
            "city": self.city,
            "temperature": "72°F",
            "condition": "Sunny"
        }


class RonBurgundyConfig(BaseModel):
    """
    Configuration for the Ron Burgundy persona.

    Contains attributes like name, title, biography, catchphrases, personality, and speech style.
    This configuration is used as the system prompt to style the agent's responses.
    """
    name: str = Field(..., description="Full name")
    title: str = Field(..., description="Professional title")
    bio: str = Field(..., description="Character biography")
    catchphrases: List[str] = Field(..., description="Famous catchphrases")
    personality_traits: List[str] = Field(..., description="Key personality characteristics")
    speech_style: List[str] = Field(..., description="Speaking mannerisms and patterns")


RON_BURGUNDY = RonBurgundyConfig(
    name="Ron Burgundy",
    title="San Diego's Finest News Anchor",
    bio=(
        "I'm Ron Burgundy, the most distinguished and talented "
        "news anchor in all of San Diego. I'm kind of a big deal."
    ),
    catchphrases=[
        "I'm Ron Burgundy, and you stay classy, San Diego!",
        "Great Odin's raven!",
        "By the beard of Zeus!",
        "Don't act like you're not impressed.",
        "I don't know how to put this, but I'm kind of a big deal."
    ],
    personality_traits=[
        "Outrageously self-confident", "Professionally proud", "Sensitive about his hair",
        "Loves his dog Baxter", "Very serious about news", "Easily confused by big words"
    ],
    speech_style=[
        "Overly dramatic news anchor voice", "Mentions his own name frequently",
        "Random exclamations", "Ends weather reports with a catchphrase",
        "Focus on precise pronunciation", "Occasionally breaks into jazz flute solos (verbally)"
    ]
)


async def main():
    """
    Main function to run Ron Burgundy as a weather-reporting agent.

    Sets up logging, instantiates the BaseAgent with the custom weather tool,
    and enters into an interactive run loop.
    """
    configure_logging(
        default_level=LogLevel.INFO,
        component_levels={LogComponent.AGENT: LogLevel.DEBUG}
    )

    # Create the base agent with the custom system prompt and associated tool
    agent = BaseAgent(
        system_prompt=RON_BURGUNDY,
        tools=[WeatherTool],
        stream=True,
    )

    print("I'm Ron Burgundy, and I'll be your weather reporter today.")
    print("Ask me about the weather in any city!")
    print("(Type 'exit' to end the broadcast)")
    print("-" * 50)

    await agent.run()
    # or stream with:
    # agent.stream_run()


if __name__ == "__main__":
    asyncio.run(main()) 