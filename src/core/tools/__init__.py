"""Tools for AI agents.

This module provides a collection of tools that can be used by AI agents
to interact with various services and perform tasks.

Available tools:
- Calculator: Evaluate mathematical expressions
- Discord: Read and interact with Discord channels
- Image: Generate images using DALL-E
"""

from typing import List, Type
from mirascope.core import BaseTool

from .calculator import CalculatorTool
from .image_generator import ImageGenerationTool
from .discord_toolkit import DiscordTools

__all__ = [
    "CalculatorTool",
    "ImageGenerationTool",
    "DiscordTools",
]

def get_available_tools() -> List[Type[BaseTool]]:
    """Get a list of all available tools.
    
    Returns:
        List[Type[BaseTool]]: List of tool classes
    """
    return [
        CalculatorTool,
        ImageGenerationTool,
        DiscordTools,
    ] 