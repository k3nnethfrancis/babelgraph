"""Tool for generating images using DALL-E 3.

This module provides a tool for generating images using OpenAI's DALL-E 3 model.
Features:
- Prompt-based image generation
- Style customization
- Error handling and logging
"""

import logging
from openai import AsyncOpenAI
from mirascope.core import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)

class ImageGenerationTool(BaseTool):
    """Tool for generating images using DALL-E 3.
    
    This tool uses OpenAI's DALL-E 3 model to generate images from text descriptions.
    It supports customization of style and provides detailed error handling.
    
    Attributes:
        prompt: The description of the image to generate
        style: Optional style guide for the image
        
    Example:
        ```python
        tool = ImageGenerationTool(prompt="A spaceship going to the moon")
        url = await tool.call()  # Returns URL to generated image
        ```
    """
    
    prompt: str = Field(
        ...,
        description="The description of the image to generate",
        examples=["A spaceship going to the moon", "A cyberpunk city at sunset"]
    )
    style: str = Field(
        default="Blend the styles of Mobeus, solarpunk, and 70s sci-fi pulp",
        description="Style guide for the image generation"
    )

    @classmethod
    def _name(cls) -> str:
        """Get the tool's name for LLM function calling."""
        return "generate_image"

    @classmethod
    def _description(cls) -> str:
        """Get the tool's description for LLM function calling."""
        return "Generate images using DALL-E 3"

    async def call(self) -> str:
        """Generate an image using DALL-E 3.
        
        Returns:
            str: URL of the generated image
            
        Raises:
            Exception: If image generation fails
        """
        try:
            # Format prompt with style guide
            formatted_prompt = f"{self.prompt}. {self.style}"
            logger.info(f"🎨 Generating image: {formatted_prompt}")
            
            # Generate image using DALL-E
            client = AsyncOpenAI()  # Uses API key from environment or client config
            response = await client.images.generate(
                model="dall-e-3",
                prompt=formatted_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            logger.info(f"✨ Generated image: {image_url}")
            
            return image_url
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {str(e)}")
            raise 