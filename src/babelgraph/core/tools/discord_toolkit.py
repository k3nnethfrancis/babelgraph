"""Discord toolkit for reading channel history.

This module provides tools for reading message history from Discord channels
via a local bot service. Features:
- Channel history retrieval with name-based lookup
- Time-based filtering
- Rich content support (embeds, attachments)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    BaseToolKit,
    openai,
    toolkit_tool
)
from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)

class DiscordTools(BaseToolKit):
    """A toolkit for reading Discord channel history.
    
    This toolkit provides tools for interacting with Discord channels and
    automatically handles channel name to ID mapping.
    
    Attributes:
        channels: Mapping of channel names to IDs
        categories: Mapping of category names to channel lists
        service_url: URL of the local Discord bot service
    """
    
    __namespace__ = "discord_tools"
    
    channels: Dict[str, str] = Field(
        description="Mapping of channel names to IDs"
    )
    categories: Dict[str, List[str]] = Field(
        description="Mapping of category names to channel lists"
    )
    service_url: str = Field(
        default="http://localhost:5000",
        description="URL of the local Discord bot service"
    )

    @property
    def available_channels(self) -> str:
        """Format the list of available channels."""
        return ', '.join(f'#{k}' for k in self.channels.keys())

    def _process_embed(self, embed: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Discord embed into a consistent format.
        
        Args:
            embed: Raw embed data from Discord
            
        Returns:
            Processed embed with validated fields
        """
        processed = {}
        
        # Required fields that should always be present
        processed["type"] = embed.get("type", "rich")  # Default to rich embed
        
        # Optional but common fields - only include if they exist
        if title := embed.get("title"):
            processed["title"] = str(title)
        if description := embed.get("description"):
            processed["description"] = str(description)
        if url := embed.get("url"):
            processed["url"] = str(url)
        if timestamp := embed.get("timestamp"):
            processed["timestamp"] = str(timestamp)
        if color := embed.get("color"):
            # Handle both hex strings and integers
            try:
                if isinstance(color, str) and color.startswith('#'):
                    processed["color"] = int(color[1:], 16)  # Convert hex to int
                else:
                    processed["color"] = int(color)  # Already an int
            except (ValueError, TypeError):
                pass  # Skip invalid colors
            
        # Handle footer - only include if it has text
        if footer := embed.get("footer"):
            if text := footer.get("text"):
                footer_data = {"text": str(text)}
                if icon_url := footer.get("icon_url"):
                    footer_data["icon_url"] = str(icon_url)
                processed["footer"] = footer_data
            
        # Handle author - only include if it has a name
        if author := embed.get("author"):
            if name := author.get("name"):
                author_data = {"name": str(name)}
                if url := author.get("url"):
                    author_data["url"] = str(url)
                if icon_url := author.get("icon_url"):
                    author_data["icon_url"] = str(icon_url)
                processed["author"] = author_data
            
        # Handle thumbnail and image - only include if they have URLs
        if thumbnail := embed.get("thumbnail"):
            if url := thumbnail.get("url"):
                processed["thumbnail"] = {"url": str(url)}
        if image := embed.get("image"):
            if url := image.get("url"):
                processed["image"] = {"url": str(url)}
            
        # Handle fields - only include non-empty ones
        if fields := embed.get("fields"):
            valid_fields = [
                {
                    "name": str(field.get("name", "")),
                    "value": str(field.get("value", "")),
                    "inline": bool(field.get("inline", False))
                }
                for field in fields
                if field.get("name") and field.get("value")
            ]
            if valid_fields:
                processed["fields"] = valid_fields
            
        return processed

    @toolkit_tool
    async def read_channel(
        self,
        channel_name: str,
        after: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Read messages from a Discord channel.

        Available channels: {self.available_channels}
        """
        # Strip # if present and look up channel ID
        clean_name = channel_name.lstrip('#')
        channel_id = self.channels.get(clean_name)
        if not channel_id:
            raise ValueError(f"Channel '{channel_name}' not found")
            
        url = f"{self.service_url}/history/{channel_id}?limit={limit}"
        if after:
            url += f"&after={after.isoformat()}"
            
        logger.info(f"📨 Reading messages from #{clean_name}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to get messages: HTTP {response.status}")
                        
                    data = await response.json()
                    
                    # Handle both list and dict responses
                    messages = data if isinstance(data, list) else data.get("messages", [])
                    
                    logger.info(f"📥 Received {len(messages)} messages")
                    # Log first message as example
                    if messages:
                        logger.info(f"📝 Example message structure:\n{messages[0]}")
                    
                    # Process messages to ensure consistent structure
                    processed_messages = []
                    for msg in messages:
                        # Process embeds - filter out None values
                        processed_embeds = []
                        for embed in msg.get("embeds", []):
                            processed_embed = {}
                            for key, value in embed.items():
                                if value is not None:  # Only include non-None values
                                    if isinstance(value, dict):
                                        # For nested dicts like footer, filter out None values
                                        processed_embed[key] = {k: v for k, v in value.items() if v is not None}
                                    else:
                                        processed_embed[key] = value
                            if processed_embed:  # Only include if we have any values
                                processed_embeds.append(processed_embed)
                        
                        processed_msg = {
                            "id": msg.get("id"),
                            "content": msg.get("content", ""),
                            "author": msg.get("author") if isinstance(msg.get("author"), str) else msg.get("author", {}).get("name", "Unknown"),
                            "timestamp": msg.get("timestamp"),
                            "embeds": processed_embeds,
                            "attachments": [att for att in msg.get("attachments", []) if att is not None]
                        }
                        processed_messages.append(processed_msg)
                    
                    return processed_messages
                    
        except aiohttp.ClientError as e:
            logger.error(f"❌ HTTP error reading Discord channel: {str(e)}")
            raise Exception(f"Failed to get messages: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error reading Discord channel: {str(e)}")
            raise

@openai.call("gpt-4o-mini")
def read_discord_history(
    channel_name: str,
    channels: Dict[str, str],
    categories: Dict[str, List[str]],
    after: Optional[datetime] = None,
    limit: int = 100
) -> BaseDynamicConfig:
    """Read message history from a Discord channel using dynamic configuration.
    
    Args:
        channel_name: Name of the channel to read from (without #)
        channels: Mapping of channel names to IDs
        categories: Mapping of category names to channel lists
        after: Optional timestamp to filter messages after
        limit: Maximum number of messages to retrieve
        
    Returns:
        BaseDynamicConfig: Configuration with tools and messages
    """
    toolkit = DiscordTools(
        channels=channels,
        categories=categories
    )
    
    return {
        "tools": toolkit.create_tools(),
        "messages": [
            BaseMessageParam(
                role="system",
                content="You are a helpful assistant that can read Discord channels. Available channels: " + 
                       ", ".join(f"#{name}" for name in channels.keys())
            ),
            BaseMessageParam(
                role="user",
                content=f"What messages are in the {channel_name} channel?"
                + (f" after {after.isoformat()}" if after else "")
            )
        ]
    } 