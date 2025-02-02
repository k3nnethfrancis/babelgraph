"""Discord Client Module.

This module provides the base Discord client implementation for Alchemist.
It handles core Discord functionality like connecting to channels and processing messages.
"""

import logging
import discord
from typing import Optional, List, Dict, Any, Callable, Awaitable
import asyncio

logger = logging.getLogger(__name__)

class DiscordClient(discord.Client):
    """Discord client implementation for Alchemist."""
    
    def __init__(self, token: str, intents: Optional[discord.Intents] = None):
        """Initialize the Discord client.
        
        Args:
            token: Discord bot token for authentication
            intents: Discord intents configuration (default: all intents)
        """
        if intents is None:
            intents = discord.Intents.all()
            
        super().__init__(intents=intents)
        self.token = token
        self.ready = asyncio.Event()
        self._message_handlers: List[Callable[[discord.Message], Awaitable[None]]] = []

    async def setup_hook(self):
        """Called when the client is done preparing data."""
        logger.info("Bot is setting up...")

    async def on_ready(self):
        """Called when the client is done preparing data after login."""
        logger.info(f"Logged in as {self.user.name} ({self.user.id})")
        self.ready.set()

    async def on_message(self, message: discord.Message):
        """Handle incoming Discord messages.
        
        Args:
            message: The incoming message
        """
        # Ignore messages from self
        if message.author == self.user:
            return
            
        # Call all registered message handlers
        for handler in self._message_handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error in message handler: {str(e)}")
    
    def add_message_handler(self, handler: Callable[[discord.Message], Awaitable[None]]):
        """Add a message handler function.
        
        Args:
            handler: Async function that takes a discord.Message parameter
        """
        self._message_handlers.append(handler)
    
    async def start(self):
        """Start the Discord client.
        
        This is a wrapper around client.start() that waits for the ready event.
        """
        try:
            logger.info("Starting Discord client...")
            await super().start(self.token)
            
            # Wait for ready with timeout
            try:
                await asyncio.wait_for(self.ready.wait(), timeout=30)
                logger.info("Discord client ready")
            except asyncio.TimeoutError:
                raise RuntimeError("Discord client failed to become ready within 30 seconds")
                
        except discord.LoginFailure as e:
            logger.error(f"Failed to login: {str(e)}")
            raise ValueError("Invalid Discord token") from e
        except Exception as e:
            logger.error(f"Error starting client: {str(e)}")
            raise
    
    async def get_channels(self) -> Dict[str, Dict[str, Any]]:
        """Get all available channels grouped by category.
        
        Returns:
            Dict with:
                channels: Dict[channel_name, channel_id]
                categories: Dict[category_name, List[channel_name]]
        """
        channels = {}
        categories = {}
        
        for guild in self.guilds:
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    channels[channel.name] = str(channel.id)
                    category_name = channel.category.name if channel.category else "No Category"
                    if category_name not in categories:
                        categories[category_name] = []
                    categories[category_name].append(channel.name)
        
        return {
            "channels": channels,
            "categories": categories
        }
    
    async def get_message_history(
        self,
        channel_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get message history from a channel.
        
        Args:
            channel_id: Discord channel ID
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with id, author, content, embeds, and timestamp
            
        Raises:
            ValueError: If channel not found
        """
        channel = self.get_channel(int(channel_id))
        if not channel:
            raise ValueError(f"Channel {channel_id} not found")
            
        messages = []
        async for msg in channel.history(limit=limit):
            # Extract all message data including embeds
            message_data = {
                "id": str(msg.id),
                "author": msg.author.name,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
                "embeds": [
                    {
                        "url": e.url,
                        "title": e.title,
                        "description": e.description,
                        "image": e.image.url if e.image else None,
                        "fields": [{"name": f.name, "value": f.value} for f in e.fields],
                        "footer": {"text": e.footer.text} if e.footer else None,
                        "author": {"name": e.author.name} if e.author else None,
                        "color": str(e.color) if e.color else None,
                        "thumbnail": {"url": e.thumbnail.url} if e.thumbnail else None
                    }
                    for e in msg.embeds
                ],
                "attachments": [
                    {
                        "filename": a.filename,
                        "url": a.url,
                        "content_type": a.content_type
                    }
                    for a in msg.attachments
                ]
            }
            messages.append(message_data)
            
        return messages 