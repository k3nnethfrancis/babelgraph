"""Discord Runtime.

This module provides runtime implementations for Discord integration:
1. DiscordRuntime: Base Discord functionality (client, channels, history)
2. DiscordChatRuntime: Chat functionality for Discord bots
3. DiscordLocalRuntime: Local chat functionality for reading Discord

The hierarchy is:
BaseRuntime -> BaseChatRuntime -> DiscordRuntime -> [DiscordChatRuntime, DiscordLocalRuntime]
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
import asyncio
import discord
from pydantic import BaseModel, Field

from alchemist.extensions.discord.client import DiscordClient
from alchemist.ai.base.runtime import BaseChatRuntime, RuntimeConfig, BaseAgent

logger = logging.getLogger(__name__)

class DiscordRuntimeConfig(BaseModel):
    """Configuration for Discord runtime.
    
    Attributes:
        bot_token: Discord bot token for authentication
        channel_ids: List of channel IDs to monitor (["*"] for all)
        runtime_config: Base runtime configuration for the chat agent
        platform_config: Additional Discord-specific configuration
    """
    
    bot_token: str = Field(..., description="Discord bot token for authentication")
    channel_ids: List[str] = Field(
        default=["*"],
        description="List of Discord channel IDs to monitor ('*' for all)"
    )
    runtime_config: Optional[RuntimeConfig] = Field(
        None,
        description="Base runtime configuration for the chat agent"
    )
    platform_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional Discord-specific configuration"
    )

class DiscordRuntime(BaseChatRuntime):
    """Base runtime for Discord integration.
    
    This runtime provides core Discord functionality:
    - Client connection and management
    - Channel monitoring and message handling
    - Message history access
    - Channel information retrieval
    """
    
    def __init__(self, config: DiscordRuntimeConfig):
        """Initialize the Discord runtime.
        
        Args:
            config: Runtime configuration
        """
        # Ensure a runtime_config is provided
        if not config.runtime_config:
            raise ValueError("runtime_config is required for DiscordRuntime")
        
        # Create the agent using the provided runtime configuration.
        # You can use the _create_agent method from BaseChatRuntime,
        # but here we directly instantiate BaseAgent with the configuration.
        # Alternatively, if you have any custom logic, factor it in here.
        agent = BaseAgent(
            system_prompt=config.runtime_config.system_prompt,
            tools=config.runtime_config.tools
        )
        # Call the base class initializer with the constructed agent.
        super().__init__(agent=agent, config=config.runtime_config)
        
        # Store Discord-specific config
        self.config = config
        self.client = DiscordClient(token=config.bot_token)
        self._task: Optional[asyncio.Task] = None
        
    def add_message_handler(self, handler: Callable[[discord.Message], Awaitable[None]]):
        """Add a message handler function.
        
        Args:
            handler: Async function that takes a discord.Message parameter
        """
        self.client.add_message_handler(handler)
    
    async def start(self):
        """Start the Discord runtime.
        
        This method starts the Discord client in a background task and waits
        for it to be ready before returning.
        """
        logger.info("Starting Discord runtime...")
        
        # Start base runtime if available
        if hasattr(super(), 'start'):
            await super().start()
        
        # Create background task for client
        self._task = asyncio.create_task(self._run_client())
        
        # Wait for client to be ready
        await self.client.ready.wait()
        logger.info("Discord runtime started successfully")
        
    async def stop(self):
        """Stop the Discord runtime.
        
        This method gracefully shuts down the Discord client and cleans up resources.
        """
        logger.info("Stopping Discord runtime...")
        
        if self._task:
            try:
                await self.client.close()
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                logger.error(f"Error stopping client: {str(e)}")
                
        # Stop base runtime if available
        if hasattr(super(), 'stop'):
            await super().stop()
            
        logger.info("Discord runtime stopped")
    
    async def _run_client(self):
        """Run the Discord client in the background."""
        try:
            await self.client.start()
        except Exception as e:
            logger.error(f"Error in client task: {str(e)}")
            raise
    
    async def get_channels(self) -> Dict[str, Dict[str, Any]]:
        """Get all available channels grouped by category.
        
        Returns:
            Dict with:
                channels: Dict[channel_name, channel_id]
                categories: Dict[category_name, List[channel_name]]
        """
        return await self.client.get_channels()
    
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
            List of message dictionaries with id, author, content, and timestamp
            
        Raises:
            ValueError: If channel not found
        """
        return await self.client.get_message_history(channel_id, limit)

class DiscordChatRuntime(DiscordRuntime):
    """Runtime for Discord chatbots.
    
    This runtime extends DiscordRuntime to add chat functionality:
    - Message processing via BaseChatRuntime
    - Automatic message handling and responses
    """
    
    def __init__(self, config: DiscordRuntimeConfig):
        """Initialize the Discord chat runtime.
        
        Args:
            config: Runtime configuration with chat settings
        """
        if not config.runtime_config:
            raise ValueError("runtime_config is required for DiscordChatRuntime")
        super().__init__(config)

class DiscordLocalRuntime(DiscordRuntime):
    """Runtime for local Discord interactions.
    
    This runtime extends DiscordRuntime for local usage:
    - No chat functionality needed
    - Focus on channel and history access
    """
    
    def __init__(self, config: DiscordRuntimeConfig):
        """Initialize the local Discord runtime.
        
        Args:
            config: Runtime configuration (no chat config needed)
        """
        super().__init__(config) 