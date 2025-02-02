"""Discord Reader Bot Service.

This script runs a Discord bot service that provides:
1. Channel information via HTTP endpoint
2. Message history access via HTTP endpoint

Before running:
1. Make sure you have set DISCORD_READER_TOKEN in your .env file
2. Run this in a separate terminal before starting the local Discord reader
"""

import asyncio
import logging
import os
from typing import Dict, List
from aiohttp import web
from dotenv import load_dotenv

from babelgraph.extensions.discord.runtime import DiscordRuntimeConfig, DiscordLocalRuntime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiscordBotService:
    """Service that provides Discord bot functionality via HTTP endpoints."""
    
    def __init__(self):
        """Initialize the Discord bot service."""
        self.runtime = None
        self.ready = asyncio.Event()
        self.channels = {}
        self.categories = {}
        
    async def _handle_message(self, message):
        """Log incoming Discord messages."""
        logger.info(f"[Discord] Message from {message.author.name}:")
        logger.info(f"Content: {message.content}")
        logger.info(f"Embeds: {message.embeds}")
        
    async def _update_channels(self):
        """Update channel information from Discord."""
        channel_info = await self.runtime.get_channels()
        self.channels = channel_info["channels"]
        self.categories = channel_info["categories"]
        
        # Log channel information
        logger.info("\nAvailable channels by category:")
        for category, channels in self.categories.items():
            logger.info(f"\nCategory: {category}")
            for channel in channels:
                logger.info(f"  - #{channel}")
    
    async def handle_channels(self, request):
        """Handle GET /channels request."""
        return web.json_response({
            "channels": self.channels,
            "categories": self.categories
        })
    
    async def handle_history(self, request):
        """Handle GET /history/{channel_id} request."""
        channel_id = request.match_info['channel_id']
        limit = int(request.query.get('limit', 100))
        
        try:
            history = await self.runtime.get_message_history(channel_id, limit)
            return web.json_response(history)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def start(self):
        """Start the Discord runtime and HTTP server."""
        try:
            # Load environment variables
            load_dotenv()
            token = os.getenv("DISCORD_BOT_TOKEN")
            if not token:
                raise ValueError(
                    "DISCORD_BOT_TOKEN not found in environment. "
                    "Please set it in your .env file."
                )
                
            logger.info("Starting Discord bot...")
            
            # Configure Discord runtime
            runtime_config = DiscordRuntimeConfig(
                bot_token=token,
                channel_ids=["*"]  # Allow access to all channels
            )
            
            self.runtime = DiscordLocalRuntime(config=runtime_config)
            
            # Add message handler for logging
            self.runtime.add_message_handler(self._handle_message)
            
            # Start Discord runtime
            await self.runtime.start()
            logger.info("Discord runtime started successfully")
            
            # Update channel information
            await self._update_channels()
            logger.info(f"Found {len(self.channels)} channels in {len(self.categories)} categories")
            self.ready.set()
            
            # Start HTTP server
            logger.info("Starting HTTP server...")
            app = web.Application()
            app.router.add_get('/channels', self.handle_channels)
            app.router.add_get('/history/{channel_id}', self.handle_history)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 5000)
            await site.start()
            
            logger.info("Discord reader bot service running on http://localhost:5000")
            logger.info("Available endpoints:")
            logger.info("  - GET /channels")
            logger.info("  - GET /history/{channel_id}?limit={limit}")
            
            # Keep the service running
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Shutting down HTTP server...")
                await runner.cleanup()
                logger.info("HTTP server stopped")
                
        except Exception as e:
            logger.error(f"Error starting service: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the Discord bot service."""
        if self.runtime:
            await self.runtime.stop()

def main():
    """Run the Discord bot service."""
    service = DiscordBotService()
    
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logger.info("\nShutting down Discord bot service...")
        asyncio.run(service.stop())
        logger.info("Service stopped")

if __name__ == "__main__":
    main() 