"""
Discord Chatbot Example

This example demonstrates a simple Discord chatbot that:
1. Uses a custom system prompt (instead of a full persona)
2. Responds to messages in a configured channel
3. Requires run_bot.py (the Discord runtime service) to be running first

Usage:
    1. Set your DISCORD_BOT_TOKEN in your .env file
    2. Run the Discord bot service: python -m examples.discord.run_bot
    3. Then run this chatbot: python -m examples.discord.chatbot
"""

import os
import asyncio
from dotenv import load_dotenv
import discord
from pydantic import BaseModel, Field

from babelgraph.core.runtime import RuntimeConfig
from babelgraph.core.agent import BaseAgent
from babelgraph.extensions.discord.runtime import DiscordRuntimeConfig, DiscordChatRuntime

# Define a basic system prompt configuration for the Discord assistant.
class DiscordAssistantConfig(BaseModel):
    """
    Basic system prompt configuration for the Discord assistant.
    
    Instructs the assistant to provide helpful and context-aware responses within Discord.
    """
    instruction: str = Field(
        default="You are a helpful and knowledgeable Discord assistant.",
        description="Discord assistant prompt"
    )

ASSISTANT_PROMPT = DiscordAssistantConfig()


async def handle_message(message: discord.Message, runtime: DiscordChatRuntime) -> None:
    """
    Handle incoming Discord messages.

    Args:
        message: The Discord message to process.
        runtime: The Discord runtime instance.
    """
    # Process messages only if they are posted in configured channels.
    if str(message.channel.id) not in runtime.config.channel_ids:
        return

    # Only process messages that mention the bot.
    if not message.mentions or runtime.client.user not in message.mentions:
        return

    # Remove the bot mention from the message.
    content = message.content.replace(f'<@{runtime.client.user.id}>', '').strip()

    try:
        # Process the message using the runtime's process_message method.
        response = await runtime.process_message(content)
        # Send the response back to Discord.
        await message.channel.send(response)
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        await message.channel.send("Sorry, I encountered an error processing your message.")


async def main() -> None:
    """
    Run the Discord chatbot.

    Loads the environment variables, configures the Discord runtime with the new system prompt,
    and starts the chatbot.
    """
    load_dotenv()
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("Error: DISCORD_BOT_TOKEN not set in .env file")
        return

    # Configure the Discord runtime using the new system prompt approach.
    discord_config = DiscordRuntimeConfig(
        bot_token=token,
        channel_ids=["1318659602115592204"],  # Change to your target channel(s)
        runtime_config=RuntimeConfig(
            provider="openpipe",
            model="openpipe:ken0-llama31-8B-instruct",
            system_prompt=ASSISTANT_PROMPT,
        )
    )

    # Create and start the Discord chat runtime with the new configuration.
    # Note: The agent is now created internally via _create_agent() in BaseChatRuntime.
    runtime = DiscordChatRuntime(config=discord_config)
    runtime.add_message_handler(lambda msg: handle_message(msg, runtime))

    await runtime.start()

    print("\nDiscord chatbot running with new system prompt!")
    print("Press Ctrl+C to exit")

    try:
        await asyncio.Future()  # Run indefinitely.
    except KeyboardInterrupt:
        await runtime.stop()
        print("\nChatbot stopped")


if __name__ == "__main__":
    asyncio.run(main()) 