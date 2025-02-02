## Discord Bot Examples with Alchemist

### How to Run Your Discord Bot

1. Create a Discord bot and obtain the token.
2. In the root of the project, create a `.env` file and set the token (DISCORD_BOT_TOKEN).
3. Run the bot service with:  
   `python -m examples.discord.run_bot`

### How to Run the Chatbot

4. Open another terminal and run the chatbot:  
   `python -m examples.discord.chatbot`

### (Optional) Run the Local Discord Reader

If you want to run a local runtime chatbot that can answer questions about your Discord server, run:  
`python -m examples.discord.local_reader_agent`

---

Note: These examples now use our new system prompt approach. Instead of a full persona, a simple validated prompt (via Pydantic) guides the agent's behavior.