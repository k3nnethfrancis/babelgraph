"""Simple Discord channel analysis workflow.

This example demonstrates a basic graph workflow that:
1. Collects messages from a Discord channel
2. Analyzes the content using LLM
3. Formats the results

The workflow is intentionally simplified to focus on:
- Proper state management
- Content formatting
- Prompt templates
- Error handling
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from babelgraph.core.graph import Graph, NodeState, Node
from babelgraph.core.nodes import TerminalNode, AgentNode
from babelgraph.tools.discord_toolkit import DiscordTools
from mirascope.core import prompt_template, BaseMessageParam
from babelgraph.core.agent import BaseAgent
from babelgraph.core.logging import AlchemistLoggingConfig, VerbosityLevel, Colors

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChannelCollectorNode(Node):
    """Collects messages from a Discord channel."""
    
    id: str = "collect"
    channel_name: str = Field(description="Name of channel to collect from")
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the collection of Discord messages.
        
        Args:
            state: Current node state
            
        Returns:
            ID of next node to execute
        """
        logger.info(f"🔍 Collecting messages from #{self.channel_name}")
        
        try:
            # Initialize Discord tools
            tools = DiscordTools(
                channels={"ai-news": "1326422578340036689"},  # AI News channel
                categories={}
            )
            
            # Get recent messages
            messages = await tools.read_channel(
                channel_name=self.channel_name,
                after=datetime.now() - timedelta(days=1),
                limit=10
            )
            
            logger.info(f"📥 Collected {len(messages)} messages")
            logger.debug(f"Message structure: {messages[0] if messages else 'No messages'}")
            
            # Update state with results
            state.results[self.id] = {
                "messages": messages,
                "count": len(messages)
            }
            
            return self.next_nodes.get("next")
            
        except Exception as e:
            logger.error(f"❌ Error collecting messages: {str(e)}", exc_info=True)
            state.errors[self.id] = str(e)
            return None

@prompt_template()
def analysis_prompt(messages: str) -> list[BaseMessageParam]:
    """Template for analyzing Discord messages."""
    return [BaseMessageParam(
        role="user",
        content=f"""Analyze the following Discord messages and provide a summary:

Messages:
{messages}

Please provide:
1. Key topics discussed
2. Important links shared
3. Overall sentiment
4. Notable trends or patterns

Format your response in markdown."""
    )]

class ChannelAnalysisNode(AgentNode):
    """Analyzes collected Discord messages."""
    
    id: str = "analyze"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the analysis of collected messages.
        
        Args:
            state: Current node state with collected messages
            
        Returns:
            ID of next node to execute
        """
        logger.info("📊 Starting message analysis")
        
        try:
            # Get messages from previous state
            messages = state.results.get("collect", {}).get("messages", [])
            if not messages:
                raise ValueError("No messages found in state")
                
            logger.debug(f"Processing {len(messages)} messages")
            
            # Format messages for analysis
            formatted_messages = []
            for msg in messages:
                content = msg.get("content", "").strip()
                embeds = msg.get("embeds", [])
                
                if content:
                    formatted_messages.append(f"Content: {content}")
                
                for embed in embeds:
                    if title := embed.get("title"):
                        formatted_messages.append(f"Title: {title}")
                    if desc := embed.get("description"):
                        formatted_messages.append(f"Description: {desc}")
                    if url := embed.get("url"):
                        formatted_messages.append(f"URL: {url}")
            
            # Generate analysis prompt
            prompt = analysis_prompt(messages="\n".join(formatted_messages))
            logger.debug(f"Analysis prompt:\n{prompt}")
            
            # Get LLM response
            response = await self.llm.generate(prompt)
            logger.info("✅ Analysis complete")
            
            # Update state
            state.results[self.id] = {
                "analysis": response,
                "message_count": len(messages)
            }
            
            return self.next_nodes.get("next")
            
        except Exception as e:
            logger.error(f"❌ Error in analysis: {str(e)}", exc_info=True)
            state.errors[self.id] = str(e)
            return None

class ResultFormatterNode(Node):
    """Formats the analysis results."""
    
    id: str = "format"
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Format the analysis results.
        
        Args:
            state: Current node state with analysis
            
        Returns:
            ID of next node to execute
        """
        logger.info("📝 Formatting results")
        
        try:
            # Get analysis from state
            analysis = state.results.get("analyze", {}).get("analysis")
            if not analysis:
                raise ValueError("No analysis found in state")
            
            # Format results
            formatted = f"""# Discord Channel Analysis

## Overview
- Channel: #ai-news
- Messages Analyzed: {state.results['analyze']['message_count']}
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis
{analysis}
"""
            logger.debug(f"Formatted output:\n{formatted}")
            
            # Update state
            state.results[self.id] = {
                "formatted": formatted
            }
            
            return self.next_nodes.get("next")
            
        except Exception as e:
            logger.error(f"❌ Error formatting results: {str(e)}", exc_info=True)
            state.errors[self.id] = str(e)
            return None

async def main():
    """Run the channel analysis workflow."""
    
    # Create graph with logging config
    graph = Graph(
        logging_config=AlchemistLoggingConfig(
            level=VerbosityLevel.INFO,
            show_llm_messages=True,
            show_node_transitions=True,
            show_tool_calls=True
        )
    )
    
    # Create nodes
    collector = ChannelCollectorNode(
        id="collect",
        channel_name="ai-news",
        next_nodes={"default": "analyze"}
    )
    
    analyzer = AgentNode(  # Updated from LLMNode
        id="analyze",
        prompt_template=analysis_prompt,
        agent=BaseAgent(),
        input_map={"messages": "node.collect.messages"},
        next_nodes={"default": "format"}
    )
    
    formatter = ResultFormatterNode(
        id="format",
        next_nodes={"default": "end"}
    )
    
    end = TerminalNode(id="end")
    
    # Add nodes to graph
    for node in [collector, analyzer, formatter, end]:
        graph.add_node(node)
    
    # Add entry point
    graph.add_entry_point("start", "collect")
    
    # Initialize state
    state = NodeState()
    
    print(f"\n{Colors.BOLD}📊 Starting Discord Channel Analysis:{Colors.RESET}")
    print(f"{Colors.INFO}Channel: #ai-news{Colors.RESET}\n")
    
    start_time = datetime.now()
    final_state = await graph.run("start", state)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if final_state.errors:
        print(f"\n{Colors.ERROR}❌ Analysis failed with errors:{Colors.RESET}")
        for node_id, error in final_state.errors.items():
            print(f"{Colors.ERROR}Node {node_id}: {error}{Colors.RESET}")
    else:
        print(f"\n{Colors.SUCCESS}✨ Analysis Complete in {elapsed:.1f}s{Colors.RESET}")
        if "format" in final_state.results:
            print(final_state.results["format"]["formatted"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 