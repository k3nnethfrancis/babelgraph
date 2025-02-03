"""Parallel Subgraphs Example

This example demonstrates:
1. Parallel execution of agent subgraphs
2. Structured story generation with multiple agents
3. Subgraph composition for story synthesis
4. Runtime injection for platform integration
5. Dynamic prompt updates and state management

The workflow:
- Two agents write stories in parallel (3 parts each)
- Each agent's story generation is its own subgraph
- A synthesis subgraph combines the stories
- Optional Discord notification of the final result
"""

import asyncio
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from babelgraph.core.agent import BaseAgent
from babelgraph.core.logging import (
    configure_logging,
    LogLevel,
    LogComponent,
    Colors,
    VerbosityLevel,
    get_logger
)
from babelgraph.core.graph import (
    Graph,
    Node,
    AgentNode
)
from babelgraph.extensions.discord.runtime import DiscordRuntime, DiscordRuntimeConfig

###################################################################
# Response Models
###################################################################

class StoryPart(BaseModel):
    """A structured part of a story."""
    content: str = Field(..., description="The actual story content")
    tone: str = Field(..., description="The emotional tone of this part")
    key_elements: List[str] = Field(..., description="Important story elements introduced")
    word_count: int = Field(..., description="Number of words in this part")

class SynthesizedStory(BaseModel):
    """The final synthesized story combining both agents' work."""
    title: str = Field(..., description="Title for the combined story")
    story_1: str = Field(..., description="Complete story from first agent")
    story_2: str = Field(..., description="Complete story from second agent")
    common_themes: List[str] = Field(..., description="Themes present in both stories")
    synthesis: str = Field(..., description="A brief synthesis comparing the stories")
    combined_word_count: int = Field(..., description="Total words in both stories")

class DiscordNotification(BaseModel):
    """Structure for Discord message."""
    channel: str = Field(..., description="Channel to post to")
    content: str = Field(..., description="Message content")
    success: bool = Field(..., description="Whether the message was sent")

###################################################################
# System Prompt Models
###################################################################

class StoryPartPrompt(BaseModel):
    """System prompt for story part generation."""
    role: str = Field(default="Creative Story Writer")
    style_guide: List[str] = Field(default=[
        "Write in a clear, engaging style",
        "Include vivid descriptions",
        "Focus on character development",
        "Maintain consistent tone"
    ])
    output_format: str = Field(default="""IMPORTANT: Return ONLY a JSON object with this structure:
{
    "content": "story text here",
    "tone": "emotional tone",
    "key_elements": ["element1", "element2", ...],
    "word_count": integer
}""")

class SynthesisPrompt(BaseModel):
    """System prompt for story synthesis."""
    role: str = Field(default="Story Editor and Synthesizer")
    analysis_points: List[str] = Field(default=[
        "Compare and contrast the stories",
        "Identify common themes",
        "Analyze different approaches",
        "Create a meaningful synthesis"
    ])
    output_format: str = Field(default="""IMPORTANT: Return ONLY a JSON object with this structure:
{
    "title": "combined story title",
    "story_1": "first complete story",
    "story_2": "second complete story",
    "common_themes": ["theme1", "theme2", ...],
    "synthesis": "synthesis text",
    "combined_word_count": integer
}""")

###################################################################
# Node Creation Helper
###################################################################

def create_story_node(
    id: str,
    part: str,
    next_node: Optional[str] = None,
    agent_number: int = 1
) -> AgentNode:
    """Create a story generation node."""
    return AgentNode(
        id=id,
        agent=BaseAgent(
            system_prompt=StoryPartPrompt(),
            response_model=StoryPart
        ),
        prompt=f"""Agent {agent_number} - Write the {part} of a story about a magical library.
Previous parts (if any): {{previous_parts}}
Make this part engaging and coherent with any existing content.""",
        response_model=StoryPart,
        input_map={"previous_parts": f"data.agent{agent_number}_story"},
        next_nodes={"default": next_node} if next_node else {}
    )

###################################################################
# Workflow Implementation
###################################################################

async def create_story_subgraph(agent_number: int) -> Graph:
    """Create a subgraph for one agent's story generation."""
    subgraph = Graph()
    
    # Create the three story part nodes
    beginning = create_story_node(
        f"beginning_{agent_number}",
        "beginning",
        f"middle_{agent_number}",
        agent_number
    )
    middle = create_story_node(
        f"middle_{agent_number}",
        "middle",
        f"end_{agent_number}",
        agent_number
    )
    end = create_story_node(
        f"end_{agent_number}",
        "end",
        None,
        agent_number
    )
    
    # Add nodes to subgraph
    for node in [beginning, middle, end]:
        subgraph.add_node(node)
    
    # Set entry point
    subgraph.set_entry_point(f"beginning_{agent_number}")
    
    return subgraph

async def create_synthesis_subgraph(discord_config: Optional[dict] = None) -> Graph:
    """Create a subgraph for story synthesis with optional Discord notification."""
    subgraph = Graph()
    
    # Create synthesis node (direct agent execution)
    synthesis_node = AgentNode(
        id="synthesis",
        agent=BaseAgent(
            system_prompt=SynthesisPrompt(),
            response_model=SynthesizedStory
        ),
        prompt="""Analyze and synthesize these two stories:
Story 1: {story1}
Story 2: {story2}
Compare their approaches, identify common themes, and create a brief synthesis.""",
        response_model=SynthesizedStory,
        input_map={
            "story1": "data.agent1_story",
            "story2": "data.agent2_story"
        },
        next_nodes={"default": "notify" if discord_config else "end"}
    )
    
    # Optionally add Discord notification node
    if discord_config and DISCORD_AVAILABLE:
        # Create Discord runtime
        discord_runtime = DiscordRuntime(
            config=DiscordRuntimeConfig(**discord_config)
        )
        
        # Create notification node with injected runtime
        notify_node = AgentNode(
            id="notify",
            agent=BaseAgent(),
            runtime=discord_runtime,
            response_model=DiscordNotification,
            prompt="Share synthesis results in Discord channel",
            input_map={"synthesis": "node.synthesis.response"},
            next_nodes={"default": "end"}
        )
        subgraph.add_node(notify_node)
    
    # Add terminal node
    end_node = TerminalNode(id="end")
    
    # Add nodes to subgraph
    subgraph.add_node(synthesis_node)
    subgraph.add_node(end_node)
    
    # Set entry point
    subgraph.set_entry_point("synthesis")
    
    return subgraph

###################################################################
# Logging Callbacks
###################################################################

def create_story_callback(agent_number: int):
    """Create a callback for story part completion."""
    async def callback(state: NodeState, node_id: str) -> None:
        if node_id in state.results:
            result = state.results[node_id].get('response', {})
            part = node_id.split('_')[0]  # beginning/middle/end
            print(f"\n{Colors.BOLD}📝 Agent {agent_number} - {part.title()}{Colors.RESET}")
            print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}")
            print(f"{Colors.INFO}{result.get('content', '')}{Colors.RESET}")
            print(f"{Colors.DIM}Tone: {result.get('tone', '')}")
            print(f"Words: {result.get('word_count', 0)}{Colors.RESET}")
    return callback

async def synthesis_callback(state: NodeState, node_id: str) -> None:
    """Callback for synthesis completion."""
    if node_id in state.results:
        result = state.results[node_id].get('response', {})
        print(f"\n{Colors.BOLD}✨ Final Synthesis{Colors.RESET}")
        print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}")
        print(f"{Colors.SUCCESS}Title: {result.get('title', '')}{Colors.RESET}")
        print(f"\n{Colors.INFO}Common Themes:{Colors.RESET}")
        for theme in result.get('common_themes', []):
            print(f"• {theme}")
        print(f"\n{Colors.INFO}Synthesis:{Colors.RESET}")
        print(result.get('synthesis', ''))
        print(f"\n{Colors.DIM}Total Words: {result.get('combined_word_count', 0)}{Colors.RESET}")

###################################################################
# Main Execution
###################################################################

async def main(discord_config: Optional[dict] = None):
    """Run the parallel story generation workflow."""
    print(f"\n{Colors.BOLD}🎭 Starting Parallel Story Generation{Colors.RESET}\n")
    
    # Configure logging
    configure_logging(LogLevel.INFO)
    logger = get_logger(LogComponent.WORKFLOW)
    
    # Create main graph
    main_graph = Graph()
    
    try:
        # Create and compose agent subgraphs
        agent1_graph = await create_story_subgraph(1)
        agent2_graph = await create_story_subgraph(2)
        synthesis_graph = await create_synthesis_subgraph(discord_config)
        
        # Add callbacks to nodes
        for node in agent1_graph.nodes.values():
            node.metadata["on_complete"] = create_story_callback(1)
        for node in agent2_graph.nodes.values():
            node.metadata["on_complete"] = create_story_callback(2)
        synthesis_graph.nodes["synthesis"].metadata["on_complete"] = synthesis_callback
        
        # Compose subgraphs into main graph
        main_graph.compose(agent1_graph, namespace="agent1")
        main_graph.compose(agent2_graph, namespace="agent2")
        main_graph.compose(synthesis_graph, namespace="final")
        
        # Initialize state
        state = NodeState()
        state.data["agent1_story"] = ""
        state.data["agent2_story"] = ""
        
        # Run agent subgraphs in parallel
        start_time = datetime.now()
        
        # Create tasks for parallel execution
        agent1_task = main_graph.run("agent1.beginning_1", state)
        agent2_task = main_graph.run("agent2.beginning_2", state)
        
        # Wait for both agent subgraphs to complete
        await asyncio.gather(agent1_task, agent2_task)
        
        # Run synthesis
        final_state = await main_graph.run("final.synthesis", state)
        
        # Show completion time
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n{Colors.SUCCESS}✨ Workflow completed in {elapsed:.1f}s{Colors.RESET}\n")
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example: To enable Discord notifications, uncomment and configure:
    # discord_config = {
    #     "bot_token": "YOUR_BOT_TOKEN",
    #     "channel_ids": ["YOUR_CHANNEL_ID"],
    #     "runtime_config": {
    #         "system_prompt": "You are a helpful Discord bot"
    #     }
    # }
    
    # Run without Discord for this example
    asyncio.run(main(None)) 