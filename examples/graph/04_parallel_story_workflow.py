"""
Parallel Haiku Generation Example

This example demonstrates:
1. Two agents generating haikus in parallel
2. Simple synthesis combining their work
3. Basic validation of haiku structure
"""

import asyncio
from typing import List
from pydantic import BaseModel, Field

from babelgraph.core.agent import BaseAgent
from babelgraph.core.graph import Graph, NodeState
from babelgraph.core.graph.nodes import AgentNode, Node, terminal_node
from babelgraph.core.graph.nodes.base.node import parallel_node, ParallelExecutionPattern
from babelgraph.core.logging import LogComponent, get_logger

# Get workflow logger
logger = get_logger(LogComponent.WORKFLOW)

###################################################################
# Models
###################################################################

class Haiku(BaseModel):
    """A haiku with 5-7-5 syllable structure."""
    lines: List[str] = Field(..., min_items=3, max_items=3)
    theme: str = Field(default="nature")

class HaikuAnalysis(BaseModel):
    """Analysis of two haikus."""
    haikus: List[Haiku]
    common_themes: List[str]
    analysis: str

###################################################################
# Nodes
###################################################################

@parallel_node(pattern=ParallelExecutionPattern.CONCURRENT)
class HaikuGeneratorNode(AgentNode):
    """Generates a single haiku."""
    
    async def process(self, state: NodeState) -> str:
        """Generate haiku with LLM validation."""
        try:
            # Get or set default prompt
            if not self.get_message_input(state):
                self.set_message_input(
                    state, 
                    content=(
                        "Create a haiku following the 5-7-5 syllable pattern. "
                        "Return ONLY a JSON object with 'lines' (array of 3 strings) and 'theme' (string)."
                    ),
                    role="user"
                )
            
            # Generate haiku
            logger.agent(f"Generating {self.id}...")
            response = await self.agent._step(self.get_message_input(state)['content'])
            haiku = Haiku.model_validate(response)
            
            # Log the generated haiku
            logger.agent(f"Generated {self.id}:\n" + 
                        "\n".join(f"  {line}" for line in haiku.lines) +
                        f"\nTheme: {haiku.theme}")
            
            # Store result
            self.set_result(state, "haiku", haiku)
            return "success"
            
        except Exception as e:
            logger.error(f"Error generating haiku in {self.id}: {str(e)}")
            return "error"

@parallel_node(pattern=ParallelExecutionPattern.JOIN)
class SynthesisNode(AgentNode):
    """Combines and analyzes two haikus."""
    
    async def process(self, state: NodeState) -> str:
        """Synthesize haikus."""
        try:
            # Get both haikus from results
            haiku1 = Haiku.model_validate(state.results["haiku1"]["haiku"])
            haiku2 = Haiku.model_validate(state.results["haiku2"]["haiku"])
            
            logger.agent("Analyzing haikus:\n" +
                        f"Haiku 1 ({haiku1.theme}):\n" +
                        "\n".join(f"  {line}" for line in haiku1.lines) +
                        f"\n\nHaiku 2 ({haiku2.theme}):\n" +
                        "\n".join(f"  {line}" for line in haiku2.lines))
            
            # Create analysis prompt
            prompt = (
                "Analyze these two haikus and return a JSON object with 'haikus' (array), "
                "'common_themes' (array of strings), and 'analysis' (string):\n\n"
                f"Haiku 1 ({haiku1.theme}):\n{chr(10).join(haiku1.lines)}\n\n"
                f"Haiku 2 ({haiku2.theme}):\n{chr(10).join(haiku2.lines)}"
            )
            
            self.set_message_input(state, content=prompt, role="user")
            
            # Generate analysis
            response = await self.agent._step(self.get_message_input(state)['content'])
            analysis = HaikuAnalysis.model_validate(response)
            
            self.set_result(state, "analysis", analysis)
            return "success"
            
        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}")
            return "error"

@terminal_node
class EndNode(Node):
    """Display final results."""
    
    async def process(self, state: NodeState) -> None:
        """Show haikus and analysis."""
        try:
            analysis = state.results.get("synthesis", {}).get("analysis")
            if analysis:
                logger.agent(f"Final Analysis:\n{analysis.analysis}\n")
                logger.agent(f"Common Themes: {', '.join(analysis.common_themes)}")
                
        except Exception as e:
            logger.error(f"Error displaying results: {str(e)}")

async def main():
    """Run parallel haiku generation."""
    try:
        # Setup graph
        graph = Graph()
        state = NodeState()
        
        # Create workflow nodes
        haiku1 = HaikuGeneratorNode(
            id="haiku1",
            agent=BaseAgent(
                system_prompt="You are a haiku master specializing in nature themes.",
                response_model=Haiku
            )
        )
        haiku2 = HaikuGeneratorNode(
            id="haiku2", 
            agent=BaseAgent(
                system_prompt="You are a haiku master specializing in emotion themes.",
                response_model=Haiku
            )
        )
        synthesis = SynthesisNode(
            id="synthesis",
            agent=BaseAgent(
                system_prompt="You are a poetry analyst who finds connections between haikus.",
                response_model=HaikuAnalysis
            )
        )
        end = EndNode(id="end")
        
        # Add nodes to graph
        for node in [haiku1, haiku2, synthesis, end]:
            graph.add_node(node)
            
        # Define parallel subgraph
        graph.add_parallel_subgraph(
            name="haiku_generation",
            nodes=["haiku1", "haiku2"],
            join_node="synthesis",
            pattern=ParallelExecutionPattern.CONCURRENT
        )
        
        # Add final edge
        graph.add_edge("synthesis", "success", "end")
        
        # Set prompts
        haiku1.set_message_input(
            state,
            content="Create a haiku about nature.",
            role="user"
        )
        haiku2.set_message_input(
            state,
            content="Create a haiku about emotions.",
            role="user"
        )
        
        logger.agent("🌸 Generating Haikus")
        
        # Run the parallel subgraph
        await graph.run_parallel("haiku_generation", state)
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 