"""
Simple Parallel Haiku Generation Example

Demonstrates:
1. Two agents generating haikus in parallel
2. Simple synthesis using Pydantic models
"""

import asyncio
from typing import List
from pydantic import BaseModel, Field

from babelgraph.core.agent import BaseAgent
from babelgraph.core.graph import Graph, NodeState
from babelgraph.core.graph.nodes import AgentNode, Node, terminal_node, state_handler
from babelgraph.core.logging import configure_logging, LogComponent, Colors, get_logger

logger = get_logger(LogComponent.WORKFLOW)

###################################################################
# Models
###################################################################

class Haiku(BaseModel):
    """A haiku with 5-7-5 syllable structure."""
    lines: List[str] = Field(..., min_items=3, max_items=3, description="The three lines of the haiku (5-7-5 syllables)")

class SyllableAnalysis(BaseModel):
    """Analysis of syllables for each line."""
    Line1: str = Field(..., description="Syllable breakdown of first line (5)")
    Line2: str = Field(..., description="Syllable breakdown of second line (7)")
    Line3: str = Field(..., description="Syllable breakdown of third line (5)")

class HaikuResponse(BaseModel):
    """Response from haiku generation."""
    haiku: Haiku
    theme: str
    syllable_analysis: SyllableAnalysis = Field(..., description="Detailed syllable breakdown for each line")

class SynthesisResponse(BaseModel):
    """Response from haiku synthesis."""
    haikus: List[Haiku]
    analysis: str

###################################################################
# System Prompts
###################################################################

class HaikuMasterConfig(BaseModel):
    """Configuration for the Haiku Master persona."""
    instruction: str = Field(
        default=(
            "You are a haiku master specializing in the 5-7-5 syllable pattern. "
            "Create haikus that strictly follow this structure, with careful syllable counting. "
            "Respond with a valid JSON object matching the schema exactly."
        ),
        description="Core instruction for the haiku master"
    )
    response_format: dict = Field(
        default={
            "haiku": {
                "lines": [
                    "autumn leaves fall down",
                    "dancing in the gentle breeze",
                    "nature's lullaby"
                ]
            },
            "theme": "seasonal change",
            "syllable_analysis": {
                "Line1": "(5): au-tumn (2) leaves (1) fall (1) down (1)",
                "Line2": "(7): dan-cing (2) in (1) the (1) gen-tle (2) breeze (1)",
                "Line3": "(5): na-ture's (2) lul-la-by (3)"
            }
        },
        description="Example of the expected JSON response format"
    )
    guidelines: List[str] = Field(
        default=[
            "Ensure exact syllable counts (5-7-5)",
            "Provide detailed syllable analysis for each line",
            "Use clear, vivid imagery",
            "Focus on the given theme",
            "Return only valid JSON matching the schema"
        ],
        description="Guidelines for haiku creation"
    )

class HaikuSynthesizerConfig(BaseModel):
    """Configuration for the Haiku Synthesizer persona."""
    instruction: str = Field(
        default=(
            "You are a poetry analyst specializing in finding connections between haikus. "
            "Analyze pairs of haikus and explain their thematic connections. "
            "Respond with a valid JSON object matching the schema exactly."
        ),
        description="Core instruction for the synthesizer"
    )
    response_format: dict = Field(
        default={
            "haikus": [
                {
                    "lines": [
                        "autumn leaves fall down",
                        "dancing in the gentle breeze",
                        "nature's lullaby"
                    ]
                },
                {
                    "lines": [
                        "heart beats like thunder",
                        "memories flood through my mind",
                        "peace comes at sunset"
                    ]
                }
            ],
            "analysis": "Both haikus explore themes of natural cycles and inner peace..."
        },
        description="Example of the expected JSON response format"
    )
    guidelines: List[str] = Field(
        default=[
            "Compare themes and imagery",
            "Identify emotional resonance",
            "Note contrasting elements",
            "Return only valid JSON matching the schema"
        ],
        description="Guidelines for analysis"
    )

# Initialize our configurations
HAIKU_MASTER = HaikuMasterConfig()
HAIKU_SYNTHESIZER = HaikuSynthesizerConfig()

###################################################################
# Nodes
###################################################################

class HaikuGeneratorNode(AgentNode):
    """Generates a single haiku."""
    
    @state_handler
    async def process(self, state: NodeState) -> str:
        """Generate haiku with LLM validation."""
        try:
            # Get or set default prompt
            if not self.get_message_input(state):
                self.set_message_input(state, content="Create a haiku.", role="user")
            
            # Generate and validate via LLM
            response = await self.agent._step(self.get_message_input(state)['content'])
            haiku = HaikuResponse.model_validate(response)
            
            # Log syllable analysis
            logger.info(f"\nSyllable Analysis:\n{haiku.syllable_analysis}")
            
            self.set_result(state, "haiku", haiku)
            return "success"
            
        except Exception as e:
            logger.error(f"Error generating haiku: {str(e)}")
            return "error"

class SynthesisNode(AgentNode):
    """Combines two haikus with analysis."""
    
    @state_handler
    async def process(self, state: NodeState) -> str:
        """Synthesize haikus."""
        try:
            # Get haikus
            haiku1 = state.results["haiku1"]["haiku"].haiku
            haiku2 = state.results["haiku2"]["haiku"].haiku
            
            # Create synthesis prompt
            prompt = f"Analyze these haikus:\n\n1:\n{chr(10).join(haiku1.lines)}\n\n2:\n{chr(10).join(haiku2.lines)}"
            self.set_message_input(state, content=prompt, role="user")
            
            # Generate synthesis
            response = await self.agent._step(self.get_message_input(state)['content'])
            synthesis = SynthesisResponse.model_validate(response)
            self.set_result(state, "synthesis", synthesis)
            return "success"
            
        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}")
            return "error"

@terminal_node
class EndNode(Node):
    """Display results."""
    
    @state_handler
    async def process(self, state: NodeState) -> None:
        """Show haikus and synthesis."""
        try:
            synthesis = state.results.get("synthesis", {}).get("synthesis")
            if synthesis:
                print(f"\n{Colors.BOLD}First Haiku:{Colors.RESET}")
                for line in synthesis.haikus[0].lines:
                    print(f"  {line}")
                    
                print(f"\n{Colors.BOLD}Second Haiku:{Colors.RESET}")
                for line in synthesis.haikus[1].lines:
                    print(f"  {line}")
                    
                print(f"\n{Colors.BOLD}Analysis:{Colors.RESET}")
                print(synthesis.analysis)
                
        except Exception as e:
            logger.error(f"Error displaying results: {str(e)}")

async def main():
    """Run parallel haiku generation."""
    configure_logging()
    
    try:
        # Setup graph
        graph = Graph()
        state = NodeState()
        
        # Create nodes
        graph.create_workflow(
            nodes={
                "haiku1": HaikuGeneratorNode(
                    id="haiku1",
                    agent=BaseAgent(
                        system_prompt=HAIKU_MASTER,
                        response_model=HaikuResponse
                    )
                ),
                "haiku2": HaikuGeneratorNode(
                    id="haiku2", 
                    agent=BaseAgent(
                        system_prompt=HAIKU_MASTER,
                        response_model=HaikuResponse
                    )
                ),
                "synthesis": SynthesisNode(
                    id="synthesis",
                    agent=BaseAgent(
                        system_prompt=HAIKU_SYNTHESIZER,
                        response_model=SynthesisResponse
                    )
                ),
                "end": EndNode(id="end")
            },
            flows=[
                ("haiku1", "success", "synthesis"),
                ("haiku2", "success", "synthesis"),
                ("synthesis", "success", "end"),
                ("haiku1", "error", "end"),
                ("haiku2", "error", "end"),
                ("synthesis", "error", "end")
            ]
        )
        
        # Set prompts
        graph.nodes["haiku1"].set_message_input(
            state,
            content="Create a nature haiku.",
            role="user"
        )
        graph.nodes["haiku2"].set_message_input(
            state,
            content="Create a haiku about emotions.",
            role="user"
        )
        
        # Run workflow
        print(f"\n{Colors.BOLD}🌸 Generating Haikus{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}\n")
        
        # Run haiku generation in parallel
        results = await asyncio.gather(
            graph.run("haiku1", state),
            graph.run("haiku2", state),
            return_exceptions=True
        )
        
        # Check for errors
        for result in results:
            if isinstance(result, Exception):
                raise result
        
        # Run synthesis
        await graph.run("synthesis", state)
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 