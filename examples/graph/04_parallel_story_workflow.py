"""Parallel Haiku Generation Workflow

This example demonstrates:
1. Parallel execution of haiku-writing agents
2. Multi-step haiku generation with validation
3. Subgraph composition for synthesis
4. Structured outputs with Pydantic

The workflow:
- Two agents write haikus in parallel (3 lines each)
- Each haiku is validated for quality and format
- Haikus are synthesized with reflection on form and meaning
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from babelgraph.core.agent import BaseAgent
from babelgraph.core.logging import (
    configure_logging,
    LogLevel,
    LogComponent,
    Colors,
    get_logger
)
from babelgraph.core.graph import (
    Graph,
    NodeState,
    AgentNode,
    TerminalNode
)

###################################################################
# Data Models
###################################################################

class HaikuPart(BaseModel):
    """A structured part of a haiku."""
    content: str = Field(..., description="The actual haiku line")
    syllables: int = Field(..., description="Number of syllables in this line")
    season_reference: Optional[str] = Field(None, description="Reference to season (kigo) if present")
    cutting_word: Optional[str] = Field(None, description="Cutting word (kireji) if present")
    themes: List[str] = Field(..., description="Themes explored in this line")

class HaikuValidation(BaseModel):
    """Validation results for a haiku line."""
    is_valid: bool = Field(..., description="Whether the line meets haiku criteria")
    syllable_count_correct: bool = Field(..., description="Whether syllable count matches haiku rules")
    feedback: str = Field(..., description="Specific feedback on what needs improvement")
    quality_score: float = Field(..., ge=0, le=1, description="Quality score between 0 and 1")
    suggested_improvements: List[str] = Field(..., description="List of suggested improvements")

class SynthesisReflection(BaseModel):
    """Reflection on the combined haikus."""
    combined_themes: List[str] = Field(..., description="Themes present across both haikus")
    follows_haiku_rules: bool = Field(..., description="Whether the combined work follows haiku rules")
    season_references: List[str] = Field(..., description="All season references found")
    cutting_words: List[str] = Field(..., description="All cutting words used")
    meets_criteria: bool = Field(..., description="Whether the synthesis meets quality bar")
    improvement_needed: Optional[str] = Field(None, description="Areas needing improvement")

class FinalSynthesis(BaseModel):
    """Final synthesized output combining both haikus."""
    title: str = Field(..., description="Title for the combined work")
    haiku_1: str = Field(..., description="First complete haiku")
    haiku_2: str = Field(..., description="Second complete haiku")
    combined_analysis: str = Field(..., description="Analysis of both haikus")
    syllable_structure: List[int] = Field(..., description="Syllable count for each line")
    seasonal_elements: List[str] = Field(..., description="Seasonal elements found")
    final_reflection: str = Field(..., description="Final thoughts on the exercise")

###################################################################
# System Prompts
###################################################################

class HaikuWriterPrompt(BaseModel):
    """System prompt for haiku generation."""
    role: str = Field(default="Haiku Master")
    style_guide: List[str] = Field(default=[
        "Follow traditional 5-7-5 syllable pattern",
        "Include seasonal reference (kigo) when possible",
        "Use cutting words (kireji) effectively",
        "Create vivid natural imagery",
        "Maintain simplicity and immediacy"
    ])
    output_format: Dict[str, str] = Field(default={
        "content": "The actual haiku line",
        "syllables": "Number of syllables in the line",
        "season_reference": "Reference to season if present (optional)",
        "cutting_word": "Cutting word if present (optional)",
        "themes": "List of themes in this line"
    })
    instructions: str = Field(default="You MUST format your response as a JSON object with the exact fields specified in the schema. No other text or explanation should be included.")

class ValidatorPrompt(BaseModel):
    """System prompt for haiku validation."""
    role: str = Field(default="Haiku Validator")
    validation_criteria: List[str] = Field(default=[
        "Correct syllable count (5-7-5)",
        "Presence of seasonal reference",
        "Effective use of cutting words",
        "Clear imagery",
        "Proper form and structure"
    ])
    output_format: Dict[str, str] = Field(default={
        "is_valid": "true/false indicating if the line meets all criteria",
        "syllable_count_correct": "true/false indicating if syllable count matches requirements",
        "feedback": "Detailed feedback explaining the validation results",
        "quality_score": "Float between 0.0 and 1.0 indicating overall quality",
        "suggested_improvements": "List of specific suggestions for improvement"
    })
    instructions: str = Field(default="You MUST format your response as a JSON object matching the exact schema below. No other text or explanation should be included:\n{\n  \"is_valid\": boolean,\n  \"syllable_count_correct\": boolean,\n  \"feedback\": \"string\",\n  \"quality_score\": float,\n  \"suggested_improvements\": [\"string\"]\n}")

class ReflectionPrompt(BaseModel):
    """System prompt for haiku synthesis reflection."""
    role: str = Field(default="Haiku Master Analyst")
    analysis_points: List[str] = Field(default=[
        "Syllable structure",
        "Seasonal elements",
        "Cutting words",
        "Imagery comparison",
        "Thematic unity"
    ])
    output_format: Dict[str, str] = Field(default={
        "combined_themes": "List of themes present across both haikus",
        "follows_haiku_rules": "true/false indicating if both haikus follow proper structure",
        "season_references": "List of seasonal references found in both haikus",
        "cutting_words": "List of cutting words used in both haikus",
        "meets_criteria": "true/false indicating if the synthesis meets quality standards",
        "improvement_needed": "String describing needed improvements, if any"
    })
    instructions: str = Field(default="You MUST format your response as a JSON object matching the exact schema below. No other text or explanation should be included:\n{\n  \"combined_themes\": [\"string\"],\n  \"follows_haiku_rules\": boolean,\n  \"season_references\": [\"string\"],\n  \"cutting_words\": [\"string\"],\n  \"meets_criteria\": boolean,\n  \"improvement_needed\": \"string\"\n}")

###################################################################
# Node Implementations
###################################################################

class HaikuNode(AgentNode):
    """Node for generating haiku lines."""
    
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'logger', get_logger(LogComponent.WORKFLOW))
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Generate a haiku line."""
        try:
            agent_id = self.metadata["agent_id"]
            line_number = self.metadata["line"]
            themes = self.metadata["themes"]
            previous_lines = state.data.get(f"{agent_id}_haiku", "")
            
            self.logger.info(f"[{self.id}] Generating line {line_number} for {agent_id}")
            self.logger.debug(f"[{self.id}] Themes: {themes}")
            
            # Construct message with syllable requirements
            syllable_count = 5 if line_number in [1, 3] else 7
            previous_context = f"\nPrevious lines:\n{previous_lines}" if previous_lines else ""
            message = f"""Write line {line_number} of a haiku with exactly {syllable_count} syllables.
Themes to incorporate: {', '.join(themes)}.{previous_context}
Remember to include seasonal references (kigo) and cutting words (kireji) where appropriate."""

            response = await self.agent._step(message)
            
            self.logger.debug(f"[{self.id}] Raw response: {response}")
            
            if not isinstance(response, HaikuPart):
                self.logger.error(f"[{self.id}] Invalid response format")
                try:
                    if isinstance(response, str):
                        import json
                        parsed = json.loads(response)
                        response = HaikuPart(**parsed)
                except Exception as e:
                    self.logger.error(f"[{self.id}] Parsing failed: {str(e)}")
                    return "error"
            
            state.results[self.id] = {"response": response}
            
            if isinstance(response, HaikuPart):
                current_haiku = state.data.get(f"{agent_id}_haiku", "")
                state.data[f"{agent_id}_haiku"] = current_haiku + "\n" + response.content
                self.logger.info(f"[{self.id}] Generated line with {response.syllables} syllables")
                if response.season_reference:
                    self.logger.debug(f"[{self.id}] Season reference: {response.season_reference}")
                if response.cutting_word:
                    self.logger.debug(f"[{self.id}] Cutting word: {response.cutting_word}")
                
                # Display progress
                print(f"\n{Colors.INFO}🎋 {agent_id} Haiku Progress:{Colors.RESET}")
                print(state.data[f"{agent_id}_haiku"].strip())
            
            return "default"
            
        except Exception as e:
            self.logger.error(f"[{self.id}] Process error: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"

class HaikuValidatorNode(AgentNode):
    """Node for validating haiku lines."""
    
    logger: Optional[Any] = Field(default=None, exclude=True)
    max_retries: int = Field(default=5)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'logger', get_logger(LogComponent.WORKFLOW))
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Validate a haiku line."""
        try:
            haiku_node_id = self.id.replace("validate_", "")
            haiku_response = state.results[haiku_node_id]["response"]
            
            if not isinstance(haiku_response, HaikuPart):
                self.logger.error(f"[{self.id}] Invalid haiku response type")
                return "error"
                
            line_content = haiku_response.content
            agent_id = self.metadata["agent_id"]
            line_number = self.metadata["line"]
            expected_syllables = 5 if line_number in [1, 3] else 7
            
            # Track validation attempts
            attempts = state.data.get(f"{self.id}_attempts", 0)
            state.data[f"{self.id}_attempts"] = attempts + 1
            
            if attempts >= self.max_retries:
                self.logger.warning(f"[{self.id}] Max validation retries ({self.max_retries}) reached")
                return "valid"  # Continue with best effort
            
            message = f"""Validate this haiku line and respond ONLY with a JSON object matching the exact schema below.
Do not include any other text or explanation outside the JSON object.

Line to validate: "{line_content}"

Line number: {line_number}
Expected syllables: {expected_syllables}
Previous lines:
{state.data.get(f"{agent_id}_haiku", "")}

Validation criteria:
1. Exactly {expected_syllables} syllables
2. Clear imagery
3. Proper seasonal reference (if present)
4. Effective use of cutting words (if present)
5. Natural flow and rhythm

Required JSON response format:
{{
    "is_valid": boolean,
    "syllable_count_correct": boolean,
    "feedback": "string with detailed feedback",
    "quality_score": float between 0.0 and 1.0,
    "suggested_improvements": ["list", "of", "improvement", "suggestions"]
}}"""

            response = await self.agent._step(message)
            
            self.logger.debug(f"[{self.id}] Validation response: {response}")
            
            state.results[self.id] = {"response": response}
            
            if isinstance(response, HaikuValidation):
                if not response.syllable_count_correct:
                    self.logger.warning(f"[{self.id}] Incorrect syllable count")
                    if attempts < self.max_retries:
                        print(f"\n{Colors.YELLOW}Retrying line {line_number} for {agent_id} (attempt {attempts + 1}/{self.max_retries}){Colors.RESET}")
                return "valid" if response.is_valid else "invalid"
            
            self.logger.error(f"[{self.id}] Invalid validation response format")
            return "error"
            
        except Exception as e:
            self.logger.error(f"[{self.id}] Process error: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"

class ReflectionNode(AgentNode):
    """Node for haiku reflection and synthesis."""
    
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'logger', get_logger(LogComponent.WORKFLOW))
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Reflect on and synthesize haikus."""
        try:
            haiku1 = state.data.get("agent1_haiku", "")
            haiku2 = state.data.get("agent2_haiku", "")
            
            self.logger.info(f"[{self.id}] Starting reflection on haikus")
            
            message = f"""Analyze these two haikus:

Haiku 1:
{haiku1}

Haiku 2:
{haiku2}

Evaluate their adherence to haiku rules and artistic merit.
Check for proper 5-7-5 syllable pattern, seasonal references (kigo),
cutting words (kireji), and thematic unity.

Format your response as a JSON object with these fields:
{{
    "combined_themes": ["themes", "across", "both"],
    "follows_haiku_rules": true/false,
    "season_references": ["spring", "autumn", etc],
    "cutting_words": ["ya", "kana", etc],
    "meets_criteria": true/false,
    "improvement_needed": "suggestions if needed"
}}"""

            response = await self.agent._step(message)
            
            self.logger.debug(f"[{self.id}] Reflection response: {response}")
            
            state.results[self.id] = {"response": response}
            
            if isinstance(response, SynthesisReflection):
                if not response.follows_haiku_rules:
                    self.logger.warning(f"[{self.id}] Haiku rules not followed: {response.improvement_needed}")
                return "meets_criteria" if response.meets_criteria else "needs_improvement"
            return "error"
            
        except Exception as e:
            self.logger.error(f"[{self.id}] Process error: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"

###################################################################
# Workflow Implementation
###################################################################

async def create_haiku_subgraph(agent_id: int, themes: List[str]) -> Graph:
    """Create a subgraph for one agent's haiku generation."""
    subgraph = Graph()
    
    # Create haiku generation nodes
    lines = [1, 2, 3]  # Three lines of haiku
    nodes = []
    
    # Initialize agents
    writer = BaseAgent(
        system_prompt=HaikuWriterPrompt(),
        response_model=HaikuPart
    )
    validator = BaseAgent(
        system_prompt=ValidatorPrompt(),
        response_model=HaikuValidation
    )
    
    for i, line in enumerate(lines):
        # Create haiku node
        haiku_node = HaikuNode(
            id=f"line{line}_{agent_id}",
            agent=writer,
            response_model=HaikuPart,
            metadata={
                "line": line,
                "themes": themes,
                "agent_id": f"agent{agent_id}"
            }
        )
        
        # Create validation node
        validator_node = HaikuValidatorNode(
            id=f"validate_line{line}_{agent_id}",
            agent=validator,
            response_model=HaikuValidation,
            metadata={
                "line": line,
                "agent_id": f"agent{agent_id}"
            }
        )
        
        nodes.extend([haiku_node, validator_node])
        
        # Connect nodes
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            validator_node.next_nodes["valid"] = f"line{next_line}_{agent_id}"
            validator_node.next_nodes["invalid"] = f"line{line}_{agent_id}"
        else:
            validator_node.next_nodes["valid"] = "/synthesis.reflection"
            validator_node.next_nodes["invalid"] = f"line{line}_{agent_id}"
        
        haiku_node.next_nodes["default"] = f"validate_line{line}_{agent_id}"
    
    for node in nodes:
        subgraph.add_node(node)
    
    subgraph.set_entry_point(f"line1_{agent_id}", "start")
    
    return subgraph

async def create_synthesis_subgraph() -> Graph:
    """Create a subgraph for haiku synthesis."""
    subgraph = Graph()
    
    synthesizer = BaseAgent(
        system_prompt=ReflectionPrompt(),
        response_model=SynthesisReflection
    )
    
    reflection = ReflectionNode(
        id="reflection",
        agent=synthesizer,
        response_model=SynthesisReflection,
        next_nodes={
            "meets_criteria": "final_synthesis",
            "needs_improvement": "reflection",
            "error": "end"
        }
    )
    
    final = ReflectionNode(
        id="final_synthesis",
        agent=synthesizer,
        response_model=FinalSynthesis,
        next_nodes={"default": "end"}
    )
    
    end = TerminalNode(id="end")
    
    for node in [reflection, final, end]:
        subgraph.add_node(node)
    
    subgraph.set_entry_point("reflection", "start")
    
    return subgraph

###################################################################
# Main Execution
###################################################################

async def main():
    """Run the parallel haiku generation workflow."""
    configure_logging(
        LogLevel.INFO,
        component_levels={
            LogComponent.AGENT: LogLevel.DEBUG,
            LogComponent.WORKFLOW: LogLevel.INFO,
            LogComponent.GRAPH: LogLevel.INFO
        }
    )
    logger = get_logger(LogComponent.WORKFLOW)
    
    start_time = datetime.now()
    
    print(f"\n{Colors.BOLD}🎋 Starting Parallel Haiku Generation{Colors.RESET}\n")
    
    try:
        main_graph = Graph()
        
        # Create haikus with traditional themes
        agent1_graph = await create_haiku_subgraph(1, ["spring", "cherry blossoms", "dawn"])
        agent2_graph = await create_haiku_subgraph(2, ["autumn", "falling leaves", "dusk"])
        synthesis_graph = await create_synthesis_subgraph()
        
        main_graph.compose(agent1_graph, "agent1")
        main_graph.compose(agent2_graph, "agent2")
        main_graph.compose(synthesis_graph, "synthesis")
        
        state = NodeState()
        state.data["agent1_haiku"] = ""
        state.data["agent2_haiku"] = ""
        
        print(f"\n{Colors.BOLD}Starting parallel haiku generation...{Colors.RESET}")
        agent1_task = main_graph.run("agent1.start", state)
        agent2_task = main_graph.run("agent2.start", state)
        
        # Wait for both haikus to complete
        await asyncio.gather(agent1_task, agent2_task)
        
        print(f"\n{Colors.BOLD}Both haikus completed:{Colors.RESET}")
        print(f"\n{Colors.INFO}Spring Haiku:{Colors.RESET}")
        print(state.data["agent1_haiku"].strip())
        print(f"\n{Colors.INFO}Autumn Haiku:{Colors.RESET}")
        print(state.data["agent2_haiku"].strip())
        
        print(f"\n{Colors.BOLD}Starting synthesis...{Colors.RESET}")
        
        # Increase max_iterations for synthesis
        final_state = await main_graph.run("synthesis.start", state, max_iterations=10)
        
        # Display final results
        if final_state and "synthesis.final_synthesis" in final_state.results:
            final_result = final_state.results["synthesis.final_synthesis"]["response"]
            if isinstance(final_result, FinalSynthesis):
                print(f"\n{Colors.SUCCESS}✨ Final Synthesis:{Colors.RESET}")
                print(f"\nTitle: {final_result.title}")
                print(f"\nHaiku 1:\n{Colors.INFO}{final_result.haiku_1}{Colors.RESET}")
                print(f"\nHaiku 2:\n{Colors.INFO}{final_result.haiku_2}{Colors.RESET}")
                print(f"\nAnalysis:\n{final_result.combined_analysis}")
                print(f"\nSeasonal Elements: {', '.join(final_result.seasonal_elements)}")
                print(f"\nReflection: {final_result.final_reflection}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n{Colors.SUCCESS}✨ Haikus completed in {elapsed:.1f}s{Colors.RESET}\n")
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 