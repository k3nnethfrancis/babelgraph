"""
Reflection Workflow Example: Multi-step reasoning with Chain of Thought.

This example demonstrates:
1. Multi-step reasoning process
2. Streaming and non-streaming responses
3. Custom callbacks for progress display
4. Chain of thought prompting
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from babelgraph.core.agent import BaseAgent
from babelgraph.core.graph import Graph, NodeState
from babelgraph.core.graph.nodes import AgentNode, Node, terminal_node, state_handler
from babelgraph.core.logging import (
    LogComponent,
    BabelLoggingConfig,
    configure_logging,
    LogLevel,
    Colors,
    get_logger
)

###################################################################
# Response Models
###################################################################

class InitialReflection(BaseModel):
    """Initial thoughts and assessment."""
    key_points: List[str] = Field(..., description="Main points identified")
    initial_thoughts: str = Field(..., description="Initial assessment of the situation")
    questions_raised: List[str] = Field(..., description="Questions that need deeper analysis")

class DeepAnalysis(BaseModel):
    """Deeper analysis of the situation."""
    analysis_steps: List[str] = Field(..., description="Step-by-step analysis")
    implications: List[str] = Field(..., description="Key implications identified")
    conclusion: str = Field(..., description="Preliminary conclusion")

class FinalSynthesis(BaseModel):
    """Final synthesized response."""
    summary: str = Field(..., description="Concise summary of findings")
    key_insights: List[str] = Field(..., description="Key insights from analysis")
    final_answer: str = Field(..., description="Final conclusive answer")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in conclusion (0-1)")

###################################################################
# System Prompts
###################################################################

class ReflectionPrompt(BaseModel):
    """Base system prompt for reflection."""
    role: str = Field(default="Expert Analytical Thinker")
    approach: List[str] = Field(default=[
        "Break down complex problems",
        "Think step by step",
        "Consider multiple angles",
        "Draw logical conclusions"
    ])
    output_format: Dict[str, str] = Field(default={
        "key_points": "List of main points identified",
        "initial_thoughts": "Initial assessment text",
        "questions_raised": "List of questions for deeper analysis"
    })
    instructions: str = Field(
        default="""
        You are an Expert Analytical Thinker who breaks down complex relationships.
        
        Your task is to analyze the input and respond with a JSON object that EXACTLY matches this structure:
        {
            "key_points": [
                "First key point about the relationship",
                "Second key point about family connections",
                "Third key point about generational links"
            ],
            "initial_thoughts": "A detailed string explaining your initial assessment of the family relationships",
            "questions_raised": [
                "What is the exact generational relationship between X and Y?",
                "How are the family lines connected through Z?",
                "What other relationships might be relevant?"
            ]
        }

        CRITICAL REQUIREMENTS:
        1. Response must be ONLY the JSON object, no other text
        2. All three fields (key_points, initial_thoughts, questions_raised) are required
        3. key_points and questions_raised must be arrays of strings
        4. initial_thoughts must be a single string
        5. Use proper JSON formatting with double quotes and commas
        6. Arrays must use square brackets []
        7. No trailing commas
        8. No comments or explanations outside the JSON
        """.strip()
    )

class AnalysisPrompt(BaseModel):
    """System prompt for deep analysis."""
    role: str = Field(default="Deep Analysis Expert")
    approach: List[str] = Field(default=[
        "Examine implications thoroughly",
        "Connect related concepts",
        "Identify patterns",
        "Draw logical conclusions"
    ])
    output_format: Dict[str, str] = Field(default={
        "analysis_steps": "List of analysis steps",
        "implications": "List of key implications",
        "conclusion": "Preliminary conclusion text"
    })
    instructions: str = Field(
        default="""
        Analyze the input deeply and respond with a JSON object that EXACTLY matches this structure:
        {
            "analysis_steps": ["step 1", "step 2", ...],
            "implications": ["implication 1", "implication 2", ...],
            "conclusion": "Your detailed conclusion as a string"
        }

        The response MUST be valid JSON and include all required fields.
        Do not include any text outside the JSON structure.
        Ensure all strings are properly quoted and arrays use square brackets.
        """.strip()
    )

class SynthesisPrompt(BaseModel):
    """System prompt for final synthesis."""
    role: str = Field(default="Integration Expert")
    approach: List[str] = Field(default=[
        "Combine insights coherently",
        "Present clear explanations",
        "Use natural language",
        "Focus on understanding"
    ])
    instructions: str = Field(
        default="""
        Write your response as a clear, natural explanation that:
        1. Shows the reasoning step by step
        2. Makes the relationship easy to understand
        3. Uses clear language without technical terms
        
        Write this as a flowing narrative that will be streamed directly to the user.
        Do not use any special formatting or JSON structure.
        """.strip()
    )

# Configure logging for better visibility
logger = get_logger(LogComponent.WORKFLOW)
configure_logging(
    default_level=LogLevel.INFO,  # Set to INFO as default
    component_levels={
        LogComponent.WORKFLOW: LogLevel.INFO,   # Keep workflow info visible
        LogComponent.AGENT: LogLevel.INFO,      # Agent level info is useful
        LogComponent.GRAPH: LogLevel.INFO,      # Graph execution info
        LogComponent.NODES: LogLevel.INFO       # Node processing info
    }
)

@terminal_node
class EndNode(Node):
    """Terminal node that displays final results."""
    
    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """Display final results."""
        if "final_synthesis" in state.results:
            result = state.results["final_synthesis"].get("response")
            if isinstance(result, FinalSynthesis):
                print(f"\n{Colors.BOLD}Final Answer:{Colors.RESET}")
                print(f"{Colors.INFO}{result.final_answer}{Colors.RESET}")
                print(f"\n{Colors.DIM}Key Insights:{Colors.RESET}")
                for insight in result.key_insights:
                    print(f"- {insight}")
                print(f"\n{Colors.SUCCESS}Confidence: {result.confidence * 100:.1f}%{Colors.RESET}")
        return None

async def run_reflection_workflow() -> None:
    """
    Build and execute the reflection workflow.
    """
    graph = Graph()
    
    # Define reflection nodes with detailed logging callback
    def log_non_streaming_step(step_name: str):
        async def callback(state: NodeState, node_id: str) -> None:
            if node_id in state.results:
                result_data = state.results[node_id].get('response', {})
                print(f"\n{Colors.BOLD}🤔 {step_name}{Colors.RESET}")
                print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}")
                
                try:
                    if node_id == "initial_reflection":
                        result = InitialReflection.model_validate(result_data)
                        print(f"{Colors.INFO}Initial Thoughts:{Colors.RESET}")
                        print(result.initial_thoughts)
                        print(f"\n{Colors.INFO}Key Points:{Colors.RESET}")
                        for point in result.key_points:
                            print(f"- {point}")
                        print(f"\n{Colors.INFO}Questions Raised:{Colors.RESET}")
                        for q in result.questions_raised:
                            print(f"- {q}")
                    
                    elif node_id == "deep_reflection":
                        result = DeepAnalysis.model_validate(result_data)
                        print(f"{Colors.INFO}Analysis Steps:{Colors.RESET}")
                        for step in result.analysis_steps:
                            print(f"- {step}")
                        print(f"\n{Colors.INFO}Implications:{Colors.RESET}")
                        for imp in result.implications:
                            print(f"- {imp}")
                        print(f"\n{Colors.INFO}Conclusion:{Colors.RESET}")
                        print(result.conclusion)
                    
                    print(f"\n{Colors.DIM}{'─' * 50}{Colors.RESET}\n")
                except Exception as e:
                    logger.error(f"Error formatting {step_name} output: {str(e)}")
                    print(f"{Colors.ERROR}Error displaying results: {str(e)}{Colors.RESET}")
        return callback

    def log_streaming_completion(step_name: str):
        async def callback(state: NodeState, node_id: str) -> None:
            if node_id in state.results:
                result = state.results[node_id].get('response', {})
                if isinstance(result, FinalSynthesis):
                    print(f"\n\n{Colors.BOLD}Summary:{Colors.RESET}")
                    print(f"{Colors.INFO}{result.summary}{Colors.RESET}")
                    print(f"\n{Colors.SUCCESS}Confidence: {result.confidence * 100:.1f}%{Colors.RESET}")
        return callback

    # Define streaming callback for final synthesis
    async def stream_final_response(chunk: str, state: NodeState, node_id: str) -> None:
        """Stream the final synthesis response."""
        print(chunk, end="", flush=True)

    # Create workflow nodes
    graph.create_workflow(
        nodes={
            "initial_reflection": AnalyzerNode(
                id="initial_reflection",
                agent=BaseAgent(
                    system_prompt=ReflectionPrompt(),
                    response_model=InitialReflection,
                    json_mode=True,  # Add this to enforce JSON output
                    debug_mode=True  # Add debug mode to see agent internals
                ),
                metadata={
                    "on_complete": log_non_streaming_step("Initial Assessment")
                }
            ),
            "deep_reflection": DeepReflectionNode(
                id="deep_reflection", 
                agent=BaseAgent(
                    system_prompt=AnalysisPrompt(),
                    response_model=DeepAnalysis
                ),
                metadata={
                    "on_complete": log_non_streaming_step("Deep Analysis")
                }
            ),
            "final_synthesis": FinalSynthesisNode(
                id="final_synthesis",
                agent=BaseAgent(
                    system_prompt=SynthesisPrompt(),
                    stream=True
                ),
                stream=True,
                metadata={
                    "on_stream": stream_final_response
                }
            ),
            "end": EndNode(id="end")
        },
        flows=[
            ("initial_reflection", "success", "deep_reflection"),
            ("initial_reflection", "error", "end"),
            ("deep_reflection", "success", "final_synthesis"),
            ("deep_reflection", "error", "end"),
            ("final_synthesis", "success", "end"),
            ("final_synthesis", "error", "end")
        ]
    )
    
    # Initialize state
    state = NodeState()
    
    # Set input for initial reflection
    initial_node = graph.nodes["initial_reflection"]
    initial_node.set_message_input(
        state,
        content="""
        If doug's dad is bob's cousin, and bob's dad is dave's dad, what is the relationship between doug and dave?
        """.strip(),
        role="user"
    )
    
    print(f"\n{Colors.BOLD}🔍 Starting Analysis:{Colors.RESET}")
    print(f"{Colors.INFO}{initial_node.get_message_input(state)['content']}{Colors.RESET}")
    print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}\n")
    
    start_time = datetime.now()
    logger.info("Starting graph execution...")
    await graph.run("initial_reflection", state)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{Colors.SUCCESS}✨ Analysis Complete in {elapsed:.1f}s{Colors.RESET}\n")

class AnalyzerNode(AgentNode):
    """Node for initial reflection."""
    
    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the initial reflection."""
        try:
            # Get message input
            message = self.get_message_input(state)
            if not message:
                raise ValueError("No message input provided")

            # Log the input message
            logger.info(f"\n{Colors.BOLD}Input to Initial Reflection:{Colors.RESET}\n{message['content']}")

            try:
                # Process with agent - pass only the content
                logger.info(f"\n{Colors.BOLD}Calling Agent...{Colors.RESET}")
                response = await self.agent._step(message['content'])
                
                # Log the raw response
                logger.info(f"\n{Colors.BOLD}Raw Agent Response:{Colors.RESET}\n{response}")
                
                # Store result and return success
                self.set_result(state, "response", response)
                return "success"
                
            except Exception as agent_error:
                # Log detailed error info
                logger.error(f"\n{Colors.ERROR}Agent Error Details:{Colors.RESET}")
                logger.error(f"Error Type: {type(agent_error)}")
                logger.error(f"Error Message: {str(agent_error)}")
                
                # If it's a RetryError, try to extract the original error
                if hasattr(agent_error, 'last_attempt'):
                    try:
                        last_error = agent_error.last_attempt.exception()
                        if last_error:
                            logger.error(f"Original Error Type: {type(last_error)}")
                            logger.error(f"Original Error Message: {str(last_error)}")
                            
                            # If it's a ValidationError, log the errors
                            if hasattr(last_error, 'errors'):
                                logger.error("Validation Errors:")
                                for error in last_error.errors():
                                    logger.error(f"- {error}")
                    except Exception as e:
                        logger.error(f"Error extracting original error: {e}")
                
                raise
            
        except Exception as e:
            logger.error(f"\n{Colors.ERROR}Error in analyzer node:{Colors.RESET}\n{str(e)}\n{Colors.DIM}Type: {type(e)}{Colors.RESET}")
            state.add_error(self.id, str(e))
            return "error"

class DeepReflectionNode(AgentNode):
    """Node for deep analysis."""
    
    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the deep analysis."""
        try:
            # Get initial reflection results and parse as Pydantic model
            initial_data = state.results['initial_reflection'].get('response', {})
            if not initial_data:
                raise ValueError("No initial reflection results found")
            
            initial = InitialReflection.model_validate(initial_data)
            logger.debug(f"Initial reflection results: {initial.model_dump_json()}")
            
            # Format message for deep analysis with proper model access
            self.set_message_input(
                state,
                content=f"""
                Initial Assessment: {initial.initial_thoughts}
                Key Points: {', '.join(initial.key_points)}
                Questions to Address: {', '.join(initial.questions_raised)}
                
                Now analyze these relationships and implications step by step.
                Consider:
                1. Family tree connections
                2. Generational relationships
                3. Common ancestors
                """.strip(),
                role="user"
            )

            try:
                # Process with agent
                logger.info(f"\n{Colors.BOLD}Calling Deep Analysis Agent...{Colors.RESET}")
                response = await self.agent._step(self.get_message_input(state)['content'])
                
                # Log the raw response
                logger.info(f"\n{Colors.BOLD}Raw Deep Analysis Response:{Colors.RESET}\n{response}")
                
                # Store result and return success
                self.set_result(state, "response", response)
                return "success"
                
            except Exception as agent_error:
                # Log detailed error info
                logger.error(f"\n{Colors.ERROR}Deep Analysis Error Details:{Colors.RESET}")
                logger.error(f"Error Type: {type(agent_error)}")
                logger.error(f"Error Message: {str(agent_error)}")
                
                # If it's a RetryError, try to extract the original error
                if hasattr(agent_error, 'last_attempt'):
                    try:
                        last_error = agent_error.last_attempt.exception()
                        if last_error:
                            logger.error(f"Original Error Type: {type(last_error)}")
                            logger.error(f"Original Error Message: {str(last_error)}")
                            
                            # If it's a ValidationError, log the errors
                            if hasattr(last_error, 'errors'):
                                logger.error("Validation Errors:")
                                for error in last_error.errors():
                                    logger.error(f"- {error}")
                    except Exception as e:
                        logger.error(f"Error extracting original error: {e}")
                
                raise
                
        except Exception as e:
            logger.error(f"Error in deep reflection: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"

class FinalSynthesisNode(AgentNode):
    """Node for final synthesis with streaming output."""
    
    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the final synthesis with streaming."""
        try:
            # Get both previous results and parse as Pydantic models
            initial_data = state.results['initial_reflection'].get('response', {})
            analysis_data = state.results['deep_reflection'].get('response', {})
            
            if not initial_data or not analysis_data:
                raise ValueError("Missing required analysis results")
            
            initial = InitialReflection.model_validate(initial_data)
            analysis = DeepAnalysis.model_validate(analysis_data)
            
            logger.debug(f"Initial results: {initial.model_dump_json()}")
            logger.debug(f"Analysis results: {analysis.model_dump_json()}")
            
            # Format message for synthesis with proper model access
            self.set_message_input(
                state,
                content=f"""
                Based on our analysis:
                Initial Assessment: {initial.initial_thoughts}
                Deep Analysis: {analysis.conclusion}
                
                Please provide a final response that:
                1. Clearly explains the relationship between Doug and Dave
                2. Shows the reasoning step by step
                3. Makes it easy to understand for anyone reading
                
                Write this as a natural, flowing response that will be streamed to the user.
                """.strip(),
                role="user"
            )
            
            try:
                # Process with streaming enabled
                logger.info(f"\n{Colors.BOLD}Starting Final Synthesis...{Colors.RESET}")
                response = ""
                async for chunk, tool in self.agent._stream_step(self.get_message_input(state)['content']):
                    if tool:
                        # Handle tool execution if needed
                        logger.info(f"Tool execution: {tool._name()}")
                        result = tool.call()
                        logger.info(f"Tool result: {result}")
                    elif chunk:
                        # Stream the chunk if callback exists
                        if "on_stream" in self.metadata:
                            await self.metadata["on_stream"](chunk, state, self.id)
                        response += chunk
                
                # Store result with logging suppressed (since we streamed it)
                self.set_result(state, "response", response, suppress_logging=True)
                return "success"
                
            except Exception as agent_error:
                # Log detailed error info
                logger.error(f"\n{Colors.ERROR}Final Synthesis Error Details:{Colors.RESET}")
                logger.error(f"Error Type: {type(agent_error)}")
                logger.error(f"Error Message: {str(agent_error)}")
                
                # If it's a RetryError, try to extract the original error
                if hasattr(agent_error, 'last_attempt'):
                    try:
                        last_error = agent_error.last_attempt.exception()
                        if last_error:
                            logger.error(f"Original Error Type: {type(last_error)}")
                            logger.error(f"Original Error Message: {str(last_error)}")
                    except Exception as e:
                        logger.error(f"Error extracting original error: {e}")
                
                raise
            
        except Exception as e:
            logger.error(f"Error in final synthesis: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"

if __name__ == "__main__":
    asyncio.run(run_reflection_workflow())