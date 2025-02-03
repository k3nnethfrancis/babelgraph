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
from babelgraph.core.graph import (
    Graph,
    NodeState,
    AgentNode,
    TerminalNode
)
from babelgraph.core.logging import (
    LogComponent,
    BabelLoggingConfig,
    configure_logging,
    LogLevel
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
    instructions: str = Field(default="Respond ONLY with a JSON object matching the schema. No other text.")

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
    instructions: str = Field(default="Respond ONLY with a JSON object matching the schema. No other text.")

class SynthesisPrompt(BaseModel):
    """System prompt for final synthesis."""
    role: str = Field(default="Integration Expert")
    approach: List[str] = Field(default=[
        "Combine insights coherently",
        "Prioritize key findings",
        "Present clear conclusions",
        "Assess confidence level"
    ])
    output_format: Dict[str, str] = Field(default={
        "summary": "Concise summary text",
        "key_insights": "List of key insights",
        "final_answer": "Final conclusion text",
        "confidence": "Float between 0 and 1"
    })
    instructions: str = Field(default="Respond ONLY with a JSON object matching the schema. No other text.")

# Configure minimal logging
configure_logging(
    default_level=LogLevel.INFO,
    component_levels={
        LogComponent.WORKFLOW: LogLevel.INFO,
        LogComponent.AGENT: LogLevel.INFO,
        LogComponent.GRAPH: LogLevel.DEBUG,
        LogComponent.NODES: LogLevel.INFO
    }
)

logger = get_logger(LogComponent.WORKFLOW)

async def run_reflection_workflow() -> None:
    """
    Build and execute the reflection workflow.
    """
    graph = Graph(
        logging_config=AlchemistLoggingConfig(
            level=VerbosityLevel.INFO,
            show_llm_messages=True,
            show_node_transitions=True,
            show_tool_calls=True
        )
    )
    
    # Define reflection nodes with detailed logging callback
    def log_non_streaming_step(step_name: str):
        async def callback(state: NodeState, node_id: str) -> None:
            if node_id in state.results:
                result = state.results[node_id].get('response', {})
                print(f"\n{Colors.BOLD}🤔 {step_name}{Colors.RESET}")
                print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}")
                
                if isinstance(result, InitialReflection):
                    print(f"{Colors.INFO}Initial Thoughts:{Colors.RESET}")
                    print(result.initial_thoughts)
                    print(f"\n{Colors.INFO}Questions Raised:{Colors.RESET}")
                    for q in result.questions_raised:
                        print(f"- {q}")
                
                elif isinstance(result, DeepAnalysis):
                    print(f"{Colors.INFO}Analysis Steps:{Colors.RESET}")
                    for step in result.analysis_steps:
                        print(f"- {step}")
                    print(f"\n{Colors.INFO}Conclusion:{Colors.RESET}")
                    print(result.conclusion)
                
                timing = state.results[node_id].get('timing', 0)
                print(f"\n{Colors.DIM}Time: {timing:.1f}s{Colors.RESET}")
                print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}\n")
        return callback

    def log_streaming_completion(step_name: str):
        async def callback(state: NodeState, node_id: str) -> None:
            if node_id in state.results:
                result = state.results[node_id].get('response', {})
                if isinstance(result, FinalSynthesis):
                    print(f"\n\n{Colors.BOLD}Summary:{Colors.RESET}")
                    print(f"{Colors.INFO}{result.summary}{Colors.RESET}")
                    print(f"\n{Colors.SUCCESS}Confidence: {result.confidence * 100:.1f}%{Colors.RESET}")
                timing = state.results[node_id].get('timing', 0)
                print(f"\n{Colors.DIM}Time: {timing:.1f}s{Colors.RESET}")
        return callback

    # Define streaming callback for final synthesis
    async def stream_final_response(chunk: str, state: NodeState, node_id: str) -> None:
        """Stream the final synthesis response."""
        try:
            # Try to parse as JSON to check if it's the full response
            import json
            data = json.loads(chunk)
            if isinstance(data, dict) and "final_answer" in data:
                # This is the structured response, don't print it
                return
        except json.JSONDecodeError:
            # This is a streaming chunk, print it
            print(chunk, end="", flush=True)

    initial_reflection = AgentNode(
        id="initial_reflection",
        agent=BaseAgent(
            system_prompt=ReflectionPrompt(),
            response_model=InitialReflection
        ),
        response_model=InitialReflection,
        prompt="""Step 1: Initial Assessment
Consider this question carefully: {user_input}

Break it down and provide your initial thoughts. Focus on:
1. Key relationships mentioned
2. Important details to consider
3. Questions that need deeper analysis""",
        input_map={"user_input": "data.input_text"},
        next_nodes={"default": "deep_reflection", "error": "end"},
        metadata={
            "on_complete": log_non_streaming_step("Initial Assessment"),
            "message_key": "message"
        }
    )

    deep_reflection = AgentNode(
        id="deep_reflection",
        agent=BaseAgent(
            system_prompt=AnalysisPrompt(),
            response_model=DeepAnalysis
        ),
        response_model=DeepAnalysis,
        prompt="""Step 2: Deep Analysis
Initial assessment: {initial_thoughts}

Now analyze the relationships and implications step by step. Consider:
1. Family tree connections
2. Generational relationships
3. Common ancestors""",
        input_map={"initial_thoughts": "node.initial_reflection.response"},
        next_nodes={"default": "final_synthesis", "error": "end"},
        metadata={
            "on_complete": log_non_streaming_step("Deep Analysis"),
            "message_key": "message"
        }
    )

    final_synthesis = AgentNode(
        id="final_synthesis",
        agent=BaseAgent(
            system_prompt=SynthesisPrompt(),
            response_model=FinalSynthesis
        ),
        response_model=FinalSynthesis,
        prompt="""Step 3: Final Synthesis
Initial Assessment: {initial_thoughts}
Deep Analysis: {deep_analysis}

Synthesize a final response that:
1. Clearly states the relationship
2. Explains the reasoning
3. Indicates confidence level""",
        stream=True,
        input_map={
            "initial_thoughts": "node.initial_reflection.response",
            "deep_analysis": "node.deep_reflection.response"
        },
        next_nodes={"default": "end", "error": "end"},
        metadata={
            "on_stream": stream_final_response,
            "on_complete": log_streaming_completion("Final Synthesis"),
            "message_key": "message"
        }
    )

    # Add nodes and set entry point with debug logging
    logger.info("Adding nodes to graph...")
    for node in [initial_reflection, deep_reflection, final_synthesis, TerminalNode(id="end")]:
        logger.info(f"Adding node: {node.id}")
        graph.add_node(node)
    
    logger.info("Setting entry point...")
    graph.set_entry_point("initial_reflection", "default")
    
    # Run workflow with better progress indication
    state = NodeState()
    state.data["input_text"] = """
    If doug's dad is bob's cousin, and bob's dad is dave's dad, what is the relationship between doug and dave?
    """.strip()
    
    # Set initial message
    state.data["message"] = state.data["input_text"]
    
    print(f"\n{Colors.BOLD}🔍 Starting Analysis:{Colors.RESET}")
    print(f"{Colors.INFO}{state.data['input_text']}{Colors.RESET}")
    print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}\n")
    
    start_time = datetime.now()
    logger.info("Starting graph execution...")
    final_state = await graph.run("default", state)
    
    # Display final synthesis if available
    if final_state and "final_synthesis" in final_state.results:
        result = final_state.results["final_synthesis"].get("response")
        if isinstance(result, FinalSynthesis):
            print(f"\n{Colors.BOLD}Final Answer:{Colors.RESET}")
            print(f"{Colors.INFO}{result.final_answer}{Colors.RESET}")
            print(f"\n{Colors.DIM}Key Insights:{Colors.RESET}")
            for insight in result.key_insights:
                print(f"- {insight}")
            print(f"\n{Colors.SUCCESS}Confidence: {result.confidence * 100:.1f}%{Colors.RESET}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{Colors.SUCCESS}✨ Analysis Complete in {elapsed:.1f}s{Colors.RESET}\n")

if __name__ == "__main__":
    asyncio.run(run_reflection_workflow())