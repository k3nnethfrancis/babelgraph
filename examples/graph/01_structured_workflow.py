"""
Structured Workflow Example

This example demonstrates:
1. Agent-based message processing with structured outputs
2. Graph-based orchestration and state management
3. Pydantic models for type safety and validation
4. Clean state passing between nodes

The workflow:
- Takes input text
- Performs structured analysis (sentiment, topics, complexity)
- Makes a decision based on the analysis
- Demonstrates proper error handling and state management
"""

import asyncio
from typing import Optional, List, Dict, Any
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
# Response Models
###################################################################

class TextAnalysis(BaseModel):
    """Structured analysis of input text."""
    sentiment: str = Field(..., description="The emotional tone (positive/negative/neutral)")
    topics: List[str] = Field(..., description="Main topics identified in the text")
    word_count: int = Field(..., description="Number of words in the text")
    complexity: int = Field(..., ge=1, le=5, description="Text complexity score (1-5)")

class ActionDecision(BaseModel):
    """Structured decision based on text analysis."""
    action: str = Field(..., description="Action to take (summarize/elaborate/simplify)")
    reason: str = Field(..., description="Explanation for the chosen action")
    priority: int = Field(..., ge=1, le=3, description="Priority level (1-3)")

###################################################################
# System Prompt Models
###################################################################

class AnalyzerPrompt(BaseModel):
    """System prompt configuration for text analysis."""
    role: str = Field(default="Expert Text Analyzer")
    capabilities: List[str] = Field(default=[
        "Sentiment analysis",
        "Topic identification",
        "Complexity assessment",
        "Word count analysis"
    ])
    output_format: Dict[str, str] = Field(default={
        "sentiment": "positive/negative/neutral indicating emotional tone",
        "topics": "List of main topics found in text",
        "word_count": "Integer count of words",
        "complexity": "Integer score from 1-5"
    })
    instructions: str = Field(default="You MUST format your response as a JSON object matching the exact schema below. No other text or explanation should be included:\n{\n  \"sentiment\": \"string\",\n  \"topics\": [\"string\"],\n  \"word_count\": integer,\n  \"complexity\": integer\n}")

class DecisionPrompt(BaseModel):
    """System prompt configuration for decision making."""
    role: str = Field(default="Strategic Decision Maker")
    decision_criteria: List[str] = Field(default=[
        "Consider text complexity when choosing action",
        "Match action to sentiment",
        "Prioritize based on topic importance"
    ])
    output_format: Dict[str, str] = Field(default={
        "action": "summarize/elaborate/simplify indicating chosen action",
        "reason": "Explanation for the decision",
        "priority": "Integer priority from 1-3"
    })
    instructions: str = Field(default="You MUST format your response as a JSON object matching the exact schema below. No other text or explanation should be included:\n{\n  \"action\": \"string\",\n  \"reason\": \"string\",\n  \"priority\": integer\n}")

###################################################################
# Node Implementations
###################################################################

class AnalyzerNode(AgentNode):
    """Analyzes input text and provides structured analysis."""
    
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'logger', get_logger(LogComponent.WORKFLOW))
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process input text and generate analysis."""
        try:
            # Get input text from state
            text = state.data.get('input_text', '')
            if not text:
                raise ValueError("No input text provided")
            
            self.logger.info(f"[{self.id}] Analyzing text ({len(text)} chars)")
            
            message = f"""Analyze this text and provide a structured response:

{text}

Focus on:
1. Overall sentiment
2. Key topics
3. Text complexity
4. Word count

Remember to respond ONLY with a JSON object matching the required schema."""
            
            # Process with agent
            response = await self.agent._step(message)
            
            self.logger.debug(f"[{self.id}] Analysis response: {response}")
            
            # Store result
            state.results[self.id] = {"response": response}
            
            if isinstance(response, TextAnalysis):
                state.data['analysis'] = response
                self.logger.info(f"[{self.id}] Analysis complete: {response.sentiment} sentiment, {len(response.topics)} topics")
                return "default"
            
            self.logger.error(f"[{self.id}] Invalid response format")
            return "error"
            
        except Exception as e:
            self.logger.error(f"[{self.id}] Process error: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"

class DecisionNode(AgentNode):
    """Makes decisions based on text analysis."""
    
    logger: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'logger', get_logger(LogComponent.WORKFLOW))
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process analysis and make decision."""
        try:
            # Get analysis from state
            analysis = state.data.get('analysis')
            if not analysis:
                raise ValueError("No analysis data available")
            
            self.logger.info(f"[{self.id}] Making decision based on analysis")
            
            message = f"""Based on this analysis, decide what action to take:

Analysis Results:
- Sentiment: {analysis.sentiment}
- Topics: {', '.join(analysis.topics)}
- Complexity: {analysis.complexity}/5
- Word Count: {analysis.word_count}

Choose between:
- summarize: Create a concise summary
- elaborate: Add more detail or context
- simplify: Reduce complexity

Remember to respond ONLY with a JSON object matching the required schema."""
            
            # Process with agent
            response = await self.agent._step(message)
            
            self.logger.debug(f"[{self.id}] Decision response: {response}")
            
            # Store result
            state.results[self.id] = {"response": response}
            
            if isinstance(response, ActionDecision):
                state.data['decision'] = response
                self.logger.info(f"[{self.id}] Decision made: {response.action} (priority: {response.priority})")
                return "default"
            
            self.logger.error(f"[{self.id}] Invalid response format")
            return "error"
            
        except Exception as e:
            self.logger.error(f"[{self.id}] Process error: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"

###################################################################
# Example Usage
###################################################################

async def main():
    """Run example workflow."""
    # Configure logging
    configure_logging(
        LogLevel.INFO,
        component_levels={
            LogComponent.AGENT: LogLevel.DEBUG,
            LogComponent.WORKFLOW: LogLevel.INFO,
            LogComponent.GRAPH: LogLevel.INFO
        }
    )
    logger = get_logger(LogComponent.WORKFLOW)
    
    print(f"\n{Colors.BOLD}🔄 Starting Structured Workflow Example{Colors.RESET}\n")
    
    # Initialize state
    state = NodeState()
    
    # Example input
    state.data['input_text'] = """
    The new AI breakthrough in quantum computing represents a significant 
    advancement in the field. Researchers have successfully demonstrated 
    quantum supremacy in a practical application, potentially revolutionizing 
    cryptography and drug discovery. However, challenges remain in scaling 
    the technology and maintaining quantum coherence.
    """
    
    try:
        # Create graph
        graph = Graph()
        
        # Create nodes
        analyzer = AnalyzerNode(
            id="analyzer",
            agent=BaseAgent(
                system_prompt=AnalyzerPrompt(),
                response_model=TextAnalysis
            ),
            response_model=TextAnalysis,
            next_nodes={
                "default": "decision",
                "error": "end"
            }
        )
        
        decision = DecisionNode(
            id="decision",
            agent=BaseAgent(
                system_prompt=DecisionPrompt(),
                response_model=ActionDecision
            ),
            response_model=ActionDecision,
            next_nodes={
                "default": "end",
                "error": "end"
            }
        )
        
        end = TerminalNode(id="end")
        
        # Add nodes to graph
        for node in [analyzer, decision, end]:
            graph.add_node(node)
        
        # Set entry point
        graph.set_entry_point("analyzer", "start")
        
        # Run workflow
        print(f"\n{Colors.INFO}Input Text:{Colors.RESET}")
        print(state.data['input_text'].strip())
        
        final_state = await graph.run("start", state)
        
        # Display results
        if 'analysis' in final_state.data:
            print(f"\n{Colors.SUCCESS}Analysis Results:{Colors.RESET}")
            analysis = final_state.data['analysis']
            print(f"Sentiment: {analysis.sentiment}")
            print(f"Topics: {', '.join(analysis.topics)}")
            print(f"Complexity: {analysis.complexity}/5")
            print(f"Word Count: {analysis.word_count}")
        
        if 'decision' in final_state.data:
            print(f"\n{Colors.SUCCESS}Decision:{Colors.RESET}")
            decision = final_state.data['decision']
            print(f"Action: {decision.action}")
            print(f"Reason: {decision.reason}")
            print(f"Priority: {decision.priority}/3")
            
        if final_state.errors:
            print(f"\n{Colors.ERROR}Errors occurred:{Colors.RESET}")
            for node_id, error in final_state.errors.items():
                print(f"- {node_id}: {error}")
                
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())