"""
Structured Workflow Example

This example demonstrates:
1. Agent-based message processing with structured outputs
2. Basic graph-based workflow
3. Pydantic models for type safety

The workflow:
- Takes input text
- Analyzes it (sentiment, topics, complexity)
- Makes a simple decision based on the analysis
"""

import asyncio
from typing import Optional, List
from pydantic import BaseModel, Field

from babelgraph.core.agent import BaseAgent
from babelgraph.core.logging import (
    configure_logging,
    LogLevel,
    LogComponent,
    Colors,
    get_logger
)
from babelgraph.core.graph.nodes import AgentNode, Node, terminal_node, state_handler
from babelgraph.core.graph import Graph, NodeState

class TextAnalysis(BaseModel):
    """Structured analysis of input text."""
    sentiment: str = Field(..., description="The emotional tone (positive/negative/neutral)")
    topics: List[str] = Field(..., description="Main topics identified")
    complexity: int = Field(..., ge=1, le=5, description="Complexity score (1-5)")

class ActionDecision(BaseModel):
    """Decision based on text analysis."""
    action: str = Field(..., description="Action to take (summarize/elaborate/simplify)")
    reason: str = Field(..., description="Explanation for the decision")

class AnalyzerNode(AgentNode):
    """Node for analyzing text with structured output."""
    
    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """Process the input text."""
        message = self.get_message_input(state)
        if not message:
            raise ValueError("No message input provided")
        return await super().process(state)

class DecisionNode(AgentNode):
    """Node for making decisions based on analysis."""
    
    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """Make a decision based on the analysis."""
        analysis = self.get_result(state, 'analyzer', 'response')
        if not analysis:
            return "error"
        
        # Pass analysis to the agent
        self.set_message_input(
            state,
            content=f"Analysis: {analysis}",
            role="user"
        )
        return await super().process(state)

@terminal_node
class EndNode(Node):
    """Terminal node that displays results."""
    
    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """Display final results."""
        analysis = self.get_result(state, 'analyzer', 'response')
        decision = self.get_result(state, 'decision', 'response')
        
        if analysis:
            print(f"\n{Colors.SUCCESS}Analysis:{Colors.RESET}")
            print(f"Sentiment: {analysis.get('sentiment')}")
            print(f"Topics: {', '.join(analysis.get('topics', []))}")
            print(f"Complexity: {analysis.get('complexity')}/5")
        
        if decision:
            print(f"\n{Colors.SUCCESS}Decision:{Colors.RESET}")
            print(f"Action: {decision.get('action')}")
            print(f"Reason: {decision.get('reason')}")
        
        return None

async def main():
    """Run the structured workflow."""
    configure_logging(default_level=LogLevel.INFO)
    logger = get_logger(LogComponent.WORKFLOW)
    logger.info("Starting workflow...")
    
    try:
        graph = Graph()
        state = NodeState()
        
        # Create simple workflow
        graph.create_workflow(
            nodes={
                "analyzer": AnalyzerNode(
                    id="analyzer",
                    agent=BaseAgent(
                        system_prompt="You are a text analysis expert.",
                        response_model=TextAnalysis
                    )
                ),
                "decision": DecisionNode(
                    id="decision",
                    agent=BaseAgent(
                        system_prompt="You are a decision-making expert.",
                        response_model=ActionDecision
                    )
                ),
                "end": EndNode(id="end")
            },
            flows=[
                ("analyzer", "success", "decision"),
                ("analyzer", "error", "end"),
                ("decision", "success", "end"),
                ("decision", "error", "end")
            ]
        )
        
        # Set input text
        analyzer = graph.nodes["analyzer"]
        analyzer.set_message_input(
            state,
            content="""
            Artificial intelligence has revolutionized many industries,
            from healthcare to transportation. Machine learning models
            can now perform complex tasks with remarkable accuracy.
            """.strip(),
            role="user"
        )
        
        # Run workflow
        print(f"\n{Colors.INFO}Input Text:{Colors.RESET}")
        print(analyzer.get_message_input(state)["content"])
        
        await graph.run("analyzer", state)
                
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())