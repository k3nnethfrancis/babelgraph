"""
Tutoring Workflow Example: 
A tutoring session between an unhinged math professor and a student that demonstrates:
1. Multi-agent interaction with personas
2. Structured outputs using response models
3. Tool usage through professor agent
4. State tracking for game rules
5. Streaming responses
"""

import asyncio
import random
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from mirascope.core import BaseTool, BaseDynamicConfig, Messages
from babelgraph.core.agent import BaseAgent
from babelgraph.core.config import AgentConfig, RuntimeConfig
from alchemist.ai.base.logging import (
    configure_logging,
    LogLevel,
    LogComponent,
    Colors,
    VerbosityLevel,
    AlchemistLoggingConfig,
    get_logger
)
from babelgraph.core.graph import Graph
from babelgraph.core.nodes.agent import AgentNode
from babelgraph.core.nodes.terminal import TerminalNode
from babelgraph.core.state import NodeState, NodeStatus
from babelgraph.core.runtime import GraphRuntime, GraphRuntimeConfig
from babelgraph.core.config import GraphConfig

# Get logger for workflow component
logger = get_logger(LogComponent.WORKFLOW)

###################################################################
# Response Models for Structured Output
###################################################################

class StudentQuizAnswer(BaseModel):
    """Structured response for student quiz answers."""
    answer: str = Field(..., description="The mathematical answer")
    explanation: str = Field(..., description="Step-by-step explanation of the solution")
    confidence: int = Field(..., description="Confidence level (1-5)", ge=1, le=5)

class StudentQuestion(BaseModel):
    """Structured response for student questions."""
    question: str = Field(..., description="The specific math question")
    needs_calculation: bool = Field(..., description="Whether the question requires calculation")
    wants_book_lookup: bool = Field(..., description="Whether the student wants to reference the book")

class TutoringResponse(BaseModel):
    """Professor's structured tutoring response."""
    explanation: str = Field(..., description="The professor's detailed mathematical explanation")
    needs_book_check: bool = Field(default=False, description="Whether a book check is needed")
    calculation_needed: bool = Field(default=False, description="Whether a calculation is needed")
    turn_count: int = Field(..., description="Current turn number (1-3)")
    book_lookup_available: bool = Field(..., description="Whether book lookup is still available")

class BookCheckResponse(BaseModel):
    """Structured output for book check responses."""
    topic: str = Field(..., description="The topic to look up (Calculus or Algebra)")
    specific_concept: str = Field(..., description="The specific concept to look up")

class QuizResponse(BaseModel):
    """Professor's structured quiz evaluation."""
    correct: bool = Field(..., description="Whether the answer is correct")
    explanation: str = Field(..., description="Explanation of why the answer is correct/incorrect")
    grade: str = Field(..., description="Final grade (Pass/Fail)")

###################################################################
# Custom Tools
###################################################################

class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations and basic derivatives."""
    expression: str = Field(..., description="The mathematical expression to evaluate")

    @classmethod
    def _name(cls) -> str:
        return "calculator"

    @classmethod
    def _description(cls) -> str:
        return "Evaluate a mathematical expression or compute basic derivatives"

    def call(self) -> float:
        """Evaluate mathematical expressions including basic derivatives."""
        if not self.expression or not isinstance(self.expression, str):
            raise ValueError("Invalid or empty expression")
            
        expr = self.expression.strip().lower()
        logger.debug(f"Calculating expression: {expr}")
        
        try:
            # Handle basic derivatives
            if expr.startswith('d/dx') or 'derivative' in expr:
                # Extract the function part
                func = expr.replace('d/dx', '').replace('derivative of', '').strip('() ')
                logger.debug(f"Extracting derivative of: {func}")
                
                # Handle basic power rule
                if '^' in func:
                    base, power = func.split('^')
                    if base.strip() == 'x':
                        try:
                            power = int(power)
                            result = f"{power}x^{power-1}" if power != 2 else "2x"
                            logger.debug(f"Power rule result: {result}")
                            return result
                        except ValueError:
                            raise ValueError(f"Invalid power in expression: {power}")
                
                # Handle basic cases
                if func == 'x':
                    return '1'
                elif func == 'x^2':
                    return '2x'
                elif func == 'x^3':
                    return '3x^2'
                    
                raise ValueError(f"Cannot compute derivative of {func}")
                
            # Regular arithmetic
            try:
                # Clean up the expression
                expr = expr.replace('^', '**')  # Replace ^ with **
                expr = expr.replace('×', '*')   # Handle multiplication symbol
                expr = expr.replace('÷', '/')   # Handle division symbol
                
                # Basic security check
                if any(unsafe in expr for unsafe in ['import', 'eval', 'exec', '__']):
                    raise ValueError("Invalid expression: contains unsafe operations")
                    
                logger.debug(f"Evaluating arithmetic: {expr}")
                result = eval(expr, {"__builtins__": {}}, {})
                
                if isinstance(result, (int, float)):
                    return result
                else:
                    raise ValueError("Expression did not evaluate to a number")
                    
            except Exception as e:
                raise ValueError(f"Invalid arithmetic expression: {expr}") from e
                
        except Exception as e:
            logger.error(f"Calculator error: {str(e)}")
            raise ValueError(f"Calculation failed: {str(e)}")

class BookLookupTool(BaseTool):
    """Tool for looking up information in textbooks."""
    topic: str = Field(..., description="The topic to look up (Calculus or Algebra)")
    concept: str = Field(..., description="The specific concept to look up")

    @classmethod
    def _name(cls) -> str:
        return "book_lookup"

    @classmethod
    def _description(cls) -> str:
        return "Look up a mathematical concept in the available textbooks"

    async def call(self) -> Dict[str, str]:
        """Look up a concept in the mock textbook database."""
        # Mock book database
        books = {
            "calculus": {
                "derivatives": "The derivative measures the rate of change of a function.",
                "integrals": "The integral computes the area under a curve.",
            },
            "algebra": {
                "equations": "Solve for x by isolating the variable.",
                "functions": "A function maps each input to exactly one output.",
            }
        }
        return {
            "topic": self.topic,
            "concept": self.concept,
            "content": books.get(self.topic.lower(), {}).get(self.concept.lower(), "Concept not found")
        }

###################################################################
# Persona Models and System Prompts
###################################################################

class StudentPersona(BaseModel):
    """Persona model for the student."""
    name: str
    learning_style: str
    knowledge_level: str
    goals: str

class ProfessorPersona(BaseModel):
    """Persona model for the unhinged math professor."""
    name: str
    personality: str
    goals: str

# Create persona instances
student_persona = StudentPersona(
    name="Alex",
    learning_style="Curious and analytical, likes to understand the 'why'",
    knowledge_level="High school math, comfortable with algebra, learning calculus",
    goals="Master key math concepts through interactive learning and practical examples"
)

professor_persona = ProfessorPersona(
    name="Dr. Mathias Eccentric",
    personality="Unhinged, passionate, and slightly deranged, with a penchant for shocking outbursts and unconventional teaching methods",
    goals="To force these unappreciative kids to grasp essential math skills—even if it means using wild and bizarre techniques. Must track turns (3 max) and enforce book lookup limit (1 per session)."
)

# Define system prompts
student_system_prompt = f"""
You are {student_persona.name}, a student learning mathematics.
Learning Style: {student_persona.learning_style}
Knowledge Level: {student_persona.knowledge_level}
Goals: {student_persona.goals}

Rules of the tutoring session:
1. You can ask 3 questions total
2. You can request ONE book lookup during the entire session
3. You can ask the professor to perform calculations (they have a calculator)
4. After 3 questions, you must take a quiz

ALWAYS format your responses as JSON matching this structure:
{{
    "question": "Your specific math question",
    "needs_calculation": true/false,
    "wants_book_lookup": true/false
}}

Example response:
{{
    "question": "Can you help me understand derivatives?",
    "needs_calculation": false,
    "wants_book_lookup": true
}}
"""

professor_system_prompt = f"""
You are {professor_persona.name}, an unhinged math professor.
Personality: {professor_persona.personality}
Goals: {professor_persona.goals}

When tutoring:
1. Track turns (max 3) and book lookup usage (max 1)
2. Use your calculator tool when students request calculations
3. Be passionate but slightly unhinged in your explanations
4. After 3 turns, give a quiz related to what was discussed

ALWAYS format your responses as JSON matching this structure:
{{
    "explanation": "Your detailed mathematical explanation (be eccentric!)",
    "needs_book_check": true/false,
    "calculation_needed": true/false,
    "turn_count": 1-3,
    "book_lookup_available": true/false
}}

Example response:
{{
    "explanation": "AHA! The quadratic formula is like a beautiful dance of numbers!",
    "needs_book_check": false,
    "calculation_needed": true,
    "turn_count": 1,
    "book_lookup_available": true
}}
"""

###################################################################
# Custom Node Implementations
###################################################################

class StudentNode(AgentNode):
    """Node for handling student interactions."""
    
    def _extract_calculation_expression(self, question: str) -> Optional[str]:
        """Extract a mathematical expression from a question.
        
        Uses common mathematical indicators and basic parsing to identify
        calculation requests in natural language questions.
        
        Args:
            question: The student's natural language question
            
        Returns:
            Optional[str]: The mathematical expression to calculate if found,
                         None otherwise
                         
        Examples:
            >>> node._extract_calculation_expression("What is 2 + 2?")
            "2 + 2"
            >>> node._extract_calculation_expression("Calculate the derivative of x^2")
            "x^2"
            >>> node._extract_calculation_expression("Tell me about calculus")
            None
        """
        # Common mathematical operation indicators
        indicators = [
            "calculate",
            "solve",
            "evaluate",
            "compute",
            "find the value",
            "what is",
            "derivative of",
            "integral of"
        ]
        
        question = question.lower()
        
        # Check if this is a calculation request
        if any(indicator in question for indicator in indicators):
            # Try to extract expression after common markers
            for word in ["of", "is", "="]:
                if word in question:
                    expression = question.split(word)[-1].strip()
                    # Clean up common terminators
                    expression = expression.replace("?", "").strip()
                    return expression
                    
        return None

    async def process(self, state: NodeState) -> Optional[str]:
        """Process student's question or quiz answer."""
        # Check if we're in quiz mode
        if state.data.get('in_quiz_mode', False):
            # Get quiz question from state
            quiz_question = state.data.get('quiz_question', '')
            
            try:
                # Use structured output with StudentQuizAnswer model
                self.agent.response_model = StudentQuizAnswer
                answer = await self.agent._step(
                    f"The professor asks: {quiz_question}\n"
                    "Please provide your answer."
                )
                
                # Store answer in state
                state.data.update({
                    'student_quiz_answer': answer.answer,
                    'answer_explanation': answer.explanation,
                    'confidence_level': answer.confidence
                })
                
                return "quiz_evaluation"
                
            except Exception as e:
                logger.error(f"Quiz answer processing error: {str(e)}")
                state.add_error(self.id, f"Quiz answer processing error: {str(e)}")
                return "end"
        
        # Normal question mode
        try:
            # Use structured output with StudentQuestion model
            self.agent.response_model = StudentQuestion
            question = await self.agent._step(
                "What would you like to ask the professor?\n"
                "Remember the rules: 3 questions max, 1 book lookup allowed."
            )
            
            # Extract calculation expression if needed
            calculation_expression = self._extract_calculation_expression(question.question)
            logger.debug(f"Extracted calculation expression: {calculation_expression}")
            
            # Update state with question
            state.data.update({
                'current_question': question.question,
                'student_wants_calculation': question.needs_calculation,
                'student_wants_book': question.wants_book_lookup,
                'calculation_expression': calculation_expression
            })
            
            return "professor"
            
        except Exception as e:
            logger.error(f"Question processing error: {str(e)}")
            state.add_error(self.id, f"Question processing error: {str(e)}")
            return "end"

class ProfessorNode(AgentNode):
    """Node for handling professor's responses and calculations."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process student's question and provide a response."""
        try:
            # Get current state
            current_turn = state.data.get('turn_count', 1)
            book_lookup_available = state.data.get('book_lookup_available', True)
            question = state.data.get('current_question', '')
            
            # Use structured output with TutoringResponse model
            self.agent.response_model = TutoringResponse
            response = await self.agent._step(
                f"The student asks: {question}\n"
                f"Current turn: {current_turn}/3\n"
                f"Book lookup {'available' if book_lookup_available else 'used'}"
            )
            
            # Update state
            state.data.update({
                'turn_count': response.turn_count,
                'book_lookup_available': response.book_lookup_available,
                'professor_needs_book': response.needs_book_check,
                'professor_needs_calculation': response.calculation_needed
            })
            
            # Handle book lookup
            if response.needs_book_check:
                if not book_lookup_available:
                    return "end"  # No more book lookups allowed
                return "book_check"
                
            # Handle calculation
            if response.calculation_needed:
                if calculation_expression := state.data.get('calculation_expression'):
                    try:
                        result = await self.agent.use_tool(CalculatorTool, {"expression": calculation_expression})
                        state.data['calculation_result'] = result
                    except Exception as e:
                        logger.error(f"Calculator error: {str(e)}")
                        
            # Check if we should move to quiz mode
            if current_turn >= 3:
                state.data['in_quiz_mode'] = True
                return "quiz"
                
            return "student"
            
        except Exception as e:
            logger.error(f"Professor response error: {str(e)}")
            state.add_error(self.id, f"Professor response error: {str(e)}")
            return "end"

class BookLookupNode(AgentNode):
    """Node for handling book lookups with structured output."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process book lookup request."""
        try:
            # Get structured response for book lookup
            response = await self.agent._step(
                "The student wants to look up information in a book.\n"
                "Format your response as JSON matching this structure:\n"
                "{\n"
                '    "topic": "Either Calculus or Algebra",\n'
                '    "specific_concept": "The concept to look up"\n'
                "}\n\n"
                "Available topics and concepts:\n"
                "Calculus: derivatives, integrals\n"
                "Algebra: equations, functions\n\n"
                "Example response:\n"
                "{\n"
                '    "topic": "Calculus",\n'
                '    "specific_concept": "derivatives"\n'
                "}"
            )
            
            # Parse response
            lookup = BookCheckResponse.model_validate_json(response)
            
            # Use book lookup tool
            result = await self.agent.use_tool(
                BookLookupTool,
                {
                    "topic": lookup.topic,
                    "concept": lookup.specific_concept
                }
            )
            
            # Store results and mark lookup as used
            state.data.update({
                "book_info": result["content"],
                "book_lookup_used": True
            })
            
        except Exception as e:
            state.add_error(self.id, f"Book lookup error: {str(e)}")
            
        return "professor"

class QuizEvaluationNode(AgentNode):
    """Node for evaluating student's quiz answers."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Evaluate student's quiz answer."""
        try:
            # Get evaluation using structured output
            result = await self.agent._step(
                f"Evaluate the student's quiz answer using the QuizResponse model.\n"
                f"Format your evaluation as JSON matching this structure:\n"
                "{\n"
                '    "correct": true/false,\n'
                '    "explanation": "Your detailed explanation of the grade",\n'
                '    "grade": "Pass/Fail"\n'
                "}\n\n"
                "Example response:\n"
                "{\n"
                '    "correct": true,\n'
                '    "explanation": "Perfect! The derivative of x^2 is indeed 2x",\n'
                '    "grade": "Pass"\n'
                "}"
            )
            
            # Parse response
            evaluation = QuizResponse.model_validate_json(result)
            
            # Store results
            state.data.update({
                'quiz_correct': evaluation.correct,
                'quiz_explanation': evaluation.explanation,
                'quiz_grade': evaluation.grade
            })
            
        except Exception as e:
            state.add_error(self.id, f"Quiz evaluation error: {str(e)}")
            
        return "end"

###################################################################
# Main Workflow Setup
###################################################################

async def tutoring_workflow() -> None:
    """Creates and runs a tutoring session using GraphRuntime with structured outputs."""
    
    # Configure logging
    configure_logging(
        default_level=LogLevel.DEBUG,  # Change to DEBUG for more details
        component_levels={
            LogComponent.GRAPH: LogLevel.DEBUG,
            LogComponent.NODES: LogLevel.DEBUG,
            LogComponent.AGENT: LogLevel.DEBUG,  # Add agent logging
            LogComponent.RUNTIME: LogLevel.DEBUG,  # Add runtime logging
            LogComponent.TOOLS: LogLevel.DEBUG,  # Add tool logging
        }
    )
    
    # Initialize session state
    state = NodeState()
    # Initialize each state value individually with logging
    logger.info("Initializing session state...")
    state.set_data("turn", 0)
    state.set_data("book_lookup_used", False)
    state.set_data("in_quiz_mode", False)
    logger.debug(f"Initial state: {state.data}")
    
    # Create main graph
    logger.info("Creating tutoring workflow graph...")
    main_graph = Graph()
    
    # Create nodes with proper prompts and configurations
    logger.info("Creating workflow nodes...")
    student_node = StudentNode(
        id="student",
        prompt=student_system_prompt,
        agent=None,  # Will be injected by runtime
        stream=True,
        response_model=StudentQuestion if not state.data.get('in_quiz_mode', False) else StudentQuizAnswer,
        json_mode=True,
        next_nodes={
            "professor": "professor",
            "quiz_evaluation": "quiz_evaluation",
            "end": "end"
        }
    )
    logger.debug(f"Created student node with response model: {student_node.response_model}")
    
    professor_node = ProfessorNode(
        id="professor",
        prompt=professor_system_prompt,
        agent=None,  # Will be injected by runtime
        stream=True,
        response_model=TutoringResponse,
        json_mode=True,
        next_nodes={
            "student": "student",
            "book_lookup": "book_lookup",
            "end": "end"
        }
    )
    logger.debug(f"Created professor node with response model: {professor_node.response_model}")
    
    book_lookup_node = BookLookupNode(
        id="book_lookup",
        prompt=(
            "The student wants to look up information in a book.\n"
            "Format your response as JSON matching this structure:\n"
            "{\n"
            '    "topic": "Either Calculus or Algebra",\n'
            '    "specific_concept": "The concept to look up"\n'
            "}\n\n"
            "Available topics and concepts:\n"
            "Calculus: derivatives, integrals\n"
            "Algebra: equations, functions\n\n"
            "Example response:\n"
            "{\n"
            '    "topic": "Calculus",\n'
            '    "specific_concept": "derivatives"\n'
            "}"
        ),
        agent=None,  # Will be injected by runtime
        stream=True,
        response_model=BookCheckResponse,
        json_mode=True,
        next_nodes={
            "professor": "professor",
            "end": "end"
        }
    )
    logger.debug(f"Created book lookup node with response model: {book_lookup_node.response_model}")
    
    quiz_evaluation_node = QuizEvaluationNode(
        id="quiz_evaluation",
        prompt=(
            "Evaluate the student's quiz answer using the QuizResponse model.\n"
            "Format your evaluation as JSON matching this structure:\n"
            "{\n"
            '    "correct": true/false,\n'
            '    "explanation": "Your detailed explanation of the grade",\n'
            '    "grade": "Pass/Fail"\n'
            "}\n\n"
            "Example response:\n"
            "{\n"
            '    "correct": true,\n'
            '    "explanation": "Perfect! The derivative of x^2 is indeed 2x",\n'
            '    "grade": "Pass"\n'
            "}"
        ),
        agent=None,  # Will be injected by runtime
        stream=True,
        response_model=QuizResponse,
        json_mode=True,
        next_nodes={"end": "end"}
    )
    logger.debug(f"Created quiz evaluation node with response model: {quiz_evaluation_node.response_model}")
    
    end_node = TerminalNode(id="end")
    
    # Add nodes to graph
    logger.info("Adding nodes to graph...")
    for node in [student_node, professor_node, book_lookup_node, quiz_evaluation_node, end_node]:
        main_graph.add_node(node)
    main_graph.add_entry_point("start", "student")
    logger.debug("Graph structure complete")
    
    # Create runtime handlers for streaming output
    async def stream_handler(chunk: str, node_id: str, state: NodeState):
        """Handle streaming output from nodes."""
        # Format based on which agent is speaking
        prefix = {
            "student": f"{Colors.BLUE}Student:{Colors.RESET} ",
            "professor": f"{Colors.GREEN}Professor:{Colors.RESET} ",
            "book_lookup": f"{Colors.YELLOW}📚 Book:{Colors.RESET} ",
            "quiz_evaluation": f"{Colors.MAGENTA}Quiz:{Colors.RESET} "
        }.get(node_id, "")
        
        print(f"{prefix}{chunk}", end="", flush=True)
        
    async def node_handler(node_id: str, status: NodeStatus, state: NodeState):
        """Handle node status changes and display relevant information."""
        if status == NodeStatus.STARTING:
            # Show turn information
            if node_id == "student" and not state.data.get('in_quiz_mode', False):
                turn = state.data.get('turn', 0)
                print(f"\n{Colors.CYAN}[Turn {turn + 1}/3]{Colors.RESET}")
                logger.debug(f"Starting turn {turn + 1}")
            elif node_id == "quiz_evaluation":
                print(f"\n{Colors.MAGENTA}[Final Quiz Evaluation]{Colors.RESET}")
                logger.debug("Starting quiz evaluation")
                
        elif status == NodeStatus.COMPLETED:
            # Show results
            if node_id == "quiz_evaluation":
                grade = state.get_data("quiz_grade")
                explanation = state.get_data("quiz_explanation")
                print(f"\n{Colors.MAGENTA}[Quiz Grade: {grade}]{Colors.RESET}")
                print(f"Explanation: {explanation}")
                logger.debug(f"Quiz completed - Grade: {grade}")
                
            elif node_id == "book_lookup":
                info = state.get_data("book_info")
                print(f"\n{Colors.YELLOW}[Book Information: {info}]{Colors.RESET}")
                logger.debug(f"Book lookup completed - Info: {info}")
                
            elif node_id == "professor" and state.get_data('calculation_result') is not None:
                result = state.get_data('calculation_result')
                print(f"\n{Colors.GREEN}[Calculation Result: {result}]{Colors.RESET}")
                logger.debug(f"Calculation completed - Result: {result}")
                
            logger.debug(f"Node {node_id} completed with status: {status}")
            logger.debug(f"Current state data: {state.data}")
            
    async def error_handler(error: Exception, node_id: str, state: NodeState):
        """Handle and display errors from nodes."""
        error_msg = f"\n{Colors.ERROR}[Error in {node_id}: {str(error)}]{Colors.RESET}"
        print(error_msg)
        logger.error(error_msg)
        if node_error := state.errors.get(node_id):
            detail_msg = f"Node error details: {node_error}"
            print(f"{Colors.ERROR}{detail_msg}{Colors.RESET}")
            logger.error(detail_msg)
            logger.debug(f"State at error: {state.data}")
            
    # Create and run the runtime with proper configuration for both agents
    logger.info("Creating graph runtime with configurations...")

    # Create default agent config
    default_agent_config = AgentConfig(
        stream=True,
        logging_config=AlchemistLoggingConfig(
            level=LogLevel.DEBUG,
            show_llm_messages=True,
            show_tool_calls=True
        )
    )

    # Create node-specific agent configs
    node_agent_configs = {
        "student": AgentConfig(
            system_prompt=student_system_prompt,
            tools=[],  # Student has no direct tool access
            json_mode=True
        ),
        "professor": AgentConfig(
            system_prompt=professor_system_prompt,
            tools=[CalculatorTool, BookLookupTool],
            json_mode=True
        ),
        "book_lookup": AgentConfig(
            system_prompt=professor_system_prompt,
            tools=[BookLookupTool],
            json_mode=True
        ),
        "quiz_evaluation": AgentConfig(
            system_prompt=professor_system_prompt,
            tools=[],
            json_mode=True
        )
    }

    runtime = GraphRuntime(
        graph=main_graph,
        config=GraphRuntimeConfig(
            # Base configuration
            logging_config=AlchemistLoggingConfig(
                level=LogLevel.DEBUG,
                show_llm_messages=True,
                show_tool_calls=True
            ),
            # Agent configuration
            default_agent_config=default_agent_config,
            node_agent_configs=node_agent_configs,
            # Event handlers
            stream_handler=stream_handler,
            node_handler=node_handler,
            error_handler=error_handler
        )
    )
    
    print("\n=== Starting Math Tutoring Session ===\n")
    print("Rules:")
    print("1. Student can ask 3 questions")
    print("2. ONE book lookup allowed per session")
    print("3. Calculator available through professor")
    print("4. Quiz after 3 questions\n")
    
    start_time = datetime.now()
    
    try:
        final_state = await runtime.start("start", state)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n=== Tutoring Session Completed in {elapsed:.1f}s ===\n")
        
        # Display final results
        if final_state.errors:
            print("\nErrors during session:")
            for node_id, error in final_state.errors.items():
                print(f"- {node_id}: {error}")
                
    except Exception as e:
        print(f"\nSession failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(tutoring_workflow()) 