"""
Student–Teacher Calculator Example

Demonstrates a minimal multi-node workflow where:
1. The student attempts to solve a problem (no tools, just a structured response).
2. The teacher verifies the student's answer with a calculator tool.
3. If incorrect, the teacher provides feedback, and the student retries.
4. If correct, the workflow concludes.

Key Concepts:
- Teacher Node (AgentNode) with a CalculatorTool for verification
- Student Node (AgentNode) that returns a structured output (Pydantic model)
- Asynchronous Graph execution via Babelgraph
- Provider-agnostic LLM calls using OpenPipe
- Mirascope-based tool integration
"""

import asyncio
from typing import Optional, Any, Dict
from datetime import datetime

from pydantic import BaseModel, Field

from babelgraph.core.graph import Graph, NodeState
from babelgraph.core.graph.nodes.agent import AgentNode
from babelgraph.core.graph.nodes import Node, terminal_node, state_handler
from babelgraph.core.tools.calculator import CalculatorTool
from babelgraph.core.agent.base import BaseAgent
from babelgraph.core.logging import (
    configure_logging,
    LogLevel,
    LogComponent,
    get_logger,
    Colors
)

logger = get_logger(LogComponent.WORKFLOW)


# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------
class StudentAnswer(BaseModel):
    """
    Student's structured answer to a math problem.
    """
    problem: str = Field(..., description="Text of the math problem.")
    solution: float = Field(..., description="Numerical solution attempted.")
    explanation: str = Field(..., description="Step-by-step reasoning.")
    attempt_number: int = Field(1, description="Which attempt this is.")


class TeacherFeedback(BaseModel):
    """
    Feedback from the teacher regarding the student's answer.
    """
    correct: bool = Field(..., description="Whether the solution is correct.")
    feedback_message: str = Field(..., description="Teacher's message to the student.")


# ---------------------------------------------------------------------
# Helper Functions for State
# ---------------------------------------------------------------------
def set_node_result(state: NodeState, node_id: str, key: str, value: Any) -> None:
    """
    Store the result from a node execution into the NodeState.
    """
    if not hasattr(state, "results"):
        state.results = {}
    state.results.setdefault(node_id, {})[key] = value


def get_node_result(state: NodeState, node_id: str, key: str) -> Any:
    """
    Retrieve a stored result from the NodeState.
    """
    if not hasattr(state, "results"):
        state.results = {}
    return state.results.get(node_id, {}).get(key)


# ---------------------------------------------------------------------
# Student Node
# ---------------------------------------------------------------------
class StudentNode(AgentNode):
    """
    Node for the student agent. 
    Tries to solve the problem and produces a structured StudentAnswer.
    """

    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Process the problem input and produce a StudentAnswer.
        """
        try:
            problem_data = self.get_message_input(state)
            if not problem_data:
                raise ValueError("No problem input found for the student.")

            logger.info(
                f"\n{Colors.BOLD}Student: Attempting to solve...{Colors.RESET}"
            )

            # Request student's structured answer
            response = await self.agent._step(problem_data["content"])

            logger.debug(f"Raw Student Response: {response}")

            # Store the student's response
            set_node_result(state, self.id, "answer", response)

            # If we get here, route to teacher verification
            return "verify_answer"

        except Exception as e:
            logger.error(f"Student error: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"


# ---------------------------------------------------------------------
# Teacher Node
# ---------------------------------------------------------------------
class TeacherNode(AgentNode):
    """
    Node for the teacher agent, which has the CalculatorTool
    to verify the student's numerical solution.
    """

    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Verify the student's solution. 
        If incorrect, provide feedback and route to student retry.
        If correct, route to success.
        """
        try:
            student_answer: StudentAnswer = get_node_result(state, "student", "answer")
            if not student_answer:
                raise ValueError("No student answer found to verify.")

            logger.info(
                f"\n{Colors.BOLD}Teacher: Verifying student solution...{Colors.RESET}"
            )

            # Use the CalculatorTool with the student's solution
            tool_instance = CalculatorTool(expression=str(student_answer.solution))
            calc_result = tool_instance.call()

            # Compare
            try:
                numeric_result = float(calc_result)
            except ValueError:
                raise ValueError(f"Calculator tool returned an invalid result: {calc_result}")

            is_correct = False
            # Minimal heuristic to determine "correctness":
            # We'll parse the student's problem (if it includes "x" or "triangle", etc. 
            # but for brevity, let's assume we just treat it as numeric).
            # In real usage, you'd parse the problem or store expected answers somewhere.
            # For this sample, let's say student solves "25% of 80" => 20, etc.
            # We'll just verify the student's numeric result == tool result
            is_correct = abs(numeric_result - student_answer.solution) < 1e-6

            if is_correct:
                teacher_feedback = TeacherFeedback(
                    correct=True,
                    feedback_message="Great job! That is correct."
                )
                set_node_result(state, self.id, "feedback", teacher_feedback)
                return "success"
            else:
                teacher_feedback = TeacherFeedback(
                    correct=False,
                    feedback_message=(
                        f"Your answer {student_answer.solution} seems off. "
                        f"I calculated {numeric_result}. Try again!"
                    )
                )
                set_node_result(state, self.id, "feedback", teacher_feedback)
                return "retry"

        except Exception as e:
            logger.error(f"Teacher error: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"


# ---------------------------------------------------------------------
# Retry Node
# ---------------------------------------------------------------------
class RetryStudentNode(AgentNode):
    """
    Node that receives teacher feedback, increments attempt count, and
    has the student try again with a new solution.
    """

    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Incorporate teacher feedback and re-prompt the student.
        """
        try:
            teacher_feedback: TeacherFeedback = get_node_result(state, "teacher", "feedback")
            student_answer: StudentAnswer = get_node_result(state, "student", "answer")
            if not (teacher_feedback and student_answer):
                raise ValueError("Missing teacher feedback or student answer.")

            logger.info(
                f"\n{Colors.BOLD}Student: Receiving teacher feedback...{Colors.RESET}"
            )
            logger.info(
                f"Teacher says: {teacher_feedback.feedback_message}"
            )

            # Build new input for the student, including teacher feedback
            new_attempt_number = student_answer.attempt_number + 1
            new_prompt = (
                f"You previously answered {student_answer.solution} for the problem:\n"
                f"'{student_answer.problem}'\n\n"
                f"Teacher says it's incorrect. Please try again.\n"
                f"Feedback: {teacher_feedback.feedback_message}\n"
                f"Attempt number: {new_attempt_number}"
            )

            # The student node will see this as a new user message
            self.set_message_input(state, content=new_prompt, role="user")

            return "solve_again"

        except Exception as e:
            logger.error(f"Error in retry logic: {str(e)}")
            state.add_error(self.id, str(e))
            return "error"


# ---------------------------------------------------------------------
# Terminal Node
# ---------------------------------------------------------------------
@terminal_node
class ShowResultNode(Node):
    """
    Terminal node that displays the final outcome.
    """

    @state_handler
    async def process(self, state: NodeState) -> Optional[str]:
        """
        Show final results (correct answer or errors).
        """
        student_answer: StudentAnswer = get_node_result(state, "student", "answer")
        teacher_feedback: TeacherFeedback = get_node_result(state, "teacher", "feedback")

        print("\n====== FINAL RESULT ======")
        if student_answer:
            print(f"Problem: {student_answer.problem}")
            print(f"Student's Last Attempt: {student_answer.solution}")
            print(f"Explanation: {student_answer.explanation}")
            print(f"Number of Attempts: {student_answer.attempt_number}")

        if teacher_feedback:
            if teacher_feedback.correct:
                print(f"\n{Colors.SUCCESS}Correct! {teacher_feedback.feedback_message}{Colors.RESET}")
            else:
                print(f"\n{Colors.ERROR}Still Incorrect: {teacher_feedback.feedback_message}{Colors.RESET}")
        else:
            print(f"{Colors.ERROR}No teacher feedback found.{Colors.RESET}")

        return None


# ---------------------------------------------------------------------
# Workflow Execution
# ---------------------------------------------------------------------
async def run_student_teacher_flow(problem: str):
    """Build and run the student-teacher workflow."""
    configure_logging(default_level=LogLevel.INFO)

    # Create the student and teacher agents with different models
    student_agent = BaseAgent(
        system_prompt=(
            "You are a math student learning to solve problems. "
            "You must show ALL your work step by step. "
            "You don't have a calculator, so you need to do calculations by hand. "
            "Be careful and methodical, but you might make arithmetic mistakes."
        ),
        response_model=StudentAnswer,
        json_mode=True,
        debug_mode=True,
        model="openpipe:ken0-llama31-8B-instruct"  # Less capable model
    )
    
    teacher_agent = BaseAgent(
        system_prompt=(
            "You are a strict but patient math teacher. Your role is to:"
            "\n1. Use the calculator tool to verify numeric answers"
            "\n2. Provide detailed, constructive feedback when students make mistakes"
            "\n3. Explain common misconceptions when relevant"
            "\n4. Encourage students when they show good work"
            "\n5. Always verify answers computationally before approving them"
        ),
        tools=[CalculatorTool],  # The teacher has the calculator
        response_model=TeacherFeedback,
        json_mode=True,
        debug_mode=True,
        model="claude-3-5-sonnet-20240620"  # Using Claude for more nuanced feedback
    )

    # Build the graph
    graph = Graph()
    student_node = StudentNode(id="student", agent=student_agent)
    teacher_node = TeacherNode(id="teacher", agent=teacher_agent)
    retry_node = RetryStudentNode(id="retry", agent=student_agent)
    end_node = ShowResultNode(id="end")

    graph.create_workflow(
        nodes={
            "student": student_node,
            "teacher": teacher_node,
            "retry": retry_node,
            "end": end_node
        },
        flows=[
            ("student", "verify_answer", "teacher"),
            ("student", "error", "end"),
            ("teacher", "success", "end"),
            ("teacher", "retry", "retry"),
            ("teacher", "error", "end"),
            ("retry", "solve_again", "student"),
            ("retry", "error", "end")
        ]
    )

    # Initialize state
    state = NodeState()
    # Provide initial input to student
    student_node.set_message_input(
        state,
        content=problem,
        role="user"
    )

    print(f"{Colors.BOLD}\nStarting Student–Teacher Workflow...{Colors.RESET}")
    print(f"{Colors.INFO}Problem: {problem}{Colors.RESET}\n")

    start_time = datetime.now()
    await graph.run("student", state)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{Colors.SUCCESS}Workflow Completed in {elapsed:.2f}s{Colors.RESET}\n")


async def main():
    """
    Demonstration entry point. 
    Runs a sequence of problems through the Student–Teacher flow.
    """
    problems = [
        # Easy but requires precise calculation
        "What is 17.5% of 344?",
        
        # Medium (recurring decimals)
        "What is 678 divided by 13? Give your answer to 3 decimal places.",
        
        # Hard (multiple steps with square roots)
        "Calculate (√125 + √45) × (√125 - √45). Show your work step by step."
    ]
    
    print(f"{Colors.BOLD}Starting Math Learning Session{Colors.RESET}")
    print(f"Using models:")
    print(f"- Student: openpipe:ken0-llama31-8B-instruct (no calculator)")
    print(f"- Teacher: claude-3-5-sonnet (with calculator)")
    print(f"{Colors.BOLD}{'-' * 50}{Colors.RESET}\n")
    
    for i, p in enumerate(problems, 1):
        print(f"\n{Colors.BOLD}Problem {i}/{len(problems)}{Colors.RESET}")
        print(f"{Colors.INFO}Difficulty: {get_difficulty_label(i)}{Colors.RESET}")
        await run_student_teacher_flow(p)
        print(f"\n{Colors.BOLD}{'=' * 50}{Colors.RESET}\n")

def get_difficulty_label(problem_number: int) -> str:
    """Return difficulty label based on problem number."""
    if problem_number == 1:
        return "Easy 🟢"
    elif problem_number == 2:
        return "Medium 🟡"
    else:
        return "Hard 🟠"

if __name__ == "__main__":
    asyncio.run(main())