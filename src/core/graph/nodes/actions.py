"""
Action Node Implementations

This module merges the functionality of ToolNode and ActionNode into a single ActionNode class.
It supports:
    - Executing a tool (sync or async).
    - Optional required state checks (required_state).
    - Optional preservation of specific keys in state.results (preserve_state).
    - Flexible input mapping from NodeState, including dotted key references.
    - Standard transitions for "default" and "error" paths.

Typical Use-Cases:
-----------------
1. Simple tool invocation with known inputs, e.g., arithmetic or search.
2. Workflow steps requiring required_state to be present in NodeState.
3. Automatic cleanup of NodeState to avoid downstream clutter.
4. More advanced action logic (pre/post hooks, chaining) if needed.

Example:
--------
>>> from alchemist.ai.tools.calculator import CalculatorTool
>>>
>>> node = ActionNode(
...     id="calc_step",
...     name="Calculator",
...     description="Adds two numbers from node state",
...     tool=CalculatorTool(),
...     required_state=["calc_args"],
...     preserve_state=["calc_args", "calc_step"],
...     next_nodes={"default": "next_node", "error": "error_node"},
... )
>>> # This node checks for 'calc_args' in state, calls the tool, then
>>> # preserves certain keys in state.results, removing others.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from pydantic import Field, model_validator
from datetime import datetime

from alchemist.ai.base.logging import get_logger, LogComponent
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.state import NodeState, NodeStatus
from alchemist.ai.base.logging import Colors

logger = get_logger(LogComponent.NODES)


class ActionNode(Node):
    """
    Unified node for executing a tool or function, with optional workflow logic.

    Features:
        - tool (Callable): The function/tool to execute, sync or async.
        - required_state (List[str]): A list of keys that must be present (in state.data or state.results).
        - preserve_state (List[str]): Keys to keep in state.results after execution. Removes other non-node keys.
        - name (str): Human-readable name for the action.
        - description (str): High-level description of the action.
        - args_key (str): (Optional) Key that references arguments in state for more complex usage patterns.
        - output_key (str): The key under which tool results are stored in state.results[node.id].
    """
    name: Optional[str] = Field(
        default=None,
        description="A human-readable name for the action."
    )
    description: Optional[str] = Field(
        default=None,
        description="A high-level description of what the action does."
    )
    tool: Callable = Field(
        ...,
        description="A callable (sync or async) that this node executes."
    )
    required_state: List[str] = Field(
        default_factory=list,
        description="Required state keys to check before execution."
    )
    preserve_state: List[str] = Field(
        default_factory=list,
        description="List of keys to preserve in results after execution. Others may be removed."
    )
    args_key: Optional[str] = Field(
        default=None,
        description="If set, references a key in state for advanced argument usage."
    )
    output_key: str = Field(
        default="result",
        description="The key under which the tool's result is stored in state.results[node.id]."
    )

    @model_validator(mode='after')
    def validate_tool_config(self) -> "ActionNode":
        """Validate the configuration of the action node."""
        if not callable(self.tool):
            raise ValueError(f"ActionNode {self.id} requires a callable 'tool'.")
        return self

    async def process(self, state: NodeState) -> Optional[str]:
        """Execute the action and handle state management."""
        try:
            start_time = datetime.now()
            state.mark_status(self.id, NodeStatus.RUNNING)
            
            # Log start of action
            logger.debug(f"\n{Colors.BOLD}🔧 Node {self.id} Starting:{Colors.RESET}")
            
            self._validate_required_state(state)
            inputs = self._prepare_input_data(state)
            
            # Handle args_key if present
            if self.args_key:
                if self.args_key in state.results:
                    inputs.update(state.results[self.args_key])
                elif self.args_key in state.data:
                    if isinstance(state.data[self.args_key], dict):
                        inputs.update(state.data[self.args_key])

            # Execute the tool
            if asyncio.iscoroutinefunction(self.tool):
                result = await self.tool(**inputs)
            else:
                result = self.tool(**inputs)

            # Store results with timing
            elapsed = (datetime.now() - start_time).total_seconds()
            state.results[self.id] = {
                self.output_key: result,
                "timing": elapsed
            }

            # Log completion with results
            logger.info(
                f"\n{Colors.SUCCESS}✓ Node '{self.id}' completed in {elapsed:.2f}s{Colors.RESET}"
                f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}"
                f"\n{Colors.INFO}{result}{Colors.RESET}"
                f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}\n"
            )

            # Handle callbacks
            if "on_complete" in self.metadata:
                await self.metadata["on_complete"](state, self.id)

            self._cleanup_state(state)
            state.mark_status(self.id, NodeStatus.COMPLETED)
            return self.get_next_node()

        except Exception as e:
            logger.error(f"Error in action node '{self.id}': {str(e)}")
            state.errors[self.id] = str(e)
            state.mark_status(self.id, NodeStatus.ERROR)
            return self.get_next_node("error")

    def _validate_required_state(self, state: NodeState) -> None:
        """
        Confirm required_state keys exist in either state.results or state.data.
        Raises ValueError if missing.
        """
        missing_keys = []
        for key in self.required_state:
            if key not in state.results and key not in state.data:
                missing_keys.append(key)
        if missing_keys:
            raise ValueError(
                f"Missing required state keys for node '{self.id}': {missing_keys}"
            )

    def _cleanup_state(self, state: NodeState) -> None:
        """
        Manage state preservation:
        1. Copy preserved values from state.data to state.results if not already present.
        2. Keep preserved values in state.results.
        3. Remove non-preserved values from state.results, except for the node's own result.
        """
        if not self.preserve_state:
            return

        # Copy preserved keys from data into results if not already in results
        for key in self.preserve_state:
            if key in state.data and key not in state.results:
                state.results[key] = state.data[key]

        # Remove non-preserved keys from results
        keys = list(state.results.keys())
        for k in keys:
            # Do not remove the node's own result store
            if k not in self.preserve_state and k != self.id:
                del state.results[k]

async def test_action_node():
    """Basic test to demonstrate usage of the merged ActionNode."""
    print("\nTesting merged ActionNode...")

    # Define a simple test tool
    async def add_tool(x: int, y: int) -> int:
        return x + y

    # Create test node with some required_state
    node = ActionNode(
        id="add",
        name="Addition Action",
        tool=add_tool,
        required_state=["calc_args"],
        preserve_state=["calc_args", "important_key"],
        input_map={"x": "node.calc_args.x", "y": "node.calc_args.y"},
        next_nodes={"default": "finished", "error": "error_node"},
        args_key=None
    )

    # Create test state
    from alchemist.ai.graph.state import NodeState
    state = NodeState()
    # Suppose the user put 'calc_args' in results
    state.results["calc_args"] = {"x": 2, "y": 3}
    # Some random other key in results
    state.results["garbage_key"] = True
    # Some data that might be used
    state.data["important_key"] = "Preserve me"

    # Run
    next_id = await node.process(state)

    # Check
    assert next_id == "finished", f"Expected 'finished', got {next_id}"
    assert state.results["add"]["result"] == 5, "Incorrect result from action"
    print("Merged ActionNode test passed! State results:", state.results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_action_node()) 