from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.state import NodeState
from alchemist.ai.graph.state import NodeStatus
from typing import Optional

class TerminalNode(Node):
    """
    A built-in terminal node that marks the end of a workflow.

    When processed, it sets its status to TERMINAL and does not return a next node.
    """

    async def process(self, state: NodeState) -> Optional[str]:
        """
        Process the terminal node.

        Args:
            state: The current NodeState.

        Returns:
            None, indicating the end of the workflow.
        """
        state.mark_status(self.id, NodeStatus.TERMINAL)
        return None 