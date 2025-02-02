"""Context Manager Node (Placeholder)

This node demonstrates how we might read/write contextual information from
external sources (e.g., a memory database, environment data, or user session).
It's intentionally minimal, as the memory system will be implemented later.

Usage Ideas:
------------
1. Pull relevant user or session data from an external store (e.g., Supabase).
2. Inject that data into NodeState for subsequent nodes (AgentNode, ActionNode).
3. Optionally transform or filter context (like summarizing stale info).
"""

from typing import Optional
from pydantic import Field
from alchemist.ai.graph.nodes.base.node import Node
from alchemist.ai.graph.state import NodeState, NodeStateProtocol

class ContextNode(Node):
    """Node that fetches and injects external context into the node state."""
    
    context_source: str = Field(
        default="supabase",
        description="ID or type of context source (e.g., 'supabase', 'redis', etc.)."
    )

    async def process(self, state: NodeStateProtocol) -> Optional[str]:
        """
        Placeholder method to demonstrate asynchronous context retrieval.

        1. Retrieve context from an external system.
        2. Update state results or data.
        3. Return next node ID, if any.
        """
        # Example: inject some placeholder info
        state.set_data("external_context", f"Fetched from {self.context_source} ...")

        # In real usage, we'd do something like:
        #  memory_records = memory_system.get_all(user_id="alice")
        #  state.set_data("memories", memory_records)

        return self.get_next_node()