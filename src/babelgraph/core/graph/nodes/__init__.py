"""Node package initialization.

Exposes node types and helpers for building workflows.
"""

from babelgraph.core.graph.nodes.base.node import (
    Node,
    terminal_node,
    state_handler
)
from babelgraph.core.graph.nodes.agent import AgentNode

__all__ = [
    # Base node types
    "Node",
    "AgentNode",
    
    # Decorators and helpers
    "terminal_node",
    "state_handler",
]
