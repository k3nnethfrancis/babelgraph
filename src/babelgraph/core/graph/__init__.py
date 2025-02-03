"""Graph package initialization.

Exposes core graph components and helpers for building workflows.
"""

from babelgraph.core.graph.base import Graph
from babelgraph.core.graph.state import NodeState, NodeStatus
from babelgraph.core.graph.nodes.base.node import (
    Node,
    terminal_node,
    state_handler
)
from babelgraph.core.graph.nodes.agent import AgentNode

__all__ = [
    # Core classes
    "Graph",
    "Node",
    "AgentNode",
    "NodeState",
    "NodeStatus",
    
    # Decorators and helpers
    "terminal_node",
    "state_handler",
]
