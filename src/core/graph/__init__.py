"""Graph system for AI workflow orchestration.

This package provides a lightweight framework for composing AI workflows using:
- Directed graph-based orchestration
- State management and persistence
- Agent and action nodes
- Subgraph composition
- Parallel execution support
"""

from alchemist.ai.graph.base import Graph
from alchemist.ai.graph.state import NodeState, NodeStatus
from alchemist.ai.graph.nodes import (
    Node,
    ActionNode,
    AgentNode,
    ContextNode,
    TerminalNode
)

__all__ = [
    # Core components
    'Graph',
    'NodeState',
    'NodeStatus',
    
    # Node types
    'Node',
    'ActionNode',
    'AgentNode',
    'ContextNode',
    'TerminalNode',
]
