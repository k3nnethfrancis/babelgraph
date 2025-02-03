"""Graph visualization tools."""

from babelgraph.core.graph.state import NodeState

class GraphVisualizer:
    """Visualize graph structure and execution."""
    def render_graph(self) -> str: ...
    def render_execution(self, state: NodeState) -> str: ... 