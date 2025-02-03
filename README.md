# 🌟 babelgraph

A lightweight graph orchestration library for building composable AI agent workflows.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

Babelgraph is a Python library that makes it easy to build, test, and deploy AI agent workflows. It provides a clean, type-safe way to orchestrate complex interactions between multiple agents, tools, and external services.

### Key Features

- 🔄 **Graph-based Orchestration**: Build complex workflows with directed graphs
- 🤖 **Agent Integration**: Seamless integration with LLMs via Mirascope
- 🛠️ **Tool System**: Extensible tool framework for agent capabilities
- 📊 **State Management**: Clean data passing between nodes
- 🎨 **Pretty Logging**: Beautiful console output with structured logging
- 🔌 **Platform Support**: Built-in Discord integration (more coming soon)

## 🚀 Quick Start

### Installation

```bash
pip install babelgraph
```

### Basic Usage

Here's a simple example of creating a workflow:

```python
from babelgraph import Graph, AgentNode, Node
from babelgraph.core.logging import configure_logging, LogLevel, LogComponent

# Configure logging
configure_logging(
    LogLevel.INFO,
    component_levels={
        LogComponent.GRAPH: LogLevel.INFO,
        LogComponent.NODES: LogLevel.DEBUG
    }
)

# Create nodes
analyzer = AgentNode(
    id="analyzer",
    agent=BaseAgent(
        system_prompt=AnalyzerPrompt(),
        response_model=TextAnalysis
    ),
    next_nodes={
        "default": "decision",
        "error": "end"
    }
)

decision = AgentNode(
    id="decision",
    agent=BaseAgent(
        system_prompt=DecisionPrompt(),
        response_model=ActionDecision
    ),
    next_nodes={
        "default": "end",
        "error": "end"
    }
)

# Terminal node
end = Node(id="end", next_nodes={})

# Create and run graph
graph = Graph()
graph.add_node(analyzer)
graph.add_node(decision)
graph.add_node(end)
graph.set_entry_point("analyzer")

# Run workflow
state = await graph.run("start")
```

## 📚 Documentation

For detailed documentation, see:
- [Development Notes](docs/dev.txt) - Architecture and design details
- [Examples](examples/) - Example workflows and patterns
- [API Reference](docs/api.md) - Detailed API documentation

## 🏗️ Core Components

### Graph System
- Build workflows with directed graphs
- Async execution and parallel processing
- Clean state management
- Event emission and monitoring

### Agent System
- Provider-agnostic LLM integration
- Structured outputs with Pydantic
- Tool integration
- Streaming support

### Logging System
- Component-based configuration
- Pretty console output
- Structured logging
- File output support

### Runtime System
- Local console runtime
- Platform-specific runtimes (Discord)
- Session management
- Tool execution context

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/babelgraph.git
cd babelgraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/graph/test_base.py

# Run with coverage
pytest --cov=babelgraph tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Add tests for your changes
5. Make sure all tests pass
6. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Mirascope](https://github.com/mirascope/mirascope) - LLM abstraction layer
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [OpenPipe](https://openpipe.ai/) - Custom model training

## 📞 Support

- 📧 Email: support@babelgraph.ai
- 💬 Discord: [Join our community](https://discord.gg/babelgraph)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/babelgraph/issues)

---

<p align="center">Made with ❤️ by the Babelgraph Team</p>
