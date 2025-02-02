from setuptools import setup, find_packages

setup(
    name="babelgraph",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "pytest",
        "pytest-asyncio",
        "mirascope",
    ],
    python_requires=">=3.8",
    # Add metadata for PyPI
    author="kenneth cavanagh",
    author_email="ken@agency42.com",
    description="lightweight graph orchestration library for multi-agent systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/k3nnethfrancis/babelgraph",
)