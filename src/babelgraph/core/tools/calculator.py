"""A simple calculator tool for basic arithmetic operations.

This tool evaluates mathematical expressions using Python's eval() function.
It supports basic arithmetic operations (+, -, *, /), exponents (**),
and parentheses for grouping.
"""

import asyncio
import logging
from mirascope.core import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)

class CalculatorTool(BaseTool):
    """A simple calculator tool for basic arithmetic operations.
    
    This tool evaluates mathematical expressions using Python's eval() function.
    It supports basic arithmetic operations (+, -, *, /), exponents (**),
    and parentheses for grouping.
    
    Attributes:
        expression: The mathematical expression to evaluate
        
    Example:
        ```python
        tool = CalculatorTool(expression="2 + 2")
        result = tool.call()  # Returns "4"
        
        tool = CalculatorTool(expression="42 ** 0.5")
        result = tool.call()  # Returns "6.48074069840786"
        ```
    """
    
    expression: str = Field(
        ...,
        description="A mathematical expression to evaluate (e.g., '2 + 2', '42 ** 0.5')"
    )

    def call(self) -> str:
        """Evaluate the mathematical expression and return result.
        
        Returns:
            str: The result of the evaluation, or an error message if evaluation fails
            
        Example:
            >>> tool = CalculatorTool(expression="2 + 2")
            >>> tool.call()
            "4"
        """
        try:
            result = eval(self.expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"