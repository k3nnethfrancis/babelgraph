"""
BaseAgent rewritten to accept a system prompt, removing direct persona references.

This module provides a provider-agnostic base class for AI Agents using Mirascope.
It handles:
 - Session state
 - Tool usage
 - Asynchronous calls
 - Prompt composition (system + conversation + user)

Message Flow:
    1. User input is received and added to history
    2. OpenPipe API call is made with:
       - System message (from persona)
       - Conversation history
       - Current user query
    3. Response is processed:
       - If no tools: response is added to history and returned
       - If tools: each tool is executed, results added to history, and step repeats
    
Key Features:
    - OpenPipe integration with gpt-4o-mini
    - Tool integration with Mirascope's BaseTool pattern
    - Conversation history management
    - Persona-based behavior configuration
    - Async-first implementation

Example:
    ```python
    from alchemist.ai.base.tools import CalculatorTool
    
    agent = BaseAgent(tools=[CalculatorTool])
    await agent.run()
    ```
"""

import logging
from typing import List, Any, Optional, Union, Callable, AsyncGenerator, Dict
from pydantic import BaseModel, Field, ValidationError
import inspect
# import lilypad
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.core import (
    BaseMessageParam,
    BaseTool,
    BaseDynamicConfig,
    openai,
    Messages,
    prompt_template,
)
from openpipe import OpenAI as OpenPipeClient
from babelgraph.tools.calculator import CalculatorTool
from babelgraph.core.logging import LogComponent, AlchemistLoggingConfig, log_verbose, VerbosityLevel
import json

# Get logger for agent component
logger = logging.getLogger(LogComponent.AGENT.value)


@prompt_template()
def create_system_prompt(config: Optional[BaseModel] = None) -> list[BaseMessageParam]:
    """Creates a formatted system prompt from a Pydantic model.
    
    Args:
        config: Optional Pydantic model containing system prompt configuration.
               If None, returns an empty system message.
        
    Returns:
        list[BaseMessageParam]: List containing the formatted system message
    """
    if not config:
        return [BaseMessageParam(role="system", content="")]
        
    def format_value(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        elif isinstance(value, dict):
            return "\n".join(f"- {k}: {v}" for k, v in value.items())
        elif isinstance(value, BaseModel):
            return format_model(value)
        return str(value)

    def format_model(model: BaseModel) -> str:
        sections = []
        for field_name, field_value in model.model_dump().items():
            field_title = field_name.replace('_', ' ').title()
            formatted_value = format_value(field_value)
            sections.append(f"{field_title}: {formatted_value}")
        return "\n\n".join(sections)

    content = format_model(config)
    
    # If BaseAgent has a response_model, append structured output instructions
    if hasattr(config, 'response_model') and config.response_model:
        content += "\n\nYou must respond with a valid JSON object matching this exact schema:\n"
        schema = config.response_model.model_json_schema()
        content += f"{schema}\n"
        content += "\nExample response format:\n"
        example = {
            field: f"<{field}>" for field in schema.get("properties", {}).keys()
        }
        content += f"{json.dumps(example, indent=2)}\n"
        content += "\nProvide ONLY the JSON response, no other text."
    
    logger.debug(f"[create_system_prompt] Formatted content:\n{content}")
    
    return [BaseMessageParam(role="system", content=content)]

class BaseAgent(BaseModel):
    """Base agent class implementing core agent functionality with persona support and tools.
    
    The agent maintains conversation history and supports tool execution through Mirascope's
    BaseTool pattern. It uses OpenPipe's gpt-4o-mini model for generating responses and
    deciding when to use tools.
    
    Message Flow:
        1. User messages are added to history
        2. System prompt (from persona) and history are sent to OpenPipe
        3. If the response includes tool calls:
           - Tools are executed in sequence
           - Results are added to history
           - Another API call is made with the tool results
        4. Final response is returned to the user
    
    Attributes:
        history: List of conversation messages (BaseMessageParam)
        system_prompt: System prompt or Pydantic model for agent configuration
        tools: List of available tool classes (not instances)
        logging_config: Logging configuration for controlling verbosity
        response_model: Optional Pydantic model for structured outputs
        json_mode: Whether to enforce JSON output format
        output_parser: Optional custom parser for response processing
        stream: Whether to use streaming mode for responses
    """
    
    history: List[BaseMessageParam] = Field(
        default_factory=list,
        description="Conversation history"
    )
    system_prompt: Optional[Union[str, BaseModel]] = None
    tools: List[type[BaseTool]] = Field(
        default_factory=list,
        description="List of tool classes available to the agent"
    )
    logging_config: AlchemistLoggingConfig = Field(
        default_factory=AlchemistLoggingConfig,
        description="Controls the verbosity and detail of agent logs."
    )
    response_model: Optional[type[BaseModel]] = Field(
        default=None,
        description="Pydantic model for structured output validation"
    )
    json_mode: bool = Field(
        default=False,
        description="Whether to enforce JSON output format"
    )
    output_parser: Optional[Callable] = Field(
        default=None,
        description="Custom parser for processing responses"
    )
    stream: bool = Field(
        default=False,
        description="Whether to use streaming mode for responses"
    )

    # @lilypad.generation()
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    @openai.call("gpt-4o-mini", client=OpenPipeClient())
    def _call(self, query: str) -> BaseDynamicConfig:
        """Make an OpenPipe API call with the current conversation state.
        
        Args:
            query: The current user input, or empty string for follow-up calls
            
        Returns:
            BaseDynamicConfig: Contains messages, tools, and output configuration
        """
        # Create system message based on type
        system_content = ""
        if isinstance(self.system_prompt, BaseModel):
            system_messages = create_system_prompt(self.system_prompt)
        else:
            system_content = self.system_prompt or ""
            if self.response_model or self.json_mode:
                # Add explicit JSON format instructions
                system_content += "\n\nYou must respond with a valid JSON object."
                if self.response_model:
                    self.json_mode = True # set json_mode to True if response_model is set
                    schema = self.response_model.model_json_schema()
                    system_content += f" Match this exact schema:\n{schema}\n"
                    system_content += "\nExample response format:\n"
                    example = {
                        field: f"<{field}>" 
                        for field in schema.get("properties", {}).keys()
                    }
                    system_content += f"{json.dumps(example, indent=2)}\n"
                system_content += "\nProvide ONLY the JSON response, no other text."
            system_messages = [BaseMessageParam(role="system", content=system_content)]
            
        messages = [
            *system_messages,
            *self.history
        ]
        
        if query:
            messages.append(BaseMessageParam(role="user", content=query))
            
        config = {
            "messages": messages,
            "tools": self.tools,
        }

        # Always set response format for JSON mode or response model
        if self.json_mode or self.response_model:
            config["response_format"] = {"type": "json_object"}

        # Add structured output configuration
        if self.response_model:
            config["response_model"] = self.response_model
            config["json_mode"] = True  # Always use json_mode with response_model
        if self.output_parser:
            config["output_parser"] = self.output_parser

        return config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _step(self, query: str) -> Union[str, BaseModel, Any]:
        """Execute a single step of agent interaction.
        
        This method handles the core interaction loop:
        1. Adds user query to history (if present)
        2. Makes API call and adds response to history
        3. If tools are in the response:
           - Executes each tool
           - Adds results to history
           - Makes another API call (recursive step)
        4. Returns final response content
        
        The flow ensures that:
        - All messages are properly added to history
        - Tools are executed in the correct order
        - Results are formatted correctly
        - Recursive steps handle follow-up responses
        
        Logging Details:
            - At VERBOSE or DEBUG levels, logs conversation messages and tool calls.
            - At INFO level, logs only high-level steps.
        
        Args:
            query: User input or empty string for follow-up steps
            
        Returns:
            Union[str, BaseModel, Any]: Response content, which could be:
                - str: Raw text response if no structured output configured
                - BaseModel: Validated Pydantic model if response_model is set
                - Any: Custom type if output_parser is used
        """
        if query:
            self.history.append(BaseMessageParam(role="user", content=query))
            if self.logging_config.show_llm_messages or \
               self.logging_config.level <= VerbosityLevel.DEBUG:
                log_verbose(logger, f"Added user query to history: {query}")
            
        response = self._call(query)
        
        # Handle structured output before adding to history
        output_content = response.content
        if self.output_parser:
            output_content = self.output_parser(response)
        elif self.response_model or self.json_mode:
            try:
                # Ensure we have a JSON string
                if isinstance(output_content, str):
                    # Skip empty responses
                    if not output_content.strip():
                        logger.error("Received empty response from LLM")
                        raise ValueError("Empty response from LLM")
                        
                    # Try to clean the response if it's not pure JSON
                    output_content = output_content.strip()
                    # Find the first { and last } if there's extra text
                    start = output_content.find('{')
                    end = output_content.rfind('}') + 1
                    if start >= 0 and end > start:
                        output_content = output_content[start:end]
                    
                # Parse JSON
                output_content = json.loads(output_content)
                
                # Validate against model if provided
                if self.response_model:
                    output_content = self.response_model.model_validate(output_content)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse response as {self.response_model.__name__ if self.response_model else 'JSON'}: {e}")
                logger.error(f"Raw response: {response.content}")
                raise
            
        # Add raw response to history
        self.history.append(response.message_param)
        
        if self.logging_config.show_llm_messages or \
           self.logging_config.level <= VerbosityLevel.DEBUG:
            log_verbose(logger, f"Agent response: {output_content}")
        
        # Handle tool calls if present
        if tools := response.tools:
            if self.logging_config.show_tool_calls or \
               self.logging_config.level <= VerbosityLevel.DEBUG:
                log_verbose(logger, f"Tools to call: {tools}")
            
            tools_and_outputs = []
            for tool in tools:
                logger.info(f"[Calling Tool '{tool._name()}' with args {tool.args}]")
                
                if inspect.iscoroutinefunction(tool.call):
                    result = await tool.call()
                else:
                    result = tool.call()
                    
                logger.info(f"Tool result: {result}")
                tools_and_outputs.append((tool, result))
            
            self.history.extend(response.tool_message_params(tools_and_outputs))
            return await self._step("")
            
        return output_content

    async def run(self) -> None:
        """Run the agent interaction loop.
        
        Uses streaming or standard mode based on the agent's stream configuration.
        
        This method:
        1. Prompts for user input
        2. Processes each query through _step() or _stream_step()
        3. Prints responses
        4. Continues until user exits
        """
        while True:
            query = input("(User): ")
            if query.lower() in ["exit", "quit"]:
                break
                
            print("(Assistant): ", end="", flush=True)
            
            if self.stream:
                async for chunk, tool in self._stream_step(query):
                    if tool:
                        print(f"\n[Calling Tool '{tool._name()}' with args {tool.args}]")
                    elif chunk:
                        print(chunk, end="", flush=True)
                print()
            else:
                result = await self._step(query)
                print(result)

    @openai.call("gpt-4o-mini", client=OpenPipeClient(), stream=True)
    def _stream(self, query: str) -> BaseDynamicConfig:
        """Make a streaming OpenPipe API call with the current conversation state.
        
        Args:
            query: The current user input
            
        Returns:
            BaseDynamicConfig: Contains messages and tools configuration
        """
        # Initialize system messages
        if isinstance(self.system_prompt, BaseModel):
            system_messages = create_system_prompt(self.system_prompt)
        else:
            system_content = self.system_prompt or ""
            system_messages = [BaseMessageParam(role="system", content=system_content)]

        messages = [
            *system_messages,
            *self.history,
            BaseMessageParam(role="user", content=query),
        ]
        return {"messages": messages, "tools": self.tools}  # Fixed tools parameter

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _stream_step(self, query: str) -> AsyncGenerator[tuple[Optional[str], Optional[BaseTool]], None]:
        """Execute a streaming step of agent interaction.
        
        Args:
            query: User input or empty string for follow-up steps
            
        Yields:
            tuple: (chunk_content, optional_tool)
                - chunk_content (str): The streamed content chunk
                - optional_tool (Optional[BaseTool]): Tool to be executed, if any
        """
        if query:
            self.history.append(BaseMessageParam(role="user", content=query))
            
        stream = self._stream(query)
        tools_and_outputs = []
        
        for chunk, tool in stream:
            if tool:
                # Yield the tool for external handling
                yield None, tool
                result = tool.call()
                tools_and_outputs.append((tool, result))
            else:
                # Yield the content chunk
                yield chunk.content, None
                
        # Add to history after stream completes
        self.history.append(stream.message_param)
        
        if tools_and_outputs:
            self.history.extend(stream.tool_message_params(tools_and_outputs))
            # Recursively stream follow-up response
            async for chunk, tool in self._stream_step(""):
                yield chunk, tool

    # async def use_tool(self, tool_class: type[BaseTool], args: Dict[str, Any]) -> Any:
    #     """Execute a tool directly with the given arguments.
        
    #     This method allows direct tool execution without going through the LLM.
    #     Useful for when we know exactly which tool to use and with what arguments.
        
    #     Args:
    #         tool_class: The tool class to instantiate
    #         args: Arguments to pass to the tool
            
    #     Returns:
    #         Any: The result of the tool execution
            
    #     Raises:
    #         ValueError: If the tool class is not available to this agent
    #         Exception: Any exception raised by the tool during execution
    #     """
    #     if tool_class not in self.tools:
    #         raise ValueError(f"Tool {tool_class.__name__} not available to this agent")
            
    #     try:
    #         # Create tool instance with args
    #         tool = tool_class(**args)
            
    #         # Log tool execution
    #         if self.logging_config.show_tool_calls or \
    #            self.logging_config.level <= VerbosityLevel.DEBUG:
    #             log_verbose(logger, f"Executing tool {tool._name()} with args: {args}")
            
    #         # Execute tool
    #         if inspect.iscoroutinefunction(tool.call):
    #             result = await tool.call()
    #         else:
    #             result = tool.call()
                
    #         # Log result
    #         logger.debug(f"Tool {tool._name()} result: {result}")
            
    #         return result
            
    #     except Exception as e:
    #         logger.error(f"Error executing tool {tool_class.__name__}: {str(e)}")
    #         raise


if __name__ == "__main__":
    import asyncio
    from alchemist.ai.base.logging import configure_logging, LogLevel
    from pydantic import BaseModel, Field
    
    # Define example response models
    class ChatResponse(BaseModel):
        """Structured chat response with sentiment and tags."""
        message: str = Field(..., description="The main response message")
        sentiment: str = Field(..., description="Emotional tone of the message")
        tags: list[str] = Field(default_factory=list, description="Relevant topic tags")

    class CalculationResponse(BaseModel):
        """Structured response for calculations."""
        result: float = Field(..., description="The calculated result")
        explanation: str = Field(..., description="Step-by-step explanation")
        
    class PersonaConfig(BaseModel):
        """Example persona configuration."""
        name: str = Field(..., description="Agent name")
        role: str = Field(..., description="Agent role")
        traits: list[str] = Field(..., description="Personality traits")
        
    async def main():
        """Run incremental tests for agent functionality."""
        configure_logging(
            default_level=LogLevel.INFO,
            component_levels={
                LogComponent.AGENT: LogLevel.DEBUG
            }
        )
        
        # Test 1: Basic streaming without tools
        # print("\n=== Test 1: Basic Streaming ===")
        # stream_agent = BaseAgent(stream=True)
        # print("Testing streaming (type 'next' to continue, 'exit' to quit):")
        # while True:
        #     query = input("(User): ")
        #     if query.lower() == 'next':
        #         break
        #     if query.lower() in ["exit", "quit"]:
        #         return
                
        #     print("(Assistant): ", end="", flush=True)
        #     async for chunk, tool in stream_agent._stream_step(query):
        #         if chunk:
        #             print(chunk, end="", flush=True)
        #     print()

        # # Test 2: Single step without tools
        # print("\n=== Test 2: Single Step ===")
        # basic_agent = BaseAgent()
        # response = await basic_agent._step("Tell me a short joke.")
        # print(f"Response: {response}")

        # # Test 3: Single step with calculator tool
        # print("\n=== Test 3: Single Step with Calculator ===")
        # tool_agent = BaseAgent(tools=[CalculatorTool])
        # response = await tool_agent._step("What is 15 * 7?")
        # print(f"Response: {response}")

        # # Test 4: Streaming with calculator tool
        # print("\n=== Test 4: Streaming with Calculator ===")
        # stream_tool_agent = BaseAgent(tools=[CalculatorTool], stream=True)
        # print("Testing streaming with tool (type 'next' to continue, 'exit' to quit):")
        # while True:
        #     query = input("(User): ")
        #     if query.lower() == 'next':
        #         break
        #     if query.lower() in ["exit", "quit"]:
        #         return
                
        #     print("(Assistant): ", end="", flush=True)
        #     async for chunk, tool in stream_tool_agent._stream_step(query):
        #         if tool:
        #             print(f"\n[Calling Tool '{tool._name()}' with args {tool.args}]")
        #         elif chunk:
        #             print(chunk, end="", flush=True)
        #     print()

        # # Test 5a: Structured output with response model and json_mode=True
        # print("\n=== Test 5a: Structured Output (Response Model + JSON Mode) ===")
        # structured_agent = BaseAgent(
        #     response_model=ChatResponse,
        #     json_mode=True
        # )
        # response = await structured_agent._step("Tell me about AI.")
        # print(f"Response Model + JSON Mode:\nMessage: {response.message}")
        # print(f"Sentiment: {response.sentiment}")
        # print(f"Tags: {', '.join(response.tags)}")

        # # Test 5b: Structured output with response model and json_mode=False
        # print("\n=== Test 5b: Structured Output (Response Model Only) ===")
        # model_agent = BaseAgent(
        #     response_model=ChatResponse,
        #     json_mode=False
        # )
        # response = await model_agent._step("Tell me about robots.")
        # print(f"Response Model Only:\nMessage: {response.message}")
        # print(f"Sentiment: {response.sentiment}")
        # print(f"Tags: {', '.join(response.tags)}")

        # Test 5c: JSON mode without response model
        # print("\n=== Test 5c: JSON Mode Only ===")
        # json_agent = BaseAgent(json_mode=True)
        # response = await json_agent._step("Tell me about machine learning.")
        # print(f"JSON Mode Only:\n{json.dumps(response, indent=2)}")

    # Run all tests
    print("\nStarting incremental agent tests...")
    asyncio.run(main())

    