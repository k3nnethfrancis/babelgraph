"""Logging Configuration with pretty formatting for Alchemist."""

import logging
from typing import Optional, Dict, Any
from enum import Enum, IntEnum
import json
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

# ANSI Color Codes
class Colors:
    """ANSI color codes for pretty terminal output."""
    HEADER = '\033[95m'      # Pink
    INFO = '\033[94m'        # Blue
    SUCCESS = '\033[92m'     # Green
    WARNING = '\033[93m'     # Yellow
    ERROR = '\033[91m'       # Red
    RESET = '\033[0m'        # Reset
    BOLD = '\033[1m'         # Bold
    DIM = '\033[2m'          # Dim
    ITALIC = '\033[3m'       # Italic
    UNDERLINE = '\033[4m'    # Underline

# Pretty format strings
PRETTY_FORMAT = (
    "%(asctime)s │ %(levelname)-8s │ %(message)s"
)

DETAILED_FORMAT = (
    f"{Colors.DIM}%(asctime)s{Colors.RESET} │ "
    f"%(colored_level)-40s │ "
    f"%(message)s"
)

class PrettyFormatter(logging.Formatter):
    """Custom formatter with colors and symbols."""
    
    level_colors = {
        'DEBUG': (Colors.DIM, '🔍'),
        'INFO': (Colors.INFO, 'ℹ️'),
        'WARNING': (Colors.WARNING, '⚠️'),
        'ERROR': (Colors.ERROR, '❌'),
        'CRITICAL': (Colors.ERROR + Colors.BOLD, '🚨')
    }

    def format(self, record):
        # Add colored level with symbol
        color, symbol = self.level_colors.get(record.levelname, (Colors.RESET, '•'))
        record.colored_level = f"{color}{symbol} {record.levelname}{Colors.RESET}"
        
        # Format the message
        message = super().format(record)
        
        # Add separator line for errors and warnings
        if record.levelno >= logging.WARNING:
            message = f"{message}\n{Colors.DIM}{'─' * 80}{Colors.RESET}"
            
        return message

class PrettyLogHandler(logging.StreamHandler):
    """Handler that adds pretty formatting to log records."""
    
    def emit(self, record):
        try:
            # Add timestamp formatting
            record.asctime = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            
            # Format and print
            msg = self.format(record)
            print(msg)
            
            # Add spacing after certain types of messages
            if record.levelno >= logging.WARNING:
                print()
                
        except Exception as e:
            print(f"{Colors.ERROR}Error in logging: {e}{Colors.RESET}")

class LogFormat(str, Enum):
    """Predefined log formats."""
    DEFAULT = PRETTY_FORMAT
    DEBUG = DETAILED_FORMAT
    SIMPLE = PRETTY_FORMAT

class LogComponent(str, Enum):
    """Components that can be logged.
    
    Each component represents a major subsystem in the Alchemist framework.
    The value is the logger name used in the logging hierarchy.
    """
    AGENT = "alchemist.ai.base.agent"          # Base agent functionality
    RUNTIME = "alchemist.ai.base.runtime"       # Runtime environment
    GRAPH = "alchemist.ai.graph.base"          # Graph system core
    NODES = "alchemist.ai.graph.nodes"         # Graph nodes
    TOOLS = "alchemist.ai.tools"               # Tool implementations
    PROMPTS = "alchemist.ai.prompts"           # Prompt management
    DISCORD = "alchemist.core.extensions.discord"  # Discord integration
    WORKFLOW = "alchemist.ai.graph.workflow"    # Workflow execution
    SESSION = "alchemist.core.session"          # Session management
    MEMORY = "alchemist.core.memory"           # Memory systems

class LogLevel(int, Enum):
    """Log levels mapped to logging module levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class VerbosityLevel(IntEnum):
    """
    Custom verbosity levels for more granular control.
    Matches or extends standard Python logging levels:
        - 10 => DEBUG
        - 15 => VERBOSE (optional custom level)
        - 20 => INFO
    """
    DEBUG = logging.DEBUG
    VERBOSE = 15  # Custom lower-than-INFO level if desired
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

logging.addLevelName(VerbosityLevel.VERBOSE, "VERBOSE")

class AlchemistLoggingConfig(BaseModel):
    """
    Pydantic model for configuring logging verbosity.

    Attributes:
        level: The main log level (DEBUG, VERBOSE, INFO, etc.).
        show_llm_messages: Whether to display full LLM messages in logs.
        show_tool_calls: Whether to display tool call details.
        show_node_transitions: Whether to display graph node transitions in logs.
    """
    level: VerbosityLevel = Field(default=VerbosityLevel.INFO)
    show_llm_messages: bool = Field(default=False)
    show_tool_calls: bool = Field(default=False)
    show_node_transitions: bool = Field(default=False)

def log_verbose(logger: logging.Logger, message: str) -> None:
    """
    Custom helper to log messages at our VERBOSE level.
    
    Args:
        logger: The logger to use.
        message: The message to log.
    """
    if logger.isEnabledFor(VerbosityLevel.VERBOSE):
        logger.log(VerbosityLevel.VERBOSE, message)

def configure_logging(
    default_level: LogLevel = LogLevel.INFO,
    component_levels: Optional[Dict[LogComponent, LogLevel]] = None,
    pretty: bool = True,
    log_file: Optional[str] = None
) -> None:
    """Configure logging with pretty formatting.
    
    Args:
        default_level: Default logging level
        component_levels: Optional component-specific levels
        pretty: Whether to use pretty formatting (default: True)
        log_file: Optional file path for logging
    """
    # Set up handlers
    handlers = []
    
    # Console handler with pretty formatting
    console_handler = PrettyLogHandler() if pretty else logging.StreamHandler()
    console_handler.setFormatter(
        PrettyFormatter(DETAILED_FORMAT if pretty else PRETTY_FORMAT)
    )
    handlers.append(console_handler)
    
    # File handler if specified (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(PRETTY_FORMAT))
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(default_level.value)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Configure component levels
    if component_levels:
        for component, level in component_levels.items():
            logger = logging.getLogger(component.value)
            logger.setLevel(level.value)

class JsonLogHandler(logging.Handler):
    """Handler that formats log records as JSON."""
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as JSON."""
        try:
            msg = self.format(record)
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
                "function": record.funcName,
                "line": record.lineno,
                "path": record.pathname
            }
            
            # Add extra fields if present
            if hasattr(record, "extra"):
                log_entry.update(record.extra)
            
            print(json.dumps(log_entry))
            
        except Exception as e:
            print(f"Error in JSON logging: {e}")

def get_logger(component: LogComponent) -> logging.Logger:
    """Get a logger for a specific component.
    
    Args:
        component: The component to get a logger for
        
    Returns:
        Logger configured for the component
    """
    return logging.getLogger(component.value)

def log_state(logger: logging.Logger, state: Dict[str, Any], prefix: str = "") -> None:
    """Log a state dictionary in a readable format.
    
    Args:
        logger: Logger to use
        state: State dictionary to log
        prefix: Optional prefix for log messages
    """
    try:
        for key, value in state.items():
            if isinstance(value, dict):
                logger.debug(f"{prefix}{key}:")
                log_state(logger, value, prefix + "  ")
            else:
                logger.debug(f"{prefix}{key}: {value}")
    except Exception as e:
        logger.error(f"Error logging state: {e}")

def set_component_level(component: LogComponent, level: LogLevel) -> None:
    """Set logging level for a specific component.
    
    Args:
        component: Component to configure
        level: Logging level to set
    """
    logging.getLogger(component.value).setLevel(level.value)

def disable_all_logging() -> None:
    """Disable logging for all components."""
    for component in LogComponent:
        logging.getLogger(component.value).setLevel(logging.CRITICAL)

def enable_debug_mode() -> None:
    """Enable debug logging for all components."""
    for component in LogComponent:
        logging.getLogger(component.value).setLevel(logging.DEBUG) 