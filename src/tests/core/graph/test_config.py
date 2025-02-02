"""Tests for graph configuration functionality.

This module tests the graph configuration system including:
- Configuration validation
- Default values
- Custom settings
- Environment variable integration
"""

import pytest
from typing import Dict, Any
import os

from alchemist.ai.graph.config import GraphConfig


@pytest.fixture
def basic_config() -> GraphConfig:
    """Fixture providing a basic graph configuration."""
    return GraphConfig()


class TestConfigInitialization:
    """Test suite for configuration initialization."""

    def test_config_init(self, basic_config: GraphConfig):
        """Test basic configuration initialization."""
        assert isinstance(basic_config.max_parallel, int)
        assert isinstance(basic_config.timeout, int)
        assert isinstance(basic_config.retry_count, int)

    def test_config_with_values(self):
        """Test configuration initialization with custom values."""
        config = GraphConfig(
            max_parallel=5,
            timeout=30,
            retry_count=3
        )
        assert config.max_parallel == 5
        assert config.timeout == 30
        assert config.retry_count == 3

    def test_config_from_dict(self):
        """Test configuration initialization from dictionary."""
        data = {
            "max_parallel": 10,
            "timeout": 60,
            "retry_count": 2
        }
        config = GraphConfig.parse_obj(data)
        assert config.max_parallel == 10
        assert config.timeout == 60
        assert config.retry_count == 2


class TestConfigValidation:
    """Test suite for configuration validation."""

    def test_max_parallel_validation(self):
        """Test max_parallel validation."""
        with pytest.raises(ValueError):
            GraphConfig(max_parallel=0)
        with pytest.raises(ValueError):
            GraphConfig(max_parallel=-1)

    def test_timeout_validation(self):
        """Test timeout validation."""
        with pytest.raises(ValueError):
            GraphConfig(timeout=0)
        with pytest.raises(ValueError):
            GraphConfig(timeout=-1)

    def test_retry_count_validation(self):
        """Test retry_count validation."""
        with pytest.raises(ValueError):
            GraphConfig(retry_count=-1)


class TestDefaultValues:
    """Test suite for default configuration values."""

    def test_default_max_parallel(self, basic_config: GraphConfig):
        """Test default max_parallel value."""
        assert basic_config.max_parallel > 0

    def test_default_timeout(self, basic_config: GraphConfig):
        """Test default timeout value."""
        assert basic_config.timeout > 0

    def test_default_retry_count(self, basic_config: GraphConfig):
        """Test default retry_count value."""
        assert basic_config.retry_count >= 0


class TestEnvironmentVariables:
    """Test suite for environment variable integration."""

    def test_env_max_parallel(self, monkeypatch):
        """Test max_parallel from environment variable."""
        monkeypatch.setenv("ALCHEMIST_MAX_PARALLEL", "8")
        config = GraphConfig()
        assert config.max_parallel == 8

    def test_env_timeout(self, monkeypatch):
        """Test timeout from environment variable."""
        monkeypatch.setenv("ALCHEMIST_TIMEOUT", "45")
        config = GraphConfig()
        assert config.timeout == 45

    def test_env_retry_count(self, monkeypatch):
        """Test retry_count from environment variable."""
        monkeypatch.setenv("ALCHEMIST_RETRY_COUNT", "5")
        config = GraphConfig()
        assert config.retry_count == 5

    def test_invalid_env_values(self, monkeypatch):
        """Test handling of invalid environment variable values."""
        monkeypatch.setenv("ALCHEMIST_MAX_PARALLEL", "invalid")
        config = GraphConfig()
        assert isinstance(config.max_parallel, int)  # Should use default value


class TestConfigSerialization:
    """Test suite for configuration serialization."""

    def test_config_to_dict(self, basic_config: GraphConfig):
        """Test configuration serialization to dictionary."""
        data = basic_config.model_dump()
        assert isinstance(data, dict)
        assert "max_parallel" in data
        assert "timeout" in data
        assert "retry_count" in data

    def test_config_json(self, basic_config: GraphConfig):
        """Test configuration JSON serialization."""
        json_str = basic_config.model_dump_json()
        assert isinstance(json_str, str)
        assert "max_parallel" in json_str
        assert "timeout" in json_str
        assert "retry_count" in json_str


class TestConfigUpdate:
    """Test suite for configuration updates."""

    def test_update_values(self, basic_config: GraphConfig):
        """Test updating configuration values."""
        basic_config.max_parallel = 15
        basic_config.timeout = 90
        basic_config.retry_count = 4
        
        assert basic_config.max_parallel == 15
        assert basic_config.timeout == 90
        assert basic_config.retry_count == 4

    def test_update_validation(self, basic_config: GraphConfig):
        """Test validation during updates."""
        with pytest.raises(ValueError):
            basic_config.max_parallel = -1
        with pytest.raises(ValueError):
            basic_config.timeout = 0
        with pytest.raises(ValueError):
            basic_config.retry_count = -2
