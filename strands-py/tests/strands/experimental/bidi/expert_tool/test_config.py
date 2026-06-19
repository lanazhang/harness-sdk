"""Unit tests for ExpertToolConfig."""

import sys

if sys.version_info < (3, 12):
    import pytest

    pytest.skip(reason="Expert Tool requires Python 3.12+", allow_module_level=True)

from unittest.mock import AsyncMock, Mock

import pytest

from strands.experimental.bidi.expert_tool.config import ExpertToolConfig


def test_config_defaults():
    """Test config creation with defaults."""
    reasoner = AsyncMock()
    config = ExpertToolConfig(reasoner=reasoner)

    assert config.reasoner is reasoner
    assert config.max_tool_iterations == 5
    assert config.streaming_strategy == "sentence"
    assert config.on_interrupted is None
    assert config.fallback_message == "I'm not sure how to help with that."


def test_config_custom_values():
    """Test config with custom values."""
    callback = Mock()
    config = ExpertToolConfig(
        reasoner=AsyncMock(),
        max_tool_iterations=10,
        streaming_strategy="token",
        on_interrupted=callback,
        fallback_message="Custom fallback.",
    )

    assert config.max_tool_iterations == 10
    assert config.streaming_strategy == "token"
    assert config.on_interrupted is callback
    assert config.fallback_message == "Custom fallback."
