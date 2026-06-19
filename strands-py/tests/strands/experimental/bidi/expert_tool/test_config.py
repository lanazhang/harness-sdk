"""Unit tests for ExpertToolConfig dataclass."""

import sys

if sys.version_info < (3, 12):
    import pytest

    pytest.skip(reason="Expert Tool requires Python 3.12+", allow_module_level=True)

from unittest.mock import AsyncMock, Mock

import pytest

from strands.experimental.bidi.expert_tool.config import ExpertToolConfig


@pytest.fixture
def mock_reasoner():
    """Create a mock ExpertToolReasoner."""
    reasoner = AsyncMock()
    reasoner.reason = AsyncMock()
    return reasoner


class TestExpertToolConfig:
    """Tests for ExpertToolConfig dataclass."""

    def test_config_with_reasoner(self, mock_reasoner):
        """Test config creation with a reasoner."""
        config = ExpertToolConfig(reasoner=mock_reasoner)

        assert config.reasoner is mock_reasoner
        assert config.max_tool_iterations == 5
        assert config.streaming_strategy == "sentence"
        assert config.on_interrupted is None
        assert config.fallback_message == "I'm not sure how to help with that."

    def test_config_custom_values(self, mock_reasoner):
        """Test config creation with custom values."""
        callback = Mock()
        config = ExpertToolConfig(
            reasoner=mock_reasoner,
            max_tool_iterations=10,
            streaming_strategy="token",
            on_interrupted=callback,
            fallback_message="Custom fallback.",
        )

        assert config.max_tool_iterations == 10
        assert config.streaming_strategy == "token"
        assert config.on_interrupted is callback
        assert config.fallback_message == "Custom fallback."

    def test_config_default_max_iterations(self, mock_reasoner):
        """Test that default max_tool_iterations is 5."""
        config = ExpertToolConfig(reasoner=mock_reasoner)
        assert config.max_tool_iterations == 5

    def test_config_default_streaming_strategy(self, mock_reasoner):
        """Test that default streaming_strategy is 'sentence'."""
        config = ExpertToolConfig(reasoner=mock_reasoner)
        assert config.streaming_strategy == "sentence"

    def test_config_on_interrupted_callable(self, mock_reasoner):
        """Test that on_interrupted accepts a callable."""
        called = []
        config = ExpertToolConfig(
            reasoner=mock_reasoner,
            on_interrupted=lambda: called.append(True),
        )
        config.on_interrupted()
        assert called == [True]
