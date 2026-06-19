"""Integration tests for Expert Tool support in BidiNovaSonicModel.

Tests the expert tool wiring within nova_sonic.py: initialization,
ExpertTool spec injection in promptStart, and event interception.
"""

import sys

if sys.version_info < (3, 12):
    import pytest

    pytest.skip(reason="Expert Tool requires Python 3.12+", allow_module_level=True)

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands.experimental.bidi.expert_tool.config import ExpertToolConfig
from strands.experimental.bidi.expert_tool.manager import _ExpertToolManager
from strands.experimental.bidi.expert_tool.reasoner import ExpertToolReasoner
from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel


@pytest.fixture
def boto_session():
    return Mock(region_name="us-east-1")


@pytest.fixture
def mock_stream():
    """Mock Nova Sonic bidirectional stream."""
    stream = AsyncMock()
    stream.input_stream = AsyncMock()
    stream.input_stream.send = AsyncMock()
    stream.input_stream.close = AsyncMock()
    stream.await_output = AsyncMock()
    return stream


@pytest.fixture
def mock_client(mock_stream):
    """Mock Bedrock Runtime client."""
    with patch("strands.experimental.bidi.models.nova_sonic.BedrockRuntimeClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_instance.invoke_model_with_bidirectional_stream = AsyncMock(return_value=mock_stream)
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_reasoner():
    """Mock ExpertToolReasoner."""

    async def _reason(messages):
        yield "Test response from reasoner."

    reasoner = Mock()
    reasoner.reason = _reason
    return reasoner


class TestNovaSonicExpertToolInit:
    """Tests for BidiNovaSonicModel initialization with expert tool."""

    def test_init_with_reasoner(self, boto_session, mock_client):
        """Test model initializes expert tool manager when reasoner is provided."""
        _ = mock_client

        reasoner = Mock()
        reasoner.reason = AsyncMock()
        model = BidiNovaSonicModel(
            client_config={"boto_session": boto_session},
            reasoner=reasoner,
        )

        assert model._expert_tool_manager is not None
        assert isinstance(model._expert_tool_manager, _ExpertToolManager)

    def test_init_with_expert_tool_config(self, boto_session, mock_client):
        """Test model initializes with ExpertToolConfig."""
        _ = mock_client

        reasoner = Mock()
        reasoner.reason = AsyncMock()
        config = ExpertToolConfig(reasoner=reasoner)
        model = BidiNovaSonicModel(
            client_config={"boto_session": boto_session},
            expert_tool=config,
        )

        assert model._expert_tool_manager is not None

    def test_init_without_expert_tool(self, boto_session, mock_client):
        """Test model initializes without expert tool manager by default."""
        _ = mock_client

        model = BidiNovaSonicModel(client_config={"boto_session": boto_session})

        assert model._expert_tool_manager is None

    def test_init_raises_on_both_reasoner_and_expert_tool(self, boto_session, mock_client):
        """Test ValueError when both reasoner and expert_tool are specified."""
        _ = mock_client

        reasoner = Mock()
        reasoner.reason = AsyncMock()
        config = ExpertToolConfig(reasoner=reasoner)

        with pytest.raises(ValueError, match="Cannot specify both"):
            BidiNovaSonicModel(
                client_config={"boto_session": boto_session},
                reasoner=reasoner,
                expert_tool=config,
            )

    def test_init_raises_expert_tool_on_v1(self, boto_session, mock_client):
        """Test ValueError when expert tool is used with Nova Sonic v1."""
        _ = mock_client

        reasoner = Mock()
        reasoner.reason = AsyncMock()

        with pytest.raises(ValueError, match="only supported in Nova Sonic v2"):
            BidiNovaSonicModel(
                model_id="amazon.nova-sonic-v1:0",
                client_config={"boto_session": boto_session},
                reasoner=reasoner,
            )


class TestNovaSonicExpertToolSpec:
    """Tests for ExpertTool spec generation and injection."""

    def test_build_expert_tool_spec(self, boto_session, mock_client, mock_reasoner):
        """Test _build_expert_tool_spec returns correct schema."""
        _ = mock_client

        model = BidiNovaSonicModel(
            client_config={"boto_session": boto_session},
            reasoner=mock_reasoner,
        )

        spec = model._build_expert_tool_spec()

        assert spec["toolSpec"]["name"] == "ExpertTool"
        assert "reasoning" in spec["toolSpec"]["description"].lower()

        # Verify input schema
        schema = json.loads(spec["toolSpec"]["inputSchema"]["json"])
        assert schema["type"] == "object"
        assert "messages" in schema["required"]
        assert schema["properties"]["messages"]["type"] == "array"

    def test_prompt_start_includes_expert_tool(self, boto_session, mock_client, mock_reasoner):
        """Test _get_prompt_start_event includes ExpertTool spec."""
        _ = mock_client

        model = BidiNovaSonicModel(
            client_config={"boto_session": boto_session},
            reasoner=mock_reasoner,
        )
        model._connection_id = "test-connection"

        event_json = model._get_prompt_start_event(tools=[])
        event = json.loads(event_json)

        # Should have tool configuration with ExpertTool
        prompt_start = event["event"]["promptStart"]
        assert "toolConfiguration" in prompt_start
        tools = prompt_start["toolConfiguration"]["tools"]
        expert_tools = [t for t in tools if t["toolSpec"]["name"] == "ExpertTool"]
        assert len(expert_tools) == 1

    def test_prompt_start_expert_tool_alongside_regular_tools(self, boto_session, mock_client, mock_reasoner):
        """Test ExpertTool is appended alongside regular tools."""
        _ = mock_client

        model = BidiNovaSonicModel(
            client_config={"boto_session": boto_session},
            reasoner=mock_reasoner,
        )
        model._connection_id = "test-connection"

        regular_tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {"json": json.dumps({"type": "object", "properties": {}})},
            }
        ]

        event_json = model._get_prompt_start_event(tools=regular_tools)
        event = json.loads(event_json)

        tools = event["event"]["promptStart"]["toolConfiguration"]["tools"]
        # Should have both regular tool + ExpertTool
        assert len(tools) == 2
        tool_names = [t["toolSpec"]["name"] for t in tools]
        assert "get_weather" in tool_names
        assert "ExpertTool" in tool_names

    def test_prompt_start_without_expert_tool(self, boto_session, mock_client):
        """Test promptStart does NOT include ExpertTool when not configured."""
        _ = mock_client

        model = BidiNovaSonicModel(client_config={"boto_session": boto_session})
        model._connection_id = "test-connection"

        event_json = model._get_prompt_start_event(tools=[])
        event = json.loads(event_json)

        prompt_start = event["event"]["promptStart"]
        # No tool configuration at all when no tools and no expert tool
        assert "toolConfiguration" not in prompt_start


class TestNovaSonicExpertToolInterception:
    """Tests for ExpertTool event interception in _convert_nova_event."""

    @pytest.fixture
    def model_with_expert(self, boto_session, mock_client, mock_reasoner):
        """Create model with expert tool manager."""
        _ = mock_client
        model = BidiNovaSonicModel(
            client_config={"boto_session": boto_session},
            reasoner=mock_reasoner,
        )
        model._connection_id = "test-connection"
        # Mock the manager's handle_invocation to avoid actual task creation
        model._expert_tool_manager.handle_invocation = AsyncMock()
        return model

    @pytest.mark.asyncio
    async def test_expert_tool_intercepted(self, model_with_expert):
        """Test ExpertTool toolUse events are intercepted and not emitted."""
        nova_event = {
            "toolUse": {
                "toolUseId": "tu-expert-1",
                "toolName": "ExpertTool",
                "content": json.dumps({"messages": [{"role": "USER", "content": [{"text": "Hi"}]}]}),
            }
        }

        result = model_with_expert._convert_nova_event(nova_event)

        # Should return None (not emitted to agent loop)
        assert result is None

    def test_regular_tool_not_intercepted(self, model_with_expert):
        """Test regular toolUse events pass through normally."""
        nova_event = {
            "toolUse": {
                "toolUseId": "tu-regular-1",
                "toolName": "get_weather",
                "content": json.dumps({"location": "Seattle"}),
            }
        }

        result = model_with_expert._convert_nova_event(nova_event)

        # Should return a ToolUseStreamEvent
        assert result is not None
        assert "delta" in result
        assert result["delta"]["toolUse"]["name"] == "get_weather"

    def test_expert_tool_without_manager(self, boto_session, mock_client):
        """Test ExpertTool events pass through when no manager configured."""
        _ = mock_client
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session})

        nova_event = {
            "toolUse": {
                "toolUseId": "tu-expert-1",
                "toolName": "ExpertTool",
                "content": json.dumps({"messages": []}),
            }
        }

        result = model._convert_nova_event(nova_event)

        # Should treat as regular tool use
        assert result is not None
        assert "delta" in result


class TestNovaSonicExpertToolStop:
    """Tests for model stop with expert tool cleanup."""

    @pytest.mark.asyncio
    async def test_stop_shuts_down_manager(self, boto_session, mock_client, mock_reasoner, mock_stream):
        """Test model.stop() calls manager.shutdown()."""
        _ = mock_client
        model = BidiNovaSonicModel(
            client_config={"boto_session": boto_session},
            reasoner=mock_reasoner,
        )

        # Mock the manager shutdown
        model._expert_tool_manager.shutdown = AsyncMock()

        # Start and stop
        await model.start()
        await model.stop()

        model._expert_tool_manager.shutdown.assert_called_once()
