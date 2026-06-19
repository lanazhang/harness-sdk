"""Integration tests for Expert Tool in BidiNovaSonicModel."""

import sys

if sys.version_info < (3, 12):
    import pytest

    pytest.skip(reason="Expert Tool requires Python 3.12+", allow_module_level=True)

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands.experimental.bidi.expert_tool.config import ExpertToolConfig
from strands.experimental.bidi.expert_tool.manager import _ExpertToolManager
from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel, NOVA_SONIC_V1_MODEL_ID


@pytest.fixture
def boto_session():
    return Mock(region_name="us-east-1")


@pytest.fixture
def mock_client():
    with patch("strands.experimental.bidi.models.nova_sonic.BedrockRuntimeClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.input_stream = AsyncMock()
        mock_stream.input_stream.send = AsyncMock()
        mock_stream.input_stream.close = AsyncMock()
        mock_instance.invoke_model_with_bidirectional_stream = AsyncMock(return_value=mock_stream)
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_reasoner():
    async def _reason(messages):
        yield "Response."

    reasoner = Mock()
    reasoner.reason = _reason
    return reasoner


class TestInit:
    def test_with_reasoner(self, boto_session, mock_client):
        reasoner = Mock()
        reasoner.reason = AsyncMock()
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=reasoner)
        assert isinstance(model._expert_tool_manager, _ExpertToolManager)

    def test_without_reasoner(self, boto_session, mock_client):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session})
        assert model._expert_tool_manager is None

    def test_raises_both_reasoner_and_config(self, boto_session, mock_client):
        reasoner = Mock()
        reasoner.reason = AsyncMock()
        with pytest.raises(ValueError, match="Cannot specify both"):
            BidiNovaSonicModel(
                client_config={"boto_session": boto_session},
                reasoner=reasoner,
                expert_tool=ExpertToolConfig(reasoner=reasoner),
            )

    def test_raises_v1_with_expert_tool(self, boto_session, mock_client):
        reasoner = Mock()
        reasoner.reason = AsyncMock()
        with pytest.raises(ValueError, match="only supported in Nova Sonic v2"):
            BidiNovaSonicModel(
                model_id=NOVA_SONIC_V1_MODEL_ID,
                client_config={"boto_session": boto_session},
                reasoner=reasoner,
            )


class TestExpertToolSpec:
    def test_prompt_start_includes_expert_tool(self, boto_session, mock_client, mock_reasoner):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=mock_reasoner)
        model._connection_id = "test-conn"
        event = json.loads(model._get_prompt_start_event(tools=[]))
        tools = event["event"]["promptStart"]["toolConfiguration"]["tools"]
        assert any(t["toolSpec"]["name"] == "ExpertTool" for t in tools)

    def test_prompt_start_no_expert_tool_without_reasoner(self, boto_session, mock_client):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session})
        model._connection_id = "test-conn"
        event = json.loads(model._get_prompt_start_event(tools=[]))
        assert "toolConfiguration" not in event["event"]["promptStart"]


class TestEventInterception:
    @pytest.mark.asyncio
    async def test_expert_tool_intercepted(self, boto_session, mock_client, mock_reasoner):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=mock_reasoner)
        model._expert_tool_manager.handle_invocation = AsyncMock()
        nova_event = {"toolUse": {"toolUseId": "t1", "toolName": "ExpertTool", "content": "{}"}}
        result = model._convert_nova_event(nova_event)
        assert result is None

    def test_regular_tool_passes_through(self, boto_session, mock_client, mock_reasoner):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=mock_reasoner)
        nova_event = {"toolUse": {"toolUseId": "t1", "toolName": "get_weather", "content": '{"city":"NYC"}'}}
        result = model._convert_nova_event(nova_event)
        assert result is not None
        assert result["delta"]["toolUse"]["name"] == "get_weather"
