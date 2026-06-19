"""Unit tests for Expert Tool support in BidiNovaSonicModel."""

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
from strands.experimental.bidi.expert_tool.reasoner import BedrockConverseReasoner, ExpertToolReasoner, StrandsAgentReasoner
from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel, NOVA_SONIC_V1_MODEL_ID


@pytest.fixture
def boto_session():
    return Mock(region_name="us-east-1")


@pytest.fixture
def mock_client():
    with patch("strands.experimental.bidi.models.nova_sonic.BedrockRuntimeClient") as cls:
        inst = AsyncMock()
        stream = AsyncMock()
        stream.input_stream = AsyncMock()
        stream.input_stream.send = AsyncMock()
        inst.invoke_model_with_bidirectional_stream = AsyncMock(return_value=stream)
        cls.return_value = inst
        yield inst


@pytest.fixture
def mock_model():
    m = AsyncMock()
    m._connection_id = "conn-1"
    m._send_nova_events = AsyncMock()
    return m


@pytest.fixture
def mock_reasoner():
    async def _reason(messages):
        yield "Hello."

    r = Mock()
    r.reason = _reason
    return r


# --- Config ---


def test_config_defaults():
    config = ExpertToolConfig(reasoner=Mock())
    assert config.max_tool_iterations == 5
    assert config.streaming_strategy == "sentence"
    assert config.fallback_message == "I'm not sure how to help with that."


# --- Protocol ---


def test_protocol_compliance():
    assert isinstance(BedrockConverseReasoner(model=Mock(config={"model_id": "x"}, client=Mock())), ExpertToolReasoner)
    assert isinstance(StrandsAgentReasoner(agent=Mock()), ExpertToolReasoner)
    assert not isinstance(object(), ExpertToolReasoner)


# --- BedrockConverseReasoner ---


class TestBedrockReasoner:
    @pytest.fixture
    def model(self):
        m = Mock()
        m.config = {"model_id": "test", "max_tokens": 1024, "temperature": 0.7}
        m.client = Mock()
        return m

    def test_convert_messages_strips_leading_assistant(self, model):
        r = BedrockConverseReasoner(model=model)
        result = r._convert_messages([
            {"role": "ASSISTANT", "content": [{"text": "Hi"}]},
            {"role": "USER", "content": [{"text": "Hello"}]},
        ])
        assert len(result) == 1 and result[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_execute_tool(self, model):
        tool = Mock(return_value='{"ok":true}')
        tool.tool_spec = {"name": "t", "description": "d", "inputSchema": {"json": {}}}
        r = BedrockConverseReasoner(model=model, tools=[tool])
        assert await r._execute_tool("t", {"x": 1}) == '{"ok":true}'
        assert "Unknown" in json.loads(await r._execute_tool("nope", {}))["error"]

    @pytest.mark.asyncio
    async def test_reason_streams_text(self, model):
        model.client.converse_stream.return_value = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "A.\nB."}}},
                {"contentBlockStop": {}},
            ]
        }
        r = BedrockConverseReasoner(model=model)
        result = [s async for s in r.reason([{"role": "USER", "content": [{"text": "hi"}]}])]
        assert "A." in result and "B." in result

    @pytest.mark.asyncio
    async def test_reason_handles_tool_call(self, model):
        tool = Mock(return_value='{"r":1}')
        tool.tool_spec = {"name": "t", "description": "d", "inputSchema": {"json": {}}}
        model.client.converse_stream.side_effect = [
            {"stream": [
                {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "id1", "name": "t"}}}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}},
                {"contentBlockStop": {}},
            ]},
            {"stream": [{"contentBlockDelta": {"delta": {"text": "Done."}}}, {"contentBlockStop": {}}]},
        ]
        r = BedrockConverseReasoner(model=model, tools=[tool])
        result = [s async for s in r.reason([{"role": "USER", "content": [{"text": "go"}]}])]
        tool.assert_called_once()
        assert "Done." in result


# --- StrandsAgentReasoner ---


class TestStrandsReasoner:
    @pytest.mark.asyncio
    async def test_reason(self):
        agent = Mock(return_value="Line 1.\nLine 2.")
        r = StrandsAgentReasoner(agent=agent)
        result = [s async for s in r.reason([{"role": "USER", "content": [{"text": "hi"}]}])]
        agent.assert_called_once_with("hi")
        assert result == ["Line 1.", "Line 2."]

    @pytest.mark.asyncio
    async def test_no_user_message_fallback(self):
        r = StrandsAgentReasoner(agent=Mock())
        result = [s async for s in r.reason([{"role": "ASSISTANT", "content": [{"text": "x"}]}])]
        assert "didn't catch" in result[0].lower()


# --- Manager ---


class TestManager:
    @pytest.mark.asyncio
    async def test_run_streams_and_closes(self, mock_model, mock_reasoner):
        mgr = _ExpertToolManager(mock_model, ExpertToolConfig(reasoner=mock_reasoner))
        await mgr._run([{"role": "USER", "content": [{"text": "Hi"}]}], "tu-1", "cn-1")
        events = [json.loads(e) for call in mock_model._send_nova_events.call_args_list for e in call.args[0]]
        assert "contentStart" in events[0]["event"]
        assert "contentEnd" in events[-1]["event"]
        assert any("toolResult" in e.get("event", {}) for e in events)

    @pytest.mark.asyncio
    async def test_cancel_sends_content_end(self, mock_model, mock_reasoner):
        mgr = _ExpertToolManager(mock_model, ExpertToolConfig(reasoner=mock_reasoner))
        mgr._active_task = asyncio.create_task(asyncio.sleep(100))
        mgr._active_content_name = "cn-1"
        mgr._content_opened = True
        mgr._data_sent = True
        mgr._content_closed = False
        await mgr._cancel_active()
        event = json.loads(mock_model._send_nova_events.call_args.args[0][0])
        assert "contentEnd" in event["event"]

    @pytest.mark.asyncio
    async def test_cancel_skips_if_closed(self, mock_model, mock_reasoner):
        mgr = _ExpertToolManager(mock_model, ExpertToolConfig(reasoner=mock_reasoner))
        mgr._active_task = asyncio.create_task(asyncio.sleep(100))
        mgr._active_content_name = "cn-1"
        mgr._content_opened = True
        mgr._data_sent = True
        mgr._content_closed = True
        await mgr._cancel_active()
        mock_model._send_nova_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_no_user_message(self, mock_model, mock_reasoner):
        mgr = _ExpertToolManager(mock_model, ExpertToolConfig(reasoner=mock_reasoner))
        await mgr.handle_invocation({"toolUseId": "t1", "content": json.dumps({"messages": [{"role": "ASSISTANT", "content": [{"text": "x"}]}]})})
        assert mgr._active_task is None


# --- Nova Sonic Integration ---


class TestNovaSonicIntegration:
    def test_init_with_reasoner(self, boto_session, mock_client, mock_reasoner):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=mock_reasoner)
        assert model._expert_tool_manager is not None

    def test_init_raises_v1(self, boto_session, mock_client, mock_reasoner):
        with pytest.raises(ValueError, match="only supported in Nova Sonic v2"):
            BidiNovaSonicModel(model_id=NOVA_SONIC_V1_MODEL_ID, client_config={"boto_session": boto_session}, reasoner=mock_reasoner)

    def test_init_raises_both(self, boto_session, mock_client, mock_reasoner):
        with pytest.raises(ValueError, match="Cannot specify both"):
            BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=mock_reasoner, expert_tool=ExpertToolConfig(reasoner=mock_reasoner))

    def test_prompt_start_includes_expert_tool(self, boto_session, mock_client, mock_reasoner):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=mock_reasoner)
        model._connection_id = "c"
        event = json.loads(model._get_prompt_start_event(tools=[]))
        tools = event["event"]["promptStart"]["toolConfiguration"]["tools"]
        assert any(t["toolSpec"]["name"] == "ExpertTool" for t in tools)

    @pytest.mark.asyncio
    async def test_expert_tool_intercepted(self, boto_session, mock_client, mock_reasoner):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=mock_reasoner)
        model._expert_tool_manager.handle_invocation = AsyncMock()
        result = model._convert_nova_event({"toolUse": {"toolUseId": "t1", "toolName": "ExpertTool", "content": "{}"}})
        assert result is None

    def test_regular_tool_passes_through(self, boto_session, mock_client, mock_reasoner):
        model = BidiNovaSonicModel(client_config={"boto_session": boto_session}, reasoner=mock_reasoner)
        result = model._convert_nova_event({"toolUse": {"toolUseId": "t1", "toolName": "weather", "content": '{}'}})
        assert result is not None
