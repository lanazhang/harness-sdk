"""Unit tests for _ExpertToolManager."""

import sys

if sys.version_info < (3, 12):
    import pytest

    pytest.skip(reason="Expert Tool requires Python 3.12+", allow_module_level=True)

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from strands.experimental.bidi.expert_tool.config import ExpertToolConfig
from strands.experimental.bidi.expert_tool.manager import _ExpertToolManager


@pytest.fixture
def mock_model():
    model = AsyncMock()
    model._connection_id = "test-conn"
    model._send_nova_events = AsyncMock()
    return model


@pytest.fixture
def mock_reasoner():
    async def _reason(messages):
        yield "Hello."
        yield "How can I help?"

    reasoner = Mock()
    reasoner.reason = _reason
    return reasoner


@pytest.fixture
def config(mock_reasoner):
    return ExpertToolConfig(reasoner=mock_reasoner)


@pytest.fixture
def manager(mock_model, config):
    return _ExpertToolManager(mock_model, config)


class TestMessageExtraction:
    def test_extract_from_json_string(self, manager):
        tool_use = {"toolUseId": "t1", "content": json.dumps({"messages": [{"role": "USER", "content": [{"text": "Hi"}]}]})}
        assert len(manager._extract_messages(tool_use)) == 1

    def test_extract_invalid_json(self, manager):
        assert manager._extract_messages({"toolUseId": "t1", "content": "bad{"}) == []

    def test_has_user_message_true(self, manager):
        assert manager._has_user_message([{"role": "USER", "content": [{"text": "Hi"}]}])

    def test_has_user_message_false(self, manager):
        assert not manager._has_user_message([{"role": "ASSISTANT", "content": [{"text": "Hi"}]}])
        assert not manager._has_user_message([{"role": "USER", "content": [{"text": ""}]}])
        assert not manager._has_user_message([])


class TestHandleInvocation:
    @pytest.mark.asyncio
    async def test_starts_task(self, manager):
        tool_use = {"toolUseId": "t1", "content": json.dumps({"messages": [{"role": "USER", "content": [{"text": "Hi"}]}]})}
        await manager.handle_invocation(tool_use)
        assert manager._active_task is not None
        await manager._active_task

    @pytest.mark.asyncio
    async def test_skips_no_user_message(self, manager):
        tool_use = {"toolUseId": "t1", "content": json.dumps({"messages": [{"role": "ASSISTANT", "content": [{"text": "Hi"}]}]})}
        await manager.handle_invocation(tool_use)
        assert manager._active_task is None


class TestRun:
    @pytest.mark.asyncio
    async def test_streams_content_lifecycle(self, manager, mock_model):
        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]
        await manager._run(messages, "tu-1", "cn-1")

        events = []
        for call in mock_model._send_nova_events.call_args_list:
            for e in call.args[0]:
                events.append(json.loads(e))

        # contentStart, 2 toolResults, contentEnd
        assert "contentStart" in events[0]["event"]
        assert "contentEnd" in events[-1]["event"]
        tool_results = [e for e in events if "toolResult" in e.get("event", {})]
        assert len(tool_results) == 2

    @pytest.mark.asyncio
    async def test_fallback_on_empty_reasoner(self, manager, mock_model):
        async def empty(messages):
            return
            yield

        manager._config.reasoner.reason = empty
        await manager._run([{"role": "USER", "content": [{"text": "Hi"}]}], "tu-1", "cn-1")

        tool_results = []
        for call in mock_model._send_nova_events.call_args_list:
            for e in call.args[0]:
                ev = json.loads(e)
                if "toolResult" in ev.get("event", {}):
                    tool_results.append(json.loads(ev["event"]["toolResult"]["content"]))

        assert any("not sure" in r["text"].lower() for r in tool_results)

    @pytest.mark.asyncio
    async def test_error_recovery(self, manager, mock_model):
        async def broken(messages):
            raise RuntimeError("fail")
            yield

        manager._config.reasoner.reason = broken
        await manager._run([{"role": "USER", "content": [{"text": "Hi"}]}], "tu-1", "cn-1")

        tool_results = []
        for call in mock_model._send_nova_events.call_args_list:
            for e in call.args[0]:
                ev = json.loads(e)
                if "toolResult" in ev.get("event", {}):
                    tool_results.append(json.loads(ev["event"]["toolResult"]["content"]))

        assert any("went wrong" in r["text"].lower() for r in tool_results)


class TestCancelActive:
    @pytest.mark.asyncio
    async def test_cancels_running_task(self, manager, mock_model):
        manager._active_task = asyncio.create_task(asyncio.sleep(100))
        manager._active_content_name = "cn-1"
        manager._content_opened = True
        manager._data_sent = True
        manager._content_closed = False

        await manager._cancel_active()
        assert manager._active_task is None

        # Should have sent contentEnd
        event = json.loads(mock_model._send_nova_events.call_args.args[0][0])
        assert "contentEnd" in event["event"]

    @pytest.mark.asyncio
    async def test_no_content_end_if_already_closed(self, manager, mock_model):
        manager._active_task = asyncio.create_task(asyncio.sleep(100))
        manager._active_content_name = "cn-1"
        manager._content_opened = True
        manager._data_sent = True
        manager._content_closed = True

        await manager._cancel_active()
        mock_model._send_nova_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_on_interrupted(self, manager, mock_model):
        callback = Mock()
        manager._config.on_interrupted = callback
        manager._active_task = asyncio.create_task(asyncio.sleep(100))
        manager._active_content_name = "cn-1"
        manager._content_opened = True
        manager._data_sent = True

        await manager._cancel_active()
        callback.assert_called_once()


class TestEventHelpers:
    @pytest.mark.asyncio
    async def test_send_tool_content_start(self, manager, mock_model):
        await manager._send_tool_content_start("cn-1", "tu-1")
        event = json.loads(mock_model._send_nova_events.call_args.args[0][0])
        cs = event["event"]["contentStart"]
        assert cs["type"] == "TOOL"
        assert cs["toolResultInputConfiguration"]["toolUseId"] == "tu-1"

    @pytest.mark.asyncio
    async def test_send_content_end(self, manager, mock_model):
        await manager._send_content_end("cn-1")
        event = json.loads(mock_model._send_nova_events.call_args.args[0][0])
        assert event["event"]["contentEnd"]["contentName"] == "cn-1"
