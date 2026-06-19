"""Unit tests for _ExpertToolManager internal lifecycle manager."""

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


@pytest.fixture
def mock_model():
    """Mock BidiNovaSonicModel with _send_nova_events and _connection_id."""
    model = AsyncMock()
    model._connection_id = "test-connection-id"
    model._send_nova_events = AsyncMock()
    return model


@pytest.fixture
def mock_reasoner():
    """Mock ExpertToolReasoner that yields sentences."""

    async def _reason(messages):
        yield "Hello from the reasoner."
        yield "How can I help?"

    reasoner = Mock()
    reasoner.reason = _reason
    return reasoner


@pytest.fixture
def config(mock_reasoner):
    """ExpertToolConfig with mock reasoner."""
    return ExpertToolConfig(
        reasoner=mock_reasoner,
        fallback_message="I'm not sure how to help with that.",
    )


@pytest.fixture
def manager(mock_model, config):
    """Create an _ExpertToolManager instance."""
    return _ExpertToolManager(mock_model, config)


class TestExpertToolManagerInit:
    """Tests for manager initialization."""

    def test_initial_state(self, manager, mock_model, config):
        """Test manager initializes with correct default state."""
        assert manager._model is mock_model
        assert manager._config is config
        assert manager._active_task is None
        assert manager._active_content_name is None
        assert manager._active_tool_use_id is None
        assert manager._content_opened is False
        assert manager._data_sent is False
        assert manager._content_closed is False


class TestExpertToolManagerMessageExtraction:
    """Tests for message extraction from toolUse events."""

    def test_extract_messages_string_content(self, manager):
        """Test extracting messages from JSON string content."""
        tool_use = {
            "toolUseId": "tu-123",
            "content": json.dumps({
                "messages": [
                    {"role": "USER", "content": [{"text": "Hello"}]},
                    {"role": "ASSISTANT", "content": [{"text": "Hi"}]},
                ]
            }),
        }
        messages = manager._extract_messages(tool_use)
        assert len(messages) == 2
        assert messages[0]["role"] == "USER"
        assert messages[0]["content"][0]["text"] == "Hello"

    def test_extract_messages_dict_content(self, manager):
        """Test extracting messages from dict content (already parsed)."""
        tool_use = {
            "toolUseId": "tu-456",
            "content": {
                "messages": [{"role": "USER", "content": [{"text": "World"}]}]
            },
        }
        messages = manager._extract_messages(tool_use)
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "World"

    def test_extract_messages_invalid_json(self, manager):
        """Test extracting messages from invalid JSON returns empty list."""
        tool_use = {"toolUseId": "tu-789", "content": "not valid json{{{"}
        messages = manager._extract_messages(tool_use)
        assert messages == []

    def test_extract_messages_no_content(self, manager):
        """Test extracting messages from missing content returns empty list."""
        tool_use = {"toolUseId": "tu-000"}
        messages = manager._extract_messages(tool_use)
        assert messages == []

    def test_extract_messages_no_messages_key(self, manager):
        """Test extracting messages when 'messages' key is missing."""
        tool_use = {"toolUseId": "tu-111", "content": json.dumps({"other": "data"})}
        messages = manager._extract_messages(tool_use)
        assert messages == []

    def test_has_user_message_true(self, manager):
        """Test _has_user_message returns True when USER message exists."""
        messages = [
            {"role": "ASSISTANT", "content": [{"text": "Hi"}]},
            {"role": "USER", "content": [{"text": "Hello"}]},
        ]
        assert manager._has_user_message(messages) is True

    def test_has_user_message_false_no_user(self, manager):
        """Test _has_user_message returns False when no USER messages."""
        messages = [{"role": "ASSISTANT", "content": [{"text": "Hi"}]}]
        assert manager._has_user_message(messages) is False

    def test_has_user_message_false_empty_text(self, manager):
        """Test _has_user_message returns False when USER message has empty text."""
        messages = [{"role": "USER", "content": [{"text": ""}]}]
        assert manager._has_user_message(messages) is False

    def test_has_user_message_false_empty_messages(self, manager):
        """Test _has_user_message returns False for empty message list."""
        assert manager._has_user_message([]) is False

    def test_has_user_message_case_insensitive(self, manager):
        """Test _has_user_message handles lowercase role."""
        # The method uses .upper() so both should work
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        assert manager._has_user_message(messages) is True


class TestExpertToolManagerHandleInvocation:
    """Tests for handle_invocation method."""

    @pytest.mark.asyncio
    async def test_handle_invocation_starts_task(self, manager):
        """Test handle_invocation creates an asyncio task."""
        tool_use = {
            "toolUseId": "tu-123",
            "content": json.dumps({"messages": [{"role": "USER", "content": [{"text": "Hi"}]}]}),
        }
        await manager.handle_invocation(tool_use)

        assert manager._active_task is not None
        assert manager._active_tool_use_id == "tu-123"
        assert manager._active_content_name is not None

        # Wait for task to complete
        await manager._active_task

    @pytest.mark.asyncio
    async def test_handle_invocation_skips_no_user_message(self, manager):
        """Test handle_invocation skips when no USER message in content."""
        tool_use = {
            "toolUseId": "tu-123",
            "content": json.dumps({"messages": [{"role": "ASSISTANT", "content": [{"text": "Hi"}]}]}),
        }
        await manager.handle_invocation(tool_use)

        # Should not start a task
        assert manager._active_task is None

    @pytest.mark.asyncio
    async def test_handle_invocation_cancels_previous(self, manager, mock_model):
        """Test handle_invocation cancels a previously active task (barge-in)."""

        # Create a slow reasoner that takes time
        async def slow_reason(messages):
            await asyncio.sleep(10)
            yield "Never reached"

        manager._config.reasoner.reason = slow_reason

        # First invocation
        tool_use_1 = {
            "toolUseId": "tu-1",
            "content": json.dumps({"messages": [{"role": "USER", "content": [{"text": "First"}]}]}),
        }
        await manager.handle_invocation(tool_use_1)
        first_task = manager._active_task
        assert first_task is not None

        # Give the task a moment to start
        await asyncio.sleep(0.01)

        # Replace reasoner with fast one for the second call
        async def fast_reason(messages):
            yield "Quick response"

        manager._config.reasoner.reason = fast_reason

        # Second invocation — should cancel first
        tool_use_2 = {
            "toolUseId": "tu-2",
            "content": json.dumps({"messages": [{"role": "USER", "content": [{"text": "Second"}]}]}),
        }
        await manager.handle_invocation(tool_use_2)

        # First task should be cancelled
        assert first_task.cancelled() or first_task.done()
        # New task should be active
        assert manager._active_tool_use_id == "tu-2"

        # Cleanup
        if manager._active_task:
            await manager._active_task


class TestExpertToolManagerRun:
    """Tests for the _run method (streaming reasoning results)."""

    @pytest.mark.asyncio
    async def test_run_streams_content(self, manager, mock_model):
        """Test _run opens content container, streams, and closes."""
        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]

        await manager._run(messages, "tu-123", "cn-abc")

        # Verify contentStart was sent
        calls = mock_model._send_nova_events.call_args_list
        assert len(calls) >= 3  # contentStart + at least one toolResult + contentEnd

        # Parse events
        all_events = []
        for call in calls:
            for event_str in call.args[0]:
                all_events.append(json.loads(event_str))

        # Check contentStart
        content_start = all_events[0]
        assert "contentStart" in content_start["event"]
        assert content_start["event"]["contentStart"]["contentName"] == "cn-abc"
        assert content_start["event"]["contentStart"]["type"] == "TOOL"

        # Check contentEnd (last event)
        content_end = all_events[-1]
        assert "contentEnd" in content_end["event"]
        assert content_end["event"]["contentEnd"]["contentName"] == "cn-abc"

    @pytest.mark.asyncio
    async def test_run_sends_tool_results(self, manager, mock_model):
        """Test _run sends toolResult events with reasoner output."""
        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]

        await manager._run(messages, "tu-123", "cn-abc")

        # Find toolResult events
        tool_results = []
        for call in mock_model._send_nova_events.call_args_list:
            for event_str in call.args[0]:
                event = json.loads(event_str)
                if "toolResult" in event.get("event", {}):
                    tool_results.append(event)

        # Should have two sentences from mock reasoner
        assert len(tool_results) == 2
        assert "Hello from the reasoner" in tool_results[0]["event"]["toolResult"]["content"]
        assert "How can I help" in tool_results[1]["event"]["toolResult"]["content"]

    @pytest.mark.asyncio
    async def test_run_sends_fallback_when_empty(self, manager, mock_model):
        """Test _run sends fallback when reasoner yields nothing."""

        async def empty_reason(messages):
            return
            yield  # Make it an async generator that yields nothing

        manager._config.reasoner.reason = empty_reason

        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]
        await manager._run(messages, "tu-123", "cn-abc")

        # Find toolResult events
        tool_results = []
        for call in mock_model._send_nova_events.call_args_list:
            for event_str in call.args[0]:
                event = json.loads(event_str)
                if "toolResult" in event.get("event", {}):
                    content = event["event"]["toolResult"]["content"]
                    tool_results.append(json.loads(content))

        # Should get fallback message
        assert len(tool_results) == 1
        assert tool_results[0]["text"] == "I'm not sure how to help with that."

    @pytest.mark.asyncio
    async def test_run_handles_reasoner_error(self, manager, mock_model):
        """Test _run sends error fallback when reasoner raises."""

        async def broken_reason(messages):
            raise RuntimeError("Model failed")
            yield  # noqa: unreachable

        manager._config.reasoner.reason = broken_reason

        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]
        await manager._run(messages, "tu-123", "cn-abc")

        # Should still send contentStart and contentEnd with error message
        tool_results = []
        for call in mock_model._send_nova_events.call_args_list:
            for event_str in call.args[0]:
                event = json.loads(event_str)
                if "toolResult" in event.get("event", {}):
                    content = event["event"]["toolResult"]["content"]
                    tool_results.append(json.loads(content))

        assert len(tool_results) == 1
        assert "something went wrong" in tool_results[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_run_sets_content_flags(self, manager, mock_model):
        """Test _run correctly sets content lifecycle flags."""
        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]

        await manager._run(messages, "tu-123", "cn-abc")

        # After successful run, flags should reflect completed state
        assert manager._content_opened is True
        assert manager._data_sent is True
        assert manager._content_closed is True

    @pytest.mark.asyncio
    async def test_run_skips_empty_sentences(self, manager, mock_model):
        """Test _run skips empty/None sentences from reasoner."""

        async def reason_with_empties(messages):
            yield "Valid sentence."
            yield ""
            yield None
            yield "Another valid one."

        manager._config.reasoner.reason = reason_with_empties

        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]
        await manager._run(messages, "tu-123", "cn-abc")

        # Should only send 2 non-empty sentences
        tool_results = []
        for call in mock_model._send_nova_events.call_args_list:
            for event_str in call.args[0]:
                event = json.loads(event_str)
                if "toolResult" in event.get("event", {}):
                    tool_results.append(event)

        assert len(tool_results) == 2


class TestExpertToolManagerCancelActive:
    """Tests for _cancel_active (barge-in handling)."""

    @pytest.mark.asyncio
    async def test_cancel_active_no_task(self, manager):
        """Test _cancel_active does nothing when no active task."""
        await manager._cancel_active()

        assert manager._active_task is None
        assert manager._content_opened is False

    @pytest.mark.asyncio
    async def test_cancel_active_cancels_running_task(self, manager, mock_model):
        """Test _cancel_active cancels a running asyncio task."""

        async def long_running():
            await asyncio.sleep(100)

        manager._active_task = asyncio.create_task(long_running())
        manager._active_content_name = "cn-test"
        manager._active_tool_use_id = "tu-test"
        manager._content_opened = True
        manager._data_sent = True
        manager._content_closed = False

        await manager._cancel_active()

        assert manager._active_task is None
        assert manager._active_content_name is None
        assert manager._content_opened is False

    @pytest.mark.asyncio
    async def test_cancel_active_sends_content_end(self, manager, mock_model):
        """Test _cancel_active sends contentEnd when content is open with data."""

        async def long_running():
            await asyncio.sleep(100)

        manager._active_task = asyncio.create_task(long_running())
        manager._active_content_name = "cn-open"
        manager._active_tool_use_id = "tu-test"
        manager._content_opened = True
        manager._data_sent = True
        manager._content_closed = False

        await manager._cancel_active()

        # Should have sent contentEnd
        mock_model._send_nova_events.assert_called()
        last_call = mock_model._send_nova_events.call_args_list[-1]
        event = json.loads(last_call.args[0][0])
        assert "contentEnd" in event["event"]
        assert event["event"]["contentEnd"]["contentName"] == "cn-open"

    @pytest.mark.asyncio
    async def test_cancel_active_no_content_end_if_already_closed(self, manager, mock_model):
        """Test _cancel_active does NOT send contentEnd if already closed."""

        async def long_running():
            await asyncio.sleep(100)

        manager._active_task = asyncio.create_task(long_running())
        manager._active_content_name = "cn-closed"
        manager._active_tool_use_id = "tu-test"
        manager._content_opened = True
        manager._data_sent = True
        manager._content_closed = True  # Already closed

        await manager._cancel_active()

        # Should NOT have sent any events (content already closed)
        mock_model._send_nova_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_active_no_content_end_if_no_data_sent(self, manager, mock_model):
        """Test _cancel_active does NOT send contentEnd if no data was sent."""

        async def long_running():
            await asyncio.sleep(100)

        manager._active_task = asyncio.create_task(long_running())
        manager._active_content_name = "cn-nodata"
        manager._active_tool_use_id = "tu-test"
        manager._content_opened = True
        manager._data_sent = False  # No data sent yet
        manager._content_closed = False

        await manager._cancel_active()

        # Should NOT have sent contentEnd (no data was streamed)
        mock_model._send_nova_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_active_calls_on_interrupted(self, manager, mock_model):
        """Test _cancel_active invokes on_interrupted callback."""
        callback = Mock()
        manager._config.on_interrupted = callback

        async def long_running():
            await asyncio.sleep(100)

        manager._active_task = asyncio.create_task(long_running())
        manager._active_content_name = "cn-int"
        manager._active_tool_use_id = "tu-test"
        manager._content_opened = True
        manager._data_sent = True
        manager._content_closed = False

        await manager._cancel_active()

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_active_handles_callback_error(self, manager, mock_model):
        """Test _cancel_active handles on_interrupted callback errors gracefully."""
        manager._config.on_interrupted = Mock(side_effect=RuntimeError("callback broke"))

        async def long_running():
            await asyncio.sleep(100)

        manager._active_task = asyncio.create_task(long_running())
        manager._active_content_name = "cn-err"
        manager._active_tool_use_id = "tu-test"
        manager._content_opened = True
        manager._data_sent = True
        manager._content_closed = False

        # Should not raise
        await manager._cancel_active()
        assert manager._active_task is None


class TestExpertToolManagerShutdown:
    """Tests for shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_active(self, manager, mock_model):
        """Test shutdown cancels any active reasoning task."""

        async def long_running():
            await asyncio.sleep(100)

        manager._active_task = asyncio.create_task(long_running())
        manager._active_content_name = "cn-shutdown"
        manager._active_tool_use_id = "tu-shutdown"
        manager._content_opened = True
        manager._data_sent = True
        manager._content_closed = False

        await manager.shutdown()

        assert manager._active_task is None

    @pytest.mark.asyncio
    async def test_shutdown_no_active_task(self, manager):
        """Test shutdown works fine when no active task."""
        await manager.shutdown()
        assert manager._active_task is None


class TestExpertToolManagerEventHelpers:
    """Tests for the event helper methods that send Nova Sonic protocol events."""

    @pytest.mark.asyncio
    async def test_send_tool_content_start(self, manager, mock_model):
        """Test _send_tool_content_start sends correct event structure."""
        await manager._send_tool_content_start("cn-123", "tu-456")

        mock_model._send_nova_events.assert_called_once()
        event_str = mock_model._send_nova_events.call_args.args[0][0]
        event = json.loads(event_str)

        cs = event["event"]["contentStart"]
        assert cs["promptName"] == "test-connection-id"
        assert cs["contentName"] == "cn-123"
        assert cs["interactive"] is False
        assert cs["type"] == "TOOL"
        assert cs["role"] == "TOOL"
        assert cs["toolResultInputConfiguration"]["toolUseId"] == "tu-456"
        assert cs["toolResultInputConfiguration"]["type"] == "TEXT"
        assert cs["toolResultInputConfiguration"]["textInputConfiguration"]["mediaType"] == "text/plain"

    @pytest.mark.asyncio
    async def test_send_tool_result(self, manager, mock_model):
        """Test _send_tool_result sends correct event structure."""
        content = json.dumps({"text": "Hello", "type": "TEXT"})
        await manager._send_tool_result("cn-123", content)

        mock_model._send_nova_events.assert_called_once()
        event_str = mock_model._send_nova_events.call_args.args[0][0]
        event = json.loads(event_str)

        tr = event["event"]["toolResult"]
        assert tr["promptName"] == "test-connection-id"
        assert tr["contentName"] == "cn-123"
        assert tr["content"] == content

    @pytest.mark.asyncio
    async def test_send_content_end(self, manager, mock_model):
        """Test _send_content_end sends correct event structure."""
        await manager._send_content_end("cn-123")

        mock_model._send_nova_events.assert_called_once()
        event_str = mock_model._send_nova_events.call_args.args[0][0]
        event = json.loads(event_str)

        ce = event["event"]["contentEnd"]
        assert ce["promptName"] == "test-connection-id"
        assert ce["contentName"] == "cn-123"
