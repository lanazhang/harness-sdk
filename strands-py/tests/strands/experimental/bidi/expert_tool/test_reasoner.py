"""Unit tests for Expert Tool reasoner implementations."""

import sys

if sys.version_info < (3, 12):
    import pytest

    pytest.skip(reason="Expert Tool requires Python 3.12+", allow_module_level=True)

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands.experimental.bidi.expert_tool.reasoner import (
    BedrockConverseReasoner,
    ExpertToolReasoner,
    StrandsAgentReasoner,
)


# --- ExpertToolReasoner Protocol Tests ---


class TestExpertToolReasonerProtocol:
    """Tests for the ExpertToolReasoner protocol."""

    def test_bedrock_converse_reasoner_is_protocol_compliant(self):
        """BedrockConverseReasoner satisfies the ExpertToolReasoner protocol."""
        mock_model = Mock()
        mock_model.config = {"model_id": "test-model"}
        mock_model.client = Mock()
        reasoner = BedrockConverseReasoner(model=mock_model)
        assert isinstance(reasoner, ExpertToolReasoner)

    def test_strands_agent_reasoner_is_protocol_compliant(self):
        """StrandsAgentReasoner satisfies the ExpertToolReasoner protocol."""
        mock_agent = Mock()
        reasoner = StrandsAgentReasoner(agent=mock_agent)
        assert isinstance(reasoner, ExpertToolReasoner)

    def test_custom_reasoner_protocol_compliance(self):
        """A custom class implementing reason() satisfies the protocol."""

        class CustomReasoner:
            async def reason(self, messages):
                yield "Hello"

        assert isinstance(CustomReasoner(), ExpertToolReasoner)

    def test_non_compliant_class_fails_protocol_check(self):
        """A class without reason() does not satisfy the protocol."""

        class NotAReasoner:
            pass

        assert not isinstance(NotAReasoner(), ExpertToolReasoner)


# --- BedrockConverseReasoner Tests ---


class TestBedrockConverseReasoner:
    """Tests for the BedrockConverseReasoner implementation."""

    @pytest.fixture
    def mock_model(self):
        """Mock Strands BedrockModel."""
        model = Mock()
        model.config = {
            "model_id": "qwen.qwen3-32b-v1:0",
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        model.client = Mock()
        return model

    @pytest.fixture
    def mock_tool(self):
        """Mock @tool decorated function."""
        tool_fn = Mock(return_value='{"temp": 72}')
        tool_fn.tool_spec = {
            "name": "get_weather",
            "description": "Get weather for a city",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }
            },
        }
        return tool_fn

    @pytest.fixture
    def reasoner(self, mock_model):
        """Create a BedrockConverseReasoner with no tools."""
        return BedrockConverseReasoner(
            model=mock_model,
            system_prompt="You are helpful.",
        )

    @pytest.fixture
    def reasoner_with_tools(self, mock_model, mock_tool):
        """Create a BedrockConverseReasoner with tools."""
        return BedrockConverseReasoner(
            model=mock_model,
            system_prompt="You are helpful.",
            tools=[mock_tool],
            max_iterations=3,
        )

    def test_initialization(self, reasoner, mock_model):
        """Test reasoner initializes with correct attributes."""
        assert reasoner.model is mock_model
        assert reasoner.system_prompt == "You are helpful."
        assert reasoner.max_iterations == 5
        assert reasoner._tool_specs == []
        assert reasoner._tool_executors == {}

    def test_initialization_with_tools(self, reasoner_with_tools, mock_tool):
        """Test reasoner initializes with tool specs and executors."""
        assert len(reasoner_with_tools._tool_specs) == 1
        assert reasoner_with_tools._tool_specs[0]["toolSpec"]["name"] == "get_weather"
        assert "get_weather" in reasoner_with_tools._tool_executors
        assert reasoner_with_tools.max_iterations == 3

    def test_build_tool_specs_with_tool_spec_attribute(self, mock_model):
        """Test _build_tool_specs handles tool_spec attribute."""
        tool_fn = Mock()
        tool_fn.tool_spec = {
            "name": "my_tool",
            "description": "A tool",
            "inputSchema": {"json": {"type": "object"}},
        }
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[tool_fn])
        assert len(reasoner._tool_specs) == 1
        assert reasoner._tool_specs[0]["toolSpec"]["name"] == "my_tool"

    def test_build_tool_specs_with_TOOL_SPEC_attribute(self, mock_model):
        """Test _build_tool_specs handles TOOL_SPEC attribute."""
        tool_fn = Mock(spec=[])
        tool_fn.TOOL_SPEC = {
            "name": "legacy_tool",
            "description": "Legacy",
            "inputSchema": {"json": {"type": "object"}},
        }
        # Ensure tool_spec is not present
        del tool_fn.tool_spec
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[tool_fn])
        assert len(reasoner._tool_specs) == 1
        assert reasoner._tool_specs[0]["toolSpec"]["name"] == "legacy_tool"

    def test_build_tool_executors(self, mock_model, mock_tool):
        """Test _build_tool_executors maps tool names to callables."""
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[mock_tool])
        assert "get_weather" in reasoner._tool_executors
        assert reasoner._tool_executors["get_weather"] is mock_tool

    def test_convert_messages_basic(self, reasoner):
        """Test _convert_messages converts Sonic messages to Bedrock format."""
        sonic_messages = [
            {"role": "USER", "content": [{"text": "Hello"}]},
            {"role": "ASSISTANT", "content": [{"text": "Hi there!"}]},
        ]
        result = reasoner._convert_messages(sonic_messages)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": [{"text": "Hello"}]}
        assert result[1] == {"role": "assistant", "content": [{"text": "Hi there!"}]}

    def test_convert_messages_empty_content(self, reasoner):
        """Test _convert_messages skips messages with no text content."""
        sonic_messages = [
            {"role": "USER", "content": []},
            {"role": "USER", "content": [{"text": "Valid"}]},
        ]
        result = reasoner._convert_messages(sonic_messages)
        # Only the message with content should be included
        assert len(result) == 1
        assert result[0]["content"][0]["text"] == "Valid"

    def test_convert_messages_lowercase_role(self, reasoner):
        """Test _convert_messages lowercases role names."""
        sonic_messages = [{"role": "USER", "content": [{"text": "test"}]}]
        result = reasoner._convert_messages(sonic_messages)
        assert result[0]["role"] == "user"

    def test_build_request_basic(self, reasoner):
        """Test _build_request constructs proper request kwargs."""
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        kwargs = reasoner._build_request(messages)

        assert kwargs["modelId"] == "qwen.qwen3-32b-v1:0"
        assert kwargs["system"] == [{"text": "You are helpful."}]
        assert kwargs["messages"] == messages
        assert kwargs["inferenceConfig"]["maxTokens"] == 1024
        assert kwargs["inferenceConfig"]["temperature"] == 0.7
        assert "toolConfig" not in kwargs

    def test_build_request_with_tools(self, reasoner_with_tools):
        """Test _build_request includes tool config when tools are registered."""
        messages = [{"role": "user", "content": [{"text": "What's the weather?"}]}]
        kwargs = reasoner_with_tools._build_request(messages)

        assert "toolConfig" in kwargs
        assert "tools" in kwargs["toolConfig"]
        assert len(kwargs["toolConfig"]["tools"]) == 1

    def test_build_request_with_guardrails(self, mock_model):
        """Test _build_request includes guardrails when configured."""
        mock_model.config["guardrail_id"] = "guard-123"
        mock_model.config["guardrail_version"] = "1"
        mock_model.config["guardrail_trace"] = "enabled"
        reasoner = BedrockConverseReasoner(model=mock_model)

        messages = [{"role": "user", "content": [{"text": "test"}]}]
        kwargs = reasoner._build_request(messages)

        assert "guardrailConfig" in kwargs
        assert kwargs["guardrailConfig"]["guardrailIdentifier"] == "guard-123"
        assert kwargs["guardrailConfig"]["guardrailVersion"] == "1"

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, reasoner_with_tools, mock_tool):
        """Test _execute_tool calls the correct tool executor."""
        result = await reasoner_with_tools._execute_tool("get_weather", {"city": "Seattle"})
        mock_tool.assert_called_once_with(city="Seattle")
        assert result == '{"temp": 72}'

    @pytest.mark.asyncio
    async def test_execute_tool_returns_string(self, mock_model):
        """Test _execute_tool returns string results directly."""
        tool_fn = Mock(return_value="plain text result")
        tool_fn.tool_spec = {"name": "simple", "description": "Simple tool", "inputSchema": {"json": {}}}
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[tool_fn])

        result = await reasoner._execute_tool("simple", {})
        assert result == "plain text result"

    @pytest.mark.asyncio
    async def test_execute_tool_json_serializes_non_string(self, mock_model):
        """Test _execute_tool JSON-serializes non-string results."""
        tool_fn = Mock(return_value={"key": "value"})
        tool_fn.tool_spec = {"name": "json_tool", "description": "JSON tool", "inputSchema": {"json": {}}}
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[tool_fn])

        result = await reasoner._execute_tool("json_tool", {})
        assert result == json.dumps({"key": "value"})

    @pytest.mark.asyncio
    async def test_execute_tool_unknown(self, reasoner):
        """Test _execute_tool returns error for unknown tool."""
        result = await reasoner._execute_tool("nonexistent", {})
        assert "error" in json.loads(result)
        assert "Unknown tool" in json.loads(result)["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_handles_exception(self, mock_model):
        """Test _execute_tool handles tool execution errors gracefully."""
        tool_fn = Mock(side_effect=ValueError("Something broke"))
        tool_fn.tool_spec = {"name": "broken", "description": "Broken tool", "inputSchema": {"json": {}}}
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[tool_fn])

        result = await reasoner._execute_tool("broken", {})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Something broke" in parsed["error"]

    @pytest.mark.asyncio
    async def test_reason_simple_text_response(self, reasoner):
        """Test reason() streams a simple text response."""
        # Mock the converse_stream response
        reasoner.model.client.converse_stream.return_value = {
            "stream": [
                {"contentBlockStart": {"start": {}}},
                {"contentBlockDelta": {"delta": {"text": "Hello there!\n"}}},
                {"contentBlockDelta": {"delta": {"text": "How can I help?"}}},
                {"contentBlockStop": {}},
            ]
        }

        messages = [{"role": "USER", "content": [{"text": "Hi"}]}]
        results = []
        async for sentence in reasoner.reason(messages):
            results.append(sentence)

        assert "Hello there!" in results
        assert "How can I help?" in results

    @pytest.mark.asyncio
    async def test_reason_with_tool_call(self, reasoner_with_tools, mock_tool):
        """Test reason() handles tool calling loop."""
        # First call: model invokes tool
        first_response = {
            "stream": [
                {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}}}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"city": "NYC"}'}}}},
                {"contentBlockStop": {}},
            ]
        }
        # Second call: model responds with text
        second_response = {
            "stream": [
                {"contentBlockStart": {"start": {}}},
                {"contentBlockDelta": {"delta": {"text": "It's 72 degrees in NYC."}}},
                {"contentBlockStop": {}},
            ]
        }

        reasoner_with_tools.model.client.converse_stream.side_effect = [first_response, second_response]

        messages = [{"role": "USER", "content": [{"text": "Weather in NYC?"}]}]
        results = []
        async for sentence in reasoner_with_tools.reason(messages):
            results.append(sentence)

        assert "72 degrees" in " ".join(results)
        mock_tool.assert_called_once_with(city="NYC")

    @pytest.mark.asyncio
    async def test_reason_max_iterations_fallback(self, mock_model):
        """Test reason() yields fallback after max iterations exceeded."""
        tool_fn = Mock(return_value='{"result": "ok"}')
        tool_fn.tool_spec = {"name": "loop_tool", "description": "Looping tool", "inputSchema": {"json": {}}}
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[tool_fn], max_iterations=2)

        # Always return a tool call (infinite loop scenario)
        tool_response = {
            "stream": [
                {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "loop_tool"}}}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}},
                {"contentBlockStop": {}},
            ]
        }
        reasoner.model.client.converse_stream.return_value = tool_response

        messages = [{"role": "USER", "content": [{"text": "loop"}]}]
        results = []
        async for sentence in reasoner.reason(messages):
            results.append(sentence)

        # Should get fallback message after max iterations
        assert any("sorry" in s.lower() for s in results)

    @pytest.mark.asyncio
    async def test_reason_yields_sentences_on_newline(self, reasoner):
        """Test reason() splits output on newlines for TTS."""
        reasoner.model.client.converse_stream.return_value = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "First sentence.\nSecond sentence.\nThird."}}},
                {"contentBlockStop": {}},
            ]
        }

        messages = [{"role": "USER", "content": [{"text": "Tell me stuff"}]}]
        results = []
        async for sentence in reasoner.reason(messages):
            results.append(sentence)

        assert "First sentence." in results
        assert "Second sentence." in results
        assert "Third." in results

    @pytest.mark.asyncio
    async def test_reason_empty_stream(self, reasoner):
        """Test reason() handles empty stream gracefully."""
        reasoner.model.client.converse_stream.return_value = {"stream": []}

        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]
        results = []
        async for sentence in reasoner.reason(messages):
            results.append(sentence)

        # No output expected (empty stream, no tool calls, exits)
        assert results == []


# --- StrandsAgentReasoner Tests ---


class TestStrandsAgentReasoner:
    """Tests for the StrandsAgentReasoner implementation."""

    @pytest.fixture
    def mock_agent(self):
        """Mock Strands Agent."""
        agent = Mock()
        agent.return_value = "The weather in Seattle is sunny and 72 degrees."
        return agent

    @pytest.fixture
    def reasoner(self, mock_agent):
        """Create a StrandsAgentReasoner."""
        return StrandsAgentReasoner(agent=mock_agent)

    def test_initialization(self, reasoner, mock_agent):
        """Test StrandsAgentReasoner initializes correctly."""
        assert reasoner.agent is mock_agent

    def test_extract_latest_user_text_basic(self, reasoner):
        """Test extracting latest user text from messages."""
        messages = [
            {"role": "USER", "content": [{"text": "First question"}]},
            {"role": "ASSISTANT", "content": [{"text": "First answer"}]},
            {"role": "USER", "content": [{"text": "Second question"}]},
        ]
        result = reasoner._extract_latest_user_text(messages)
        assert result == "Second question"

    def test_extract_latest_user_text_case_insensitive(self, reasoner):
        """Test extraction handles uppercase USER role from Sonic."""
        messages = [{"role": "USER", "content": [{"text": "Hello"}]}]
        result = reasoner._extract_latest_user_text(messages)
        assert result == "Hello"

    def test_extract_latest_user_text_empty_messages(self, reasoner):
        """Test extraction returns empty string for no messages."""
        result = reasoner._extract_latest_user_text([])
        assert result == ""

    def test_extract_latest_user_text_no_user_messages(self, reasoner):
        """Test extraction returns empty string when no USER messages."""
        messages = [{"role": "ASSISTANT", "content": [{"text": "Hi"}]}]
        result = reasoner._extract_latest_user_text(messages)
        assert result == ""

    def test_extract_latest_user_text_empty_content(self, reasoner):
        """Test extraction returns empty when content has no text."""
        messages = [{"role": "USER", "content": []}]
        result = reasoner._extract_latest_user_text(messages)
        assert result == ""

    def test_split_sentences_basic(self, reasoner):
        """Test splitting text into sentences on newlines."""
        text = "First line.\nSecond line.\nThird line."
        result = reasoner._split_sentences(text)
        assert result == ["First line.", "Second line.", "Third line."]

    def test_split_sentences_strips_whitespace(self, reasoner):
        """Test splitting strips whitespace from sentences."""
        text = "  Hello.  \n  World.  "
        result = reasoner._split_sentences(text)
        assert result == ["Hello.", "World."]

    def test_split_sentences_skips_empty_lines(self, reasoner):
        """Test splitting skips empty lines."""
        text = "Hello.\n\n\nWorld."
        result = reasoner._split_sentences(text)
        assert result == ["Hello.", "World."]

    def test_split_sentences_single_line(self, reasoner):
        """Test splitting returns single line as-is."""
        text = "Just one line."
        result = reasoner._split_sentences(text)
        assert result == ["Just one line."]

    def test_split_sentences_empty_text(self, reasoner):
        """Test splitting handles empty text."""
        result = reasoner._split_sentences("")
        # Should return the stripped text (empty)
        assert result == [""]

    @pytest.mark.asyncio
    async def test_reason_invokes_agent(self, reasoner, mock_agent):
        """Test reason() invokes the agent with latest user text."""
        messages = [{"role": "USER", "content": [{"text": "What's the weather?"}]}]
        results = []
        async for sentence in reasoner.reason(messages):
            results.append(sentence)

        mock_agent.assert_called_once_with("What's the weather?")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_reason_yields_agent_response_sentences(self, mock_agent):
        """Test reason() yields agent response split into sentences."""
        mock_agent.return_value = "Line one.\nLine two.\nLine three."
        reasoner = StrandsAgentReasoner(agent=mock_agent)

        messages = [{"role": "USER", "content": [{"text": "Tell me"}]}]
        results = []
        async for sentence in reasoner.reason(messages):
            results.append(sentence)

        assert results == ["Line one.", "Line two.", "Line three."]

    @pytest.mark.asyncio
    async def test_reason_no_user_message_yields_fallback(self, reasoner):
        """Test reason() yields fallback when no user message found."""
        messages = [{"role": "ASSISTANT", "content": [{"text": "I said something"}]}]
        results = []
        async for sentence in reasoner.reason(messages):
            results.append(sentence)

        assert len(results) == 1
        assert "didn't catch" in results[0].lower()

    @pytest.mark.asyncio
    async def test_reason_handles_multiline_agent_response(self, mock_agent):
        """Test reason() handles agent response with multiple paragraphs."""
        mock_agent.return_value = "First paragraph.\n\nSecond paragraph.\nThird line."
        reasoner = StrandsAgentReasoner(agent=mock_agent)

        messages = [{"role": "USER", "content": [{"text": "Explain"}]}]
        results = []
        async for sentence in reasoner.reason(messages):
            results.append(sentence)

        # Empty lines should be skipped
        assert "First paragraph." in results
        assert "Second paragraph." in results
        assert "Third line." in results
