"""Unit tests for Expert Tool reasoner implementations."""

import sys

if sys.version_info < (3, 12):
    import pytest

    pytest.skip(reason="Expert Tool requires Python 3.12+", allow_module_level=True)

import json
from unittest.mock import Mock

import pytest

from strands.experimental.bidi.expert_tool.reasoner import (
    BedrockConverseReasoner,
    ExpertToolReasoner,
    StrandsAgentReasoner,
)


# --- Protocol Tests ---


class TestExpertToolReasonerProtocol:
    def test_bedrock_reasoner_compliant(self):
        model = Mock(config={"model_id": "test"}, client=Mock())
        assert isinstance(BedrockConverseReasoner(model=model), ExpertToolReasoner)

    def test_strands_reasoner_compliant(self):
        assert isinstance(StrandsAgentReasoner(agent=Mock()), ExpertToolReasoner)

    def test_custom_class_compliant(self):
        class Custom:
            async def reason(self, messages):
                yield "hi"

        assert isinstance(Custom(), ExpertToolReasoner)

    def test_non_compliant_fails(self):
        assert not isinstance(object(), ExpertToolReasoner)


# --- BedrockConverseReasoner Tests ---


class TestBedrockConverseReasoner:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.config = {"model_id": "test-model", "max_tokens": 1024, "temperature": 0.7}
        model.client = Mock()
        return model

    @pytest.fixture
    def mock_tool(self):
        tool_fn = Mock(return_value='{"temp": 72}')
        tool_fn.tool_spec = {
            "name": "get_weather",
            "description": "Get weather",
            "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}}},
        }
        return tool_fn

    def test_initialization_with_tools(self, mock_model, mock_tool):
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[mock_tool], max_iterations=3)
        assert len(reasoner._tool_specs) == 1
        assert "get_weather" in reasoner._tool_executors
        assert reasoner.max_iterations == 3

    def test_convert_messages_strips_leading_assistant(self, mock_model):
        reasoner = BedrockConverseReasoner(model=mock_model)
        sonic_messages = [
            {"role": "ASSISTANT", "content": [{"text": "Hello!"}]},
            {"role": "USER", "content": [{"text": "Hi"}]},
        ]
        result = reasoner._convert_messages(sonic_messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_convert_messages_lowercases_role(self, mock_model):
        reasoner = BedrockConverseReasoner(model=mock_model)
        result = reasoner._convert_messages([{"role": "USER", "content": [{"text": "test"}]}])
        assert result[0]["role"] == "user"

    def test_build_request_includes_tools(self, mock_model, mock_tool):
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[mock_tool])
        kwargs = reasoner._build_request([{"role": "user", "content": [{"text": "hi"}]}])
        assert "toolConfig" in kwargs

    def test_build_request_includes_guardrails(self, mock_model):
        mock_model.config["guardrail_id"] = "g-123"
        mock_model.config["guardrail_version"] = "1"
        reasoner = BedrockConverseReasoner(model=mock_model)
        kwargs = reasoner._build_request([{"role": "user", "content": [{"text": "hi"}]}])
        assert kwargs["guardrailConfig"]["guardrailIdentifier"] == "g-123"

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_model, mock_tool):
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[mock_tool])
        result = await reasoner._execute_tool("get_weather", {"city": "NYC"})
        assert result == '{"temp": 72}'

    @pytest.mark.asyncio
    async def test_execute_tool_unknown(self, mock_model):
        reasoner = BedrockConverseReasoner(model=mock_model)
        result = await reasoner._execute_tool("nope", {})
        assert "Unknown tool" in json.loads(result)["error"]

    @pytest.mark.asyncio
    async def test_reason_simple_response(self, mock_model):
        mock_model.client.converse_stream.return_value = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "Hello!\nWorld."}}},
                {"contentBlockStop": {}},
            ]
        }
        reasoner = BedrockConverseReasoner(model=mock_model)
        results = [s async for s in reasoner.reason([{"role": "USER", "content": [{"text": "Hi"}]}])]
        assert "Hello!" in results
        assert "World." in results

    @pytest.mark.asyncio
    async def test_reason_with_tool_call(self, mock_model, mock_tool):
        first = {
            "stream": [
                {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}}}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"city": "NYC"}'}}}},
                {"contentBlockStop": {}},
            ]
        }
        second = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "72 degrees."}}},
                {"contentBlockStop": {}},
            ]
        }
        mock_model.client.converse_stream.side_effect = [first, second]
        reasoner = BedrockConverseReasoner(model=mock_model, tools=[mock_tool])
        results = [s async for s in reasoner.reason([{"role": "USER", "content": [{"text": "weather?"}]}])]
        assert any("72" in s for s in results)
        mock_tool.assert_called_once_with(city="NYC")


# --- StrandsAgentReasoner Tests ---


class TestStrandsAgentReasoner:
    def test_extract_latest_user_text(self):
        reasoner = StrandsAgentReasoner(agent=Mock())
        messages = [
            {"role": "USER", "content": [{"text": "First"}]},
            {"role": "ASSISTANT", "content": [{"text": "Reply"}]},
            {"role": "USER", "content": [{"text": "Second"}]},
        ]
        assert reasoner._extract_latest_user_text(messages) == "Second"

    def test_extract_latest_user_text_empty(self):
        reasoner = StrandsAgentReasoner(agent=Mock())
        assert reasoner._extract_latest_user_text([]) == ""

    def test_split_sentences(self):
        reasoner = StrandsAgentReasoner(agent=Mock())
        assert reasoner._split_sentences("A.\n\nB.\nC.") == ["A.", "B.", "C."]

    @pytest.mark.asyncio
    async def test_reason_invokes_agent(self):
        agent = Mock(return_value="Line one.\nLine two.")
        reasoner = StrandsAgentReasoner(agent=agent)
        messages = [{"role": "USER", "content": [{"text": "hello"}]}]
        results = [s async for s in reasoner.reason(messages)]
        agent.assert_called_once_with("hello")
        assert results == ["Line one.", "Line two."]

    @pytest.mark.asyncio
    async def test_reason_no_user_message_fallback(self):
        reasoner = StrandsAgentReasoner(agent=Mock())
        results = [s async for s in reasoner.reason([{"role": "ASSISTANT", "content": [{"text": "hi"}]}])]
        assert len(results) == 1
        assert "didn't catch" in results[0].lower()
