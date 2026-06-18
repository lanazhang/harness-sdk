"""Expert Tool Reasoner protocol and built-in implementations.

Provides:
- ExpertToolReasoner: Protocol for custom reasoning implementations.
- BedrockConverseReasoner: Built-in reasoner using Bedrock ConverseStream.
- StrandsAgentReasoner: Wraps any Strands Agent as a reasoner.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ExpertToolReasoner(Protocol):
    """Protocol for Expert Tool reasoning implementations.

    Implement this to bring any LLM as the reasoning layer for Nova Sonic.
    The SDK calls `reason()` when Nova Sonic emits an Expert Tool invocation.

    The contract is simple: receive conversation messages, yield text sentences.
    Each yielded string is streamed back to Sonic for TTS.
    """

    async def reason(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream reasoning response sentence by sentence.

        Args:
            messages: Conversation messages from Nova Sonic (USER/ASSISTANT turns).
                Format: [{"role": "USER", "content": [{"text": "..."}]}, ...]

        Yields:
            Text chunks (sentences) to stream back to Sonic for TTS.
            Each yield becomes a separate toolResult event.
        """
        ...


class BedrockConverseReasoner:
    """Built-in reasoner using Bedrock ConverseStream.

    Wraps any Bedrock Converse-compatible model (Claude, Nova Pro, Qwen,
    custom/fine-tuned models, etc.) and handles:
    - Streaming text output sentence-by-sentence
    - Tool calling loops (up to max_iterations)
    - Error recovery with fallback responses

    The model parameter accepts a Strands BedrockModel instance, giving full
    control over region, credentials, guardrails, and inference config.
    """

    def __init__(
        self,
        model: Any,
        system_prompt: str = "You are a helpful voice assistant.",
        tools: list[Any] | None = None,
        max_iterations: int = 5,
    ):
        """Initialize BedrockConverseReasoner.

        Args:
            model: A Strands BedrockModel instance.
            system_prompt: System prompt for the reasoning LLM.
            tools: List of @tool decorated functions the reasoner can call.
            max_iterations: Max tool call loops per turn before fallback.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self._tools = tools or []
        self._tool_specs = self._build_tool_specs(self._tools)
        self._tool_executors = self._build_tool_executors(self._tools)

    async def reason(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream reasoning with automatic tool calling loops.

        Converts Sonic messages to Bedrock Converse format, streams the response,
        and handles tool calls internally. Yields sentences for TTS.

        Args:
            messages: Conversation messages from Nova Sonic Expert Tool invocation.

        Yields:
            Text sentences ready for TTS.
        """
        logger.debug("reasoner_model=<%s> | Expert Tool invoked", self.model.config["model_id"])
        logger.debug("messages_count=<%d> | conversation context received", len(messages))
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            text = ""
            for block in msg.get("content", []):
                if "text" in block:
                    text = block["text"][:100]
                    break
            logger.debug("  [%d] role=<%s> text=<%s>", i, role, text)

        working_messages = self._convert_messages(messages)

        for _ in range(self.max_iterations):
            full_text = ""
            buffer = ""
            tool_requests: list[dict[str, Any]] = []
            current_tool: dict[str, Any] | None = None

            request_kwargs = self._build_request(working_messages)
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.client.converse_stream(**request_kwargs)
            )

            for event in response.get("stream", []):
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"].get("start", {})
                    if "toolUse" in start:
                        current_tool = {
                            "toolUseId": start["toolUse"]["toolUseId"],
                            "name": start["toolUse"]["name"],
                            "_raw": "",
                        }
                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        token = delta["text"]
                        full_text += token
                        buffer += token
                        # Yield complete sentences (split on newline)
                        while "\n" in buffer:
                            sentence, buffer = buffer.split("\n", 1)
                            if sentence.strip():
                                yield sentence.strip()
                    elif "toolUse" in delta and current_tool is not None:
                        inp = delta["toolUse"].get("input", "")
                        if isinstance(inp, str):
                            current_tool["_raw"] += inp
                elif "contentBlockStop" in event:
                    if current_tool is not None:
                        raw = current_tool.pop("_raw", "{}")
                        try:
                            current_tool["input"] = json.loads(raw)
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_requests.append(current_tool)
                        current_tool = None

            # Yield remaining buffer
            if buffer.strip():
                yield buffer.strip()

            # No tool calls — done
            if not tool_requests:
                return

            # Execute tools and continue the loop
            assistant_content: list[dict[str, Any]] = []
            if full_text:
                assistant_content.append({"text": full_text})
            for tr in tool_requests:
                assistant_content.append({
                    "toolUse": {
                        "toolUseId": tr["toolUseId"],
                        "name": tr["name"],
                        "input": tr["input"],
                    }
                })
            working_messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for tr in tool_requests:
                result = await self._execute_tool(tr["name"], tr["input"])
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tr["toolUseId"],
                        "content": [{"text": result}],
                    }
                })
            working_messages.append({"role": "user", "content": tool_results})

        # Fallback after max iterations
        yield "I'm sorry, I wasn't able to complete that request."

    def _build_request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the converse_stream request kwargs."""
        kwargs: dict[str, Any] = {
            "modelId": self.model.config["model_id"],
            "system": [{"text": self.system_prompt}],
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.model.config.get("max_tokens", 1024),
                "temperature": self.model.config.get("temperature", 0.7),
            },
        }
        if self._tool_specs:
            kwargs["toolConfig"] = {"tools": self._tool_specs}

        # Include guardrails if configured on the model
        if self.model.config.get("guardrail_id") and self.model.config.get("guardrail_version"):
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self.model.config["guardrail_id"],
                "guardrailVersion": self.model.config["guardrail_version"],
                "trace": self.model.config.get("guardrail_trace", "enabled"),
            }

        return kwargs

    def _convert_messages(self, sonic_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Sonic Expert Tool messages to Bedrock Converse format."""
        converted = []
        for msg in sonic_messages:
            role = msg.get("role", "USER").lower()
            content_blocks = msg.get("content", [])
            bedrock_content = []
            for block in content_blocks:
                if "text" in block:
                    bedrock_content.append({"text": block["text"]})
            if bedrock_content:
                converted.append({"role": role, "content": bedrock_content})
        return converted

    async def _execute_tool(self, name: str, input_data: dict[str, Any]) -> str:
        """Execute a registered tool by name."""
        executor = self._tool_executors.get(name)
        if executor:
            try:
                result = executor(**input_data)
                return result if isinstance(result, str) else json.dumps(result)
            except Exception as e:
                logger.error(f"Tool execution error: {name} - {e}")
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _build_tool_specs(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert @tool functions to Bedrock tool spec format."""
        specs = []
        for t in tools:
            if hasattr(t, "tool_spec"):
                spec = t.tool_spec
                specs.append({"toolSpec": spec})
            elif hasattr(t, "TOOL_SPEC"):
                specs.append({"toolSpec": t.TOOL_SPEC})
        return specs

    def _build_tool_executors(self, tools: list[Any]) -> dict[str, Any]:
        """Map tool names to callable executors."""
        executors: dict[str, Any] = {}
        for t in tools:
            if hasattr(t, "tool_spec"):
                name = t.tool_spec.get("name", "")
                executors[name] = t
            elif hasattr(t, "TOOL_SPEC"):
                name = t.TOOL_SPEC.get("name", "")
                executors[name] = t
        return executors


class StrandsAgentReasoner:
    """Wraps any Strands Agent as an ExpertToolReasoner.

    This gives Expert Tool access to all Strands-supported model providers
    (Bedrock, OpenAI, Gemini, Anthropic, Ollama, LiteLLM, Mistral, etc.)
    without writing custom reasoner implementations.

    The Agent handles tool calling internally — the reasoner just invokes it
    and yields the response text.

    Usage:
        from strands import Agent
        from strands.models.openai import OpenAIModel

        openai_agent = Agent(
            model=OpenAIModel(model_id="gpt-4o"),
            tools=[get_weather],
            system_prompt="You are a helpful voice assistant.",
        )
        reasoner = StrandsAgentReasoner(agent=openai_agent)
        model = BidiNovaSonicModel(reasoner=reasoner)
    """

    def __init__(self, agent: Any):
        """Initialize with a Strands Agent instance.

        Args:
            agent: A Strands Agent instance (with model, tools, system_prompt configured).
        """
        self.agent = agent

    async def reason(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Invoke the Strands Agent and yield response sentences.

        Extracts the latest user text from Sonic messages, invokes the agent,
        and splits the response into sentences for TTS.

        Args:
            messages: Conversation messages from Nova Sonic Expert Tool invocation.

        Yields:
            Text sentences from the agent response.
        """
        user_text = self._extract_latest_user_text(messages)
        if not user_text:
            yield "I didn't catch that. Could you say it again?"
            return

        # Invoke the agent (runs synchronously with its own tool loops)
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.agent(user_text)
        )

        # Yield response text sentence by sentence
        response_text = str(result)
        for sentence in self._split_sentences(response_text):
            yield sentence

    def _extract_latest_user_text(self, messages: list[dict[str, Any]]) -> str:
        """Extract the most recent user text from Sonic messages."""
        for m in reversed(messages):
            if m.get("role", "").upper() == "USER":
                parts = m.get("content", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        return ""

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for TTS-friendly streaming."""
        sentences = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped:
                sentences.append(stripped)
        return sentences if sentences else [text.strip()]
