"""Expert Tool configuration for BidiNovaSonicModel.

Provides the ExpertToolConfig dataclass for advanced Expert Tool configuration.
Most developers should use the `reasoner` shorthand on BidiNovaSonicModel instead.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .reasoner import ExpertToolReasoner


@dataclass
class ExpertToolConfig:
    """Advanced configuration for Expert Tool integration.

    Most developers should use `BidiNovaSonicModel(reasoner=...)` directly.
    Use ExpertToolConfig for fine-grained control over interruption callbacks,
    tool iteration limits, and fallback messages.

    Attributes:
        reasoner: The reasoning implementation (required).
        max_tool_iterations: Max tool call loops per turn before fallback.
        on_interrupted: Optional callback when barge-in cancels active reasoning.
        fallback_message: Message to return if reasoner yields nothing or errors.
    """

    reasoner: ExpertToolReasoner
    max_tool_iterations: int = 5
    on_interrupted: Callable[[], None] | None = None
    fallback_message: str = "I'm not sure how to help with that."
