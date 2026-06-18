"""Expert Tool support for BidiNovaSonicModel.

Enables bring-your-own-LLM reasoning with Nova Sonic. The Expert Tool is a
Nova Sonic-specific feature that allows customers to use an external reasoning
LLM while Sonic handles ASR/TTS.

Usage:
    from strands.experimental.bidi.expert_tool import BedrockConverseReasoner, ExpertToolConfig

    reasoner = BedrockConverseReasoner(
        model=BedrockModel(model_id="qwen.qwen3-32b-v1:0"),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
    )

    model = BidiNovaSonicModel(reasoner=reasoner)
    agent = BidiAgent(model=model)
"""

from .config import ExpertToolConfig
from .reasoner import BedrockConverseReasoner, ExpertToolReasoner, StrandsAgentReasoner

__all__ = [
    "ExpertToolConfig",
    "ExpertToolReasoner",
    "BedrockConverseReasoner",
    "StrandsAgentReasoner",
]
