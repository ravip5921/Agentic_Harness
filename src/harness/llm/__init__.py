"""Provider selection and the public text-generation interface."""

from harness.llm.base import Completion, GenerationError, TextGenerator
from harness.llm.settings import LLMSettings


def create_generator(*, model=None, provider=None, client=None) -> TextGenerator:
    """Inject a provider-native client in tests, or configure one from the environment."""
    settings = LLMSettings.from_environment(provider=provider, model=model)
    if settings.provider == "groq":
        from harness.llm.groq import GroqGenerator
        return GroqGenerator(settings, client)
    from harness.llm.http import GeminiGenerator, CompatibleGenerator
    adapter = GeminiGenerator if settings.provider == "gemini" else CompatibleGenerator
    return adapter(settings, client)


__all__ = ["Completion", "GenerationError", "TextGenerator", "LLMSettings", "create_generator"]
