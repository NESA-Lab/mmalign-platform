"""Intern API adapter with native thinking control and complete-answer checks."""

from mmalign.core.backend import IncompatibleBackendError
from .openai import OpenAIBackend, OpenAIBackendError


class InternBackend(OpenAIBackend):
    thinking_control = True

    def __init__(self, *args, thinking_style="official", min_output_tokens=16384, **kwargs):
        super().__init__(*args, **kwargs)
        if thinking_style not in {"official", "chat_template"}:
            raise ValueError("Unknown Intern thinking style")
        self.thinking_style = thinking_style
        self.min_output_tokens = int(min_output_tokens)

    async def generate(self, request, **kwargs):
        if self._api_style != "chat_completions":
            raise IncompatibleBackendError("Intern requires chat_completions")
        request = request.model_copy(update={
            "max_output_tokens": max(request.max_output_tokens or 0, self.min_output_tokens)})
        enabled = request.metadata.get("enable_thinking")
        if enabled is not None:
            if type(enabled) is not bool:
                raise ValueError("enable_thinking must be boolean")
            body = dict(kwargs.pop("extra_body", {}) or {})
            if self.thinking_style == "official":
                body["thinking_mode"] = enabled
            else:
                template = dict(body.get("chat_template_kwargs", {}))
                template["enable_thinking"] = enabled
                body["chat_template_kwargs"] = template
            kwargs["extra_body"] = body
        result = await super().generate(request, **kwargs)
        if not result.text().strip():
            raise OpenAIBackendError("Intern returned no final answer")
        if "length" in result.metadata.get("finish_reasons", []):
            raise OpenAIBackendError("Intern answer truncated; increase output limit")
        result.metadata.update(thinking_requested=enabled, thinking_style=self.thinking_style)
        return result
