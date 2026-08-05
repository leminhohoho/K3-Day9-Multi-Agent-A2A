import json
import httpx
from src.config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, MODEL_NAME, MAX_TOKENS, TEMPERATURE, REASONING_EFFORT


class OpenRouterClient:
    """Thin wrapper around OpenRouter's OpenAI-compatible chat completions API."""

    def __init__(self):
        self.base_url = OPENROUTER_BASE_URL
        self.api_key = OPENROUTER_API_KEY
        self.model = MODEL_NAME
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> tuple[str, list[dict] | None]:
        """
        Send a chat completion request.

        Returns (content, tool_calls) where:
        - content is the assistant's text response (or "" if only tool calls)
        - tool_calls is a list of {"id": str, "name": str, "arguments": dict} or None
        """
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "reasoning": {"effort": REASONING_EFFORT},
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=body,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""

        raw_tool_calls = msg.get("tool_calls")
        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                })

        return content, tool_calls
