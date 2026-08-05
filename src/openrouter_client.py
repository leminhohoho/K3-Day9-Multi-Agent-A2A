import json
import time
import httpx
from src.config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, MODEL_NAME, MAX_TOKENS, TEMPERATURE, REASONING_EFFORT


class OpenRouterClient:
    """Thin wrapper around OpenRouter's OpenAI-compatible chat completions API."""

    def __init__(self, max_retries: int = 4):
        self.base_url = OPENROUTER_BASE_URL
        self.api_key = OPENROUTER_API_KEY
        self.model = MODEL_NAME
        self.max_retries = max_retries
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

        response = None
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=body,
                    timeout=120,
                )
                if response.status_code in (429, 408, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    time.sleep(0.25 * (2 ** attempt))  # exponential backoff
                    continue
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                if e.response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                raise
            except httpx.HTTPError as e:
                last_error = str(e)
                time.sleep(0.25 * (2 ** attempt))
                continue

        if response is None or response.status_code >= 400:
            raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")

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
