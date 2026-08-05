import json
from src.openrouter_client import OpenRouterClient


class BaseAgent:
    """
    ReAct agent: system prompt + scoped tools -> structured JSON output.

    Subclasses set system_prompt and tools. The run() method:
    1. Sends the prompt + input to the LLM
    2. If tool calls are returned, executes them, appends results, loops
    3. Extracts JSON from the final response
    """

    def __init__(self, name: str, system_prompt: str, tools: list[dict] | None = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_defs = tools or []  # OpenAI-compatible tool definitions
        self.client = OpenRouterClient()

    def run(self, input_data: dict, trace_callback=None) -> dict:
        """
        Execute the agent's ReAct loop.

        Args:
            input_data: dict with keys the agent expects (e.g. order_id, dataframes)
            trace_callback: optional fn(name, step, data) for logging

        Returns: structured JSON dict (agent-specific schema)
        """
        messages = [{"role": "user", "content": json.dumps(input_data, default=str)}]
        max_rounds = 5  # prevent infinite loops

        for round_idx in range(max_rounds):
            if trace_callback:
                trace_callback(self.name, "llm_call", {"round": round_idx})

            content, tool_calls, usage = self.client.chat(
                system=self.system_prompt,
                messages=messages,
                tools=self.tool_defs if self.tool_defs else None,
                tool_choice="auto" if self.tool_defs else None,
            )

            if tool_calls:
                # Execute each tool call and append results
                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["arguments"]
                    if trace_callback:
                        trace_callback(self.name, "tool_call", {"tool": tool_name, "args": tool_args})

                    result = self._execute_tool(tool_name, tool_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, default=str),
                    })
            else:
                # No tool calls — parse JSON from content
                if trace_callback:
                    trace_callback(self.name, "response", {"content": content})
                return self._extract_json(content)

        # Fallback: try to extract JSON from last message
        return self._extract_json(messages[-1].get("content", "{}"))

    def _execute_tool(self, name: str, args: dict) -> dict:
        """Execute a tool by name. Subclasses override with scoped tool dict."""
        raise NotImplementedError("Subclasses must implement _execute_tool")

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response text (handles markdown fences)."""
        text = text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            start = text.find("{")
            if start == -1:
                start = text.find("[")
            if start >= 0:
                text = text[start:]
            end = text.rfind("}")
            if end >= 0:
                text = text[: end + 1]
            elif text.rfind("]") >= 0:
                text = text[: text.rfind("]") + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Log the raw text for debugging
            print(f"[{self.name}] Failed to parse JSON from: {text[:500]}")
            return {}
