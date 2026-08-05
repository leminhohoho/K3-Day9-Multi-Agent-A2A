import pytest
from src.openrouter_client import OpenRouterClient
from src.agent_base import BaseAgent


def test_openrouter_client_returns_response():
    """Client should return a complete response for a simple prompt."""
    client = OpenRouterClient()
    content, tool_calls = client.chat(
        system="You are a helpful assistant. Say 'hello'.",
        messages=[{"role": "user", "content": "Say hello"}],
        tools=None,
    )
    assert isinstance(content, str)
    assert len(content) > 0
    assert tool_calls is None


def test_base_agent_returns_structured_output():
    """Agent should produce a JSON dict matching the output_schema."""

    class TestAgent(BaseAgent):
        def _execute_tool(self, name, args):
            return {"result": "ok"}

    agent = TestAgent(
        name="test_agent",
        system_prompt="You are a test agent. Always return this exact JSON: {\"test\": \"value\"}",
        tools=None,
    )
    result = agent.run(input_data={"task": "return test value"})
    assert isinstance(result, dict)
    assert result.get("test") == "value"