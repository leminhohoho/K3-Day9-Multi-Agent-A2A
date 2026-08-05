import pytest
from src.agents.order_agent import OrderAgent
from src.loader import load_all_data
from src.models import OrderFinding


@pytest.fixture(scope="module")
def data():
    return load_all_data()


def test_order_agent_returns_valid_finding(data):
    """OrderAgent should produce a valid OrderFinding for a real order."""
    agent = OrderAgent(data)
    result = agent.run({"order_id": "e2a03ccf5ea816036608b2d8c3ab8e60"})
    finding = OrderFinding(**result)
    assert finding.order_status in (
        "delivered", "shipped", "canceled", "unavailable",
        "processing", "invoiced", "created", "approved",
    )
    assert isinstance(finding.item_total_brl, float)
    assert isinstance(finding.freight_total_brl, float)


def test_order_agent_missing_order(data):
    """OrderAgent should handle missing order gracefully."""
    agent = OrderAgent(data)
    result = agent.run({"order_id": "NONEXISTENT"})
    assert result["item_total_brl"] == 0.0
    assert result["freight_total_brl"] == 0.0
    assert result["items"] == []