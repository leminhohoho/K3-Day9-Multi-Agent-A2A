import pytest
from src.agents.delivery_agent import DeliveryAgent
from src.loader import load_all_data
from src.models import DeliveryFinding


@pytest.fixture(scope="module")
def data():
    return load_all_data()


def test_delivery_agent_returns_valid_finding(data):
    """DeliveryAgent should produce a valid DeliveryFinding for a real order."""
    agent = DeliveryAgent(data)
    result = agent.run({"order_id": "e2a03ccf5ea816036608b2d8c3ab8e60"})
    finding = DeliveryFinding(**result)
    assert isinstance(finding.delivered_late, bool)
    assert finding.responsible in ("none", "seller", "logistics_provider")


def test_delivery_agent_missing_order(data):
    """DeliveryAgent should handle missing order gracefully."""
    agent = DeliveryAgent(data)
    result = agent.run({"order_id": "NONEXISTENT"})
    assert result["delivered_late"] is False