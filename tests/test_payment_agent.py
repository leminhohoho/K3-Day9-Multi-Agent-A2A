import pytest
from src.agents.payment_agent import PaymentAgent
from src.loader import load_all_data
from src.models import PaymentFinding


@pytest.fixture(scope="module")
def data():
    return load_all_data()


def test_payment_agent_returns_valid_finding(data):
    """PaymentAgent should produce a valid PaymentFinding for a real order."""
    agent = PaymentAgent(data)
    result = agent.run({"order_id": "e2a03ccf5ea816036608b2d8c3ab8e60"})
    finding = PaymentFinding(**result)
    assert isinstance(finding.payment_total_brl, float)
    assert isinstance(finding.expected_total_brl, float)
    assert isinstance(finding.reconciled, bool)


def test_payment_agent_no_payments(data):
    """PaymentAgent should handle orders with no payments."""
    agent = PaymentAgent(data)
    result = agent.run({"order_id": "NONEXISTENT"})
    assert result["payment_total_brl"] == 0.0
    assert result["payment_rows"] == []