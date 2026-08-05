import pytest
from src.agents.policy_agent import PolicyAgent


def test_policy_agent_late_delivery_seller():
    """PolicyAgent should detect late_delivery_seller when carrier received late."""
    merged = {
        "order_status": "delivered",
        "delivered_late": True,
        "carrier_received_late": True,
        "responsible": "seller",
        "seller_ids": ["seller_1"],
        "item_total_brl": 100.0,
        "freight_total_brl": 15.0,
        "payment_total_brl": 115.0,
        "reconciled": True,
    }
    agent = PolicyAgent()
    result = agent.run({"findings": merged, "policy_version": "EC_POLICY_V1"})
    assert result["primary_issue"] == "late_delivery_seller"
    assert result["case_status"] == "action_required"
    assert result["financial_resolution"]["recommended_refund_brl"] == 15.0


def test_policy_agent_canceled_order():
    """PolicyAgent should detect canceled_order_paid."""
    merged = {
        "order_status": "canceled",
        "payment_total_brl": 100.0,
        "item_total_brl": 85.0,
        "freight_total_brl": 15.0,
        "reconciled": True,
    }
    agent = PolicyAgent()
    result = agent.run({"findings": merged, "policy_version": "EC_POLICY_V1"})
    assert result["primary_issue"] == "canceled_order_paid"
    assert result["case_status"] == "action_required"
    assert result["financial_resolution"]["recommended_refund_brl"] == 100.0


def test_policy_agent_unsupported_late_claim():
    """PolicyAgent should reject claims where delivery was on time and payment matches."""
    merged = {
        "order_status": "delivered",
        "delivered_late": False,
        "reconciled": True,
        "payment_total_brl": 115.0,
        "item_total_brl": 100.0,
        "freight_total_brl": 15.0,
    }
    agent = PolicyAgent()
    result = agent.run({"findings": merged, "policy_version": "EC_POLICY_V1"})
    assert result["primary_issue"] == "unsupported_late_claim"
    assert result["case_status"] == "no_action"