import pytest
from src.agents.verifier_agent import VerifierAgent, VerifierError


def _valid_candidate():
    return {
        "case_id": "EC_001",
        "assessment": {"primary_issue": "late_delivery_seller", "case_status": "action_required", "confidence": 0.92},
        "affected_entities": {
            "order_ids": ["abc123"],
            "item_ids": ["abc123:1"],
            "seller_ids": ["seller_1"],
            "payment_ids": ["abc123:1"],
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": "SELLER_HANDOFF_AFTER_LIMIT", "rank": 1}],
            "responsible_parties": [{"party_type": "seller", "party_id": "seller_1"}],
        },
        "evidence_ids": ["order:abc123", "item:abc123:1", "payment:abc123:1", "seller:seller_1", "policy:SELLER_HANDOFF_AFTER_LIMIT"],
        "financial_resolution": {
            "currency": "BRL", "item_total_brl": 100.0, "freight_total_brl": 15.0,
            "payment_total_brl": 115.0, "recommended_refund_brl": 15.0,
        },
        "resolution_actions": ["refund_freight"],
    }


def test_verifier_accepts_valid_output():
    """VerifierAgent should accept a well-formed output."""
    agent = VerifierAgent()
    result = agent.run({"candidate": _valid_candidate(), "order_id": "abc123"})
    assert result["case_id"] == "EC_001"


def test_verifier_rejects_bad_confidence():
    """VerifierAgent should reject confidence outside [0, 1]."""
    candidate = _valid_candidate()
    candidate["assessment"]["confidence"] = 1.5
    agent = VerifierAgent()
    with pytest.raises(VerifierError):
        agent.run({"candidate": candidate, "order_id": "abc123"})


def test_verifier_rejects_too_many_entities():
    """VerifierAgent should reject >5 entities in any set."""
    candidate = _valid_candidate()
    candidate["affected_entities"]["order_ids"] = [f"o{i}" for i in range(6)]
    agent = VerifierAgent()
    with pytest.raises(VerifierError):
        agent.run({"candidate": candidate, "order_id": "abc123"})


def test_verifier_rejects_bad_evidence_id():
    """VerifierAgent should reject evidence IDs not matching the allowed format."""
    candidate = _valid_candidate()
    candidate["evidence_ids"] = ["invalid:format:extra:part"]
    agent = VerifierAgent()
    with pytest.raises(VerifierError):
        agent.run({"candidate": candidate, "order_id": "abc123"})


def test_verifier_rejects_bad_case_status():
    """VerifierAgent should reject invalid case_status."""
    candidate = _valid_candidate()
    candidate["assessment"]["case_status"] = "bogus"
    agent = VerifierAgent()
    with pytest.raises(VerifierError):
        agent.run({"candidate": candidate, "order_id": "abc123"})