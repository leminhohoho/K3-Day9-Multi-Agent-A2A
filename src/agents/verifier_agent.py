import re
from src.agent_base import BaseAgent


class VerifierError(Exception):
    """Raised when a candidate output fails verification."""
    pass


VERIFIER_SYSTEM_PROMPT = """You are a Verifier Agent for Olist e-commerce dispute resolution.

Your job: validate a candidate output dict against the schema, entity limits, evidence ID format, and rounding rules.

VALIDATION RULES:
1. confidence must be in [0, 1]
2. At most 5 IDs per entity set (order_ids, item_ids, seller_ids, payment_ids)
3. At most 10 evidence_ids total
4. At most 3 root causes
5. At most 3 responsible parties
6. At most 5 resolution actions
7. Evidence IDs must match one of these patterns:
   - order:<order_id>
   - item:<order_id>:<order_item_id>
   - payment:<order_id>:<payment_sequential>
   - seller:<seller_id>
   - policy:<root_cause_code>
8. case_status must be "action_required" or "no_action"

INPUT: {"candidate": {...}, "order_id": "..."}

OUTPUT: Return a JSON object with these exact fields:
{
  "valid": true,
  "errors": [],
  "candidate": { ...the candidate dict exactly as given... }
}

If valid, copy the candidate into the "candidate" field WITHOUT adding, removing,
or modifying any field inside it. If invalid, keep the candidate as-is and list the
errors in "errors".

Be strict — reject anything that doesn't match the spec.
"""


class VerifierAgent(BaseAgent):
    """VerifierAgent validates candidate output. No data tools needed."""

    def __init__(self):
        super().__init__(
            name="VerifierAgent",
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            tools=None,
        )

    def run(self, input_data: dict, trace_callback=None) -> dict:
        candidate = input_data["candidate"]
        # Deterministic schema/format validation (no LLM needed — all checks
        # are exact rules on limits, formats, and rounding).
        self._validate_deterministic(candidate)
        if trace_callback:
            trace_callback(self.name, "complete", {"valid": True})
        return candidate

    def _validate_deterministic(self, candidate: dict):
        """Run deterministic checks that don't need an LLM."""
        errors = []

        # confidence range
        conf = candidate.get("assessment", {}).get("confidence", -1)
        if not (0 <= conf <= 1):
            errors.append(f"confidence {conf} out of range [0, 1]")

        # monetary rounding: all BRL values should be rounded to 2 decimals
        fin = candidate.get("financial_resolution", {})
        for key in ("item_total_brl", "freight_total_brl", "payment_total_brl", "recommended_refund_brl"):
            val = fin.get(key, 0.0)
            if isinstance(val, float) and round(val, 2) != val:
                errors.append(f"financial_resolution.{key} must be rounded to 2 decimal places (current value: {val})")

        # entity limits
        entities = candidate.get("affected_entities", {})
        for key in ("order_ids", "item_ids", "seller_ids", "payment_ids"):
            if len(entities.get(key, [])) > 5:
                errors.append(f"{key}: {len(entities[key])} > 5 max")

        # evidence limit
        evidence = candidate.get("evidence_ids", [])
        if len(evidence) > 10:
            errors.append(f"evidence_ids: {len(evidence)} > 10 max")

        # root cause limit
        causes = candidate.get("root_cause_analysis", {}).get("ranked_causes", [])
        if len(causes) > 3:
            errors.append(f"ranked_causes: {len(causes)} > 3 max")

        # responsible parties limit
        parties = candidate.get("root_cause_analysis", {}).get("responsible_parties", [])
        if len(parties) > 3:
            errors.append(f"responsible_parties: {len(parties)} > 3 max")

        # actions limit
        actions = candidate.get("resolution_actions", [])
        if len(actions) > 5:
            errors.append(f"resolution_actions: {len(actions)} > 5 max")

        # evidence ID format
        valid_patterns = [
            re.compile(r"^order:[A-Za-z0-9_]+$"),
            re.compile(r"^item:[A-Za-z0-9_]+:\d+$"),
            re.compile(r"^payment:[A-Za-z0-9_]+:\d+$"),
            re.compile(r"^seller:[A-Za-z0-9_]+$"),
            re.compile(r"^policy:[A-Z_]+$"),
        ]
        for ev in evidence:
            if not any(p.match(ev) for p in valid_patterns):
                errors.append(f"invalid evidence ID format: {ev}")

        # case_status
        status = candidate.get("assessment", {}).get("case_status", "")
        if status not in ("action_required", "no_action"):
            errors.append(f"invalid case_status: {status}")

        if errors:
            raise VerifierError("; ".join(errors))