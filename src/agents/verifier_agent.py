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
8. All monetary values must be rounded to 2 decimal places
9. case_status must be "action_required" or "no_action"

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
        # First pass: deterministic validations before LLM call
        self._validate_deterministic(candidate)
        # Second pass: LLM-based validation for nuanced checks
        result = super().run(input_data, trace_callback)

        # Unwrap the LLM's {valid, errors, candidate} envelope if present
        if isinstance(result, dict) and "valid" in result:
            if result.get("valid") is False:
                errors = result.get("errors", ["Unknown validation error"])
                raise VerifierError("; ".join(errors))
            result = result.get("candidate", result)

        # Ensure only schema fields survive (strip hallucinated extras)
        allowed = {
            "case_id", "assessment", "affected_entities",
            "root_cause_analysis", "evidence_ids",
            "financial_resolution", "resolution_actions",
        }
        result = {k: v for k, v in result.items() if k in allowed}

        return result

    def _validate_deterministic(self, candidate: dict):
        """Run deterministic checks that don't need an LLM."""
        errors = []

        # confidence range
        conf = candidate.get("assessment", {}).get("confidence", -1)
        if not (0 <= conf <= 1):
            errors.append(f"confidence {conf} out of range [0, 1]")

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