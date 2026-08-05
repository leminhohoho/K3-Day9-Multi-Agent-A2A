import json
from src.agent_base import BaseAgent
from src.tools.payment_tools import PAYMENT_TOOL_MAP, lookup_payments, sum_payments, reconcile_payment

PAYMENT_SYSTEM_PROMPT = """You are a Payment Agent analyzing Olist payment data.

You receive real payment rows plus the item+freight reconciliation. Analyze
them and return ONLY this compact JSON (no explanation, no markdown):
{"payment_rows":[{"sequential":1,"type":"credit_card","value":99.33}],"payment_total_brl":99.33,"expected_total_brl":72.19,"reconciled":false,"discrepancy_brl":27.14}

- payment_rows: each row with sequential, type, value (2 decimals).
- reconciled: true if |payment_total - expected_total| <= 0.10.
If no payments exist, payment_rows=[] and payment_total_brl=0.0.
"""


class PaymentAgent(BaseAgent):
    """
    Payment Agent — single-shot: pulls real payment data via its scoped tools
    (deterministic), then the LLM reasons over it and emits a compact
    finding.
    """

    def __init__(self, data: dict):
        self.data = data
        super().__init__(
            name="PaymentAgent",
            system_prompt=PAYMENT_SYSTEM_PROMPT,
            tools=None,
        )

    def _execute_tool(self, name: str, args: dict) -> dict:
        fn = PAYMENT_TOOL_MAP.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}
        if name == "reconcile_payment":
            return fn(self.data["payments"], self.data["items"], args["order_id"])
        return fn(self.data["payments"], args["order_id"])

    def run(self, input_data: dict, trace_callback=None) -> dict:
        order_id = input_data["order_id"]
        if trace_callback:
            trace_callback(self.name, "query", {"order_id": order_id})

        # Gather real data via scoped tools (deterministic)
        rows_raw = lookup_payments(self.data["payments"], order_id)
        total = sum_payments(self.data["payments"], order_id)
        reconciliation = reconcile_payment(self.data["payments"], self.data["items"], order_id)

        if not rows_raw:
            result = {
                "payment_rows": [], "payment_total_brl": 0.0,
                "expected_total_brl": reconciliation["expected_total_brl"],
                "reconciled": reconciliation["reconciled"],
                "discrepancy_brl": reconciliation["discrepancy_brl"],
            }
            if trace_callback:
                trace_callback(self.name, "complete", result)
            return result

        context = {
            "payment_rows": rows_raw,
            "item_plus_freight_total": reconciliation["expected_total_brl"],
        }
        if trace_callback:
            trace_callback(self.name, "llm_call", {"round": 0})
        content, _ = self.client.chat(
            system=self.system_prompt,
            messages=[{"role": "user", "content": json.dumps(context, default=str)}],
            tools=None,
        )
        finding = self._extract_json(content)

        # Merge LLM finding with deterministic reconciliation
        payment_rows = finding.get("payment_rows") or [
            {
                "sequential": int(r.get("payment_sequential", 0)),
                "type": str(r.get("payment_type", "")),
                "value": round(float(r.get("payment_value", 0.0)), 2),
            } for r in rows_raw
        ]
        result = {
            "payment_rows": payment_rows,
            "payment_total_brl": total,
            "expected_total_brl": reconciliation["expected_total_brl"],
            "reconciled": reconciliation["reconciled"],
            "discrepancy_brl": reconciliation["discrepancy_brl"],
        }
        if trace_callback:
            trace_callback(self.name, "complete", {"result_keys": list(result.keys())})
        return result