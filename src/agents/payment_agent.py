from src.agent_base import BaseAgent
from src.tools.payment_tools import PAYMENT_TOOL_DEFS, PAYMENT_TOOL_MAP

PAYMENT_SYSTEM_PROMPT = """You are a Payment Data Agent for Olist e-commerce dispute resolution.

Your job: given an order_id, investigate the payment records and reconcile against item + freight totals.

TOOLS:
- lookup_payments: get all payment rows (sequential, type, installments, value)
- sum_payments: compute total payment value
- reconcile_payment: reconcile payment total vs item + freight totals

PROCEDURE:
1. Call lookup_payments to view all payment transactions.
2. Call sum_payments to get the total paid.
3. Call reconcile_payment to check if payment matches item+freight.

OUTPUT: Return a JSON object with these exact fields:
{
  "payment_rows": [{"sequential": 1, "type": "credit_card", "value": 99.33}],
  "payment_total_brl": 99.33,
  "expected_total_brl": 72.19,
  "reconciled": false,
  "discrepancy_brl": 27.14
}

If no payments exist, set payment_rows=[] and payment_total_brl=0.0.
"""


class PaymentAgent(BaseAgent):
    def __init__(self, data: dict):
        self.data = data
        super().__init__(
            name="PaymentAgent",
            system_prompt=PAYMENT_SYSTEM_PROMPT,
            tools=PAYMENT_TOOL_DEFS,
        )

    def _execute_tool(self, name: str, args: dict) -> dict:
        fn = PAYMENT_TOOL_MAP.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}
        order_id = args["order_id"]
        if name == "lookup_payments":
            return fn(self.data["payments"], order_id)
        elif name == "sum_payments":
            return fn(self.data["payments"], order_id)
        elif name == "reconcile_payment":
            return fn(self.data["payments"], self.data["items"], order_id)
        return {"error": f"Unhandled tool: {name}"}