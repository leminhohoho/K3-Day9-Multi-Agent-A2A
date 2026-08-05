from src.agent_base import BaseAgent
from src.tools.payment_tools import lookup_payments, sum_payments, reconcile_payment


class PaymentAgent(BaseAgent):
    """
    Payment Data Agent.

    Retrieves payment data deterministically from the scoped tools,
    then formats the finding.
    """

    def __init__(self, data: dict):
        self.data = data
        super().__init__(
            name="PaymentAgent",
            system_prompt="",
            tools=None,
        )

    def run(self, input_data: dict, trace_callback=None) -> dict:
        order_id = input_data["order_id"]
        if trace_callback:
            trace_callback(self.name, "query", {"order_id": order_id})

        rows_raw = lookup_payments(self.data["payments"], order_id)
        payment_rows = []
        for r in rows_raw:
            payment_rows.append({
                "sequential": int(r.get("payment_sequential", 0)),
                "type": str(r.get("payment_type", "")),
                "value": round(float(r.get("payment_value", 0.0)), 2),
            })

        total = sum_payments(self.data["payments"], order_id)
        reconciliation = reconcile_payment(self.data["payments"], self.data["items"], order_id)

        if trace_callback:
            trace_callback(self.name, "complete", {"rows": len(payment_rows), "total": total})

        return {
            "payment_rows": payment_rows,
            "payment_total_brl": total,
            "expected_total_brl": reconciliation["expected_total_brl"],
            "reconciled": reconciliation["reconciled"],
            "discrepancy_brl": reconciliation["discrepancy_brl"],
        }