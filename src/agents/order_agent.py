import json
from src.agent_base import BaseAgent
from src.tools.order_tools import ORDER_TOOL_MAP, lookup_order, lookup_items, lookup_sellers, sum_item_totals

ORDER_SYSTEM_PROMPT = """You are an Order Agent analyzing Olist order data.

You receive real data from the order, items, and sellers tables. Analyze it and
return ONLY this compact JSON (no explanation, no markdown):
{"order_status":"...","item_total_brl":0.0,"freight_total_brl":0.0,"sellers":["seller_id1"]}

- order_status: exact status string from the data.
- item_total_brl / freight_total_brl: sums of price / freight_value over all items, 2 decimals.
- sellers: unique seller_id values.
If the order has no items, use 0.0 totals and empty sellers array.
"""


class OrderAgent(BaseAgent):
    """
    Order Agent — single-shot: pulls real order data via its scoped tools
    (deterministic), then the LLM reasons over it and emits a compact
    finding. The coordinator expands details from source data afterwards.
    """

    def __init__(self, data: dict):
        self.data = data
        super().__init__(
            name="OrderAgent",
            system_prompt=ORDER_SYSTEM_PROMPT,
            tools=None,
        )

    def _execute_tool(self, name: str, args: dict) -> dict:
        fn = ORDER_TOOL_MAP.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}
        return fn(self.data["orders"] if name == "lookup_order" else self.data["items"], args["order_id"])

    def run(self, input_data: dict, trace_callback=None) -> dict:
        order_id = input_data["order_id"]
        if trace_callback:
            trace_callback(self.name, "query", {"order_id": order_id})

        # Gather real data via scoped tools (deterministic)
        order = lookup_order(self.data["orders"], order_id)
        items = lookup_items(self.data["items"], order_id)
        sellers = lookup_sellers(self.data["items"], order_id)
        totals = sum_item_totals(self.data["items"], order_id)

        if order.get("error") == "not_found":
            result = {
                "order_status": "unknown", "purchase_ts": None, "approved_ts": None,
                "delivered_carrier_ts": None, "delivered_customer_ts": None,
                "estimated_delivery_ts": None, "items": [], "sellers": [],
                "item_total_brl": 0.0, "freight_total_brl": 0.0,
            }
            if trace_callback:
                trace_callback(self.name, "complete", result)
            return result

        context = {
            "order": order,
            "items": items,
            "totals": totals,
        }
        if trace_callback:
            trace_callback(self.name, "llm_call", {"round": 0})
        content, _ = self.client.chat(
            system=self.system_prompt,
            messages=[{"role": "user", "content": json.dumps(context, default=str)}],
            tools=None,
        )
        finding = self._extract_json(content)

        # Merge LLM finding with full deterministic details
        result = {
            "order_status": finding.get("order_status", str(order.get("order_status", "unknown"))),
            "purchase_ts": str(order.get("order_purchase_timestamp", "")) or None,
            "approved_ts": str(order.get("order_approved_at", "")) or None,
            "delivered_carrier_ts": str(order.get("order_delivered_carrier_date", "")) or None,
            "delivered_customer_ts": str(order.get("order_delivered_customer_date", "")) or None,
            "estimated_delivery_ts": str(order.get("order_estimated_delivery_date", "")) or None,
            "items": [{
                "item_id": int(it.get("order_item_id", 0)),
                "seller_id": str(it.get("seller_id", "")),
                "price": round(float(it.get("price", 0.0)), 2),
                "freight_value": round(float(it.get("freight_value", 0.0)), 2),
                "shipping_limit_ts": str(it.get("shipping_limit_date", "")) or None,
            } for it in items],
            "sellers": sellers,
            "item_total_brl": totals["item_total_brl"],
            "freight_total_brl": totals["freight_total_brl"],
        }
        if trace_callback:
            trace_callback(self.name, "complete", {"result_keys": list(result.keys())})
        return result