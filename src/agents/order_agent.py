from src.agent_base import BaseAgent
from src.tools.order_tools import lookup_order, lookup_items, lookup_sellers, sum_item_totals


class OrderAgent(BaseAgent):
    """
    Order & Seller Data Agent.

    Retrieves order data deterministically from the scoped tools (no LLM
    required for data extraction), then formats the finding.
    """

    def __init__(self, data: dict):
        self.data = data
        super().__init__(
            name="OrderAgent",
            system_prompt="",
            tools=None,
        )

    def run(self, input_data: dict, trace_callback=None) -> dict:
        order_id = input_data["order_id"]
        if trace_callback:
            trace_callback(self.name, "query", {"order_id": order_id})

        # Look up order metadata
        order = lookup_order(self.data["orders"], order_id)
        if order.get("error") == "not_found":
            return {
                "order_status": "unknown",
                "purchase_ts": None,
                "approved_ts": None,
                "delivered_carrier_ts": None,
                "delivered_customer_ts": None,
                "estimated_delivery_ts": None,
                "items": [],
                "sellers": [],
                "item_total_brl": 0.0,
                "freight_total_brl": 0.0,
            }

        # Items
        items_raw = lookup_items(self.data["items"], order_id)
        items = []
        for it in items_raw:
            items.append({
                "item_id": int(it.get("order_item_id", 0)),
                "seller_id": str(it.get("seller_id", "")),
                "price": round(float(it.get("price", 0.0)), 2),
                "freight_value": round(float(it.get("freight_value", 0.0)), 2),
                "shipping_limit_ts": str(it.get("shipping_limit_date", "")) or None,
            })

        sellers = lookup_sellers(self.data["items"], order_id)
        totals = sum_item_totals(self.data["items"], order_id)

        if trace_callback:
            trace_callback(self.name, "complete", {"items": len(items), "sellers": len(sellers)})

        return {
            "order_status": str(order.get("order_status", "unknown")),
            "purchase_ts": str(order.get("order_purchase_timestamp", "")) or None,
            "approved_ts": str(order.get("order_approved_at", "")) or None,
            "delivered_carrier_ts": str(order.get("order_delivered_carrier_date", "")) or None,
            "delivered_customer_ts": str(order.get("order_delivered_customer_date", "")) or None,
            "estimated_delivery_ts": str(order.get("order_estimated_delivery_date", "")) or None,
            "items": items,
            "sellers": sellers,
            "item_total_brl": totals["item_total_brl"],
            "freight_total_brl": totals["freight_total_brl"],
        }