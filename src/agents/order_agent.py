from src.agent_base import BaseAgent
from src.tools.order_tools import ORDER_TOOL_DEFS, ORDER_TOOL_MAP

ORDER_SYSTEM_PROMPT = """You are an Order & Seller Data Agent for Olist e-commerce dispute resolution.

Your job: given an order_id, investigate the order's status, items, sellers, and financial totals.

TOOLS:
- lookup_order: get order metadata (status, purchase/approval/delivery timestamps)
- lookup_items: get all items (product_id, seller_id, price, freight, shipping_limit_date)
- lookup_sellers: get unique seller IDs for this order
- sum_item_totals: compute sum of item prices and freight values

PROCEDURE:
1. Call lookup_order to get the order status and timestamps.
2. Call lookup_items to get all items and their details.
3. Call lookup_sellers to identify all sellers.
4. Call sum_item_totals to compute financial totals.

OUTPUT: Return a JSON object with these exact fields:
{
  "order_status": "string",
  "purchase_ts": "string or null",
  "approved_ts": "string or null",
  "delivered_carrier_ts": "string or null",
  "delivered_customer_ts": "string or null",
  "estimated_delivery_ts": "string or null",
  "items": [{"item_id": 1, "seller_id": "s1", "price": 58.90, "freight_value": 13.29, "shipping_limit_ts": "..."}],
  "sellers": ["seller_id1"],
  "item_total_brl": 58.90,
  "freight_total_brl": 13.29
}

If the order has no items, set items=[], sellers=[], item_total_brl=0.0, freight_total_brl=0.0.
If timestamps are null/missing, set them to null.
"""


class OrderAgent(BaseAgent):
    def __init__(self, data: dict):
        self.data = data
        super().__init__(
            name="OrderAgent",
            system_prompt=ORDER_SYSTEM_PROMPT,
            tools=ORDER_TOOL_DEFS,
        )

    def _execute_tool(self, name: str, args: dict) -> dict:
        fn = ORDER_TOOL_MAP.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}
        if name == "lookup_order":
            return fn(self.data["orders"], args["order_id"])
        elif name in ("lookup_items", "lookup_sellers", "sum_item_totals"):
            return fn(self.data["items"], args["order_id"])
        return {"error": f"Unhandled tool: {name}"}