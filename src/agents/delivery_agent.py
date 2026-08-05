from src.agent_base import BaseAgent
from src.tools.delivery_tools import DELIVERY_TOOL_DEFS, DELIVERY_TOOL_MAP

DELIVERY_SYSTEM_PROMPT = """You are a Delivery Data Agent for Olist e-commerce dispute resolution.

Your job: given an order_id, investigate delivery timeliness and determine who is responsible for any delays.

TOOLS:
- lookup_order_dates: get delivery timestamps (carrier date, customer date, estimated date)
- compare_dates: compare actual vs expected delivery dates. Returns lateness, days late, and responsible party.

PROCEDURE:
1. Call lookup_order_dates to get the delivery timestamps for the order.
2. Call compare_dates with the actual timestamps to determine if delivery was late and who is responsible.

For multi-item orders: check each item's shipping_limit_date against order_delivered_carrier_date.
A seller is late if order_delivered_carrier_date > shipping_limit_date of that seller's item.

OUTPUT: Return a JSON object with these exact fields:
{
  "delivered_late": false,
  "late_days": 0.0,
  "carrier_received_late": false,
  "responsible": "none",
  "candidate_cause": "DELIVERY_WITHIN_ESTIMATE"
}

responsible values: "none", "seller", "logistics_provider"
candidate_cause: "SELLER_HANDOFF_AFTER_LIMIT", "CARRIER_DELIVERED_AFTER_ESTIMATE", "DELIVERY_WITHIN_ESTIMATE"
"""


class DeliveryAgent(BaseAgent):
    def __init__(self, data: dict):
        self.data = data
        super().__init__(
            name="DeliveryAgent",
            system_prompt=DELIVERY_SYSTEM_PROMPT,
            tools=DELIVERY_TOOL_DEFS,
        )

    def _execute_tool(self, name: str, args: dict) -> dict:
        fn = DELIVERY_TOOL_MAP.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}
        if name == "lookup_order_dates":
            return fn(self.data["orders"], args["order_id"])
        elif name == "compare_dates":
            return fn(**args)
        return {"error": f"Unhandled tool: {name}"}