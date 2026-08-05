from src.agent_base import BaseAgent
from src.tools.delivery_tools import lookup_order_dates, compare_dates, _parse_ts
from src.tools.order_tools import lookup_items


class DeliveryAgent(BaseAgent):
    """
    Delivery Agent — deterministic date comparison. Delivery lateness is a
    purely deterministic calculation (compare timestamps), so no LLM is
    needed (and the LLM was observed to hallucinate dates). The agent
    still runs as a scoped, named agent with division of labor.
    """

    def __init__(self, data: dict):
        self.data = data
        super().__init__(
            name="DeliveryAgent",
            system_prompt="",
            tools=None,
        )

    def run(self, input_data: dict, trace_callback=None) -> dict:
        order_id = input_data["order_id"]
        if trace_callback:
            trace_callback(self.name, "query", {"order_id": order_id})

        dates = lookup_order_dates(self.data["orders"], order_id)
        if not dates:
            result = {
                "delivered_late": False, "late_days": 0.0, "carrier_received_late": False,
                "responsible": "none", "candidate_cause": "DELIVERY_WITHIN_ESTIMATE",
            }
            if trace_callback:
                trace_callback(self.name, "complete", result)
            return result

        delivered_customer = dates.get("order_delivered_customer_date")
        estimated = dates.get("order_estimated_delivery_date")
        carrier_date = dates.get("order_delivered_carrier_date")

        # Customer delivery vs estimate
        result = compare_dates(
            delivered_customer_ts=delivered_customer,
            estimated_delivery_ts=estimated,
        )

        # Carrier receipt vs each item's shipping limit
        carrier_received_late = False
        if carrier_date:
            items_raw = lookup_items(self.data["items"], order_id)
            for it in items_raw:
                limit_ts = it.get("shipping_limit_date")
                if limit_ts:
                    c = _parse_ts(carrier_date)
                    l = _parse_ts(limit_ts)
                    if c and l and c > l:
                        carrier_received_late = True
                        break

        result["carrier_received_late"] = carrier_received_late
        if carrier_received_late:
            result["responsible"] = "seller"
            result["candidate_cause"] = "SELLER_HANDOFF_AFTER_LIMIT"

        if trace_callback:
            trace_callback(self.name, "complete", result)
        return result