from src.agent_base import BaseAgent
from src.tools.delivery_tools import lookup_order_dates, compare_dates, _parse_ts
from src.tools.order_tools import lookup_items


class DeliveryAgent(BaseAgent):
    """
    Delivery Data Agent.

    Compares delivery timestamps against estimated dates and shipping limits
    deterministically. The compare_dates tool handles the actual comparison;
    lookup_items is also called to check per-item shipping limits for
    multi-order cases.
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

        # Get delivery timestamps from the order record
        dates = lookup_order_dates(self.data["orders"], order_id)
        if not dates:
            return {
                "delivered_late": False,
                "late_days": 0.0,
                "carrier_received_late": False,
                "responsible": "none",
                "candidate_cause": "DELIVERY_WITHIN_ESTIMATE",
            }

        # Check lateness vs estimated delivery date
        delivered_customer = dates.get("order_delivered_customer_date")
        estimated = dates.get("order_estimated_delivery_date")
        carrier_date = dates.get("order_delivered_carrier_date")

        # Check carrier receipt vs shipping limit for each item's seller
        carrier_received_late = False
        if carrier_date:
            items_raw = lookup_items(self.data["items"], order_id)
            for it in items_raw:
                shipping_limit = it.get("shipping_limit_date")
                if shipping_limit:
                    carrier_ts = _parse_ts(carrier_date)
                    limit_ts = _parse_ts(shipping_limit)
                    if carrier_ts and limit_ts and carrier_ts > limit_ts:
                        carrier_received_late = True
                        break

        # Use compare_dates for the customer delivery comparison
        result = compare_dates(
            delivered_customer_ts=delivered_customer,
            estimated_delivery_ts=estimated,
            delivered_carrier_ts=carrier_date,
            shipping_limit_ts=None,  # We already checked seller lateness above
        )

        # Override carrier_received_late with our per-item check above
        result["carrier_received_late"] = carrier_received_late
        if carrier_received_late:
            result["responsible"] = "seller"
            result["candidate_cause"] = "SELLER_HANDOFF_AFTER_LIMIT"

        if trace_callback:
            trace_callback(self.name, "complete", result)

        return result