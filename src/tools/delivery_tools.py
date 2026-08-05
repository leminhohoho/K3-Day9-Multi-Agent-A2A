import pandas as pd
from datetime import datetime


def _parse_ts(ts: str) -> datetime | None:
    """Parse a timestamp string, handling various formats."""
    if pd.isna(ts) or not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(ts).strip(), fmt)
        except ValueError:
            continue
    return None


def compare_dates(
    delivered_customer_ts: str | None = None,
    estimated_delivery_ts: str | None = None,
    delivered_carrier_ts: str | None = None,
    shipping_limit_ts: str | None = None,
) -> dict:
    """Compare actual delivery dates against expected dates."""
    result = {
        "delivered_late": False,
        "late_days": 0.0,
        "carrier_received_late": False,
        "responsible": "none",
        "candidate_cause": "DELIVERY_WITHIN_ESTIMATE",
    }

    delivered = _parse_ts(delivered_customer_ts) if delivered_customer_ts else None
    estimated = _parse_ts(estimated_delivery_ts) if estimated_delivery_ts else None

    # Check if delivery is late vs estimated
    if delivered and estimated and delivered > estimated:
        result["delivered_late"] = True
        result["late_days"] = round((delivered - estimated).total_seconds() / 86400, 1)
        result["candidate_cause"] = "CARRIER_DELIVERED_AFTER_ESTIMATE"

    # Check if carrier received late (seller handoff after limit)
    carrier_ts = _parse_ts(delivered_carrier_ts) if delivered_carrier_ts else None
    limit_ts = _parse_ts(shipping_limit_ts) if shipping_limit_ts else None
    if carrier_ts and limit_ts and carrier_ts > limit_ts:
        result["carrier_received_late"] = True
        result["responsible"] = "seller"
        result["candidate_cause"] = "SELLER_HANDOFF_AFTER_LIMIT"

    return result


def lookup_order_dates(df_orders: pd.DataFrame, order_id: str) -> dict:
    """Get delivery-related timestamps for an order."""
    row = df_orders[df_orders["order_id"] == order_id]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "order_delivered_carrier_date": r.get("order_delivered_carrier_date"),
        "order_delivered_customer_date": r.get("order_delivered_customer_date"),
        "order_estimated_delivery_date": r.get("order_estimated_delivery_date"),
    }


DELIVERY_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order_dates",
            "description": "Get delivery timestamps for an order (carrier, customer, estimated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The Olist order ID"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_dates",
            "description": "Compare actual delivery dates against expected dates. Determine if late, by how many days, and who is responsible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delivered_customer_ts": {"type": "string", "description": "Actual delivery timestamp to customer"},
                    "estimated_delivery_ts": {"type": "string", "description": "Estimated delivery date"},
                    "delivered_carrier_ts": {"type": "string", "description": "Timestamp when carrier received the package"},
                    "shipping_limit_ts": {"type": "string", "description": "Seller's shipping limit date"},
                },
                "required": [],
            },
        },
    },
]

DELIVERY_TOOL_MAP = {
    "lookup_order_dates": lookup_order_dates,
    "compare_dates": compare_dates,
}