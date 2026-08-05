import pandas as pd


def lookup_order(df_orders: pd.DataFrame, order_id: str) -> dict:
    """Look up an order's metadata by order_id."""
    row = df_orders[df_orders["order_id"] == order_id]
    if row.empty:
        return {"order_id": order_id, "order_status": "unknown", "error": "not_found"}
    return row.iloc[0].to_dict()


def lookup_items(df_items: pd.DataFrame, order_id: str) -> list[dict]:
    """Look up all items for an order."""
    rows = df_items[df_items["order_id"] == order_id]
    if rows.empty:
        return []
    return rows.to_dict("records")


def lookup_sellers(df_items: pd.DataFrame, order_id: str) -> list[str]:
    """Get unique seller IDs for an order."""
    rows = df_items[df_items["order_id"] == order_id]
    if rows.empty:
        return []
    return rows["seller_id"].unique().tolist()


def sum_item_totals(df_items: pd.DataFrame, order_id: str) -> dict:
    """Sum item prices and freight values for an order."""
    rows = df_items[df_items["order_id"] == order_id]
    if rows.empty:
        return {"item_total_brl": 0.0, "freight_total_brl": 0.0}
    return {
        "item_total_brl": round(float(rows["price"].sum()), 2),
        "freight_total_brl": round(float(rows["freight_value"].sum()), 2),
    }


# OpenAI-compatible tool definitions
ORDER_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order's metadata (status, timestamps) by order_id.",
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
            "name": "lookup_items",
            "description": "Look up all items (product_id, seller_id, price, freight, shipping_limit) for an order.",
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
            "name": "lookup_sellers",
            "description": "Get unique seller IDs involved in an order.",
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
            "name": "sum_item_totals",
            "description": "Compute sum of item prices and freight values for an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The Olist order ID"},
                },
                "required": ["order_id"],
            },
        },
    },
]

ORDER_TOOL_MAP = {
    "lookup_order": lookup_order,
    "lookup_items": lookup_items,
    "lookup_sellers": lookup_sellers,
    "sum_item_totals": sum_item_totals,
}