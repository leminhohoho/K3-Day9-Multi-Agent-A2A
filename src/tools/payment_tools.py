import pandas as pd


def lookup_payments(df_payments: pd.DataFrame, order_id: str) -> list[dict]:
    """Look up all payment rows for an order."""
    rows = df_payments[df_payments["order_id"] == order_id]
    if rows.empty:
        return []
    return rows.to_dict("records")


def sum_payments(df_payments: pd.DataFrame, order_id: str) -> float:
    """Sum all payment values for an order."""
    rows = df_payments[df_payments["order_id"] == order_id]
    if rows.empty:
        return 0.0
    return round(float(rows["payment_value"].sum()), 2)


def reconcile_payment(df_payments: pd.DataFrame, df_items: pd.DataFrame, order_id: str) -> dict:
    """Reconcile payment total vs item + freight total."""
    pay_rows = df_payments[df_payments["order_id"] == order_id]
    pay_total = round(float(pay_rows["payment_value"].sum()), 2) if not pay_rows.empty else 0.0

    item_rows = df_items[df_items["order_id"] == order_id]
    item_total = round(float(item_rows["price"].sum()), 2) if not item_rows.empty else 0.0
    freight_total = round(float(item_rows["freight_value"].sum()), 2) if not item_rows.empty else 0.0
    expected = round(item_total + freight_total, 2)

    discrepancy = round(pay_total - expected, 2)
    reconciled = abs(discrepancy) <= 0.10

    return {
        "payment_total_brl": pay_total,
        "expected_total_brl": expected,
        "reconciled": reconciled,
        "discrepancy_brl": discrepancy,
    }


PAYMENT_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_payments",
            "description": "Look up all payment rows for an order (sequential, type, installments, value).",
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
            "name": "sum_payments",
            "description": "Sum all payment values for an order.",
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
            "name": "reconcile_payment",
            "description": "Reconcile payment total against item + freight totals. Returns whether they match within 0.10 BRL.",
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

PAYMENT_TOOL_MAP = {
    "lookup_payments": lookup_payments,
    "sum_payments": sum_payments,
    "reconcile_payment": reconcile_payment,
}