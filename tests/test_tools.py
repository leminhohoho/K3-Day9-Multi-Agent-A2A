import pytest
import pandas as pd
from src.tools.order_tools import lookup_order, lookup_items, lookup_sellers, sum_item_totals
from src.tools.order_tools import ORDER_TOOL_DEFS, ORDER_TOOL_MAP
from src.tools.payment_tools import lookup_payments, sum_payments, reconcile_payment
from src.tools.payment_tools import PAYMENT_TOOL_DEFS, PAYMENT_TOOL_MAP
from src.tools.delivery_tools import compare_dates, lookup_order_dates
from src.tools.delivery_tools import DELIVERY_TOOL_DEFS, DELIVERY_TOOL_MAP
from src.loader import load_all_data


@pytest.fixture(scope="module")
def data():
    return load_all_data()


def test_lookup_order_finds_by_id(data):
    order_id = "e2a03ccf5ea816036608b2d8c3ab8e60"
    result = lookup_order(data["orders"], order_id)
    assert result["order_id"] == order_id
    assert "order_status" in result


def test_lookup_items_returns_list(data):
    order_id = "e2a03ccf5ea816036608b2d8c3ab8e60"
    items = lookup_items(data["items"], order_id)
    assert isinstance(items, list)


def test_lookup_sellers_returns_list(data):
    order_id = "e2a03ccf5ea816036608b2d8c3ab8e60"
    sellers = lookup_sellers(data["items"], order_id)
    assert isinstance(sellers, list)


def test_sum_item_totals_computes(data):
    order_id = "e2a03ccf5ea816036608b2d8c3ab8e60"
    result = sum_item_totals(data["items"], order_id)
    assert isinstance(result["item_total_brl"], float)
    assert isinstance(result["freight_total_brl"], float)


def test_lookup_payments_returns_rows(data):
    order_id = "e2a03ccf5ea816036608b2d8c3ab8e60"
    payments = lookup_payments(data["payments"], order_id)
    assert isinstance(payments, list)


def test_sum_payments_computes_total(data):
    order_id = "e2a03ccf5ea816036608b2d8c3ab8e60"
    total = sum_payments(data["payments"], order_id)
    assert isinstance(total, float)


def test_reconcile_payment_returns_dict(data):
    order_id = "e2a03ccf5ea816036608b2d8c3ab8e60"
    result = reconcile_payment(data["payments"], data["items"], order_id)
    assert isinstance(result["reconciled"], bool)
    assert isinstance(result["discrepancy_brl"], float)


def test_compare_dates_late():
    result = compare_dates(
        delivered_customer_ts="2018-10-20 10:00:00",
        estimated_delivery_ts="2018-10-15 00:00:00",
    )
    assert result["delivered_late"] is True
    assert result["late_days"] == pytest.approx(5.0, abs=0.6)


def test_compare_dates_on_time():
    result = compare_dates(
        delivered_customer_ts="2018-10-10 10:00:00",
        estimated_delivery_ts="2018-10-15 00:00:00",
    )
    assert result["delivered_late"] is False


def test_lookup_order_dates(data):
    order_id = "e2a03ccf5ea816036608b2d8c3ab8e60"
    result = lookup_order_dates(data["orders"], order_id)
    assert "order_delivered_carrier_date" in result


def test_tool_defs_have_correct_structure():
    for tool_def in ORDER_TOOL_DEFS + PAYMENT_TOOL_DEFS + DELIVERY_TOOL_DEFS:
        assert "function" in tool_def
        assert "name" in tool_def["function"]
        assert "parameters" in tool_def["function"]


def test_tool_maps_cover_all_defs():
    for td in ORDER_TOOL_DEFS:
        assert td["function"]["name"] in ORDER_TOOL_MAP
    for td in PAYMENT_TOOL_DEFS:
        assert td["function"]["name"] in PAYMENT_TOOL_MAP
    for td in DELIVERY_TOOL_DEFS:
        assert td["function"]["name"] in DELIVERY_TOOL_MAP