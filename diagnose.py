"""Diagnostic: recompute ideal output per case from source data, diff vs current output."""
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd
from src.loader import load_all_data, load_input_case


def parse_ts(ts):
    if pd.isna(ts) or not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(ts).strip(), fmt)
        except ValueError:
            continue
    return None


def ideal_output(data, case):
    oid = case["customer_request"]["claimed_order_id"]
    case_id = case["case_id"]
    o = data["orders"][data["orders"]["order_id"] == oid]
    items = data["items"][data["items"]["order_id"] == oid]
    pays = data["payments"][data["payments"]["order_id"] == oid]

    if o.empty:
        return {"problem": "ORDER NOT FOUND", "order_id": oid}

    r = o.iloc[0]
    status = r["order_status"]
    item_total = round(float(items["price"].sum()), 2) if not items.empty else 0.0
    freight_total = round(float(items["freight_value"].sum()), 2) if not items.empty else 0.0
    pay_total = round(float(pays["payment_value"].sum()), 2) if not pays.empty else 0.0
    expected_total = round(item_total + freight_total, 2)
    reconciled = abs(pay_total - expected_total) <= 0.10
    n_payrows = len(pays)

    delivered_cust = r.get("order_delivered_customer_date")
    estimated = r.get("order_estimated_delivery_date")
    carrier = r.get("order_delivered_carrier_date")
    dc = parse_ts(delivered_cust)
    est = parse_ts(estimated)
    delivered_late = bool(dc and est and dc > est)
    carrier_late = False
    if carrier:
        ca = parse_ts(carrier)
        for _, it in items.iterrows():
            lim = parse_ts(it.get("shipping_limit_date"))
            if ca and lim and ca > lim:
                carrier_late = True
                break

    sellers = items["seller_id"].unique().tolist() if not items.empty else []
    item_ids = [f"{oid}:{int(it)}" for it in items["order_item_id"]] if not items.empty else []
    pay_ids = [f"{oid}:{int(p)}" for p in pays["payment_sequential"]] if not pays.empty else []

    # Rules priority
    if status == "canceled" and pay_total > 0:
        issue = "canceled_order_paid"; case_status = "action_required"
        cause = "ORDER_CANCELED_AFTER_PAYMENT"; parties = [{"party_type": "platform", "party_id": "OLIST_PLATFORM"}]
        refund = pay_total; actions = ["issue_full_refund"]
    elif status == "unavailable" and pay_total > 0:
        issue = "unavailable_order_paid"; case_status = "action_required"
        cause = "ORDER_UNAVAILABLE_AFTER_PAYMENT"; parties = [{"party_type": "platform", "party_id": "OLIST_PLATFORM"}]
        refund = pay_total; actions = ["issue_full_refund"]
    elif delivered_late and carrier_late:
        seller_id = sellers[0] if sellers else "MISSING"
        issue = "late_delivery_seller"; case_status = "action_required"
        cause = "SELLER_HANDOFF_AFTER_LIMIT"; parties = [{"party_type": "seller", "party_id": seller_id}]
        refund = freight_total; actions = ["refund_freight"]
    elif delivered_late:
        issue = "late_delivery_logistics"; case_status = "action_required"
        cause = "CARRIER_DELIVERED_AFTER_ESTIMATE"; parties = [{"party_type": "logistics_provider", "party_id": "LOGISTICS_PROVIDER"}]
        refund = freight_total; actions = ["refund_freight"]
    elif n_payrows >= 2 and reconciled:
        issue = "valid_split_payment"; case_status = "no_action"
        cause = "MULTIPLE_PAYMENTS_RECONCILED"; parties = []
        refund = 0.0; actions = ["explain_valid_split_payment"]
    elif not delivered_late and reconciled:
        issue = "unsupported_late_claim"; case_status = "no_action"
        cause = "DELIVERY_WITHIN_ESTIMATE"; parties = []
        refund = 0.0; actions = ["reject_late_refund"]
    else:
        issue = "UNKNOWN"; case_status = "UNKNOWN"; cause = "?";
        parties = []; refund = 0.0; actions = ["?"]

    evidence = [f"order:{oid}"] + [f"item:{x}" for x in item_ids] + [f"payment:{x}" for x in pay_ids] + [f"seller:{s}" for s in sellers] + [f"policy:{cause}"]

    return {
        "case_id": case_id,
        "order_id": oid,
        "status": status,
        "primary_issue": issue,
        "case_status": case_status,
        "confidence": 0.95,
        "order_ids": [oid],
        "item_ids": item_ids,
        "seller_ids": sellers,
        "payment_ids": pay_ids,
        "ranked_causes": [{"cause_code": cause, "rank": 1}],
        "responsible_parties": parties,
        "evidence_ids": evidence,
        "item_total_brl": item_total,
        "freight_total_brl": freight_total,
        "payment_total_brl": pay_total,
        "recommended_refund_brl": round(refund, 2),
        "resolution_actions": actions,
        "delivered_late": delivered_late,
        "carrier_late": carrier_late,
        "n_payrows": n_payrows,
        "reconciled": reconciled,
    }


data = load_all_data()
issues = Counter()
diffs = []
for p in sorted(Path("input").glob("EC_*.json")):
    case = json.loads(p.read_text())
    ideal = ideal_output(data, case)
    out_path = Path("output") / f"{case['case_id']}.json"
    if not out_path.exists():
        diffs.append((case["case_id"], f"MISSING OUTPUT", ideal))
        continue
    out = json.loads(out_path.read_text())

    problems = []
    # primary issue
    if out["assessment"]["primary_issue"] != ideal["primary_issue"]:
        problems.append(f"ISSUE: got={out['assessment']['primary_issue']} ideal={ideal['primary_issue']}")
    # case_status
    if out["assessment"]["case_status"] != ideal["case_status"]:
        problems.append(f"STATUS: got={out['assessment']['case_status']} ideal={ideal['case_status']}")
    # entities
    ae = out["affected_entities"]
    if set(ae["order_ids"]) != set(ideal["order_ids"]):
        problems.append(f"ORDER_IDS: got={ae['order_ids']} ideal={ideal['order_ids']}")
    if set(ae["item_ids"]) != set(ideal["item_ids"]):
        problems.append(f"ITEM_IDS: got={ae['item_ids']} ideal={ideal['item_ids']}")
    if set(ae["seller_ids"]) != set(ideal["seller_ids"]):
        problems.append(f"SELLER_IDS: got={ae['seller_ids']} ideal={ideal['seller_ids']}")
    if set(ae["payment_ids"]) != set(ideal["payment_ids"]):
        problems.append(f"PAYMENT_IDS: got={ae['payment_ids']} ideal={ideal['payment_ids']}")
    # root causes
    rca = out["root_cause_analysis"]
    got_causes = [c["cause_code"] for c in rca["ranked_causes"]]
    if got_causes != [ideal["ranked_causes"][0]["cause_code"]]:
        problems.append(f"CAUSES: got={got_causes} ideal={[ideal['ranked_causes'][0]['cause_code']]}")
    got_parties = rca["responsible_parties"]
    if got_parties != ideal["responsible_parties"]:
        problems.append(f"PARTIES: got={got_parties} ideal={ideal['responsible_parties']}")
    # evidence
    if set(out["evidence_ids"]) != set(ideal["evidence_ids"]):
        problems.append(f"EVIDENCE: got={out['evidence_ids']} ideal={ideal['evidence_ids']}")
    # financial
    fin = out["financial_resolution"]
    if abs(fin["item_total_brl"] - ideal["item_total_brl"]) > 0.01:
        problems.append(f"ITEM_TOTAL: got={fin['item_total_brl']} ideal={ideal['item_total_brl']}")
    if abs(fin["freight_total_brl"] - ideal["freight_total_brl"]) > 0.01:
        problems.append(f"FREIGHT_TOTAL: got={fin['freight_total_brl']} ideal={ideal['freight_total_brl']}")
    if abs(fin["payment_total_brl"] - ideal["payment_total_brl"]) > 0.01:
        problems.append(f"PAY_TOTAL: got={fin['payment_total_brl']} ideal={ideal['payment_total_brl']}")
    if abs(fin["recommended_refund_brl"] - ideal["recommended_refund_brl"]) > 0.01:
        problems.append(f"REFUND: got={fin['recommended_refund_brl']} ideal={ideal['recommended_refund_brl']}")
    # actions
    if set(out["resolution_actions"]) != set(ideal["resolution_actions"]):
        problems.append(f"ACTIONS: got={out['resolution_actions']} ideal={ideal['resolution_actions']}")

    if problems:
        diffs.append((case["case_id"], problems, ideal))

print(f"=== Cases with diffs: {len(diffs)} ===")
for cid, probs, ideal in diffs[:80]:
    print(f"\n{cid} (status={ideal.get('status')}, issue={ideal.get('primary_issue')}, n_payrows={ideal.get('n_payrows')}, carrier_late={ideal.get('carrier_late')}, delivered_late={ideal.get('delivered_late')}):")
    for pr in probs:
        print(f"   {pr}")