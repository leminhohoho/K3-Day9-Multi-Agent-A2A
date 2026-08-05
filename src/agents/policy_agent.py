class PolicyAgent:
    """
    Policy decision engine applying EC_POLICY_V1 business rules.

    The business rules are a fully deterministic priority-ordered table, so
    they are applied in code (not via an LLM) to guarantee correctness.
    It consumes the merged Order/Payment/Delivery findings and emits the
    decision fields consumed by the coordinator.
    """

    def __init__(self):
        self.name = "PolicyAgent"

    def run(self, input_data: dict, trace_callback=None) -> dict:
        findings = input_data["findings"]
        if trace_callback:
            trace_callback(self.name, "apply_rules", {"policy": input_data.get("policy_version")})
        result = self._apply_rules(findings)
        if trace_callback:
            trace_callback(self.name, "complete", {"primary_issue": result.get("primary_issue")})
        return result

    def _apply_rules(self, f: dict) -> dict:
        order_status = f.get("order_status", "")
        payment_total = round(float(f.get("payment_total_brl", 0.0)), 2)
        freight_total = round(float(f.get("freight_total_brl", 0.0)), 2)
        item_total = round(float(f.get("item_total_brl", 0.0)), 2)
        delivered_late = bool(f.get("delivered_late", False))
        carrier_received_late = bool(f.get("carrier_received_late", False))
        reconciled = bool(f.get("reconciled", False))
        payment_rows = f.get("payment_rows", []) or []
        seller_ids = f.get("seller_ids", []) or []

        fin = {
            "currency": "BRL",
            "item_total_brl": item_total,
            "freight_total_brl": freight_total,
            "payment_total_brl": payment_total,
            "recommended_refund_brl": 0.0,
        }

        # Rule 1: canceled_order_paid
        if order_status == "canceled" and payment_total > 0:
            fin["recommended_refund_brl"] = payment_total
            return {
                "primary_issue": "canceled_order_paid",
                "case_status": "action_required",
                "confidence": 0.95,
                "ranked_causes": [{"cause_code": "ORDER_CANCELED_AFTER_PAYMENT", "rank": 1}],
                "responsible_parties": [{"party_type": "platform", "party_id": "OLIST_PLATFORM"}],
                "financial_resolution": fin,
                "resolution_actions": ["issue_full_refund"],
            }

        # Rule 2: unavailable_order_paid
        if order_status == "unavailable" and payment_total > 0:
            fin["recommended_refund_brl"] = payment_total
            return {
                "primary_issue": "unavailable_order_paid",
                "case_status": "action_required",
                "confidence": 0.95,
                "ranked_causes": [{"cause_code": "ORDER_UNAVAILABLE_AFTER_PAYMENT", "rank": 1}],
                "responsible_parties": [{"party_type": "platform", "party_id": "OLIST_PLATFORM"}],
                "financial_resolution": fin,
                "resolution_actions": ["issue_full_refund"],
            }

        # Late-delivery rules require a delivered order
        if delivered_late:
            # Rule 3: late_delivery_seller
            if carrier_received_late:
                party_id = seller_ids[0] if seller_ids else "UNKNOWN_SELLER"
                fin["recommended_refund_brl"] = freight_total
                return {
                    "primary_issue": "late_delivery_seller",
                    "case_status": "action_required",
                    "confidence": 0.95,
                    "ranked_causes": [{"cause_code": "SELLER_HANDOFF_AFTER_LIMIT", "rank": 1}],
                    "responsible_parties": [{"party_type": "seller", "party_id": party_id}],
                    "financial_resolution": fin,
                    "resolution_actions": ["refund_freight"],
                }
            # Rule 4: late_delivery_logistics
            fin["recommended_refund_brl"] = freight_total
            return {
                "primary_issue": "late_delivery_logistics",
                "case_status": "action_required",
                "confidence": 0.95,
                "ranked_causes": [{"cause_code": "CARRIER_DELIVERED_AFTER_ESTIMATE", "rank": 1}],
                "responsible_parties": [{"party_type": "logistics_provider", "party_id": "LOGISTICS_PROVIDER"}],
                "financial_resolution": fin,
                "resolution_actions": ["refund_freight"],
            }

        # Not late -> check payment reconciliation
        # Rule 5: valid_split_payment
        if len(payment_rows) >= 2 and reconciled:
            return {
                "primary_issue": "valid_split_payment",
                "case_status": "no_action",
                "confidence": 0.95,
                "ranked_causes": [{"cause_code": "MULTIPLE_PAYMENTS_RECONCILED", "rank": 1}],
                "responsible_parties": [],
                "financial_resolution": fin,
                "resolution_actions": ["explain_valid_split_payment"],
            }

        # Rule 6: unsupported_late_claim (delivered on time, payment reconciled)
        if reconciled:
            return {
                "primary_issue": "unsupported_late_claim",
                "case_status": "no_action",
                "confidence": 0.95,
                "ranked_causes": [{"cause_code": "DELIVERY_WITHIN_ESTIMATE", "rank": 1}],
                "responsible_parties": [],
                "financial_resolution": fin,
                "resolution_actions": ["reject_late_refund"],
            }

        # Fallback: delivered on time but payment does not reconcile
        return {
            "primary_issue": "unsupported_late_claim",
            "case_status": "no_action",
            "confidence": 0.90,
            "ranked_causes": [{"cause_code": "DELIVERY_WITHIN_ESTIMATE", "rank": 1}],
            "responsible_parties": [],
            "financial_resolution": fin,
            "resolution_actions": ["reject_late_refund"],
        }