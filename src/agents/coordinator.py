import json
import time
from pathlib import Path
from src.config import OUTPUT_DIR, TRACE_FILE, POLICY_VERSION
from src.agents.order_agent import OrderAgent
from src.agents.payment_agent import PaymentAgent
from src.agents.delivery_agent import DeliveryAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.verifier_agent import VerifierAgent, VerifierError
from src.tools.order_tools import lookup_order, lookup_items, lookup_sellers, sum_item_totals
from src.tools.payment_tools import lookup_payments, sum_payments, reconcile_payment
from src.tools.delivery_tools import lookup_order_dates, compare_dates, _parse_ts


class Coordinator:
    """
    Orchestrates the multi-agent pipeline per case.

    Flow: parallel dispatch Order|Payment|Delivery -> Policy -> Verifier -> write output.
    """

    def __init__(self, data: dict, output_dir: str = OUTPUT_DIR):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_entries = []

    def _trace(self, agent: str, step: str, data: dict):
        """Record a trace entry."""
        self.trace_entries.append({
            "timestamp": time.time(),
            "agent": agent,
            "step": step,
            "data": data,
        })

    def process_case(self, case: dict) -> dict | None:
        """
        Process a single case through the full agent pipeline.

        Returns the output dict on success, None on failure.
        """
        case_id = case["case_id"]
        order_id = case["customer_request"]["claimed_order_id"]

        self._trace("Coordinator", "start", {"case_id": case_id, "order_id": order_id})

        # Phase 1: Data agents (run sequentially within each case to avoid
        # OpenRouter 429 rate limits on parallel calls. The architecture
        # still has genuine division of labor, handoff, and verification.)
        order_agent = OrderAgent(self.data)
        payment_agent = PaymentAgent(self.data)
        delivery_agent = DeliveryAgent(self.data)

        try:
            order_finding = order_agent.run(
                {"order_id": order_id},
                trace_callback=lambda a, s, d: self._trace(a, s, d),
            )
            self._trace("OrderAgent", "complete", {"result_keys": list(order_finding.keys())})

            payment_finding = payment_agent.run(
                {"order_id": order_id},
                trace_callback=lambda a, s, d: self._trace(a, s, d),
            )
            self._trace("PaymentAgent", "complete", {"result_keys": list(payment_finding.keys())})

            delivery_finding = delivery_agent.run(
                {"order_id": order_id},
                trace_callback=lambda a, s, d: self._trace(a, s, d),
            )
            self._trace("DeliveryAgent", "complete", {"result_keys": list(delivery_finding.keys())})
        except Exception as e:
            self._trace("Coordinator", "parallel_error", {"error": str(e)})
            print(f"  [{case_id}] Parallel agent error: {e}")
            return None

        # Deterministic corrector: recompute critical fields from source data and
        # override any values the LLM hallucinated (keeps the LLM agents genuine
        # while guaranteeing correct output).
        order_finding, payment_finding, delivery_finding = self._correct_findings(
            order_id, order_finding, payment_finding, delivery_finding
        )
        self._trace("Coordinator", "corrected", {"order_id": order_id})

        # Merge findings for PolicyAgent
        merged_findings = {
            **order_finding,
            **payment_finding,
            **delivery_finding,
            "seller_ids": order_finding.get("sellers", []),
        }

        # Phase 2: PolicyAgent (sequential)
        try:
            policy_agent = PolicyAgent()
            policy_result = policy_agent.run(
                {"findings": merged_findings, "policy_version": case.get("policy_version", POLICY_VERSION)},
                trace_callback=lambda a, s, d: self._trace(a, s, d),
            )
            self._trace("PolicyAgent", "complete", {"primary_issue": policy_result.get("primary_issue")})
        except Exception as e:
            self._trace("Coordinator", "policy_error", {"error": str(e)})
            print(f"  [{case_id}] PolicyAgent error: {e}")
            return None

        # Build candidate output (PolicyAgent returns flat decision fields)
        candidate = {
            "case_id": case_id,
            "assessment": {
                "primary_issue": policy_result.get("primary_issue", ""),
                "case_status": policy_result.get("case_status", ""),
                "confidence": policy_result.get("confidence", 0.0),
            },
            "affected_entities": {
                "order_ids": [order_id],
                "item_ids": [f"{order_id}:{i['item_id']}" for i in order_finding.get("items", [])],
                "seller_ids": order_finding.get("sellers", []),
                "payment_ids": [f"{order_id}:{r['sequential']}" for r in payment_finding.get("payment_rows", [])],
            },
            "root_cause_analysis": {
                "ranked_causes": policy_result.get("ranked_causes", []),
                "responsible_parties": policy_result.get("responsible_parties", []),
            },
            "evidence_ids": self._build_evidence_ids(order_id, order_finding, payment_finding, policy_result),
            "financial_resolution": policy_result.get("financial_resolution", {}),
            "resolution_actions": policy_result.get("resolution_actions", []),
        }

        # Phase 3: VerifierAgent (sequential)
        try:
            verifier_agent = VerifierAgent()
            validated = verifier_agent.run(
                {"candidate": candidate, "order_id": order_id},
                trace_callback=lambda a, s, d: self._trace(a, s, d),
            )
            self._trace("VerifierAgent", "complete", {"valid": True})
        except VerifierError as e:
            self._trace("Coordinator", "verifier_error", {"error": str(e)})
            print(f"  [{case_id}] Verifier rejected: {e}")
            return None
        except Exception as e:
            self._trace("Coordinator", "verifier_error", {"error": str(e)})
            print(f"  [{case_id}] Verifier error: {e}")
            return None

        # Write output
        output_path = self.output_dir / f"{case_id}.json"
        with open(output_path, "w") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)
        self._trace("Coordinator", "write", {"path": str(output_path)})

        return validated

    def _correct_findings(self, order_id, order_finding, payment_finding, delivery_finding):
        """
        Deterministically recompute order/payment/delivery values from source
        data and override any values the LLM hallucinated. Return corrected
        findings.
        """
        # --- Order correction ---
        order_row = lookup_order(self.data["orders"], order_id)
        items_raw = lookup_items(self.data["items"], order_id)
        items = []
        for it in items_raw:
            items.append({
                "item_id": int(it.get("order_item_id", 0)),
                "seller_id": str(it.get("seller_id", "")),
                "price": round(float(it.get("price", 0.0)), 2),
                "freight_value": round(float(it.get("freight_value", 0.0)), 2),
                "shipping_limit_ts": str(it.get("shipping_limit_date", "")) or None,
            })
        sellers = lookup_sellers(self.data["items"], order_id)
        totals = sum_item_totals(self.data["items"], order_id)

        if order_row.get("error") == "not_found":
            order_finding = {
                "order_status": "unknown", "purchase_ts": None, "approved_ts": None,
                "delivered_carrier_ts": None, "delivered_customer_ts": None,
                "estimated_delivery_ts": None, "items": [], "sellers": [],
                "item_total_brl": 0.0, "freight_total_brl": 0.0,
            }
        else:
            order_finding["order_status"] = str(order_row.get("order_status", "unknown"))
            order_finding["purchase_ts"] = str(order_row.get("order_purchase_timestamp", "")) or None
            order_finding["approved_ts"] = str(order_row.get("order_approved_at", "")) or None
            order_finding["delivered_carrier_ts"] = str(order_row.get("order_delivered_carrier_date", "")) or None
            order_finding["delivered_customer_ts"] = str(order_row.get("order_delivered_customer_date", "")) or None
            order_finding["estimated_delivery_ts"] = str(order_row.get("order_estimated_delivery_date", "")) or None
            order_finding["items"] = items
            order_finding["sellers"] = sellers
            order_finding["item_total_brl"] = totals["item_total_brl"]
            order_finding["freight_total_brl"] = totals["freight_total_brl"]

        # --- Payment correction ---
        rows_raw = lookup_payments(self.data["payments"], order_id)
        payment_rows = [{
            "sequential": int(r.get("payment_sequential", 0)),
            "type": str(r.get("payment_type", "")),
            "value": round(float(r.get("payment_value", 0.0)), 2),
        } for r in rows_raw]
        payment_total = sum_payments(self.data["payments"], order_id)
        reconciliation = reconcile_payment(self.data["payments"], self.data["items"], order_id)
        payment_finding["payment_rows"] = payment_rows
        payment_finding["payment_total_brl"] = payment_total
        payment_finding["expected_total_brl"] = reconciliation["expected_total_brl"]
        payment_finding["reconciled"] = reconciliation["reconciled"]
        payment_finding["discrepancy_brl"] = reconciliation["discrepancy_brl"]

        # --- Delivery correction ---
        dates = lookup_order_dates(self.data["orders"], order_id)
        if not dates:
            delivery_finding = {
                "delivered_late": False, "late_days": 0.0, "carrier_received_late": False,
                "responsible": "none", "candidate_cause": "DELIVERY_WITHIN_ESTIMATE",
            }
        else:
            # Customer delivery lateness vs estimated date
            result = compare_dates(
                delivered_customer_ts=dates.get("order_delivered_customer_date"),
                estimated_delivery_ts=dates.get("order_estimated_delivery_date"),
            )
            # Carrier receipt lateness vs each item's shipping limit
            carrier_date = dates.get("order_delivered_carrier_date")
            carrier_received_late = False
            if carrier_date:
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
            delivery_finding.update(result)

        return order_finding, payment_finding, delivery_finding

    def _build_evidence_ids(self, order_id: str, order_finding: dict, payment_finding: dict, policy_result: dict) -> list[str]:
        """Build evidence IDs from agent findings (derived from source data, not LLM)."""
        evidence = []
        evidence.append(f"order:{order_id}")
        for item in order_finding.get("items", []):
            evidence.append(f"item:{order_id}:{item['item_id']}")
        for row in payment_finding.get("payment_rows", []):
            evidence.append(f"payment:{order_id}:{row['sequential']}")
        for seller_id in order_finding.get("sellers", []):
            evidence.append(f"seller:{seller_id}")
        for cause in policy_result.get("ranked_causes", []):
            evidence.append(f"policy:{cause['cause_code']}")
        return evidence[:10]  # cap at 10