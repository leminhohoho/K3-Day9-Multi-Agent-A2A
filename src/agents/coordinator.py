import json
import asyncio
import time
from pathlib import Path
from src.config import OUTPUT_DIR, TRACE_FILE, POLICY_VERSION
from src.agents.order_agent import OrderAgent
from src.agents.payment_agent import PaymentAgent
from src.agents.delivery_agent import DeliveryAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.verifier_agent import VerifierAgent, VerifierError


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

        # Phase 1: Parallel data agents
        order_agent = OrderAgent(self.data)
        payment_agent = PaymentAgent(self.data)
        delivery_agent = DeliveryAgent(self.data)

        # Run data agents in parallel using asyncio
        async def run_parallel():
            async def run_agent(agent, name):
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: agent.run(
                        {"order_id": order_id},
                        trace_callback=lambda a, s, d: self._trace(a, s, d),
                    ),
                )
                self._trace(name, "complete", {"result_keys": list(result.keys())})
                return result

            results = await asyncio.gather(
                run_agent(order_agent, "OrderAgent"),
                run_agent(payment_agent, "PaymentAgent"),
                run_agent(delivery_agent, "DeliveryAgent"),
            )
            return results

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            order_finding, payment_finding, delivery_finding = loop.run_until_complete(run_parallel())
            loop.close()
        except Exception as e:
            self._trace("Coordinator", "parallel_error", {"error": str(e)})
            print(f"  [{case_id}] Parallel agent error: {e}")
            return None

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