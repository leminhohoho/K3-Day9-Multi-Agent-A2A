# Design Spec: Multi-Agent E-commerce Dispute Resolution

## Problem

Process 50 customer support cases against Olist data. Each case requires cross-referencing orders, sellers, payments, delivery timelines, and business rules — with evidence, responsible parties, and refund amounts — output as structured JSON.

## Constraints

- Each agent must use a model ≤10B parameters.
- No single prompt can do all the work (README penalty).
- Evidence IDs must be derivable from source CSVs.
- Output schema is fixed (README §6).

## Design Decisions

| Decision | Choice | Rationale |
| -------- | ------ | --------- |
| Pattern | Coordinator + Parallel fan-out + Sequential policy/verifier | Parallel data agents (no inter-dependency), sequential post-processing (Policy needs all data). |
| Runtime | OpenRouter (OpenAI-compatible API) | User choice. ≤10B, no local GPU required. |
| Agent execution | ReAct + system prompt + scoped tools | Each agent is a full model call, avoids "one prompt" penalty. |
| Model calls | One per agent per case | 50 cases × 6 agents = 300 calls per run. |
| Tool scoping | Per-agent scoped tools | Stronger isolation, each agent sees only its domain. |

## Pattern

**Coordinator + Parallel Fan-out (data agents) → Sequential Policy → Sequential Verifier.**

The data agents (`Order`, `Payment`, `Delivery`) have **no runtime dependency on each other**: each queries the source CSVs directly through its own scoped tools. This lets them run **truly in parallel**. `Policy` runs after all three finish (it needs all their findings). `Verifier` runs last as a validation gate.

The coordinator is the only agent that calls other agents. No agent calls another agent directly.

```
                 ┌──────────────────────────────┐
                 │         Coordinator           │
                 │  reads case, orchestrates,    │
                 │  merges, writes output        │
                 └──────────────┬───────────────┘
                                │ dispatch (parallel)
        ┌──────────────┬────────┴────────┬──────────────┐
        ▼              ▼                 ▼
  ┌───────────┐  ┌───────────┐    ┌───────────┐
  │ OrderAgent │  │PaymentAgent│   │DeliveryAgent│
  │  (scoped   │  │ (scoped    │   │ (scoped     │
  │  tools)    │  │  tools)    │   │  tools)     │
  └───────────┘  └───────────┘    └───────────┘
        └──────────────┬────────┴────────┘
                       ▼
              ┌──────────────┐
              │  PolicyAgent │  (sequential)
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ VerifierAgent│  (sequential)
              └──────┬───────┘
                     ▼
              coordinator writes output/
```

## Agent Roles & Responsibilities

| Agent | Responsibility | Runs |
| ----- | -------------- | ---- |
| **Coordinator** | Read input case, extract `order_id`, hand shared dataframes to agents, merge results, invoke Verifier, write output JSON. | main loop |
| **OrderAgent** | Resolve order status, items, sellers, dates (approved / carrier / delivered / estimated), per-seller `shipping_limit_date`, item & freight totals. | parallel phase |
| **PaymentAgent** | Enumerate payment rows, sum `payment_value`, reconcile total vs item + freight, report discrepancy. | parallel phase |
| **DeliveryAgent** | Compare actual delivery timestamps vs `order_estimated_delivery_date` and carrier receipt vs `shipping_limit_date`; classify late + responsible party. | parallel phase |
| **PolicyAgent** | Apply `EC_POLICY_V1` business rules in priority order; emit primary issue, status, confidence, root causes, parties, refund, actions. | sequential |
| **VerifierAgent** | Validate evidence ID format, entity limits, schema, amount rounding, `confidence` range before write. | sequential |

## Model & Runtime

- **Runtime:** OpenRouter (OpenAI-compatible API), each agent = **one separate model call**.
- **Model:** ≤10B parameters. Concrete model name declared as a constant in code and recorded in `metadata.json` (not in `.env`).
- **Agent execution style:** ReAct (Reason → Act → Observe) with a **system prompt + scoped tools**. Each agent selects tools to call and produces a structured JSON finding.

## Data Flow (per case)

1. Coordinator loads 9 CSVs into pandas dataframes once at startup.
2. Reads `input/EC_XXX.json`, extracts `claimed_order_id`.
3. Dispatches `OrderAgent | PaymentAgent | DeliveryAgent` **in parallel**, each receiving `order_id` + its scoped dataframes.
4. Awaits all three, merges their findings into a case context.
5. Calls `PolicyAgent` with the merged context → decision fields.
6. Calls `VerifierAgent` on the candidate output.
7. On success writes `output/EC_XXX.json`; on failure logs error and does not write a bad file.

## Agent Input / Output Contracts

### Coordinator
- **Input:** `case_id`, `input` JSON (with `claimed_order_id`), `dict[str, DataFrame]`.
- **Output:** final output dict (README §6 schema) written to `output/<case_id>.json`.

### OrderAgent
- **Input:** `order_id`, `df_orders`, `df_items`, `df_sellers`.
- **Output:**
  ```json
  {
    "order_status": "delivered",
    "purchase_ts": "...", "approved_ts": "...",
    "delivered_carrier_ts": "...", "delivered_customer_ts": "...",
    "estimated_delivery_ts": "...",
    "items": [{"item_id": 1, "seller_id": "s1", "price": 58.90, "freight_value": 13.29, "shipping_limit_ts": "..."}],
    "sellers": ["s1"],
    "item_total_brl": 58.90, "freight_total_brl": 13.29
  }
  ```

### PaymentAgent
- **Input:** `order_id`, `df_payments`, `df_items`.
- **Output:**
  ```json
  {
    "payment_rows": [{"sequential": 1, "type": "credit_card", "value": 99.33}],
    "payment_total_brl": 99.33,
    "expected_total_brl": 72.19,
    "reconciled": false,
    "discrepancy_brl": 27.14
  }
  ```

### DeliveryAgent
- **Input:** `order_id`, `df_orders`, `df_items`.
- **Output:**
  ```json
  {
    "delivered_late": false,
    "late_days": 0.0,
    "carrier_received_late": false,
    "responsible": "none",
    "candidate_cause": "DELIVERY_WITHIN_ESTIMATE"
  }
  ```

### PolicyAgent
- **Input:** merged `OrderFindings + PaymentFindings + DeliveryFindings`, `policy_version`.
- **Output:** decision fields only (primary_issue, case_status, confidence, ranked_causes, responsible_parties, financial_resolution, resolution_actions).

### VerifierAgent
- **Input:** candidate output dict + source order data.
- **Output:** validated output dict; raises on schema/ID/rounding/limit violation.

## Tool Scoping

Tools are **per-agent scoped** — each agent only sees the read-only query functions relevant to its domain. All tools are read-only over the loaded dataframes.

| Agent | Tools |
| ----- | ----- |
| OrderAgent | `lookup_order`, `lookup_items`, `lookup_sellers`, `sum_item_totals` |
| PaymentAgent | `lookup_payments`, `sum_payments`, `reconcile_payment` |
| DeliveryAgent | `lookup_order` (dates), `lookup_items` (shipping_limit), `compare_dates` |
| PolicyAgent | none (receives structured findings as context) |
| VerifierAgent | none (validates a candidate dict) |

## Error Handling

- **Missing order / no item rows:** item & seller lists empty, item/freight totals `0.0` (per README).
- **Null delivery timestamps:** treated as not-yet-delivered; DeliveryAgent flags accordingly.
- **Multiple sellers:** follow README rule (`order_delivered_carrier_date > shipping_limit_date` of that seller's item ⇒ that seller is late). Official set has no ambiguous multi-seller cases, but Verifier still guards limits.
- **Verifier failure:** coordinator logs reason, abandons bad write, moves to next case.

## Testing

- **Per-agent unit tests:** each agent's finding given a known-case input.
- **Reconciliation test:** PaymentAgent total vs item + freight within 0.10 BRL.
- **Schema test:** Verifier rejects malformed evidence IDs, out-of-range confidence, >5 entities, >10 evidence, >3 causes, >3 parties, >5 actions.
- **End-to-end:** run all 50 cases, assert 50 valid JSON files in `output/`.

## Files

```
.
├── architecture.md          # pointer to this spec (required at repo root)
├── specs/                   # design specs
│   └── 2025-08-05-multi-agent-architecture-design.md
├── main.py                  # entry point
├── agents/                  # (to be created)
│   ├── coordinator.py
│   ├── order_agent.py
│   ├── payment_agent.py
│   ├── delivery_agent.py
│   ├── policy_agent.py
│   └── verifier_agent.py
├── tools/                   # scoped tools
│   ├── order_tools.py
│   ├── payment_tools.py
│   └── delivery_tools.py
└── data/                    # CSV source
```