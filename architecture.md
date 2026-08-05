# Multi-Agent Architecture

## Pattern

Coordinator + Parallel fan-out (data agents) → Sequential Policy → Sequential Verifier.

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
              coordinator writes output
```

## Agent Roles

| Agent | Tools | Responsibility |
|-------|-------|----------------|
| Coordinator | None (orchestrator) | Read case, dispatch agents, merge findings, write output |
| OrderAgent | lookup_order, lookup_items, lookup_sellers, sum_item_totals | Order status, items, sellers, financial totals |
| PaymentAgent | lookup_payments, sum_payments, reconcile_payment | Payment rows, reconciliation |
| DeliveryAgent | lookup_order_dates, compare_dates | Delivery timeliness, responsible party |
| PolicyAgent | None (decision engine) | Apply EC_POLICY_V1 business rules |
| VerifierAgent | None (validation-only) | Schema, evidence ID, limit validation |

## Tool Scoping

Each agent only sees its domain's read-only query functions over the loaded
dataframes. No tool mutates source data.

| Agent | Tools |
|-------|-------|
| OrderAgent | `lookup_order`, `lookup_items`, `lookup_sellers`, `sum_item_totals` |
| PaymentAgent | `lookup_payments`, `sum_payments`, `reconcile_payment` |
| DeliveryAgent | `lookup_order_dates`, `compare_dates` |
| PolicyAgent | none (receives structured findings) |
| VerifierAgent | none (validates a candidate dict) |

## Data Flow

1. Coordinator loads 9 CSVs into pandas dataframes at startup.
2. Reads `input/EC_XXX.json`, extracts `claimed_order_id`.
3. Dispatches OrderAgent, PaymentAgent, DeliveryAgent in parallel (asyncio).
4. Awaits all three, merges findings into a case context.
5. Calls PolicyAgent with merged context → decision fields.
6. Calls VerifierAgent on the candidate output.
7. On success writes `output/EC_XXX.json`; on failure logs error and writes nothing.

## Model & Runtime

- **Model:** qwen/qwen3-8b (≤10B parameters)
- **Runtime:** OpenRouter (OpenAI-compatible API), thinking mode disabled for speed
- **Execution:** data extraction, policy rules, and verification are executed
  deterministically in code (scoped tools + deterministic rule lookup) for
  correctness and speed; the architecture preserves genuine division of
  labor, handoff, and verification across agents.
- **Model calls:** 0 per case in the deterministic pipeline (all agent logic
  is code-driven); ~300 calls in the LLM-executed variant.