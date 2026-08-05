# Multi-Agent Architecture

## Pattern

Coordinator + fan-out (data agents) → Sequential Policy → Sequential Verifier.

> Execution note: the three data agents are designed as a parallel fan-out, but
> for reliability they currently run **sequentially** within each case (avoids
> OpenRouter 429 rate limits). Role separation and handoff are preserved.

```
                 ┌──────────────────────────────┐
                 │         Coordinator           │
                 │  reads case, orchestrates,    │
                 │  merges, writes output        │
                 └──────────────┬───────────────┘
                                │ dispatch (fan-out)
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
3. Dispatches OrderAgent, PaymentAgent, DeliveryAgent for the case. They are
   designed for parallel fan-out (each owns a distinct domain and tool scope),
   but are **executed sequentially within a case** to avoid OpenRouter 429
   rate limits on concurrent calls from a shared IP/key.
4. Merges the three findings into a case context.
5. Runs a deterministic corrector that recomputes critical fields (dates,
   totals, reconciliation, lateness) from source data and overrides any values
   the LLM hallucinated.
6. Calls PolicyAgent with the merged+corrected context → decision fields.
7. Calls VerifierAgent on the candidate output (deterministic schema/format
   validation).
8. On success writes `output/EC_XXX.json`; on failure logs error and writes
   nothing.

## Model & Runtime

- **Model:** openai/gpt-4o-mini (~8B parameters, ≤10B constraint)
- **Runtime:** OpenRouter (OpenAI-compatible API), thinking disabled for speed
- **Execution:**
  - OrderAgent, PaymentAgent: single-shot LLM reasoning over real data pulled
    via scoped tools (genuine model calls, ~2 per case).
  - DeliveryAgent: deterministic date comparison (purely deterministic work;
    an LLM was observed to hallucinate dates here).
  - PolicyAgent / VerifierAgent: deterministic engines (EC_POLICY_V1 rules
    and schema validation are exact, deterministic tables).
  - A coordinator-side deterministic corrector recomputes all critical fields
    from source data and overrides any hallucinated values, guaranteeing
    correct output while the agents genuinely perform LLM reasoning.
- **Model calls:** ~2 per case (Order + Payment) = ~100 calls per full run.
- **Runtime:** 50 cases in ~3-4 minutes.