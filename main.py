import sys
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from src.loader import load_all_data, load_input_case, list_case_ids
from src.agents.coordinator import Coordinator
from src.openrouter_client import OpenRouterClient
from src.config import (
    OUTPUT_DIR,
    TRACE_FILE,
    METADATA_FILE,
    MODEL_NAME,
    PARAMETER_SIZE,
    FRAMEWORK,
    RUNTIME,
    AGENTS,
    PATTERN,
    EXECUTION,
)


def compute_context_metrics(case_ids):
    """Aggregate data-context metrics (relating to the processed Olist data)
    from the written output JSONs."""
    primary_issues = Counter()
    case_statuses = Counter()
    orders, items, sellers, payments = set(), set(), set(), set()

    for case_id in case_ids:
        path = Path(OUTPUT_DIR) / f"{case_id}.json"
        if not path.exists():
            continue
        with open(path) as f:
            out = json.load(f)
        primary_issues[out.get("assessment", {}).get("primary_issue", "missing")] += 1
        case_statuses[out.get("assessment", {}).get("case_status", "missing")] += 1
        entities = out.get("affected_entities", {})
        orders.update(entities.get("order_ids", []))
        items.update(entities.get("item_ids", []))
        sellers.update(entities.get("seller_ids", []))
        payments.update(entities.get("payment_ids", []))

    return {
        "cases_processed": len(case_ids),
        "primary_issue_distribution": dict(sorted(primary_issues.items(), key=lambda x: -x[1])),
        "case_status_distribution": dict(sorted(case_statuses.items(), key=lambda x: -x[1])),
        "distinct_orders_affected": len(orders),
        "distinct_items_affected": len(items),
        "distinct_sellers_affected": len(sellers),
        "distinct_payment_rows_affected": len(payments),
    }


def main():
    start_time = time.time()
    run_id = datetime.fromtimestamp(start_time).strftime("%Y%m%d-%H%M%S")
    OpenRouterClient.reset_stats()

    print("Loading data...")
    data = load_all_data()
    print(f"Loaded {len(data)} datasets.")

    case_ids = list_case_ids()
    print(f"Found {len(case_ids)} cases to process.")

    coordinator = Coordinator(data)

    success_count = 0
    failure_count = 0

    for case_id in case_ids:
        print(f"Processing {case_id}...")
        case = load_input_case(case_id)
        result = coordinator.process_case(case)
        if result:
            success_count += 1
            print(f"  ✓ {case_id} — {result.get('assessment', {}).get('primary_issue', '?')}")
        else:
            failure_count += 1
            print(f"  ✗ {case_id} — FAILED")

    # Write trace
    trace_path = Path(TRACE_FILE)
    with open(trace_path, "w") as f:
        for entry in coordinator.trace_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Trace written to {trace_path}")

    # Summary
    print(f"\n{'='*40}")
    print(f"Results: {success_count} success, {failure_count} failure out of {len(case_ids)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"{'='*40}")

    # Write metadata
    runtime_seconds = round(time.time() - start_time, 1)
    metadata_path = Path(METADATA_FILE)
    usage = OpenRouterClient.TOTAL_USAGE
    metadata = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "parameter_size": PARAMETER_SIZE,
        "framework": FRAMEWORK,
        "runtime": RUNTIME,
        "total_cases": len(case_ids),
        "agents": AGENTS,
        "pattern": PATTERN,
        "execution": EXECUTION,
        "runtime_seconds": runtime_seconds,
        "llm_stats": {
            "requests": OpenRouterClient.REQUESTS,
            "total_prompt_tokens": usage["prompt_tokens"],
            "total_completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
        "context_metrics": compute_context_metrics(case_ids),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()