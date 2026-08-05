import sys
import json
from pathlib import Path
from src.loader import load_all_data, load_input_case, list_case_ids
from src.agents.coordinator import Coordinator
from src.config import OUTPUT_DIR, TRACE_FILE, MODEL_NAME


def main():
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


if __name__ == "__main__":
    main()