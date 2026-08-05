import pytest
import json
from pathlib import Path
from src.loader import load_all_data, load_input_case
from src.agents.coordinator import Coordinator


@pytest.fixture(scope="module")
def data():
    return load_all_data()


def test_coordinator_processes_case(data, tmp_path):
    """Coordinator should process a single case and write valid output."""
    case = load_input_case("EC_001")
    coord = Coordinator(data, output_dir=str(tmp_path))
    result = coord.process_case(case)
    assert result["case_id"] == "EC_001"
    assert "assessment" in result
    assert "affected_entities" in result
    assert "evidence_ids" in result

    # Check file was written
    output_file = tmp_path / "EC_001.json"
    assert output_file.exists()
    with open(output_file) as f:
        written = json.load(f)
    assert written["case_id"] == "EC_001"


def test_coordinator_trace_entries(data, tmp_path):
    """Coordinator should record trace entries for each agent step."""
    case = load_input_case("EC_001")
    coord = Coordinator(data, output_dir=str(tmp_path))
    coord.process_case(case)
    assert len(coord.trace_entries) > 0
    agents = {e["agent"] for e in coord.trace_entries}
    assert "Coordinator" in agents
    assert "OrderAgent" in agents
    assert "PaymentAgent" in agents
    assert "DeliveryAgent" in agents
    assert "PolicyAgent" in agents
    assert "VerifierAgent" in agents