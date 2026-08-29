import json

import pytest

from abem.generator_identification import GATE_STATUSES, gate


def test_gate_schema_is_tristate_and_json_serializable():
    gates = {
        "G0": gate("PASS", "ok"),
        "G1": gate("FAIL", "falsified"),
        "G2": gate("NOT_EVALUATED", "stopped after G1 failure"),
    }
    assert {item["status"] for item in gates.values()} == GATE_STATUSES
    assert "pass" not in json.dumps(gates)


def test_gate_schema_rejects_boolean_style_status():
    with pytest.raises(ValueError, match="invalid gate status"):
        gate("false", "legacy representation")
