"""Read existing evidence metadata without evaluating a model or inspecting test outcomes."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

for rel in ["paper_evidence_review/confirmatory/results.json",
            "paper_evidence_review/confirmatory/evaluation_cases.json",
            "paper_evidence_review/confirmatory/method_freeze.json",
            "paper_evidence_review/candidate_audit.json",
            "paper_evidence_review/remaining_public_long/results.json"]:
    value = read(rel)
    print(rel, type(value).__name__, list(value)[:20] if isinstance(value, dict) else len(value))
    if "confirmatory/results" in rel:
        records = value if isinstance(value, list) else value.get("records", value.get("results", []))
        print("record_count", len(records))
        if records:
            print("record_keys", list(records[0]))
            print("metadata_only", [{k: r.get(k) for k in ["split", "case", "case_id", "arm", "policy_id"] if k in r} for r in records[-6:]])
    elif "evaluation_cases" in rel:
        if isinstance(value, dict):
            print("collection_sizes", {k: len(v) for k, v in value.items() if isinstance(v, (dict, list))})
    elif "candidate_audit" in rel:
        print("candidate_labels", [x.get("label") for x in value.get("candidates", [])])
        for item in value.get("candidates", []):
            if item.get("label") == "worst_guard":
                print("candidate_keys", list(item))
                recs = item.get("records", [])
                print("record_example", recs[0] if recs else None)
    elif "remaining_public" in rel:
        print(json.dumps(value, ensure_ascii=False)[:3500])

state = read("当前执行状态.json")
print("state_top_keys", list(state))
