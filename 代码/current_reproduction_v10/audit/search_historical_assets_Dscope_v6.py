"""Read only the six explicitly requested D-drive recovery locations."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import pickletools
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "audit/historical_asset_recovery_search_v5.json"
INVENTORY = ROOT / "audit/historical_asset_recovery_search_v6_inventory.json"
OUTPUT = ROOT / "audit/historical_asset_recovery_search_v6.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def longpath(path):
    text = str(path)
    return text if text.startswith("\\\\?\\") else "\\\\?\\" + text


def main():
    old = json.loads(OLD.read_text("utf-8"))
    inventory = json.loads(INVENTORY.read_text("utf-8"))
    targets = {r["relative"]: dict(relative=r["relative"], expected_sha256=r["expected_sha256"],
        confirmed_locations=[], uncertain_candidates=[], rejected_candidates=[]) for r in old["targets"]}
    by_base = {}
    for name in targets:
        by_base.setdefault(name.rsplit("/", 1)[-1].lower(), []).append(name)
    counters, errors, uninspected, archive_rows = Counter(), [], [], []
    allfiles = [f for group in inventory for f in group["files"]]

    def examine(location, normalized, getter):
        base = normalized.rsplit("/", 1)[-1].lower()
        if base not in by_base:
            return
        counters["basename_candidates"] += 1
        try:
            data = getter()
            info = dict(location=location, bytes=len(data), sha256=sha(data))
            if base.endswith(".npz"):
                with zipfile.ZipFile(io.BytesIO(data)) as item:
                    info["array_keys"] = [n[:-4] for n in item.namelist() if n.endswith(".npy")]
            elif base.endswith((".pt", ".pth")) and zipfile.is_zipfile(io.BytesIO(data)):
                # Inspect pickle strings only: do not import torch or execute pickle.
                with zipfile.ZipFile(io.BytesIO(data)) as item:
                    pickle_name = next((n for n in item.namelist() if n.endswith("data.pkl")), None)
                    strings = []
                    if pickle_name:
                        for opcode, argument, _ in pickletools.genops(item.read(pickle_name)):
                            if opcode.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE") and isinstance(argument, str):
                                if argument.endswith((".pt", ".npz")) or "seed" in argument or "training" in argument or "source" in argument:
                                    strings.append(argument)
                    info["static_pickle_identity_strings"] = strings[:60]
            for target in by_base[base]:
                item = dict(info, exact_tail=normalized.lower().endswith(target.lower()))
                expected = targets[target]["expected_sha256"]
                if expected and item["sha256"] == expected:
                    item["identity"] = "exact historical SHA-256 match"
                    targets[target]["confirmed_locations"].append(item)
                    print("CONFIRMED", target, location, flush=True)
                elif expected:
                    item["reason"] = "historical SHA-256 differs"
                    targets[target]["rejected_candidates"].append(item)
                elif target.startswith("farvector_round17_data/") and "original_observations" not in item.get("array_keys", []):
                    item["reason"] = "required original_observations array missing"
                    targets[target]["rejected_candidates"].append(item)
                else:
                    item["reason"] = "No expected historical hash; different registered path and filename/schema alone cannot establish identity"
                    targets[target]["uncertain_candidates"].append(item)
        except Exception as exc:
            errors.append(dict(location=location, error=f"{type(exc).__name__}: {exc}"))

    for file in allfiles:
        if file.lower().endswith((".npz", ".pt", ".pth")):
            counters["filesystem_binary_entries"] += 1
            examine(file, file.replace("\\", "/"), lambda f=file: Path(longpath(f)).read_bytes())

    def scan_archive(file, label, depth=0):
        try:
            with zipfile.ZipFile(file) as archive:
                names = archive.namelist()
                counters["archive_containers"] += 1
                archive_rows.append(dict(location=label, entries=len(names), depth=depth))
                for name in names:
                    norm = name.replace("\\", "/")
                    if name.lower().endswith((".npz", ".pt", ".pth")):
                        counters["archive_binary_entries"] += 1
                        examine(label + "::" + name, norm, lambda n=name: archive.read(n))
                    elif name.lower().endswith(".zip"):
                        entry = archive.getinfo(name)
                        if depth >= 3 or entry.file_size > 128 * 1024 * 1024:
                            uninspected.append(dict(location=label + "::" + name, bytes=entry.file_size,
                                reason="nested archive depth/128 MiB bound; explicitly unresolved, not claimed searched"))
                        else:
                            scan_archive(io.BytesIO(archive.read(name)), label + "::" + name, depth + 1)
        except Exception as exc:
            errors.append(dict(location=label, error=f"{type(exc).__name__}: {exc}"))

    archives = [f for f in allfiles if f.lower().endswith(".zip")]
    for index, file in enumerate(archives, 1):
        scan_archive(longpath(file), file)
        if index % 200 == 0:
            print("ZIP_PROGRESS", index, "of", len(archives), flush=True)
    for target in targets.values():
        target["status"] = ("confirmed" if target["confirmed_locations"] else
                            "unresolved_basename_candidates" if target["uncertain_candidates"] else "not_found_within_scope")
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        operation="Read-only target filename, archive-directory, hash and static schema inspection. No source replacement or model unpickling.",
        scope=[r["root"] for r in inventory], prior_scope_not_rescanned=old["scope"],
        target_source=str(OLD), target_source_sha256=sha(OLD.read_bytes()),
        inventory=str(INVENTORY), inventory_sha256=sha(INVENTORY.read_bytes()),
        target_count=len(targets), counters=dict(counters), top_level_zip_count=len(archives),
        scan_errors=errors, uninspected_nested_archives=uninspected, archive_index=archive_rows,
        targets=list(targets.values()), confirmed_target_count=sum(bool(t["confirmed_locations"]) for t in targets.values()),
        unresolved_candidate_target_count=sum(bool(t["uncertain_candidates"]) and not t["confirmed_locations"] for t in targets.values()),
        copied_assets=[], interpretation="A same-basename candidate is not a recovered training-chain asset. No model/trajectory experiment was executed.")
    OUTPUT.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("confirmed_target_count", "unresolved_candidate_target_count", "counters", "top_level_zip_count")}))


if __name__ == "__main__":
    main()
