import csv
import json
from pathlib import Path

from .constants import NTU120_XSUB_TRAIN, NTU60_XSUB_TRAIN, NTU_NAME_RE


def meta_from_path(path):
    match = NTU_NAME_RE.search(Path(path).stem)
    if not match:
        return None
    return {k: int(v) for k, v in match.groupdict().items()}


def split_name(meta, protocol, ntu_version):
    protocol = protocol.lower()
    if protocol == "xview":
        return "train" if meta["camera"] in {2, 3} else "val"
    if protocol == "xset":
        return "train" if meta["setup"] % 2 == 0 else "val"
    train_subjects = NTU120_XSUB_TRAIN if ntu_version == 120 else NTU60_XSUB_TRAIN
    return "train" if meta["subject"] in train_subjects else "val"


def build_manifest(skeleton_dir, out_csv, protocol="xsub", ntu_version="auto"):
    rows = []
    for path in sorted(Path(skeleton_dir).rglob("*.npz")):
        meta = meta_from_path(path)
        if not meta:
            continue
        version = 120 if (ntu_version == 120 or meta["action"] > 60) else 60
        rows.append({
            "path": str(path.resolve()),
            "video": path.stem,
            "setup": meta["setup"],
            "camera": meta["camera"],
            "subject": meta["subject"],
            "replication": meta["replication"],
            "action": meta["action"],
            "label": meta["action"] - 1,
            "split": split_name(meta, protocol, version),
        })
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "path", "video", "setup", "camera", "subject",
            "replication", "action", "label", "split",
        ])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "total": len(rows),
        "train": sum(1 for r in rows if r["split"] == "train"),
        "val": sum(1 for r in rows if r["split"] == "val"),
    }
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("manifest {} {}".format(out_csv, summary))
    return out_csv
