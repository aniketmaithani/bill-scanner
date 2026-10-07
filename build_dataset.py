#!/usr/bin/env python3
"""
Phase 3c: turn reviewed labels into MLX chat-format JSONL.

Source of truth is labels_review.csv (edit it, then re-run this).
Copies of the same invoice (same vendor + number + total) are kept in one
split so the test set never contains an invoice the model trained on.

Outputs data/{train,valid,test}.jsonl and data/test_files.csv.
"""

import csv
import json
import random

from invoice_schema import BASE, FIELDS, INSTRUCTIONS, load_text, to_json, user_message

REVIEW = BASE / "labels_review.csv"
DATA = BASE / "data"
SEED = 42
TEST_DOCS = 25
VALID_DOCS = 15
# No vendor may dominate training, or the model learns to default to it
MAX_PER_VENDOR = 16

NUMERIC = {"subtotal", "tax", "total"}


def parse(row):
    record = {}
    for field in FIELDS:
        value = row[field].strip()
        if field == "is_invoice":
            record[field] = value == "True"
        elif value == "":
            record[field] = None
        elif field in NUMERIC:
            number = round(float(value), 2)
            record[field] = int(number) if number.is_integer() else number
        else:
            record[field] = value
    return record


def example(sha, record):
    return {
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user_message(load_text(sha))},
            {"role": "assistant", "content": to_json(record)},
        ]
    }


def main():
    with REVIEW.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    groups = {}
    for row in rows:
        record = parse(row)
        key = (
            (str(record["vendor"]).lower(), record["invoice_number"], record["total"])
            if record["invoice_number"]
            else row["sha256"]
        )
        groups.setdefault(key, []).append((row, record))

    keys = sorted(groups, key=str)
    random.Random(SEED).shuffle(keys)

    # Keep an existing test set fixed so rounds stay comparable
    pinned = set()
    if (DATA / "test_files.csv").exists():
        with (DATA / "test_files.csv").open(encoding="utf-8") as handle:
            pinned = {row["sha256"] for row in csv.DictReader(handle)}

    splits = {"test": [], "valid": [], "train": []}
    for key in keys:
        if pinned:
            in_test = any(row["sha256"] in pinned for row, _ in groups[key])
            name = "test" if in_test else ("valid" if len(splits["valid"]) < VALID_DOCS else "train")
            splits[name].extend(groups[key])
            continue
        if len(splits["test"]) < TEST_DOCS:
            name = "test"
        elif len(splits["valid"]) < VALID_DOCS:
            name = "valid"
        else:
            name = "train"
        splits[name].extend(groups[key])

    capped, per_vendor = [], {}
    for row, record in splits["train"]:
        vendor = str(record["vendor"]).lower()
        per_vendor[vendor] = per_vendor.get(vendor, 0) + 1
        if per_vendor[vendor] <= MAX_PER_VENDOR:
            capped.append((row, record))
    dropped = len(splits["train"]) - len(capped)
    splits["train"] = capped
    print(f"train capped at {MAX_PER_VENDOR} per vendor: dropped {dropped}")

    DATA.mkdir(exist_ok=True)
    for name, items in splits.items():
        with (DATA / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row, record in items:
                handle.write(json.dumps(example(row["sha256"], record), ensure_ascii=False) + "\n")
        print(f"{name:<6}: {len(items)}")

    with (DATA / "test_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "sha256"])
        for row, _ in splits["test"]:
            writer.writerow([row["file"], row["sha256"]])

    print(f"groups: {len(groups)}  (duplicates kept together)")


if __name__ == "__main__":
    main()
