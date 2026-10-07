#!/usr/bin/env python3
"""
Phase 3b: sanity-check teacher labels locally, before training on them.

Every label is checked against the document text:
    total_in_text      the total appears as a number in the text
    number_in_text     the invoice number is printed in the text
    sum_ok             subtotal + tax == total (when all three exist)
    date_ok            invoice_date parses as YYYY-MM-DD

Writes labels_review.csv sorted so failing rows come first; build_dataset.py
trains from that sheet. For corrections that must survive a re-run, add them to
label_corrections.json as {"<sha256>": {"field": value}} - they are applied to
the teacher labels every time this script runs.
"""

import csv
import json
import re
from datetime import date

from invoice_schema import BASE, FIELDS, LABELS, TEXT_CACHE

REVIEW = BASE / "labels_review.csv"
# Hand corrections, re-applied on every run so a relabel never loses them
CORRECTIONS = BASE / "label_corrections.json"
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text):
    values = set()
    for match in NUMBER.findall(text):
        try:
            values.add(round(float(match.replace(",", "")), 2))
        except ValueError:
            pass
    return values


def squash(value):
    return re.sub(r"[\s\-_/]", "", value).lower()


def audit(sha, record):
    text = (TEXT_CACHE / f"{sha}.txt").read_text(encoding="utf-8")
    numbers = numbers_in(text)
    problems = []

    total = record["total"]
    if total is None:
        problems.append("no total")
    elif round(total, 2) not in numbers:
        problems.append("total not in text")

    number = record["invoice_number"]
    if number and squash(number) not in squash(text):
        problems.append("invoice number not in text")

    subtotal, tax = record["subtotal"], record["tax"]
    if None not in (subtotal, tax, total) and abs(subtotal + tax - total) > 1:
        problems.append("subtotal+tax != total")

    if record["invoice_date"]:
        try:
            date.fromisoformat(record["invoice_date"])
        except ValueError:
            problems.append("bad date")

    if not record["is_invoice"]:
        problems.append("teacher says not an invoice")

    return problems


def main():
    corrections = json.loads(CORRECTIONS.read_text()) if CORRECTIONS.exists() else {}
    rows = []
    for path in sorted(LABELS.glob("*.json")):
        record = json.loads(path.read_text())
        for field, value in corrections.get(path.stem, {}).items():
            if not field.startswith("_"):
                record[field] = value
        problems = audit(path.stem, record)
        rows.append({
            "problems": "; ".join(problems),
            "approved": "" if problems else "auto",
            "file": record["_source_file"],
            **{field: record[field] for field in FIELDS},
            "sha256": path.stem,
        })

    rows.sort(key=lambda r: (r["problems"] == "", r["file"]))
    with REVIEW.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flagged = [r for r in rows if r["problems"]]
    print(f"labels: {len(rows)}  clean: {len(rows) - len(flagged)}  flagged: {len(flagged)}")
    tally = {}
    for row in flagged:
        for problem in row["problems"].split("; "):
            tally[problem] = tally.get(problem, 0) + 1
    for problem, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}  {problem}")
    print(f"review sheet: {REVIEW}")


if __name__ == "__main__":
    main()
