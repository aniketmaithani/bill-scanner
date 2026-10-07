#!/usr/bin/env python3
"""
Phase 3a: auto-label bill candidates with a teacher model (OpenAI).

Only documents classified as "bill" or "review" AND not flagged sensitive
are sent. Only the extracted text is sent, never the PDF itself.
Results are cached in work/labels/<sha>.json, so re-runs resume.

Usage:
    python label_with_gpt.py --limit 1      # smoke test
    python label_with_gpt.py                # label everything pending
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from invoice_schema import (
    BASE,
    CLASSIFICATION,
    INSTRUCTIONS,
    JSON_SCHEMA,
    LABELS,
    load_text,
    user_message,
)

MODEL = "gpt-6.1-sol"
WORKERS = 6


def load_env():
    for line in (BASE / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def candidates(include_dir=None):
    with CLASSIFICATION.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["verdict"] in ("encrypted", "unreadable") or row["sensitive"] != "False":
                continue
            forced = include_dir and row["file"].startswith(include_dir.rstrip("/") + "/")
            if forced or row["verdict"] in ("bill", "review"):
                yield row


def label(client, row):
    output = LABELS / f"{row['sha256']}.json"
    if output.exists():
        return row["file"], "cached"

    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=user_message(load_text(row["sha256"])),
        text={
            "format": {
                "type": "json_schema",
                "name": "invoice",
                "schema": JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    record = json.loads(response.output_text)
    record["_source_file"] = row["file"]
    record["_teacher"] = MODEL
    output.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return row["file"], "labelled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-dir",
        help="folder under PDF_Collection whose files are labelled whatever their score",
    )
    args = parser.parse_args()

    load_env()
    LABELS.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    rows = list(candidates(args.include_dir))[: args.limit]
    print(f"Candidates to label: {len(rows)}  (model: {MODEL})", flush=True)

    failures = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(label, client, row) for row in rows]
        for done, future in enumerate(futures, 1):
            try:
                name, status = future.result()
                print(f"[{done}/{len(rows)}] {status:<9} {name}", flush=True)
            except Exception as error:
                failures += 1
                print(f"[{done}/{len(rows)}] FAILED    {error}", flush=True)

    print(f"\nDone. failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
