#!/usr/bin/env python3
"""
Extract invoice fields from PDFs with the local fine-tuned Qwen model.

Fully offline: text layer or Tesseract OCR, then Qwen2.5-1.5B + LoRA adapter.
Each row gets a "check" column: whether the predicted total actually appears
in the document text, and whether subtotal + tax adds up.

Usage:
    python extract_invoices.py                 # all PDFs in Bills/
    python extract_invoices.py some/folder -o out.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from audit_labels import numbers_in
from classify_bills import SUPPORTED, extract_text
from evaluate import ADAPTER, MODEL, as_number, parse_json
from invoice_schema import BASE, FIELDS, INSTRUCTIONS, prepare, user_message


# Domestic GSTIN, or the OIDAR form issued to foreign online-service
# suppliers (e.g. 9926USA29037OS3 for DigitalOcean)
GSTIN = re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]|\d{4}[A-Z]{3}\d{5}[A-Z]{2}[A-Z\d]")


def checks(record, text):
    notes = []
    gstin = record.get("gstin")
    if gstin and not GSTIN.fullmatch(str(gstin).strip()):
        record["gstin"] = None
        notes.append("invalid gstin dropped")
    total = as_number(record.get("total"))
    if isinstance(total, float) or isinstance(total, int):
        if round(total, 2) not in numbers_in(text):
            notes.append("total not found in text")
    else:
        notes.append("no total")
    subtotal, tax = as_number(record.get("subtotal")), as_number(record.get("tax"))
    if all(isinstance(v, (int, float)) for v in (subtotal, tax, total)):
        if abs(subtotal + tax - total) > 1:
            # The model often reports one GST half (CGST only); the difference
            # is right whenever subtotal and total are both grounded in the text
            derived = round(total - subtotal, 2)
            if derived >= 0 and {round(subtotal, 2), round(total, 2)} <= numbers_in(text):
                record["tax"] = derived
                notes.append("tax derived as total - subtotal")
            else:
                notes.append("subtotal+tax != total")
    return "; ".join(notes) or "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=str(BASE / "Bills"))
    parser.add_argument("-o", "--output", default=str(BASE / "invoices.csv"))
    args = parser.parse_args()

    pdfs = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in SUPPORTED)
    model, tokenizer = load(str(MODEL), adapter_path=str(ADAPTER))
    sampler = make_sampler(temp=0.0)

    rows = []
    for index, pdf in enumerate(pdfs, 1):
        text, _, _, status = extract_text(pdf)
        row = {"file": pdf.name, **{field: None for field in FIELDS}}

        if status != "ok":
            row["check"] = status
        else:
            messages = [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": user_message(prepare(text))},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            output = generate(model, tokenizer, prompt=prompt, max_tokens=300, sampler=sampler)
            record = parse_json(output)
            if record is None:
                row["check"] = "model output not JSON"
            else:
                check = checks(record, text)
                row.update({field: record.get(field) for field in FIELDS})
                row["check"] = check

        rows.append(row)
        print(f"[{index}/{len(pdfs)}] {row['check']:<24} {pdf.name}", flush=True)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
