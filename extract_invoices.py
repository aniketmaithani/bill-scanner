#!/usr/bin/env python3
"""
Extract invoice fields from PDFs with the local fine-tuned Qwen model.

Fully offline: text layer or Tesseract OCR, then Qwen2.5-1.5B + LoRA adapter,
run with MLX on Apple Silicon or PyTorch on any other CPU (see backends.py).
Each row gets a "check" column: whether the predicted total actually appears
in the document text, and whether subtotal + tax adds up.

Two safety nets sit on top of the model:
  * amounts that lost or gained a digit are snapped to the printed amount
  * if the answer is not valid JSON, the fields are filled one at a time

Usage:
    python extract_invoices.py                        # all PDFs in Bills/
    python extract_invoices.py bill.pdf -o bill.csv   # one file
    python extract_invoices.py a.pdf b.jpg some/folder -o out.csv
    python extract_invoices.py bill.pdf --backend torch   # force the CPU runtime
"""

import argparse
import csv
import json
import re
from pathlib import Path

from audit_labels import numbers_in
from backends import load_backend
from classify_bills import SUPPORTED, extract_text
from invoice_schema import (
    BASE,
    CATEGORIES,
    FIELDS,
    INSTRUCTIONS,
    as_number,
    parse_json,
    prepare,
    user_message,
)


# Domestic GSTIN, or the OIDAR form issued to foreign online-service
# suppliers (e.g. 9926USA29037OS3 for DigitalOcean)
GSTIN = re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]|\d{4}[A-Z]{3}\d{5}[A-Z]{2}[A-Z\d]")


def _digits(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _one_digit_apart(a, b):
    """True when one string is the other with a single character added."""
    if abs(len(a) - len(b)) != 1:
        return False
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def snap_to_printed(record, numbers):
    """
    The model sometimes drops or repeats a zero (125000 -> 12500). Subtotal and
    total are always printed on a bill, so if exactly one printed amount is a
    single digit away, use it. Tax is left alone: it is often a sum that is
    not printed anywhere, and is derived from subtotal and total instead.
    """
    notes = []
    for field in ("subtotal", "total"):
        value = as_number(record.get(field))
        if not isinstance(value, (int, float)) or round(value, 2) in numbers:
            continue
        near = [n for n in numbers if _one_digit_apart(_digits(value), _digits(n))]
        if len(near) == 1:
            record[field] = near[0]
            notes.append(f"{field} snapped to printed amount")
    return notes


def checks(record, text):
    numbers = numbers_in(text)
    notes = snap_to_printed(record, numbers)
    gstin = record.get("gstin")
    if gstin and not GSTIN.fullmatch(str(gstin).strip()):
        record["gstin"] = None
        notes.append("invalid gstin dropped")
    total = as_number(record.get("total"))
    if isinstance(total, float) or isinstance(total, int):
        if round(total, 2) not in numbers:
            notes.append("total not found in text")
    else:
        notes.append("no total")
    subtotal, tax = as_number(record.get("subtotal")), as_number(record.get("tax"))
    if all(isinstance(v, (int, float)) for v in (subtotal, tax, total)):
        if abs(subtotal + tax - total) > 1:
            # The model often reports one GST half (CGST only); the difference
            # is right whenever subtotal and total are both grounded in the text
            derived = round(total - subtotal, 2)
            if derived >= 0 and {round(subtotal, 2), round(total, 2)} <= numbers:
                record["tax"] = derived
                notes.append("tax derived as total - subtotal")
            else:
                notes.append("subtotal+tax != total")
    return "; ".join(notes) or "ok"


def _value_end(text):
    """Index where a JSON value ends at the top level (",", "}"), else None."""
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in ",}":
            return index
    return None


def fill_fields(backend, prompt):
    """
    Fallback when the free-form answer is not valid JSON: write each key
    ourselves and let the model complete only its value. The result always
    has every field and is always well formed.
    """
    record, prefix = {}, "{"
    for index, field in enumerate(FIELDS):
        prefix += (", " if index else "") + json.dumps(field) + ": "
        value_text = ""
        for piece in backend.stream(prompt + prefix, max_tokens=60):
            value_text += piece
            end = _value_end(value_text)
            if end is not None:
                value_text = value_text[:end]
                break
        try:
            value = json.loads(value_text.strip())
        except json.JSONDecodeError:
            value = None
        record[field] = value
        prefix += json.dumps(value, ensure_ascii=False)
    return record


def clean(record):
    """Coerce field types so a bad value becomes empty instead of wrong."""
    if not isinstance(record.get("is_invoice"), bool):
        record["is_invoice"] = None
    for field in ("subtotal", "tax", "total"):
        value = as_number(record.get(field))
        record[field] = value if isinstance(value, (int, float)) else None
    if record.get("category") not in CATEGORIES:
        record["category"] = None
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="*", default=[str(BASE / "Bills")],
        help="PDF/photo files and/or folders (default: Bills/)",
    )
    parser.add_argument("-o", "--output", default=str(BASE / "invoices.csv"))
    parser.add_argument(
        "--backend", choices=["mlx", "torch"],
        help="model runtime (default: mlx on Apple Silicon, torch elsewhere)",
    )
    args = parser.parse_args()

    pdfs = []
    for item in map(Path, args.inputs):
        found = sorted(item.iterdir()) if item.is_dir() else [item]
        pdfs += [p for p in found if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not pdfs:
        parser.error("no supported PDF or image files found")
    backend = load_backend(args.backend)
    print(f"runtime: {backend.name}", flush=True)

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
            prompt = backend.chat_prompt(messages)
            output = backend.generate(prompt, max_tokens=300)
            record = parse_json(output)
            guided = record is None or not set(FIELDS) <= set(record)
            if guided:
                record = fill_fields(backend, prompt)
            record = clean(record)
            check = checks(record, text)
            if guided:
                check = "fields filled one at a time" + ("" if check == "ok" else "; " + check)
            for field in ("subtotal", "tax", "total"):
                if isinstance(record[field], float) and record[field].is_integer():
                    record[field] = int(record[field])
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
