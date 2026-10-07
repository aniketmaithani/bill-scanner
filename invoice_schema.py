"""
Shared definitions for labelling, training and inference.

The teacher (GPT) and the student (Qwen) must see exactly the same input
text and produce exactly the same output format, so both import from here.
"""

import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEXT_CACHE = BASE / "work" / "text"
LABELS = BASE / "work" / "labels_v3"
CLASSIFICATION = BASE / "classification.csv"

# Long documents keep their head and tail: totals are usually at the end
MAX_CHARS = 6000
HEAD_CHARS = 4000

CATEGORIES = [
    "Cloud & Hosting",
    "Software & SaaS",
    "Telecom & Internet",
    "Utilities",
    "Travel",
    "Food & Dining",
    "Shopping",
    "Professional Services",
    "Government & Taxes",
    "Finance & Banking",
    "Rent",
    "Healthcare",
    "Education",
    "Other",
]

FIELDS = [
    "is_invoice",
    "customer",
    "vendor",
    "invoice_number",
    "invoice_date",
    "currency",
    "subtotal",
    "tax",
    "total",
    "gstin",
    "description",
    "category",
]


def _nullable(kind):
    return {"type": [kind, "null"]}


JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": FIELDS,
    "properties": {
        "is_invoice": {"type": "boolean"},
        "customer": _nullable("string"),
        "vendor": _nullable("string"),
        "invoice_number": _nullable("string"),
        "invoice_date": _nullable("string"),
        "currency": _nullable("string"),
        "subtotal": _nullable("number"),
        "tax": _nullable("number"),
        "total": _nullable("number"),
        "gstin": _nullable("string"),
        "description": _nullable("string"),
        "category": {"type": ["string", "null"], "enum": CATEGORIES + [None]},
    },
}

INSTRUCTIONS = f"""You extract structured data from the text of a bill, invoice or receipt.
The text may come from OCR and contain noise.

Return JSON with exactly these keys:
- is_invoice: true if this document is a bill, invoice, receipt or payment demand; else false
- customer: who the bill is addressed to (Bill To, Billed To, Recipient, Customer, Name)
- vendor: the seller who issued the bill (From, Supplier, Sold By, or the business named
  in the header next to its own GSTIN). Never the customer. Use the short brand name
  without legal suffix or country, e.g. "Amazon Web Services", "Airtel", "DigitalOcean"
- invoice_number: the invoice / bill / receipt number as printed
- invoice_date: issue date as YYYY-MM-DD
- currency: ISO 4217 code, e.g. INR, USD
- subtotal: amount before tax, as a number
- tax: total tax (sum of CGST, SGST, IGST, VAT, etc.), as a number
- total: amount charged by this document including tax, as a number (not a running
  account balance that includes earlier unpaid dues)
- gstin: the vendor's GSTIN if printed
- description: what was billed, at most 12 words
- category: one of {", ".join(CATEGORIES)}

Rules:
- Numbers have no currency symbols or thousands separators.
- Fill subtotal and tax whenever the document shows them separately, even if the tax
  is split into CGST and SGST lines (add them up). If only one amount is shown, put it
  in total and leave subtotal and tax null.
- Decide is_invoice from how the document titles itself. Titled invoice, tax invoice,
  bill, bill of supply or receipt (including combined titles such as "Tax Invoice /
  Proforma") -> true, even if it also contains an account summary. Titled only as a
  preview, estimate, quotation, proforma or statement -> false. When false, still fill
  vendor, currency and every other field that is present.
- customer and vendor are different parties. A person's name can be either: it is the
  vendor only when the document says it is from them.
- Use null for anything not present in the text. Never guess."""


# 1,234,567.89 (western) or 12,34,567.89 (Indian lakh) style grouping
GROUPED_NUMBER = re.compile(
    r"(?<![\d,.])(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3})(?=\.\d|(?![\d,]))"
)


def normalize_numbers(text: str) -> str:
    """Drop thousands separators: the model copies digits reliably, commas it does not."""
    return GROUPED_NUMBER.sub(lambda match: match.group(0).replace(",", ""), text)


def load_text(sha256: str) -> str:
    return prepare((TEXT_CACHE / f"{sha256}.txt").read_text(encoding="utf-8"))


def prepare(text: str) -> str:
    """Exactly what both teacher and student see as the document."""
    return truncate(normalize_numbers(text))


def truncate(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    tail = MAX_CHARS - HEAD_CHARS
    return text[:HEAD_CHARS] + "\n[...]\n" + text[-tail:]


def user_message(text: str) -> str:
    return f"DOCUMENT:\n{text}"


def to_json(record: dict) -> str:
    """Canonical compact output the student model is trained to emit."""
    return json.dumps({key: record[key] for key in FIELDS}, ensure_ascii=False)
