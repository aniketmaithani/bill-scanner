#!/usr/bin/env python3
"""
Phase 1: find PDFs that look like bills / invoices and COPY them out.

No LLM. Text comes from the PDF text layer, or Tesseract OCR for scanned
pages. Each unique file (by SHA-256) is scored with weighted keyword rules.

Outputs (all next to this script):
    Bills/               score >= BILL_THRESHOLD
    Bills_Review/        REVIEW_THRESHOLD <= score < BILL_THRESHOLD
    work/text/<sha>.txt  cached extracted text (reused by later phases)
    classification.csv   one row per unique file, with score and reasons

Originals in PDF_Collection are never modified, moved or deleted.
"""

import csv
import hashlib
import io
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf as fitz
import pytesseract
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "PDF_Collection"
BILLS = BASE / "Bills"
REVIEW = BASE / "Bills_Review"
TEXT_CACHE = BASE / "work" / "text"
REPORT = BASE / "classification.csv"

BILL_THRESHOLD = 8
REVIEW_THRESHOLD = 4

# Pages read for classification and cached for extraction
MAX_PAGES = 4
# Below this many characters a page is treated as scanned and OCR'd
MIN_TEXT_CHARS = 50
OCR_DPI = 300
SHORT_OCR_CHARS = 600

# Photographed bills are OCR'd directly
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".avif"}
SUPPORTED = IMAGE_SUFFIXES | {".pdf"}


# ============================================================
# SCORING RULES  (pattern, weight, label) - each counted once
# ============================================================

POSITIVE = [
    (r"\btax invoice\b", 4, "tax invoice"),
    (r"\bbill of supply\b", 4, "bill of supply"),
    (r"\binvoice\b", 3, "invoice"),
    (r"\breceipt\b", 3, "receipt"),
    (r"\b(invoice|bill|receipt)\s*(no\.?|number|#|id)\b", 4, "invoice number"),
    (r"\bamount (due|payable)\b", 3, "amount due"),
    (r"\bgrand total\b", 3, "grand total"),
    (r"\btotal amount\b", 2, "total amount"),
    (r"\bsub\s?-?total\b", 2, "subtotal"),
    (r"\bgstin\b", 2, "gstin"),
    (r"\b(hsn|sac)\b", 2, "hsn/sac"),
    (r"\b(cgst|sgst|igst|utgst)\b", 2, "gst split"),
    (r"\b(amount paid|payment received|paid on)\b", 2, "payment"),
    (r"\b(billing|bill) (period|date|cycle)\b", 2, "billing period"),
    (r"\bdue date\b", 1, "due date"),
    (r"\border (id|no\.?|number|#)\b", 1, "order id"),
    (r"\bchallan\b", 2, "challan"),
    (r"\b(ending|closing|outstanding) balance\b", 2, "balance"),
]

# "Total ... 1,234.56" style line: a labelled amount is strong evidence
TOTAL_WITH_AMOUNT = re.compile(
    r"(grand total|total amount|amount due|amount payable|total due|\btotal\b)"
    r"[^\n]{0,40}?(₹|rs\.?|inr|\$|usd|€|eur)?\s*\d[\d,]*\.\d{2}",
    re.I,
)

NEGATIVE = [
    (r"\babstract\b", -2, "abstract"),
    (r"\bchapter\b", -2, "chapter"),
    (r"\btable of contents\b", -3, "table of contents"),
    (r"\breferences\b", -1, "references"),
    (r"\b(pay ?slip|salary slip|earnings|deductions)\b", -4, "payslip"),
    (r"\b(examination report|trade ?mark|registration certificate)\b", -3, "legal"),
    (r"\b(balance sheet|profit (and|&) loss)\b", -3, "financials"),
]

FILENAME_HINT = re.compile(r"invoice|receipt|bill(?!ion)", re.I)

# Documents that must never leave the machine (e.g. for API labelling)
SENSITIVE = re.compile(
    r"annual information statement|form 26as|statement of account|"
    r"account statement|bank statement|aadhaar|passport|"
    r"credit card statement|card statement",
    re.I,
)

POSITIVE = [(re.compile(p, re.I), w, l) for p, w, l in POSITIVE]
NEGATIVE = [(re.compile(p, re.I), w, l) for p, w, l in NEGATIVE]


# ============================================================
# TEXT EXTRACTION
# ============================================================

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ocr_page(page) -> str:
    pixmap = page.get_pixmap(dpi=OCR_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    text = pytesseract.image_to_string(image, lang="eng")
    # Automatic layout analysis can drop whole tables on bills; when little
    # text comes back, retry treating the page as one uniform block
    if len(text.strip()) < SHORT_OCR_CHARS:
        block = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
        if len(block.strip()) > len(text.strip()):
            text = block
    return text


def ocr_photo(path: Path) -> str:
    """Phone photos: fix rotation, greyscale, upscale small shots, boost contrast."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("L")
    if image.width < 1500:
        scale = 1500 / image.width
        image = image.resize((1500, int(image.height * scale)), Image.LANCZOS)
    image = ImageOps.autocontrast(image)
    # psm 4: a single column of text of variable sizes, which suits receipts
    return pytesseract.image_to_string(image, lang="eng", config="--psm 4")


def extract_text(path: Path):
    """
    Return (text, page_count, ocr_pages, status).
    status is one of: ok, encrypted, unreadable
    """

    if path.suffix.lower() in IMAGE_SUFFIXES:
        try:
            return f"--- page 1 ---\n{ocr_photo(path)}", 1, 1, "ok"
        except Exception:
            return "", 0, 0, "unreadable"

    try:
        document = fitz.open(path)
    except Exception:
        return "", 0, 0, "unreadable"

    if document.needs_pass:
        return "", document.page_count, 0, "encrypted"

    parts = []
    ocr_pages = 0

    for index in range(min(MAX_PAGES, document.page_count)):
        page = document[index]
        text = page.get_text()

        if len(text.strip()) < MIN_TEXT_CHARS:
            try:
                text = ocr_page(page)
                ocr_pages += 1
            except Exception:
                text = ""

        parts.append(f"--- page {index + 1} ---\n{text}")

    return "\n".join(parts), document.page_count, ocr_pages, "ok"


# ============================================================
# SCORING
# ============================================================

def score_text(text: str, filename: str, page_count: int):
    score = 0
    reasons = []

    for pattern, weight, label in POSITIVE + NEGATIVE:
        if pattern.search(text):
            score += weight
            reasons.append(f"{label}({weight:+d})")

    if TOTAL_WITH_AMOUNT.search(text):
        score += 3
        reasons.append("total with amount(+3)")

    if FILENAME_HINT.search(filename):
        score += 3
        reasons.append("filename(+3)")

    if page_count > 30:
        score -= 6
        reasons.append("pages>30(-6)")
    elif page_count > 10:
        score -= 3
        reasons.append("pages>10(-3)")

    return score, reasons


# ============================================================
# PER-FILE WORKER
# ============================================================

def process(item):
    digest, path = item
    cache = TEXT_CACHE / f"{digest}.txt"

    text, page_count, ocr_pages, status = extract_text(path)

    if status == "ok":
        cache.write_text(text, encoding="utf-8")
        score, reasons = score_text(text, path.name, page_count)
        sensitive = bool(SENSITIVE.search(text))
    else:
        score, reasons, sensitive = 0, [], False

    if status != "ok":
        verdict = status
    elif score >= BILL_THRESHOLD:
        verdict = "bill"
    elif score >= REVIEW_THRESHOLD:
        verdict = "review"
    else:
        verdict = "not_bill"

    return {
        "sha256": digest,
        "file": str(path.relative_to(SOURCE)),
        "verdict": verdict,
        "score": score,
        "pages": page_count,
        "ocr_pages": ocr_pages,
        "sensitive": sensitive,
        "reasons": " ".join(reasons),
    }




# ============================================================
# MAIN
# ============================================================

def main():
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    BILLS.mkdir(exist_ok=True)
    REVIEW.mkdir(exist_ok=True)

    pdfs = sorted(
        p for p in SOURCE.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    # Keep the first file name seen for each distinct content hash
    unique = {}
    for path in pdfs:
        unique.setdefault(sha256(path), path)

    print(f"PDFs: {len(pdfs)}  unique: {len(unique)}", flush=True)

    rows = []
    with ProcessPoolExecutor() as pool:
        for done, row in enumerate(pool.map(process, unique.items()), 1):
            rows.append(row)
            if done % 50 == 0:
                print(f"  processed {done}/{len(unique)}", flush=True)

    for row in rows:
        target = {"bill": BILLS, "review": REVIEW}.get(row["verdict"])
        # Skip files already copied, so re-runs are idempotent
        name = Path(row["file"]).name
        if target and not (target / name).exists():
            shutil.copy2(unique[row["sha256"]], target / name)

    rows.sort(key=lambda r: (-r["score"], r["file"]))
    with REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    print()
    for verdict, count in sorted(counts.items()):
        print(f"{verdict:<12}: {count}")
    print(f"sensitive   : {sum(r['sensitive'] for r in rows)}")
    print(f"report      : {REPORT}")


if __name__ == "__main__":
    sys.exit(main())
