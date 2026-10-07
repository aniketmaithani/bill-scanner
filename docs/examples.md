---
title: Examples
nav_order: 4
---

# Examples
{: .no_toc }

1. TOC
{:toc}

---

## A GST invoice

The [Quickstart](quickstart.html) walks through `samples/sample_gst_invoice.pdf`
field by field. In short: every value comes out right, including the GSTIN and the
lakh-formatted amounts, after the extractor corrects a dropped zero in the
subtotal and works out the tax.

## A café receipt photo
{: #a-cafe-receipt-photo }

`samples/sample_cafe_receipt.jpg` is a phone photo of a till receipt: a little
tilted, a little blurry, with some grain.

```bash
python extract_invoices.py samples/sample_cafe_receipt.jpg -o receipt.csv
```

OCR reads it almost perfectly. The model does not:

| Column | Got | Should be |
|---|---|---|
| vendor | Shop 3 | Corner Cafe |
| customer | 1 | (none) |
| invoice_number | 1 | CC-7781 |
| invoice_date | (empty) | 2026-10-02 |
| currency | INR | INR |
| subtotal | 500 | 500 |
| tax | 25 | 25 |
| total | 525 | 525 |
| check | `fields filled one at a time; tax derived as total - subtotal` | |

The money is right, the names are not, and the `check` column flags the row. I'm
keeping this example in on purpose. The model was trained on invoices, and a till
receipt with no "Bill To" and no invoice heading is outside what it has seen. If
your bills are mostly receipts like this, [retrain](training.html#retrain-on-your-own-bills)
with some of them in the training set.

## Sort a messy folder first

If your bills are mixed in with everything else, let `classify_bills.py` find them.
It reads from a folder called `PDF_Collection` inside the repository (subfolders
are fine) and never touches the originals.

```bash
mkdir PDF_Collection
cp ~/Downloads/*.pdf PDF_Collection/
python classify_bills.py
```

```
PDFs: 897  unique: 669
  processed 50/669
  ...
bill        : 162
encrypted   : 22
not_bill    : 474
review      : 9
unreadable  : 2
```

Those are the numbers from my own laptop. Likely bills are copied to `Bills/`,
borderline files to `Bills_Review/`, and `classification.csv` lists every file with
its score and the words that decided it. Then extract:

```bash
python extract_invoices.py Bills Bills_Review -o bills.csv
```

## Total spend per vendor

`bills.csv` is a plain CSV, so a few lines of Python will summarise it. No extra
packages needed:

```python
import csv
from collections import defaultdict

totals = defaultdict(float)
with open("bills.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["is_invoice"] != "True" or not row["total"]:
            continue
        totals[(row["vendor"], row["currency"])] += float(row["total"])

for (vendor, currency), amount in sorted(totals.items(), key=lambda kv: -kv[1]):
    print(f"{vendor:<30} {currency} {amount:>12,.2f}")
```

Rows where `is_invoice` is `False` are skipped. Those are previews, quotations and
account statements, which would otherwise double count the real invoices.

## Only the bills addressed to you

The `customer` column holds whoever the bill is addressed to. That separates bills
you paid from invoices you sent:

```python
import csv

with open("bills.csv", newline="", encoding="utf-8") as f:
    mine = [r for r in csv.DictReader(f) if "your name" in r["customer"].lower()]

print(len(mine), "bills addressed to you")
```

## Copies of the same bill

Downloads often leave you with `invoice.pdf`, `invoice (1).pdf` and
`invoice-edited.pdf`. `classify_bills.py` drops exact duplicates, but files that
differ by a byte all get extracted. Before adding things up, keep one row per
vendor, invoice number and total:

```python
import csv

seen, unique = set(), []
with open("bills.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["vendor"].lower(), row["invoice_number"] or row["file"], row["total"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
```

On my own bills this took 170 rows down to 149, and the 128 real invoices down
to 108. Without it, my USD total came out almost three times too high.

## Choose the runtime yourself

The runtime is picked automatically, but you can force it. On an Apple Silicon Mac
this runs the CPU path, which is handy for checking that both give the same answer:

```bash
python extract_invoices.py samples/sample_gst_invoice.pdf --backend torch -o cpu.csv
```

`--backend mlx` only works on Apple Silicon.
