---
title: About
nav_order: 1
permalink: /
---

# bill-scanner

Point it at a folder of PDFs and phone photos. It works out which ones are bills,
then pulls out who billed whom, when, for what, and how much, into a CSV. All of it
runs on your own computer. Nothing is uploaded anywhere.

[Install it](installation.html){: .btn .btn-primary }
[Try it in two minutes](quickstart.html){: .btn }

---

## Why this exists

I had a little under nine hundred PDFs scattered across my laptop. Most were noise:
manuals, papers, exported slides. Somewhere in that pile were about 170 bills: cloud
hosting, SaaS subscriptions, hotel stays, GST invoices from Indian vendors, and the
invoices I send to my own clients.

I wanted all of them in one spreadsheet. I did not want to upload my bank-adjacent
paperwork to a hosted API to get there. So the extraction runs locally, on a small
language model that I fine-tuned for exactly this job.

## What it does

**1. Sorts the pile.** `classify_bills.py` reads every PDF (and OCRs the scanned ones),
drops duplicates, and scores each file on words like "Tax Invoice", "GSTIN" and
"Amount Due". Likely bills are copied to `Bills/`, unclear ones to `Bills_Review/`.
Your originals are never moved or changed.

**2. Reads each bill.** `extract_invoices.py` gives the text to the fine-tuned model
and gets back these fields:

| Field | Example |
|---|---|
| `is_invoice` | `True` (previews, quotes and statements come back `False`) |
| `customer`, `vendor` | `Demo Customer LLP`, `Acme Cloud Services` |
| `invoice_number`, `invoice_date` | `ACS/2026/0042`, `2026-09-15` |
| `currency`, `subtotal`, `tax`, `total` | `INR`, `125000`, `22500`, `147500` |
| `gstin` | `29ABCDE1234F1Z5` |
| `description`, `category` | `Kubernetes cluster, object storage...`, `Cloud & Hosting` |

**3. Checks its own answers.** Every row has a `check` column. The extractor confirms
the total is actually printed on the bill, fixes amounts where the model dropped a
digit, works out the tax when the model only reported half of the GST, and throws
away GSTINs that are not in a valid format. If something still looks off, the row
tells you so instead of quietly being wrong.

## How well it works

Measured on 26 of my bills that the model never saw during training:

| Field | Untrained Qwen2.5-1.5B | Fine-tuned |
|---|---|---|
| Vendor | 77% | **100%** |
| Invoice date | 88% | **100%** |
| Is it an invoice | 77% | **96%** |
| GSTIN | 73% | **96%** |
| Subtotal | 73% | **92%** |
| Total | 85% | **92%** |
| Customer | 73% | **88%** |
| Category | 35% | **88%** |
| Tax | **77%** | 73% |
| Every field right on a bill | 0% | **46%** |

Two things to keep in mind. The test set is small, so one bill moves a number by
about four points. And the table shows the raw model; the checks described above
fix most of the tax misses before anything reaches your CSV.

It runs on a Mac with Apple Silicon (about 8 seconds a bill), and on Linux and
Windows on a plain CPU (about 20 seconds a bill on an 8 core machine). See
[Installation](installation.html) for what was tested.

## Where it falls short

- **Shop receipts.** The model learned from invoices, not till receipts. On the café
  receipt in the [examples](examples.html#a-cafe-receipt-photo) it gets the amounts
  right and the names wrong.
- **Handwriting.** Tesseract cannot read it, so handwritten bills come out mostly empty.
- **Logos.** When the vendor's name only appears inside a logo image, the vendor
  field is empty.
- **Narrow training set.** About 110 bills from one person. Bills that look very
  different from mine will be less accurate. [Retraining](training.html#retrain-on-your-own-bills)
  on your own bills is the fix.

## How the model was made

A larger model (OpenAI's GPT 6.1 Sol) labelled my bills once. Scripts checked
every label against the bill text, I corrected the handful that were wrong, and
that became the training set for a LoRA fine-tune of Qwen2.5-1.5B-Instruct. It
took four rounds to get here, each fixing a specific failure. The
[training page](training.html) walks through all of them.

## Privacy

- Extraction is completely offline.
- The teacher model was only used once, to label training data. It saw extracted
  text from bills, never the files, and never anything that looked like a bank
  statement, tax form or ID.
- The repository contains code, the adapter weights and two made-up sample bills.
  No real bills, OCR text or labels.
