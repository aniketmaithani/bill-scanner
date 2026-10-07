---
title: Quickstart
nav_order: 3
---

# Quickstart
{: .no_toc }

This assumes you have finished [Installation](installation.html) and your virtual
environment is active (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1`
on Windows).

1. TOC
{:toc}

---

## Run it on the sample bills

The repository ships two made-up bills in `samples/`:

- `sample_gst_invoice.pdf`, a typical Indian GST invoice with CGST and SGST
- `sample_cafe_receipt.jpg`, a phone photo of a café receipt, slightly tilted and grainy

```bash
python extract_invoices.py samples/sample_gst_invoice.pdf samples/sample_cafe_receipt.jpg -o samples.csv
```

You'll see something like this:

```
runtime: mlx
[1/2] subtotal snapped to printed amount; tax derived as total - subtotal sample_gst_invoice.pdf
[2/2] fields filled one at a time; tax derived as total - subtotal sample_cafe_receipt.jpg

wrote 2 rows to samples.csv
```

`runtime` says `torch` on Linux and Windows. The first run takes a little longer
while the model loads.

## Read the result

`samples.csv` opens in any spreadsheet app. Here is the GST invoice row:

| Column | Value |
|---|---|
| is_invoice | True |
| customer | Demo Customer LLP |
| vendor | Acme Cloud Services |
| invoice_number | ACS/2026/0042 |
| invoice_date | 2026-09-15 |
| currency | INR |
| subtotal | 125000 |
| tax | 22500 |
| total | 147500 |
| gstin | 29ABCDE1234F1Z5 |
| description | Kubernetes cluster, object storage, and premium support plan |
| category | Cloud & Hosting |

Every value matches the PDF. The `check` column for this row reads:

```
subtotal snapped to printed amount; tax derived as total - subtotal
```

That means the safety nets did some work. The model wrote the subtotal as `12500`,
one zero short. Exactly one amount printed on the bill, `1,25,000.00`, is a single
digit away, so the extractor used that. The model also reported only one GST half,
so the tax was worked out as total minus subtotal: 147500 - 125000 = 22500.

The café receipt is a weaker result, and the [examples page](examples.html#a-cafe-receipt-photo)
goes through it honestly.

## What the check column means

| Message | What happened | What to do |
|---|---|---|
| `ok` | Nothing needed fixing | Nothing |
| `subtotal snapped to printed amount` (or `total`) | The model's number was one digit off a printed amount and was replaced by it | Nothing, usually right |
| `tax derived as total - subtotal` | Subtotal and total are both printed on the bill, so tax was calculated from them | Nothing, usually right |
| `fields filled one at a time` | The model's first answer wasn't valid JSON, so each field was asked for separately | Look at this row |
| `total not found in text` | The total isn't printed anywhere on the bill | Check the total |
| `subtotal+tax != total` | The three amounts don't add up | Check the amounts |
| `invalid gstin dropped` | The GSTIN wasn't a valid format and was cleared | Fill it in if you need it |
| `encrypted` | The PDF is password protected | Unlock it and run again |

## Run it on your own bills

One file:

```bash
python extract_invoices.py ~/Downloads/invoice.pdf -o invoice.csv
```

A whole folder, PDFs and photos mixed:

```bash
python extract_invoices.py ~/Documents/bills -o bills.csv
```

Files and folders together:

```bash
python extract_invoices.py march.pdf receipt.jpg ~/Documents/bills -o all.csv
```

Supported formats: PDF, JPG, PNG, HEIC, WEBP, TIFF and AVIF. The output file is
overwritten if it already exists.

## How long it takes

From the test runs in [Installation](installation.html#what-was-tested):

| Machine | First bill, including model load | Each extra bill |
|---|---|---|
| MacBook Pro, M3 Pro | 8 s | about 5 s |
| Linux PC, 8 cores | 19 s | about 15 to 20 s |
| Windows, 2 cores | 58 s | about 50 to 60 s |

A scanned PDF or a photo adds a second or two for OCR. A bill that needs the
`fields filled one at a time` fallback takes longer, between about 1.2 and 3 times
as long in testing.

## Next steps

- [Examples](examples.html) for sorting a big folder first and totalling the results
- [Training](training.html) to see how the model was built, or to retrain it on your own bills
