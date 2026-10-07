---
title: Training
nav_order: 5
---

# How the model was trained
{: .no_toc }

The adapter in `adapters/invoice-qwen1.5b` came out of four rounds of training.
Each round fixed one specific way the previous model was failing. This page goes
through what changed and why, and then how to train your own.

1. TOC
{:toc}

---

## Setup

| | |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0) |
| Method | LoRA: rank 16, scale 20, dropout 0.05, all layers, loss on the answer only |
| Steps | 240, batch size 2 with 2 accumulation steps, learning rate 1e-4, AdamW |
| Trainable parameters | 18.5 million, about 1.2% of the model |
| Labels | OpenAI GPT 6.1 Sol filling a strict JSON schema, then checked by script |
| Data | 171 labelled bills: about 110 to train on, 15 to validate, 26 held back for testing |
| Hardware | One H100 80 GB through `mlx[cuda13]`, about a minute per run |

The same config also trains on an Apple Silicon Mac. On an 18 GB M3 Pro I'd expect
20 to 40 minutes, and memory gets tight.

Every round was scored on the same 26 test bills, which the model never trained
on. With a set that small, one bill moves a number by about four points, so read
small differences as noise.

## Round 1: a first fine-tune

The teacher labelled the raw OCR text and the model trained on 119 examples.
Invoice numbers went from 77% to 100%, GSTINs from 62% to 92%, categories from
58% to 88%.

Two problems showed up:

- Indian amounts lost a digit. An amount written like `4,12,345.00` would come
  back as `41234`. The comma grouping of lakhs breaks the number into pieces the
  model copies badly.
- The teacher was inconsistent about DigitalOcean "Invoice Previews", calling some
  of them invoices and some not.

## Round 2: cleaner input, clearer rules

- **Numbers lose their commas before either model sees them.** `normalize_numbers()`
  turns `4,12,345.00` and `1,234,567.89` into plain digits. The dropped digit
  problem mostly went away.
- **A document is an invoice if it calls itself one.** "Tax Invoice / Proforma"
  counts. "Statement", "Preview", "Estimate" and "Quotation" do not. My first
  version of this rule said "statements are not invoices", which wrongly caught
  AWS invoices that simply contain an account summary section.
- **Total means this bill's charges**, not a running balance that includes old dues.

One surprise: in most DigitalOcean previews the word "DigitalOcean" doesn't appear
in the text at all. It's only in the logo. So a missing vendor is the right answer
there, and the earlier model had been guessing.

## Round 3: more bills, better OCR

- Added a new batch of bills, including a phone photo and a scanned hotel bill.
- **OCR retries.** When Tesseract's automatic page layout returns less than 600
  characters, it reads the page again treating it as a single block
  (`--psm 6`) and keeps the longer result. That recovered the hotel bill's room
  rent table, and improved 46 other scanned pages.

Bills with every field right went from 35% to 50%. But a new problem appeared: the
model started naming the person a bill was addressed to as the vendor.

## Round 4: telling customer and vendor apart

About a quarter of my bills are invoices I sent to my own clients. That made my
name the most common vendor in the training data, and the model learned to fall
back on it.

- **A new `customer` field**, written before `vendor` in the JSON. Once the model
  has written down who the bill is addressed to, it stops reusing that name as
  the seller.
- **Vendor is the seller**: the From or Supplier line, or the business named in the
  header next to its own GSTIN. Never the Bill To party. Short brand names only,
  so "Amazon Web Services" rather than "AWS India Pvt Ltd".
- **No vendor may dominate training.** I capped each vendor at 8 examples first.
  That left only 86 examples and the model became unstable: it got stuck writing
  `"gstin": "000000..."` and only 92% of answers were valid JSON. Raising the cap
  to 16 (round 4b, the one that ships) kept 108 examples and fixed it.

| Field | Untrained | Round 3 | Round 4, cap 8 | Round 4b, cap 16 |
|---|---|---|---|---|
| Valid JSON | 100% | 96% | 92% | **100%** |
| Is it an invoice | 77% | 92% | 81% | **96%** |
| Customer | 73% | n/a | 77% | **88%** |
| Vendor | 77% | 62% | 85% | **100%** |
| Invoice date | 88% | 88% | 92% | **100%** |
| Subtotal | 73% | 81% | 81% | **92%** |
| Total | 85% | 92% | 85% | **92%** |
| GSTIN | 73% | 92% | 88% | **96%** |
| Tax | 77% | 81% | 85% | 73% |

All columns are scored against the round 4 labels. "Untrained" is the base model
with the same prompt and number cleanup. Round 3 never learned the customer field.

## What I learned

- **Clean the input, not just the labels.** A small model copies plain digits
  reliably and comma-grouped numbers badly.
- **Write labelling rules from the document's own words.** Rules about what I
  meant ("statements aren't invoices") went wrong. Rules about what the page says
  ("it calls itself a statement") didn't.
- **Check the teacher before blaming the student.** Several of the model's
  "mistakes" were inconsistent labels. `audit_labels.py` checks every label
  against the bill text and catches most of them.
- **Balance matters more than size, up to a point.** Capping the dominant vendor
  fixed the vendor mix-ups. Capping too hard broke the model.
- **Let code do the arithmetic.** The model often reports only the CGST half of
  the tax. Calculating tax as total minus subtotal, when both are printed on the
  bill, is more reliable than teaching it to add.
- **MLX on CUDA needs `MLX_CUDA_GRAPH_CACHE_SIZE=8000`.** Without it, training on
  examples of different lengths crashes with "Cache thrashing". With it, MLX
  processes about 1,300 tokens per second on an H100.
- **GSTINs come in two shapes.** Foreign online services registered in India get
  numbers like `9926USA29037OS3`, and a validator that only knows the domestic
  format throws them away.

---

## Retrain on your own bills

Training is worth it if your bills look different from mine: mostly receipts, a
different country's tax format, or vendors I never used. Labelling needs an OpenAI
API key. Training needs an Apple Silicon Mac or a Linux machine with an NVIDIA GPU.

**1. Collect and sort.** Put your PDFs and photos under `PDF_Collection/` and run:

```bash
python classify_bills.py
```

**2. Label with a teacher model.** Copy `.env.example` to `.env` and add your key.
Try one bill first:

```bash
python label_with_gpt.py --limit 1
python label_with_gpt.py
```

Only bills are sent, as extracted text. Anything that looks like a bank statement,
tax statement, Aadhaar or passport is skipped automatically. To label a folder
regardless of its keyword score, pass `--include-dir <folder under PDF_Collection>`.

**3. Check the labels.**

```bash
python audit_labels.py
```

This writes `labels_review.csv`, with the rows that failed a check at the top.
To make a correction stick, even if you relabel later, add it to
`label_corrections.json`:

```json
{"<sha256 of the file>": {"total": 45657}}
```

**4. Build the dataset.**

```bash
python build_dataset.py
```

After the first build, `data/test_files.csv` pins the test bills, so later rounds
are scored on the same set.

**5. Train.** On a Mac:

```bash
python -m mlx_lm lora -c lora_config.yaml
```

On Linux with an NVIDIA GPU, install `mlx[cuda13]` and `mlx-lm` instead of the
default requirements, then:

```bash
MLX_CUDA_GRAPH_CACHE_SIZE=8000 python -m mlx_lm lora -c lora_config.yaml
```

**6. Score it.**

```bash
python evaluate.py
```

This prints the untrained and fine-tuned accuracy side by side for every field.
The new adapter lands in `adapters/invoice-qwen1.5b`, and `extract_invoices.py`
uses it straight away, on every platform.

A word on privacy: LoRA weights can remember pieces of what they were trained to
output, like names and invoice numbers. If you train on private bills, keep your
adapter private.
