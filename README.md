# bill-scanner

**Find the bills in a pile of PDFs and photos, then pull out vendor, customer,
date, subtotal, tax, total, GSTIN and category, fully offline.**

There are no API calls at inference time. Text comes from the PDF itself or from
Tesseract OCR. Fields are extracted by **Qwen2.5-1.5B-Instruct fine-tuned with
LoRA** (this repo ships the adapter). The model was trained on labels distilled
from a large teacher model, and it is tuned for Indian GST invoices as well as
international SaaS and cloud bills.

```
PDFs / photos ──► classify_bills.py ──► Bills/ ──► extract_invoices.py ──► invoices.csv
                  (rules + OCR, no LLM)            (Qwen 1.5B + LoRA, offline)
```

## Results

Measured on 26 held-out bills that were never used in training:

| Field | Base Qwen2.5-1.5B | **Fine-tuned (this adapter)** |
|---|---|---|
| Valid JSON | 100% | **100%** |
| Is it an invoice? | 77% | **96%** |
| Customer (bill-to) | 73% | **88%** |
| Vendor (seller) | 77% | **100%** |
| Invoice number | 85% | **88%** |
| Invoice date | 88% | **100%** |
| Currency | 92% | 92% |
| Subtotal | 73% | **92%** |
| Tax | **77%** | 73%* |
| Total | 85% | **92%** |
| GSTIN | 73% | **96%** |
| Category | 35% | **88%** |
| **Every field right on a document** | 0% | **46%** |
| Seconds per bill (M3 Pro) | 4.4 | 4.9 |

\* Tax is the weakest field: the model often reports a single GST half (CGST
only). `extract_invoices.py` fixes this in code: it sets tax = total − subtotal
whenever both of those amounts are actually printed in the document. The table
above shows the raw model output, before that fix.

The test set is small (26 documents, so one document ≈ 4 points). See
[docs/TRAINING.md](docs/TRAINING.md) for the round-by-round history and what
each change fixed.

## Quick start (Apple Silicon)

```bash
brew install tesseract
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Base model (~3 GB). The LoRA adapter in adapters/ is applied on top.
.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct

# Extract every PDF / photo in a folder into a CSV
.venv/bin/python extract_invoices.py path/to/bills -o invoices.csv
```

Supported inputs: `.pdf`, `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.webp`, `.tif`, `.avif`.

### Output columns

| Column | Meaning |
|---|---|
| `is_invoice` | `false` for previews, estimates, quotations and account statements |
| `customer` | who the bill is addressed to |
| `vendor` | who issued it (short brand name) |
| `invoice_number`, `invoice_date` | as printed; the date is ISO `YYYY-MM-DD` |
| `currency` | ISO 4217 code |
| `subtotal`, `tax`, `total` | plain numbers; tax is CGST + SGST + IGST + VAT |
| `gstin` | the vendor's GSTIN, domestic or OIDAR format (e.g. `9926USA29037OS3`) |
| `description`, `category` | what was billed; one of 14 fixed categories |
| `check` | `ok`, or the reason to double-check this row (see below) |

`check` flags rows you should verify by hand:
- `total not found in text`: the predicted total isn't printed anywhere in the document.
- `subtotal+tax != total`: the three amounts don't add up.
- `invalid gstin dropped`: the predicted GSTIN wasn't a valid format, so it was cleared.
- `model output not JSON`: the model's answer couldn't be read.
- `tax derived as total - subtotal`: the arithmetic fix described above was applied.

## Pipeline

| Step | Script | Uses an LLM? |
|---|---|---|
| 1. Find the bills | `classify_bills.py`: removes duplicates by SHA-256, OCRs scanned pages, scores keywords, copies bills to `Bills/` and unclear files to `Bills_Review/` | No |
| 2. Label (training only) | `label_with_gpt.py`: a teacher model fills the JSON schema; sensitive documents are never sent | Teacher (OpenAI) |
| 3. Check labels | `audit_labels.py`: is the total printed in the text? does the invoice number appear? do the amounts add up? is the date valid? | No |
| 4. Build dataset | `build_dataset.py`: chat-format JSONL, a fixed test set, duplicates kept in the same split, at most 16 examples per vendor | No |
| 5. Train | `mlx_lm lora -c lora_config.yaml` | – |
| 6. Evaluate | `evaluate.py`: field accuracy, base vs fine-tuned | No |
| 7. Extract | `extract_invoices.py`: offline extraction with grounding checks | Local Qwen only |

`invoice_schema.py` is the single source of truth for the prompt, the JSON schema,
the categories and the text preprocessing. The teacher, training and inference all
import it, so they always see identical input.

## Retraining on your own bills

```bash
# Put your PDFs/photos under PDF_Collection/ (subfolders are fine)
.venv/bin/python classify_bills.py
cp .env.example .env    # add OPENAI_KEY
.venv/bin/python label_with_gpt.py --limit 1   # smoke test first
.venv/bin/python label_with_gpt.py
.venv/bin/python audit_labels.py               # review labels_review.csv
.venv/bin/python build_dataset.py

# Mac
.venv/bin/python -m mlx_lm lora -c lora_config.yaml
# Linux + NVIDIA: pip install "mlx[cuda13]" mlx-lm, then
MLX_CUDA_GRAPH_CACHE_SIZE=8000 python -m mlx_lm lora -c lora_config.yaml

.venv/bin/python evaluate.py
```

Training takes about 1 minute on an H100 and an estimated 20–40 minutes on an 18 GB M3 Pro.

- **Durable corrections:** to fix a teacher label so it survives re-labelling, add it to
  `label_corrections.json` as `{"<sha256>": {"field": value}}`.
- **Fixed test set:** after the first build, `data/test_files.csv` pins the test
  documents, so later rounds stay comparable.

## Privacy

- **Extraction is fully local.** Nothing leaves the machine.
- **Teacher labelling** sends only the extracted *text* of documents classified as
  bills, never the files. Anything matching the sensitive filter (tax statements,
  bank and card statements, Aadhaar, passports) is skipped automatically.
- `.gitignore` excludes every document, the OCR text, labels, datasets, outputs and
  `.env`. **This repo contains no training data.**
- The published adapter was trained on the author's own bills. LoRA weights can
  memorise fragments of the outputs they were trained on (vendor and customer names,
  invoice numbers, amounts). **If you retrain on your own private bills, keep your
  adapter private.**

## Limitations

- **Handwriting:** Tesseract cannot read handwritten bills, so amounts on
  handwritten estimates come out empty or wrong.
- **Small test set:** 26 documents. Expect noise of about ±4 points per document.
- **Narrow training data:** about 110 examples from one person's bills. Vendors and
  layouts unlike these will be less accurate, so retrain with your own bills for
  the best results.
- **Logo-only vendors:** when the vendor's name appears only in a logo image,
  the vendor field is empty.

## Credits

- Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache 2.0).
- Training and inference: [MLX / mlx-lm](https://github.com/ml-explore/mlx-lm).
- OCR: [Tesseract](https://github.com/tesseract-ocr/tesseract) and [PyMuPDF](https://pymupdf.readthedocs.io).
