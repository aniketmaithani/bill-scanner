# bill-scanner

Point it at a folder of PDFs and phone photos. It finds the bills and pulls out
vendor, customer, invoice number, date, subtotal, tax, total, GSTIN and category
into a CSV. Everything runs on your own machine.

**Documentation: [aniketmaithani.net/bill-scanner](https://aniketmaithani.net/bill-scanner/)**

Under the hood it's Tesseract for OCR and a small model, Qwen2.5-1.5B-Instruct,
fine-tuned with LoRA on real bills. It handles Indian GST invoices (lakh-formatted
amounts, CGST and SGST, GSTINs) as well as international SaaS and cloud invoices.

## Try it

```bash
git lfs install
git clone https://github.com/aniketmaithani/bill-scanner.git
cd bill-scanner
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct

python extract_invoices.py samples/sample_gst_invoice.pdf -o result.csv
```

That's the Mac version. Linux needs the CPU build of torch first, and Windows
needs Tesseract added to PATH. The [installation guide](https://aniketmaithani.net/bill-scanner/installation.html)
has the exact steps for each.

Only 4 GB of RAM? Download the
[8-bit model](https://huggingface.co/aniketmaithani/bill-scanner-invoice-qwen1.5b-GGUF)
and run it with llama.cpp instead. It needs about 2 GB. The guide has a
[Windows walkthrough](https://aniketmaithani.net/bill-scanner/installation.html#windows-with-4-gb-of-ram).

## Where it runs

| Machine | Runtime | One bill, including model load |
|---|---|---|
| Mac with Apple Silicon | MLX on the GPU | 8 s on an M3 Pro |
| Linux | PyTorch on the CPU | 19 s on 8 cores |
| Windows | PyTorch on the CPU | 58 s on 2 cores |

Tested from a fresh clone on macOS 26, Ubuntu 22.04 and 24.04, and Windows Server
2025. All of them give the same answers.

## How accurate

On 26 bills held back from training, the fine-tuned model gets the vendor and date
right on every one, the GSTIN on 96%, and subtotal and total on 92%. The untrained
model manages 73% to 88% on the same fields. Every row also gets a `check` column
that fixes or flags amounts that don't match what's printed on the bill.

It's weaker on shop till receipts and can't read handwriting. The
[About page](https://aniketmaithani.net/bill-scanner/) has the full numbers
and the known gaps.

## What's in the repository

| Path | What it does |
|---|---|
| `extract_invoices.py` | Reads bills and writes the CSV. This is the one you'll use. |
| `classify_bills.py` | Finds the bills in a big mixed folder of PDFs |
| `backends.py` | Runs the model with MLX, PyTorch or a llama.cpp server |
| `invoice_schema.py` | The prompt, fields, categories and text cleanup, shared by everything |
| `label_with_gpt.py`, `audit_labels.py`, `build_dataset.py`, `evaluate.py` | Labelling, checking, dataset building and scoring, for retraining |
| `lora_config.yaml` | Training settings |
| `adapters/invoice-qwen1.5b/` | The fine-tuned LoRA weights (Git LFS) |
| `samples/` | Two made-up bills to try it on, and the script that draws them |
| `docs/` | The documentation site |

## Privacy

Extraction never leaves your machine. The repository has no real bills, OCR text
or labels in it, only code, the adapter weights and two made-up samples. The
adapter was trained on my own bills; if you retrain on yours, keep your adapter
private.

## Credits

Base model [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
(Apache 2.0). Runs on [MLX](https://github.com/ml-explore/mlx-lm) and
[PyTorch](https://pytorch.org). OCR by [Tesseract](https://github.com/tesseract-ocr/tesseract)
and [PyMuPDF](https://pymupdf.readthedocs.io).
