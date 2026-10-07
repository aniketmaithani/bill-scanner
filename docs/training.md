# Training history

How the shipped adapter (`adapters/invoice-qwen1.5b`) was produced, round by round,
and what each round taught us. Every round was scored on the **same 26 held-out
documents**, which were never trained on. On a set that small one document is
worth about 4 percentage points, so treat small differences as noise.

## Setup

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0) |
| Method | LoRA, rank 16, scale 20, dropout 0.05, all layers, prompt masked |
| Steps | 240 at batch 2 × 2 accumulation, learning rate 1e-4, AdamW |
| Trainable parameters | 18.5M (1.2% of the model) |
| Teacher | OpenAI `gpt-6.1-sol`, strict JSON schema output |
| Corpus | 171 labelled bills: ~110 train, ~15 validation, 26 test |
| Hardware | 1× H100 80 GB via `mlx[cuda13]`, about 1 minute per run (peak 20–24 GB) |

The same config also trains on an Apple Silicon Mac. An 18 GB M3 Pro is estimated
at 20–40 minutes and is very tight on memory.

## Rounds

### Round 1: baseline fine-tune
Teacher labels on raw OCR text, 119 training examples.

- Invoice number went from 77% to 100%, GSTIN from 62% to 92%, category from 58% to 88%.
- **Problem:** Indian lakh-grouped numbers lost a digit (e.g. `4,12,345.00` came out as `41234`).
- **Problem:** the teacher labelled DigitalOcean "Invoice Previews" inconsistently.

### Round 2: clean numbers and stricter rules
- `normalize_numbers()` strips thousands separators (western and lakh) before
  the text reaches either model. The digit-drop errors disappeared.
- `is_invoice` is now decided by how the document titles itself. "Tax Invoice /
  Proforma" counts as an invoice; "Statement", "Preview", "Estimate" and
  "Quotation" do not.
- "total" means *this document's charges*, not a running balance that includes arrears.
- A finding: in most DigitalOcean previews the word "DigitalOcean" never appears in
  the text (it is only in the logo), so a null vendor label is correct there.

### Round 3: more data and better OCR
- Added a batch of new bills, including a photographed one.
- OCR fallback: when Tesseract's automatic layout returns under 600 characters,
  retry with `--psm 6` and keep the longer result. This recovered a hotel bill's
  room-rent table and improved 46 other scanned pages.
- Fields correct on every document rose from 35% to 50%.
- **Problem:** the model started naming the *recipient* as the vendor.

### Round 4 / 4b: separate customer and vendor (shipped)
Root cause: about 25% of the corpus is invoices the owner issued *themselves*, so
one person's name was the most common vendor and became the model's default.

- New `customer` field, emitted **before** `vendor` in the JSON, so the model
  names the addressee first.
- Vendor = the seller (From / Supplier / the business named in the header),
  never the Bill-To party. Use the short brand name.
- Training capped at N examples per vendor so no vendor dominates.
  - **N = 8 (round 4) was unstable:** with only 86 training examples the model
    looped (`"gstin": "000000…"`) and dropped from 100% to 92% valid JSON.
  - **N = 16 (round 4b, shipped)** kept 108 examples. Vendor accuracy reached 100%.

| Field | Base | R3 | R4 (cap 8) | **R4b (shipped)** |
|---|---|---|---|---|
| Valid JSON | 100% | 96% | 92% | **100%** |
| is_invoice | 77% | 92% | 81% | **96%** |
| customer | 73% | n/a | 77% | **88%** |
| vendor | 77% | 62% | 85% | **100%** |
| invoice_date | 88% | 88% | 92% | **100%** |
| subtotal | 73% | 81% | 81% | **92%** |
| total | 85% | 92% | 85% | **92%** |
| gstin | 73% | 92% | 88% | **96%** |
| tax | 77% | 81% | 85% | 73% |

All columns are scored against the round-4 labels. "Base" is the untrained model
with the same prompt and number normalisation. Round 3 never learned the
`customer` field.

## Lessons

- **Normalise the text, not just the labels.** Small models copy digits reliably
  but not comma-grouped numbers.
- **Write labelling rules from the document's own words** (its title), not from
  your intent. Vague rules like "statements are not invoices" swept up real AWS
  invoices that merely contain an account-summary section.
- **Check teacher labels before blaming the student.** Several "errors" were
  label inconsistencies; grounding checks (`audit_labels.py`) catch most of them.
- **Class balance matters more than size, up to a point.** Capping the dominant
  vendor fixed vendor confusion, but capping too hard caused degenerate loops.
- **Leave arithmetic to code.** The model often reports one GST half (CGST only);
  `extract_invoices.py` derives tax = total − subtotal when both are grounded in
  the text.
- **MLX on CUDA** needs `MLX_CUDA_GRAPH_CACHE_SIZE=8000` (variable sequence lengths
  otherwise crash with "Cache thrashing"). With it, MLX trains at about 1,300
  tokens/s on an H100.
- **Foreign online-service suppliers** registered in India have GSTINs in a
  different format, e.g. `9926USA29037OS3`. Validators must accept it.
