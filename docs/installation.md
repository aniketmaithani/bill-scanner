---
title: Installation
nav_order: 2
---

# Installation
{: .no_toc }

1. TOC
{:toc}

---

## What you need

- **Python** 3.10 to 3.14
- **Tesseract**, the OCR engine used for scanned PDFs and photos
- **Git with Git LFS**, because the model weights are stored with LFS
- **About 4 GB of disk** for the base model and packages
- **RAM:** 8 GB is comfortable. The CPU runtime holds the model in float32, which
  takes a little over 6 GB. With 4 GB, use the
  [llama.cpp setup](#windows-with-4-gb-of-ram) instead, which needs about 2 GB.

The code picks a runtime for your machine on its own:

| Machine | Runtime | Notes |
|---|---|---|
| Mac with Apple Silicon (M1 and later) | MLX, on the GPU | Fastest option |
| Linux, x86 | PyTorch, on the CPU | Uses every core |
| Windows 10 / 11, x86 | PyTorch, on the CPU | Same code path as Linux |
| Intel Mac | PyTorch, on the CPU | Should work, not tested |
| Any machine with 4 GB of RAM | llama.cpp, on the CPU | Opt in with `--backend llamacpp`, [see below](#windows-with-4-gb-of-ram) |

### What was tested

Every setup below was installed from a fresh clone and run on the two
[sample bills](quickstart.html). All three gave the same answers.

| Setup | Hardware | Install time | One bill |
|---|---|---|---|
| macOS 26, Python 3.12 | MacBook Pro, M3 Pro, 18 GB | n/a | 8 s |
| Ubuntu 24.04, Python 3.12 | 8 cores of a Xeon Platinum 8468 | about 1 minute | 19 s |
| Ubuntu 22.04, Python 3.10 | 8 cores of a Xeon Platinum 8468 | about 1 minute | 19 s |
| Windows Server 2025, Python 3.12 | GitHub Actions runner, 2 cores | about 1.5 minutes | 58 s |

"One bill" is the full run for a single bill, including loading the model, which
is the slow part. Each further bill in the same run adds roughly the same time
again minus the load.

Python 3.14 was tested separately, with Python 3.14.7 on macOS (the same M3 Pro),
on Ubuntu 24.04 and on Windows Server 2025 (both GitHub Actions runners). It
installed from ready-made packages everywhere, with nothing to compile, and gave
the same answers as 3.12.

---

## macOS (Apple Silicon)

You need [Homebrew](https://brew.sh).

```bash
brew install python@3.12 tesseract git-lfs
git lfs install

git clone https://github.com/aniketmaithani/bill-scanner.git
cd bill-scanner

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
```

That last command fetches the 3 GB base model from Hugging Face. The fine-tuned
adapter is already in the repository and gets applied on top of it.

---

## Ubuntu 22.04 / 24.04

```bash
sudo apt update
sudo apt install -y git git-lfs tesseract-ocr python3 python3-venv
git lfs install

git clone https://github.com/aniketmaithani/bill-scanner.git
cd bill-scanner

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
```

Install torch from the CPU index first, as shown. A plain `pip install torch` on
Linux pulls the CUDA build, which is several gigabytes you don't need for this.

Other distributions work the same way. Install Tesseract, Git LFS and Python with
your package manager and follow the rest as written.

---

## Windows 10 / 11

1. Install **Python** (3.10 to 3.14) from [python.org](https://www.python.org/downloads/).
   Tick "Add python.exe to PATH" on the first screen of the installer.
2. Install **[Git for Windows](https://git-scm.com/download/win)**. Git LFS comes
   with it.
3. Install **Tesseract**. Either of these works:

   ```powershell
   winget install --id UB-Mannheim.TesseractOCR
   ```

   ```powershell
   choco install tesseract
   ```

   Then add `C:\Program Files\Tesseract-OCR` to your PATH (Start, search for
   "environment variables", edit `Path`, add the folder). Open a new terminal and
   check that `tesseract --version` prints a version.

4. In PowerShell:

   ```powershell
   git lfs install
   git clone https://github.com/aniketmaithani/bill-scanner.git
   cd bill-scanner

   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
   ```

   If PowerShell refuses to run `Activate.ps1`, allow local scripts once with
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and try again.

On Windows, `pip install -r requirements.txt` installs the CPU build of torch by
default, so there is no extra step.

---

## Windows with 4 GB of RAM

The normal setup needs over 6 GB of RAM, which a 4 GB laptop doesn't have. On
those machines, run the model with [llama.cpp](https://github.com/ggml-org/llama.cpp)
instead. You download one ready-made file, the same fine-tuned model with the
adapter already merged in and stored at 8 bits, and llama.cpp's server runs it.
You don't need PyTorch, the 3 GB base model or Git LFS.

In testing, the llama.cpp server peaked at 1.9 GB of RAM after reading both sample
bills.

### 1. Install Python and Tesseract

Do steps 1 and 3 of the [Windows section above](#windows-10--11): Python, then
Tesseract added to your PATH. You can skip Git if you download the repository as a
ZIP from GitHub (Code, Download ZIP).

### 2. Get bill-scanner

In PowerShell:

```powershell
git clone https://github.com/aniketmaithani/bill-scanner.git
cd bill-scanner

py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-llamacpp.txt
```

`requirements-llamacpp.txt` is the short list: PDF reading, OCR and photo
support, and nothing else.

### 3. Download the model

The model is one 1.65 GB file on Hugging Face, with no account needed. Either
click
[invoice-qwen1.5b-q8_0.gguf](https://huggingface.co/aniketmaithani/bill-scanner-invoice-qwen1.5b-GGUF/resolve/main/invoice-qwen1.5b-q8_0.gguf?download=true)
and save it into the `bill-scanner` folder, or run this from inside it:

```powershell
curl.exe -L -o invoice-qwen1.5b-q8_0.gguf "https://huggingface.co/aniketmaithani/bill-scanner-invoice-qwen1.5b-GGUF/resolve/main/invoice-qwen1.5b-q8_0.gguf?download=true"
```

To check the download, run `Get-FileHash invoice-qwen1.5b-q8_0.gguf`. The hash
should start with `6F6B9C16BB46A86B`.

### 4. Download llama.cpp

From the [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases),
download the file named `llama-b<number>-bin-win-cpu-x64.zip` and unzip it, for
example to `C:\llama.cpp`. Use `win-cpu-arm64` instead on an ARM laptop. It's a
ready-made program, so there is nothing to build.

### 5. Start the model server

Open a PowerShell window in the `bill-scanner` folder and leave it running:

```powershell
C:\llama.cpp\llama-server.exe -m invoice-qwen1.5b-q8_0.gguf -c 4096 --port 8080
```

It's ready when it prints `listening on http://127.0.0.1:8080`.
`-c 4096` is enough room for the longest bill bill-scanner sends.

### 6. Read a bill

In a second PowerShell window:

```powershell
cd bill-scanner
.venv\Scripts\Activate.ps1
python extract_invoices.py samples\sample_gst_invoice.pdf samples\sample_cafe_receipt.jpg --backend llamacpp -o samples.csv
```

You should see:

```
runtime: llamacpp
[1/2] subtotal snapped to printed amount; tax derived as total - subtotal sample_gst_invoice.pdf
[2/2] fields filled one at a time; tax derived as total - subtotal sample_cafe_receipt.jpg

wrote 2 rows to samples.csv
```

Open `samples.csv`. The GST invoice row reads:

| Column | Value |
|---|---|
| customer | Demo Customer LLP |
| vendor | Acme Cloud Services |
| invoice_number | ACS/2026/0042 |
| invoice_date | 2026-09-15 |
| currency | INR |
| subtotal | 125000 |
| tax | 22500 |
| total | 147500 |
| gstin | 29ABCDE1234F1Z5 |
| category | Cloud & Hosting |

Every value matches the PDF and the [Quickstart](quickstart.html#read-the-result).
Only the free-text `description` is worded a little differently. For your own
bills, point it at a folder:

```powershell
python extract_invoices.py C:\Users\you\Documents\bills --backend llamacpp -o bills.csv
```

If the server runs on another port or machine, set `LLAMA_SERVER_URL`, for example
`$env:LLAMA_SERVER_URL = "http://127.0.0.1:9000"`, before running the command.

### How it compares

The 8-bit model is not identical to the full model. On the GST invoice it gives
the same amounts as the full-precision model before and after the `check` step. On
the café receipt it gets the money right (500, 25, 525) but, like the full model,
gets the names wrong, and it also gets the currency wrong. It hasn't been scored on
the 26 held-out bills yet, so the accuracy figures on the [About page](index.html)
belong to the full model.

The test machine was Ubuntu, not Windows, using 2 threads of a server CPU to stand
in for a small laptop. The GST invoice took 34 seconds with the model already
loaded. A real 4 GB laptop will differ.

---

## Check that it works

```bash
python extract_invoices.py samples/sample_gst_invoice.pdf -o check.csv
```

You should see `runtime: mlx` on an Apple Silicon Mac or `runtime: torch`
everywhere else, followed by one line for the bill. Open `check.csv` and the
total should read `147500`. The [Quickstart](quickstart.html) explains the output.

---

## Troubleshooting

**`TesseractNotFoundError`**
: Tesseract is installed but not on your PATH. On Windows, add
  `C:\Program Files\Tesseract-OCR` to PATH and open a new terminal.

**An error loading `adapters.safetensors`, or the file is only a few hundred bytes**
: Git LFS wasn't set up before cloning, so you got a pointer file instead of the
  weights. Run `git lfs install` and then `git lfs pull` inside the repository.

**`No such file or directory: models/Qwen2.5-1.5B-Instruct`**
: The base model hasn't been downloaded yet. Run the `hf download` command from
  your platform's section, from inside the repository folder.

**Very slow on Linux inside Docker or a VM with a CPU limit**
: A limit like `docker run --cpus 8` still shows PyTorch every core on the host,
  so it starts too many threads and they fight over the limit. In testing that
  made a 19 second bill take almost 7 minutes. Pin real cores instead
  (`--cpuset-cpus 0-7`) or set `OMP_NUM_THREADS` to the number of cores you have.

**A photo or scanned PDF comes back `unreadable`**
: The line above the result says why, for example
  `warning: could not read receipt.jpg: UnidentifiedImageError: ...`. A broken or
  truncated file is the usual reason. In one Windows test run a photo that reads
  fine was reported unreadable once and worked on the next run, which points at
  something briefly locking Tesseract's temporary file, such as an antivirus scan.
  Running the file again is worth a try.

**`no llama-server at http://127.0.0.1:8080`**
: `--backend llamacpp` couldn't reach the model server. Start `llama-server.exe` as
  in [step 5](#start-the-model-server) and wait for `listening on`. If it
  runs on another port, set `LLAMA_SERVER_URL` to match.

**`DLL load failed` on Windows**
: You are on the MLX code path, which only applies to Apple Silicon. Make sure you
  installed from `requirements.txt` in a fresh virtual environment, or pass
  `--backend torch`.
