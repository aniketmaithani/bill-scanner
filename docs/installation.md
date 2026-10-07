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
  takes a little over 6 GB.

The code picks a runtime for your machine on its own:

| Machine | Runtime | Notes |
|---|---|---|
| Mac with Apple Silicon (M1 and later) | MLX, on the GPU | Fastest option |
| Linux, x86 | PyTorch, on the CPU | Uses every core |
| Windows 10 / 11, x86 | PyTorch, on the CPU | Same code path as Linux |
| Intel Mac | PyTorch, on the CPU | Should work, not tested |

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

**`DLL load failed` on Windows**
: You are on the MLX code path, which only applies to Apple Silicon. Make sure you
  installed from `requirements.txt` in a fresh virtual environment, or pass
  `--backend torch`.
