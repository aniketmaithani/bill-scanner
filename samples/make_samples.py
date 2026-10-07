#!/usr/bin/env python3
"""
Generate the synthetic sample bills used by the Quickstart.

All names, numbers and GSTINs are fictional.

    python samples/make_samples.py
        -> samples/sample_gst_invoice.pdf   (text PDF, Indian GST invoice)
        -> samples/sample_cafe_receipt.jpg  (phone-photo style, needs OCR)
"""

import io
import random
from pathlib import Path

import pymupdf
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent

INVOICE = [
    ("TAX INVOICE", 20, True),
    ("Acme Cloud Services Pvt Ltd", 13, True),
    ("12 MG Road, Bengaluru, Karnataka 560001", 10, False),
    ("GSTIN: 29ABCDE1234F1Z5", 10, False),
    ("", 10, False),
    ("Invoice No: ACS/2026/0042          Invoice Date: 15-09-2026", 11, False),
    ("", 10, False),
    ("Bill To:", 11, True),
    ("Demo Customer LLP", 11, False),
    ("44 Park Street, Kolkata, West Bengal 700016", 10, False),
    ("", 10, False),
    ("Description                         HSN/SAC    Qty      Amount (INR)", 10, True),
    ("Managed Kubernetes cluster          998315     1        80,000.00", 10, False),
    ("Object storage (5 TB)               998315     1        30,000.00", 10, False),
    ("Premium support plan                998315     1        15,000.00", 10, False),
    ("", 10, False),
    ("Subtotal                                                1,25,000.00", 10, False),
    ("CGST @ 9%                                                 11,250.00", 10, False),
    ("SGST @ 9%                                                 11,250.00", 10, False),
    ("Grand Total                                           ₹ 1,47,500.00", 11, True),
    ("", 10, False),
    ("Amount in words: One Lakh Forty Seven Thousand Five Hundred Only", 9, False),
]

RECEIPT = [
    ("CORNER CAFE", 16, True),
    ("Shop 3, Linking Road, Mumbai", 9, False),
    ("GSTIN 27AAAAA0000A1Z5", 9, False),
    ("Bill No: CC-7781   Date: 02/10/2026", 9, False),
    ("--------------------------------", 9, False),
    ("Masala Chai x2            120.00", 9, False),
    ("Veg Sandwich x1           180.00", 9, False),
    ("Cold Coffee x1            200.00", 9, False),
    ("--------------------------------", 9, False),
    ("Sub Total                 500.00", 9, False),
    ("CGST 2.5%                  12.50", 9, False),
    ("SGST 2.5%                  12.50", 9, False),
    ("TOTAL                 Rs. 525.00", 11, True),
    ("Thank you! Visit again", 9, False),
]


def render(lines, width, height, path=None):
    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    y = 50
    for text, size, bold in lines:
        page.insert_text((40, y), text, fontsize=size, fontname="cour" if not bold else "cobo")
        y += size + 8
    if path:
        document.save(path)
    return document


def main():
    render(INVOICE, 595, 842, HERE / "sample_gst_invoice.pdf")

    # Rasterise the receipt and make it look like a phone photo:
    # slight rotation, blur and sensor noise
    receipt = render(RECEIPT, 300, 420)
    pixmap = receipt[0].get_pixmap(dpi=200)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("L")
    image = image.rotate(2.0, expand=True, fillcolor=235).filter(ImageFilter.GaussianBlur(0.8))
    random.seed(7)
    pixels = image.load()
    for _ in range(image.width * image.height // 25):
        x, y = random.randrange(image.width), random.randrange(image.height)
        pixels[x, y] = max(0, min(255, pixels[x, y] + random.randint(-60, 60)))
    image.convert("RGB").save(HERE / "sample_cafe_receipt.jpg", quality=80)

    print("wrote samples/sample_gst_invoice.pdf and samples/sample_cafe_receipt.jpg")


if __name__ == "__main__":
    main()
