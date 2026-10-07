#!/usr/bin/env python3
"""
Phase 5: compare base Qwen vs the LoRA fine-tune on the held-out test set.

Usage:
    python evaluate.py                       # base and fine-tuned
    python evaluate.py --adapter-only
"""

import argparse
import json
import re
import time

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from invoice_schema import BASE, FIELDS

MODEL = BASE / "models" / "Qwen2.5-1.5B-Instruct"
ADAPTER = BASE / "adapters" / "invoice-qwen1.5b"
TEST = BASE / "data" / "test.jsonl"

SCORED = [f for f in FIELDS if f != "description"]
NUMERIC = {"subtotal", "tax", "total"}


def parse_json(output):
    match = re.search(r"\{.*\}", output, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def as_number(value):
    if value is None or isinstance(value, (int, float)):
        return value
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return "unparseable"


def norm(value):
    return None if value is None else re.sub(r"[^a-z0-9]", "", str(value).lower())


def field_correct(field, predicted, expected):
    if field in NUMERIC:
        predicted = as_number(predicted)
        if predicted is None or expected is None:
            return predicted is expected
        return predicted != "unparseable" and abs(predicted - expected) <= 0.01
    if field == "vendor" and predicted and expected:
        p, e = norm(predicted), norm(expected)
        return p in e or e in p
    return norm(predicted) == norm(expected)


def run(model_path, adapter_path, examples):
    model, tokenizer = load(str(model_path), adapter_path=adapter_path and str(adapter_path))
    sampler = make_sampler(temp=0.0)
    correct = {field: 0 for field in SCORED}
    valid_json = 0
    exact_docs = 0
    seconds = 0.0
    misses = []

    for messages in examples:
        expected = json.loads(messages[-1]["content"])
        prompt = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False
        )
        start = time.time()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=300, sampler=sampler)
        seconds += time.time() - start

        predicted = parse_json(output)
        if predicted is None:
            misses.append(("invalid json", output[:80]))
            continue
        valid_json += 1

        all_right = True
        for field in SCORED:
            if field_correct(field, predicted.get(field), expected[field]):
                correct[field] += 1
            else:
                all_right = False
                misses.append((field, f"got {predicted.get(field)!r} want {expected[field]!r}"))
        exact_docs += all_right

    n = len(examples)
    return {
        "valid_json": valid_json / n,
        "fields": {field: correct[field] / n for field in SCORED},
        "all_fields_right": exact_docs / n,
        "sec_per_doc": seconds / n,
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument("--adapter", default=str(ADAPTER))
    args = parser.parse_args()

    examples = [json.loads(line)["messages"] for line in TEST.open(encoding="utf-8")]
    print(f"test documents: {len(examples)}\n")

    runs = {} if args.adapter_only else {"base": run(MODEL, None, examples)}
    runs["fine-tuned"] = run(MODEL, args.adapter, examples)

    names = list(runs)
    print(f"{'metric':<18}" + "".join(f"{name:>12}" for name in names))
    print(f"{'valid JSON':<18}" + "".join(f"{runs[n]['valid_json']:>12.0%}" for n in names))
    for field in SCORED:
        print(f"{field:<18}" + "".join(f"{runs[n]['fields'][field]:>12.0%}" for n in names))
    print(f"{'ALL fields right':<18}" + "".join(f"{runs[n]['all_fields_right']:>12.0%}" for n in names))
    print(f"{'sec / doc':<18}" + "".join(f"{runs[n]['sec_per_doc']:>12.1f}" for n in names))

    print("\nfine-tuned misses:")
    for field, detail in runs["fine-tuned"]["misses"]:
        print(f"  {field:<15} {detail}")


if __name__ == "__main__":
    main()
