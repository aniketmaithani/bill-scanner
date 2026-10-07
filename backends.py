"""
Runtimes that turn a prompt into the model's answer.

mlx    Apple Silicon Macs. Runs on the Metal GPU and loads the LoRA adapter
       as is.
torch  Everything else (Linux, Windows, Intel Macs), on the CPU. Loads the
       same base model with transformers and folds the LoRA adapter into its
       weights once at startup, so the answers match the MLX runtime.

MLX can also run on a plain CPU, but its CPU backend needs several minutes
per bill, which is why other machines use PyTorch.

Both runtimes decode greedily and expose the same two calls:
    generate(prompt, max_tokens) -> str
    stream(prompt, max_tokens)   -> iterator of text pieces
"""

import json
import platform

from invoice_schema import ADAPTER, MODEL


def default_backend():
    on_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return "mlx" if on_apple_silicon else "torch"


def load_backend(name=None):
    name = name or default_backend()
    if name == "mlx":
        return MLXBackend()
    if name == "torch":
        return TorchBackend()
    raise ValueError(f"unknown backend: {name}")


class MLXBackend:
    name = "mlx"

    def __init__(self):
        import windows_compat  # noqa: F401  (must come before mlx_lm)
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tokenizer = load(str(MODEL), adapter_path=str(ADAPTER))
        self.sampler = make_sampler(temp=0.0)

    def chat_prompt(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate(self, prompt, max_tokens):
        from mlx_lm import generate

        return generate(
            self.model, self.tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=self.sampler,
        )

    def stream(self, prompt, max_tokens):
        from mlx_lm import stream_generate

        for response in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=self.sampler
        ):
            yield response.text


class TorchBackend:
    name = "torch"

    def __init__(self):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # float32 is the fast path on ordinary CPUs; bfloat16 is only fast
        # on chips with AMX / AVX512-BF16
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)

        # MLX LoRA computes y = Wx + scale * (x @ A @ B), with A (in, r) and
        # B (r, out). Folding it in: W (out, in) += scale * (A @ B)^T
        config = json.loads((ADAPTER / "adapter_config.json").read_text())
        scale = config["lora_parameters"]["scale"]
        weights = load_file(str(ADAPTER / "adapters.safetensors"))
        modules = dict(self.model.named_modules())
        with torch.no_grad():
            for key, lora_a in weights.items():
                if not key.endswith(".lora_a"):
                    continue
                name = key[: -len(".lora_a")]
                lora_b = weights[name + ".lora_b"]
                delta = scale * (lora_a.float() @ lora_b.float()).T
                modules[name].weight += delta.to(modules[name].weight.dtype)
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])

    def chat_prompt(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def stream(self, prompt, max_tokens):
        torch = self.torch
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        past, tokens, emitted = None, [], ""
        with torch.inference_mode():
            for _ in range(max_tokens):
                output = self.model(input_ids=input_ids, past_key_values=past, use_cache=True)
                past = output.past_key_values
                token = int(output.logits[0, -1].argmax())
                if token in self.stop_ids:
                    break
                tokens.append(token)
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                yield text[len(emitted):]
                emitted = text
                input_ids = torch.tensor([[token]])

    def generate(self, prompt, max_tokens):
        return "".join(self.stream(prompt, max_tokens))
