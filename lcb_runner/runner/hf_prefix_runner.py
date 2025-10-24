from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from lcb_runner.runner.base_runner import BaseRunner


def _trim_stop_sequences(text: str, stops: Iterable[str]) -> str:
    """Trim output at the first occurrence of any stop sequence."""
    cutoff = len(text)
    for stop in stops:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1:
            cutoff = min(cutoff, idx)
    return text[:cutoff].rstrip()


class HFPrefixRunner(BaseRunner):
    """
    Minimal runner using Hugging Face transformers + PEFT to load prefix-tuned adapters.
    """

    def __init__(self, args, model):
        super().__init__(args, model)
        if args.local_model_path is None:
            raise ValueError(
                "Prefix runner requires --local_model_path pointing to the adapter directory."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        adapter_path = Path(args.local_model_path).expanduser()
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path {adapter_path} does not exist.")
        self.adapter_path = adapter_path

        metadata_path = adapter_path / "prefix_metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            base_model_name = metadata.get("base_model", model.model_name)
        else:
            base_model_name = model.model_name

        tokenizer_source: str | Path
        if (adapter_path / "tokenizer_config.json").exists():
            tokenizer_source = adapter_path
        else:
            tokenizer_source = base_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=args.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        base_model.to(self.device)

        self.hf_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=dtype,
        )
        self.hf_model.to(self.device)
        self.hf_model.eval()

    def _format_prompt(self, prompt: str | list[dict[str, str]]) -> str:
        if isinstance(prompt, list):
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            # Fallback: concatenate role/content pairs.
            return "\n".join(f"{turn['role']}: {turn['content']}" for turn in prompt) + "\nassistant:"
        return prompt

    @torch.inference_mode()
    def _run_single(self, prompt: str | list[dict[str, str]]) -> List[str]:
        prompt_text = self._format_prompt(prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        do_sample = self.args.temperature > 0
        if not do_sample and self.args.n > 1:
            do_sample = True
        outputs = self.hf_model.generate(
            **encoded,
            do_sample=do_sample,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_new_tokens=self.args.max_tokens,
            num_return_sequences=self.args.n,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        prompt_len = encoded["input_ids"].shape[-1]
        generated = sequences[:, prompt_len:]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        completions = []
        for text in texts:
            completion = _trim_stop_sequences(text, self.args.stop)
            completions.append(completion.strip())
        return completions
