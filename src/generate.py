"""Generate Gemma responses and save final-prompt-token hidden states.

Designed for a free Kaggle GPU. Activations are collected once per prompt;
multiple sampled responses share the same pre-generation representation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_utils import (
    infer_finish_reason,
    resolve_eos_token_id,
    resolve_generation_counts,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/ladders.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/run_main"))
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--generation-count-field",
        help="Dataset field containing a prespecified per-prompt generation count.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--base-seed", type=int, default=20260724)
    return parser.parse_args()


def choose_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. This prototype is intended for a Kaggle GPU session."
        )
    # Native bfloat16 tensor-core support starts with Ampere (compute capability
    # 8.x). Newer PyTorch versions may report bfloat16 as available through
    # emulation on a T4, but float16 is the appropriate fast dtype there.
    compute_capability, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if compute_capability >= 8 else torch.float16


def main() -> None:
    args = parse_args()
    if args.num_generations < 1:
        raise ValueError("--num-generations must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")

    prompts = read_jsonl(args.dataset)
    required = {"prompt_id", "ladder_id", "category", "rung", "prompt"}
    for row in prompts:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{row.get('prompt_id', '<unknown>')} lacks {sorted(missing)}")
    prompt_ids = [row["prompt_id"] for row in prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt_id values must be unique")
    generation_counts = resolve_generation_counts(
        prompts,
        args.num_generations,
        args.generation_count_field,
    )
    seed_stride = max(generation_counts)
    total_generations = sum(generation_counts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = choose_dtype()
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading {args.model} with {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Gemma 2 Instruct uses both <eos> (1) and <end_of_turn> (107) as stopping
    # tokens. tokenizer.eos_token_id contains only the former, while the model's
    # generation config preserves the full list.
    generation_eos_token_id = resolve_eos_token_id(model, tokenizer)

    activation_rows: list[np.ndarray] = []
    activation_token_ids: list[int] = []
    finish_reason_counts = {"eos": 0, "length": 0, "other": 0}
    response_path = args.output_dir / "responses.jsonl"

    with response_path.open("w", encoding="utf-8") as response_file:
        for prompt_index, item in enumerate(prompts):
            messages = [{"role": "user", "content": item["prompt"]}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            attention_mask = torch.ones_like(input_ids)

            # hidden_states contains the embedding output followed by one residual
            # stream tensor for every transformer layer.
            with torch.inference_mode():
                forward = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            final_prompt_states = torch.stack(
                [state[0, -1, :].detach().to(torch.float16).cpu() for state in forward.hidden_states]
            )
            activation_rows.append(final_prompt_states.numpy())
            activation_token_ids.append(int(input_ids[0, -1].item()))
            del forward, final_prompt_states

            prompt_generation_count = generation_counts[prompt_index]
            for generation_id in range(prompt_generation_count):
                seed = args.base_seed + prompt_index * seed_stride + generation_id
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                do_sample = args.temperature > 0
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": generation_eos_token_id,
                    "use_cache": True,
                }
                if do_sample:
                    generation_kwargs.update(
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                with torch.inference_mode():
                    generated = model.generate(**generation_kwargs)
                new_token_ids = generated[0, input_ids.shape[1] :]
                response = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
                token_count = int(new_token_ids.shape[0])
                final_token_id = (
                    int(new_token_ids[-1].item()) if token_count > 0 else None
                )
                finish_reason = infer_finish_reason(
                    token_count=token_count,
                    max_new_tokens=args.max_new_tokens,
                    final_token_id=final_token_id,
                    eos_token_id=generation_eos_token_id,
                )
                hit_token_cap = finish_reason == "length"
                finish_reason_counts[finish_reason] += 1
                record = {
                    **item,
                    "generation_id": generation_id,
                    "seed": seed,
                    "response": response,
                    "response_length_tokens": token_count,
                    "response_length_chars": len(response),
                    "hit_token_cap": hit_token_cap,
                    "finish_reason": finish_reason,
                }
                response_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                response_file.flush()
                print(
                    f"[{prompt_index + 1:02d}/{len(prompts)}] "
                    f"{item['prompt_id']} generation "
                    f"{generation_id + 1}/{prompt_generation_count}"
                )
                del generated, new_token_ids

    activation_array = np.stack(activation_rows, axis=0)
    np.savez_compressed(
        args.output_dir / "activations.npz",
        activations=activation_array,
        prompt_ids=np.asarray(prompt_ids),
        final_prompt_token_ids=np.asarray(activation_token_ids, dtype=np.int64),
    )

    config = {
        "model": args.model,
        "dataset": str(args.dataset),
        "num_prompts": len(prompts),
        "num_generations": args.num_generations,
        "generation_count_field": args.generation_count_field,
        "generation_count_minimum": min(generation_counts),
        "generation_count_maximum": max(generation_counts),
        "total_generations": total_generations,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "base_seed": args.base_seed,
        "generation_eos_token_id": generation_eos_token_id,
        "finish_reason_counts": finish_reason_counts,
        "num_capped_responses": finish_reason_counts["length"],
        "capped_response_rate": (
            finish_reason_counts["length"] / total_generations
        ),
        "activation_position": "last token of chat template with generation prompt",
        "activation_shape": list(activation_array.shape),
        "activation_dtype_on_disk": str(activation_array.dtype),
        "compute_dtype": str(dtype),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print(f"Finished. Results saved under {args.output_dir}")


if __name__ == "__main__":
    main()
