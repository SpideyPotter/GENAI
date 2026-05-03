"""
QLoRA fine-tune Llama 3.1 8B — tuned for 1x H200 (~70GB VRAM), 16 CPU cores, ~30GB RAM.

Fixes TRL SFTConfig incompatibilities (e.g. group_by_length / eval_strategy) by only
passing arguments supported by your installed `trl` version.

Copy over container `/workspace/finetune_stage1.py` or run this file directly.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir: str = "./llama31-agro-finetuned"

    data_path: str = "finetune_QA_stage1/master_dataset.jsonl"
    include_instruction_in_prompt: bool = True
    val_split_ratio: float = 0.05
    max_seq_length: int = 2048

    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    logging_steps: int = 25
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "none"
    hf_token: Optional[str] = None


def _format_llama31_messages_manual(messages: List[Dict[str, str]]) -> str:
    parts = ["<|begin_of_text|>"]
    for m in messages:
        role = m["role"]
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
        )
    return "".join(parts)


def format_prompt(
    example: Dict[str, Any],
    tokenizer,
    include_instruction_in_prompt: bool,
) -> str:
    user_msg = (example.get("input") or "").strip()
    assistant = (example.get("output") or "").strip()
    if include_instruction_in_prompt:
        system_msg = (example.get("instruction") or "").strip()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant},
        ]
    else:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant},
        ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return _format_llama31_messages_manual(messages)


def load_jsonl(path: str) -> Dataset:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %s records from %s", len(records), path)
    return Dataset.from_list(records)


def prepare_datasets(cfg: FinetuneConfig, tokenizer):
    raw = load_jsonl(cfg.data_path)

    def to_text(example: dict) -> dict:
        text = format_prompt(
            example,
            tokenizer,
            include_instruction_in_prompt=cfg.include_instruction_in_prompt,
        )
        return {"text": text}

    dataset = raw.map(to_text, remove_columns=raw.column_names)
    split = dataset.train_test_split(test_size=cfg.val_split_ratio, seed=42)
    logger.info("Train: %s | Val: %s", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


def load_tokenizer(cfg: FinetuneConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        token=cfg.hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(cfg: FinetuneConfig):
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )
    logger.info("Loading model: %s", cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=cfg.hf_token,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _build_sft_config(cfg: FinetuneConfig) -> SFTConfig:
    params = inspect.signature(SFTConfig.__init__).parameters
    eval_kw: Dict[str, str] = {}
    if "eval_strategy" in params:
        eval_kw["eval_strategy"] = "steps"
    elif "evaluation_strategy" in params:
        eval_kw["evaluation_strategy"] = "steps"

    candidate = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        optim=cfg.optim,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        save_strategy="steps",
        report_to=cfg.report_to,
        dataset_text_field="text",
        packing=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        **eval_kw,
    )
    if cfg.dataloader_num_workers > 0 and "dataloader_prefetch_factor" in params:
        candidate["dataloader_prefetch_factor"] = cfg.dataloader_prefetch_factor
    if "group_by_length" in params:
        candidate["group_by_length"] = True

    if "max_length" in params:
        candidate["max_length"] = cfg.max_seq_length
    elif "max_seq_length" in params:
        candidate["max_seq_length"] = cfg.max_seq_length

    allowed = {k: v for k, v in candidate.items() if k in params}
    dropped = sorted(set(candidate) - set(allowed))
    if dropped:
        logger.warning("SFTConfig omitted unsupported args for this TRL build: %s", dropped)
    return SFTConfig(**allowed)


def build_trainer(cfg: FinetuneConfig, model, tokenizer, train_ds, eval_ds):
    st_params = inspect.signature(SFTTrainer.__init__).parameters
    kwargs = dict(
        model=model,
        args=_build_sft_config(cfg),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    if "processing_class" in st_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in st_params:
        kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "SFTTrainer API not recognized (no processing_class or tokenizer)."
        )
    allowed = {k: v for k, v in kwargs.items() if k in st_params}
    return SFTTrainer(**allowed)


def merge_and_save(cfg: FinetuneConfig, trainer, tokenizer):
    merged_dir = os.path.join(cfg.output_dir, "merged")
    logger.info("Merging LoRA adapters -> %s", merged_dir)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merged model saved.")


def main():
    parser = argparse.ArgumentParser(description="QLoRA Llama 3.1 8B (H200-oriented)")
    parser.add_argument("--data_path", default="finetune_QA_stage1/master_dataset.jsonl")
    parser.add_argument("--output_dir", default="./llama31-agro-finetuned")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        help="Ignore JSONL 'instruction'; train user question + answer only",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (lower if RAM OOM / 137)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()

    cfg = FinetuneConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        data_path=args.data_path,
        include_instruction_in_prompt=not args.no_instruction,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        max_seq_length=args.max_seq_len,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        report_to=args.report_to,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info("GPU %s: %s | %.1f GB", i, p.name, p.total_memory / 1e9)
    else:
        logger.warning("No CUDA.")

    tokenizer = load_tokenizer(cfg)
    train_ds, eval_ds = prepare_datasets(cfg, tokenizer)
    model = load_model(cfg)
    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    logger.info("Starting training")
    trainer.train()

    adapter_dir = os.path.join(cfg.output_dir, "lora_adapters")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("LoRA adapters saved to %s", adapter_dir)

    if args.merge:
        merge_and_save(cfg, trainer, tokenizer)

    logger.info("Done.")


if __name__ == "__main__":
    main()
