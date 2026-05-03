"""
Fine-tuning Script: Llama 3.1 8B on Agricultural QA Dataset
Hardware: DGX H200 | 71GB VRAM | 16 cores | 32GB RAM
Method: QLoRA (4-bit quantization + LoRA adapters)
"""

import os
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass
class FinetuneConfig:
    # Model
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir: str = "./llama31-agro-finetuned"

    # Data
    data_path: str = "finetune_QA_stage1/master_dataset.jsonl"          # path to your .jsonl file
    val_split_ratio: float = 0.05             # 5% for validation
    max_seq_length: int = 2048

    # QLoRA
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # H200 supports bf16 natively
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True             # double quantization

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training — tuned for 71GB VRAM
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2     # effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True                        # H200 native
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
    group_by_length: bool = True             # speeds up training
    logging_steps: int = 25
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "none"                 # set "wandb" if you use W&B

    # HuggingFace token (needed for gated Llama model)
    hf_token: Optional[str] = None


# ─────────────────────────────────────────────
# Prompt formatting (Llama 3 chat template)
# ─────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = "{instruction}"

def format_prompt(example: dict, tokenizer) -> str:
    """
    Converts a {instruction, input, output} record into
    the Llama-3 chat format with <|begin_of_text|> tokens.
    """
    system_msg = example.get("instruction", "").strip()
    user_msg   = example.get("input", "").strip()
    assistant  = example.get("output", "").strip()

    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": assistant},
    ]
    # apply_chat_template adds the special tokens automatically
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ─────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────
def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {path}")
    return Dataset.from_list(records)


def prepare_datasets(cfg: FinetuneConfig, tokenizer):
    raw = load_jsonl(cfg.data_path)

    def tokenize(example):
        text = format_prompt(example, tokenizer)
        return {"text": text}

    dataset = raw.map(tokenize, remove_columns=raw.column_names)

    split = dataset.train_test_split(test_size=cfg.val_split_ratio, seed=42)
    logger.info(
        f"Train: {len(split['train'])} | Val: {len(split['test'])} samples"
    )
    return split["train"], split["test"]


# ─────────────────────────────────────────────
# Model & tokenizer
# ─────────────────────────────────────────────
def load_tokenizer(cfg: FinetuneConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        token=cfg.hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"          # critical for causal LM
    return tokenizer


def load_model(cfg: FinetuneConfig):
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )

    logger.info(f"Loading base model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",                    # spreads across all GPUs if multi-GPU
        token=cfg.hf_token,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2",  # H200 fully supports FA2
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare for k-bit training (freezes base, casts norms to fp32, etc.)
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


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
def build_trainer(cfg: FinetuneConfig, model, tokenizer, train_ds, eval_ds):
    training_args = SFTConfig(
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
        group_by_length=cfg.group_by_length,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=cfg.report_to,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=False,                        # set True to pack short sequences
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer


# ─────────────────────────────────────────────
# Merge & save (optional — fuses LoRA into base)
# ─────────────────────────────────────────────
def merge_and_save(cfg: FinetuneConfig, trainer, tokenizer):
    merged_dir = os.path.join(cfg.output_dir, "merged")
    logger.info(f"Merging LoRA adapters → {merged_dir}")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merged model saved successfully.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B (QLoRA)")
    parser.add_argument("--data_path",   default="finetune_QA_stage1/master_dataset.jsonl",           help="Path to .jsonl training file")
    parser.add_argument("--output_dir",  default="./llama31-agro-finetuned", help="Output directory")
    parser.add_argument("--model_name",  default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--lora_r",      type=int,   default=64)
    parser.add_argument("--max_seq_len", type=int,   default=2048)
    parser.add_argument("--hf_token",    default=None, help="HuggingFace access token for gated models")
    parser.add_argument("--merge",       action="store_true", help="Merge LoRA into base model after training")
    parser.add_argument("--report_to",   default="none", help="Logging backend: none | wandb | tensorboard")
    args = parser.parse_args()

    cfg = FinetuneConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        data_path=args.data_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        max_seq_length=args.max_seq_len,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        report_to=args.report_to,
    )

    # ── GPU info ──
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB VRAM")
    else:
        logger.warning("No CUDA GPU detected — training will be very slow on CPU!")

    # ── Pipeline ──
    tokenizer = load_tokenizer(cfg)
    train_ds, eval_ds = prepare_datasets(cfg, tokenizer)
    model    = load_model(cfg)
    trainer  = build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    logger.info("Starting training …")
    trainer.train()

    # Save LoRA adapters
    adapter_dir = os.path.join(cfg.output_dir, "lora_adapters")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"LoRA adapters saved to: {adapter_dir}")

    # Optionally merge
    if args.merge:
        merge_and_save(cfg, trainer, tokenizer)

    logger.info("Fine-tuning complete ✓")


if __name__ == "__main__":
    main()
