"""
Fine-tune base Llama 3.1 8B on YouTube QA only.

Defaults:
- Base model (local HF cache snapshot):
  /workspace/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
- Training data:
  greenagriculture_QA/all_qa_pairs.jsonl
- Output:
  FineTunedmodels/Finetuned_YoutubeData
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import trl as _trl
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
except ImportError:
    from trl import SFTTrainer
    DataCollatorForCompletionOnlyLM = None

_USE_SFT_CONFIG = hasattr(_trl, "SFTConfig")
if _USE_SFT_CONFIG:
    from trl import SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "As an experienced agronomist proficient in farming techniques, crop management, "
    "and disease-resistant crop cultivation, answer only from given knowledge."
)
RESPONSE_TEMPLATE = "### Answer:\n"


@dataclass
class Config:
    data_path: str = "greenagriculture_QA/all_qa_pairs.jsonl"
    base_model: str = "/workspace/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
    output_dir: str = "FineTunedmodels/Finetuned_YoutubeData"
    val_ratio: float = 0.05
    seed: int = 42
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    report_to: str = "none"
    run_name: str = "llama31-8b-youtube-only"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True


CFG = Config()


def format_prompt(sample: dict) -> str:
    q = sample["input"].strip()
    a = sample["output"].strip()
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Question:\n{q}\n\n"
        f"{RESPONSE_TEMPLATE}{a}"
    )


def load_data(path: str, val_ratio: float, seed: int) -> DatasetDict:
    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Training data not found: {in_path}")

    rows: list[dict] = []
    if in_path.suffix.lower() == ".jsonl":
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                q = item.get("question")
                a = item.get("answer")
                if q and a:
                    rows.append({"input": q, "output": a})
    else:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            q = item.get("question") or item.get("input")
            a = item.get("answer") or item.get("output")
            if q and a:
                rows.append({"input": q, "output": a})

    if len(rows) < 2:
        raise ValueError("Need at least 2 QA rows to create train/validation split.")

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def load_model_and_tokenizer(cfg: Config):
    base = Path(cfg.base_model)
    if not base.exists():
        raise FileNotFoundError(f"Base model path does not exist: {cfg.base_model}")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=compute_dtype,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.print_trainable_parameters()
    return model, tokenizer


def train(cfg: Config):
    dataset = load_data(cfg.data_path, cfg.val_ratio, cfg.seed)
    model, tokenizer = load_model_and_tokenizer(cfg)

    def _add_text(sample: dict) -> dict:
        sample["text"] = format_prompt(sample)
        return sample

    dataset["train"] = dataset["train"].map(_add_text, num_proc=4)
    dataset["validation"] = dataset["validation"].map(_add_text, num_proc=4)

    collator = None
    if DataCollatorForCompletionOnlyLM is not None:
        try:
            collator = DataCollatorForCompletionOnlyLM(
                response_template=RESPONSE_TEMPLATE,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            log.warning("Completion-only collator unavailable (%s); using default.", exc)

    common_args = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        seed=cfg.seed,
    )

    if _USE_SFT_CONFIG:
        args = SFTConfig(
            **common_args,
            max_seq_length=cfg.max_seq_length,
            packing=False,
            dataset_text_field="text",
        )
        sft_extra = {}
    else:
        args = TrainingArguments(**common_args)
        sft_extra = dict(
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="text",
            packing=False,
            dataset_num_proc=4,
        )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        **sft_extra,
    )

    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer

    allowed = {k: v for k, v in trainer_kwargs.items() if k in params or k in {"model", "args", "train_dataset", "eval_dataset", "data_collator"}}
    trainer = SFTTrainer(**allowed)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters found (all requires_grad=False). "
            "LoRA adapters were not activated correctly."
        )
    log.info("Trainable parameter count: %s", trainable)

    trainer.create_optimizer()
    wrapped = trainer.optimizer
    base_opt = getattr(wrapped, "optimizer", wrapped)
    if not hasattr(base_opt, "train"):
        base_opt.train = types.MethodType(lambda self: None, base_opt)
    if not hasattr(base_opt, "eval"):
        base_opt.eval = types.MethodType(lambda self: None, base_opt)

    log.info("Starting training from base model: %s", cfg.base_model)
    stats = trainer.train()
    log.info("Train done. Runtime: %.2fs", stats.metrics.get("train_runtime", 0.0))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_dir = output_dir / "lora_adapter"
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    log.info("Saved LoRA adapter to %s", lora_dir)

    try:
        merged_dir = output_dir / "merged_16bit"
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        log.info("Saved merged model to %s", merged_dir)
    except Exception as exc:
        log.warning("Merge skipped: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune base Llama 3.1 8B on YouTube QA")
    parser.add_argument("--data_path", default=CFG.data_path)
    parser.add_argument("--base_model", default=CFG.base_model)
    parser.add_argument("--output_dir", default=CFG.output_dir)
    parser.add_argument("--epochs", type=int, default=CFG.num_train_epochs)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--batch_size", type=int, default=CFG.per_device_train_batch_size)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    CFG.data_path = args.data_path
    CFG.base_model = args.base_model
    CFG.output_dir = args.output_dir
    CFG.num_train_epochs = args.epochs
    CFG.learning_rate = args.lr
    CFG.per_device_train_batch_size = args.batch_size
    train(CFG)

