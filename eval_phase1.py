"""
Run eval_loss (and related metrics) on the held-out split using a trained LoRA adapter.

Uses the same JSONL → text formatting and train/test split seed (42) as
finetune_stage1_llama31_8b_base.py so the eval set matches training.

Example:
  HF_TOKEN=... python eval_phase1.py \\
    --adapter_path ./llama31-8b-base-agro-qlora/lora_adapters \\
    --data_path finetune_QA_stage1/master_dataset.jsonl
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from typing import Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import finetune_stage1_llama31_8b_base as ft

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Eval LoRA checkpoint on Phase 1 val split")
    p.add_argument("--adapter_path", required=True, help="Directory with adapter_config.json")
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B")
    p.add_argument("--data_path", default="finetune_QA_stage1/master_dataset.jsonl")
    p.add_argument("--include_instruction", action="store_true")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--eval_batch_size", type=int, default=8)
    args = p.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cfg = ft.FinetuneConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        include_instruction_in_prompt=args.include_instruction,
        max_seq_length=args.max_seq_len,
        hf_token=hf_token,
    )

    tokenizer = ft.load_tokenizer(cfg)
    _, eval_ds = ft.prepare_datasets(cfg, tokenizer)

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    st_params = inspect.signature(SFTTrainer.__init__).parameters
    sft_params = inspect.signature(SFTConfig.__init__).parameters

    eval_kw: Dict[str, str] = {}
    if "eval_strategy" in sft_params:
        eval_kw["eval_strategy"] = "no"
    elif "evaluation_strategy" in sft_params:
        eval_kw["evaluation_strategy"] = "no"

    cand = dict(
        output_dir="/tmp/eval_phase1_trash",
        max_steps=0,
        per_device_eval_batch_size=args.eval_batch_size,
        bf16=torch.cuda.is_available(),
        report_to="none",
        dataset_text_field="text",
        packing=False,
        **eval_kw,
    )
    if "max_length" in sft_params:
        cand["max_length"] = cfg.max_seq_length
    elif "max_seq_length" in sft_params:
        cand["max_seq_length"] = cfg.max_seq_length

    sft_args = SFTConfig(**{k: v for k, v in cand.items() if k in sft_params})

    kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=eval_ds,
        eval_dataset=eval_ds,
    )
    if "processing_class" in st_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in st_params:
        kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**{k: v for k, v in kwargs.items() if k in st_params})

    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
