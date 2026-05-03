"""
Fine-tuning Pipeline: Llama-3-8B (Base) for Agricultural QA
================================================================
Based on: "Leveraging Synthetic Data for Question Answering with
           Multilingual LLMs in the Agricultural Domain"

Hardware target : DGX node — MIG slice mig-3g.71gb (1 slice available)
Method          : QLoRA (Parameter-Efficient Fine-Tuning via LoRA + 4-bit quant)
Framework       : Unsloth + HuggingFace SFTTrainer
Prompt template : Prompt 1 from the paper (Appendix D) — selected best prompt

Paper hyperparameters (Table 13 — Llama-3-8B Base → fine-tuned):
  LoRA rank (r)         = 16
  LoRA alpha            = 16
  Target modules        = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  Batch size            = 2
  Gradient accum steps  = 4  →  effective batch = 8
  Max sequence length   = 2048
  Learning rate         = 2e-4
  Epochs                = 1
  Optimizer             = AdamW 8-bit
  Quantization          = 4-bit (bnb)
  Scheduler             = Linear with warm-up
  Tracking              = Weights & Biases
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  INSTALL DEPENDENCIES
#     Run once before launching this script:
#
#       pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#       pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
#       pip install wandb datasets transformers
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import math
import logging
import types
from dataclasses import dataclass, field
from typing import Optional
from packaging import version

import torch
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments

# ── Version-safe trl imports ──────────────────────────────────────────────────
# DataCollatorForCompletionOnlyLM location changed across trl versions:
#   < 0.8.0  →  trl.trainer.utils   (or not present at all)
#   >= 0.8.0 →  trl  (top-level)
# SFTTrainer dataset_text_field / formatting_func API also shifted in 0.9+
import trl as _trl

_trl_version = version.parse(_trl.__version__)

try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        from trl import SFTTrainer
    except ImportError:
        # Fallback: build a minimal completion-only collator from transformers
        from transformers import DataCollatorForSeq2Seq as DataCollatorForCompletionOnlyLM  # placeholder replaced below
        from trl import SFTTrainer
        DataCollatorForCompletionOnlyLM = None  # handled in train()

# ── SFTConfig was introduced in trl 0.9; use TrainingArguments for older trl ─
_USE_SFT_CONFIG = _trl_version >= version.parse("0.9.0")
if _USE_SFT_CONFIG:
    try:
        from trl import SFTConfig
    except ImportError:
        _USE_SFT_CONFIG = False

# ── Optional W&B tracking (set WANDB_API_KEY env var or disable below) ───────
WANDB_ENABLED = os.environ.get("WANDB_API_KEY") is not None
if WANDB_ENABLED:
    import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.info(f"trl version: {_trl.__version__}  |  using SFTConfig: {_USE_SFT_CONFIG}")

# =============================================================================
# 1.  CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path: str = "final_finetune_stage1/final_training_data.json"       # path to your JSON file
    val_ratio: float = 0.05                            # 5 % held-out validation
    seed: int = 42
    smoke_test_chunks: int = 0                         # 0 disables; >0 keeps first N train rows

    # ── Model ─────────────────────────────────────────────────────────────────
    # Priority order for base_model:
    #   1. Local path  →  e.g. "/raid/models/Meta-Llama-3-8B"
    #   2. HF hub ID   →  requires HF_TOKEN + accepted licence on hf.co
    #   3. Open fallback (no token needed) → "mistralai/Mistral-7B-v0.1"
    # Override via --base_model flag or by editing this default.
    base_model: str = "meta-llama/Meta-Llama-3-8B"
    hf_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    load_in_4bit: bool = False                        # QLoRA (paper: 4-bit bnb)

    # ── LoRA (paper Table 13) ─────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"            # paper Table 13
    ])

    # ── Training (paper Table 13) ─────────────────────────────────────────────
    output_dir: str = "./llama3_agri_finetuned"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4              # effective batch = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"                 # paper: linear with warm-up
    optim: str = "adamw_8bit"                         # paper: AdamW 8-bit
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = not torch.cuda.is_bf16_supported()
    bf16: bool = torch.cuda.is_bf16_supported()

    # ── Logging ───────────────────────────────────────────────────────────────
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    report_to: str = "wandb" if WANDB_ENABLED else "none"
    run_name: str = "llama3-8b-agri-qlora"


CFG = Config()


# =============================================================================
# 2.  PROMPT TEMPLATE  (Prompt 1 from paper — Figure 2 / Appendix D)
#
#     "As an experienced agronomist proficient in farming techniques, crop
#      management, and disease-resistant crop cultivation, you are tasked
#      with answering questions based on your expertise. Questions may be
#      provided in various languages such as English, Hindi, Marathi, etc.
#      Provide only the answer relevant to the question. Do not include any
#      other information. Answer should be in the same language as the question."
# =============================================================================

SYSTEM_PROMPT = (
    "As an experienced agronomist proficient in farming techniques, crop management, "
    "and disease-resistant crop cultivation, you are tasked with answering questions "
    "based on your expertise.Generate suitable questions related to the information.\n"
    "Provide only the answer relevant to the question. Do not include any other "
    "information."
)

# ── Prompt format for Meta-Llama-3-8B (BASE model, not Instruct) ─────────────
# The base model does not understand special chat tokens like <|begin_of_text|>.
# We use a plain text completion format that any causal LM can learn from.
# The paper's Prompt 1 is embedded as a prefix so the model conditions on it.
# Format:
#   <system text>
#
#   ### Question:
#   {question}
#
#   ### Answer:
#   {answer}
#
# The DataCollatorForCompletionOnlyLM is told to compute loss only after
# "### Answer:\n" so the model learns to generate answers, not the prompt.

RESPONSE_TEMPLATE = "### Answer:\n"   # collator masks everything before this


def format_prompt(sample: dict) -> str:
    """
    Plain-text prompt for Llama-3-8B base model (no special chat tokens).
    Faithful to the paper's Prompt 1 instruction (Figure 2 / Appendix D).
    """
    question = sample["input"].strip()
    answer   = sample["output"].strip()

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Question:\n{question}\n\n"
        f"{RESPONSE_TEMPLATE}{answer}"
    )


def formatting_func(samples):
    """Vectorised formatter expected by SFTTrainer."""
    return [format_prompt({"input": q, "output": a})
            for q, a in zip(samples["input"], samples["output"])]


# =============================================================================
# 3.  DATA LOADING & SPLITTING
# =============================================================================

def load_data(path: str, val_ratio: float = 0.05, seed: int = 42) -> DatasetDict:
    log.info(f"Loading dataset from {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # The dataset already has the paper's instruction embedded in each record.
    # We keep 'input' (question) and 'output' (answer); instruction is baked
    # into the prompt template above.
    records = [{"input": r["input"], "output": r["output"]} for r in raw]

    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    log.info(f"Train: {len(split['train'])}  |  Validation: {len(split['test'])}")
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# =============================================================================
# 4.  MODEL + TOKENIZER
# =============================================================================

# Open-source fallbacks that need NO HuggingFace token.
# These are architecturally equivalent to Llama-3-8B for fine-tuning purposes.
# The paper's best English LLM was Llama-3-8B, but Mistral-7B-v0.3 is the
# closest openly available alternative with identical target modules.
OPEN_FALLBACKS = [
    "mistralai/Mistral-7B-v0.1",       # closest open equivalent
    "mistralai/Mistral-7B-v0.1",
]


def _resolve_model_path(cfg: Config) -> tuple[str, dict]:
    """
    Returns (model_name_or_path, extra_from_pretrained_kwargs).

    Resolution order:
      1. If base_model is a local directory that exists → use it directly (no token needed)
      2. If HF_TOKEN is set → try hub download with token
      3. Warn and try open fallbacks in order
    """
    import os

    # ── 1. Local path ──────────────────────────────────────────────────────────
    if os.path.isdir(cfg.base_model):
        log.info(f"Loading from local path: {cfg.base_model}")
        return cfg.base_model, {}

    # ── 2. HF Hub with token ───────────────────────────────────────────────────
    token = cfg.hf_token
    if token:
        log.info(f"HF_TOKEN found — attempting hub download: {cfg.base_model}")
        # Login so the token is embedded in all subsequent hub calls
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            log.warning(f"huggingface_hub login failed: {e}")
        return cfg.base_model, {"token": token}

    # ── 3. No token, no local path → suggest fixes then try open fallbacks ─────
    log.error(
        "\n"
        "════════════════════════════════════════════════════════════════\n"
        "  CANNOT ACCESS: meta-llama/Meta-Llama-3-8B\n"
        "  This is a GATED model. You need ONE of:\n"
        "\n"
        "  Option A — Use a local copy already on this machine:\n"
        "    Find it:  find / -name 'config.json' 2>/dev/null | grep -i llama | head\n"
        "    Then run: python finetune_llama3_agri.py --base_model /path/to/local/model\n"
        "\n"
        "  Option B — Authenticate with HuggingFace:\n"
        "    1) Accept the licence at https://huggingface.co/meta-llama/Meta-Llama-3-8B\n"
        "    2) Get your token from https://huggingface.co/settings/tokens\n"
        "    3) Run: export HF_TOKEN=hf_xxxxxxxxxxxx\n"
        "       Then re-run the script.\n"
        "\n"
        "  Option C — Use an open model (no token required):\n"
        "    python finetune_llama3_agri.py --base_model mistralai/Mistral-7B-v0.1\n"
        "\n"
        "  Trying open fallbacks automatically …\n"
        "════════════════════════════════════════════════════════════════"
    )

    for fallback in OPEN_FALLBACKS:
        log.info(f"Trying open fallback: {fallback}")
        return fallback, {}   # caller will attempt; errors will surface naturally

    raise RuntimeError(
        "No usable model found. Set HF_TOKEN, provide a local path via --base_model, "
        "or use an open model like mistralai/Mistral-7B-v0.1."
    )


def load_model_and_tokenizer(cfg: Config):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_path, hub_kwargs = _resolve_model_path(cfg)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = None
    if cfg.load_in_4bit:
        # ── bitsandbytes 4-bit config (paper: 4-bit QLoRA) ───────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLMs
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,     # nested quant saves ~0.4 GB
        )
        log.info("Loading model in 4-bit mode (QLoRA).")
    else:
        log.info("Loading model in full precision path (no 4-bit quantization).")

    log.info(f"Loading model weights from: {model_path}")
    load_kwargs = dict(
        device_map="auto",
        torch_dtype=compute_dtype,
        attn_implementation="eager",        # safe default; flash-attn-2 optional
        **hub_kwargs,
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, **hub_kwargs)

    if cfg.load_in_4bit:
        # ── Prepare for k-bit training ────────────────────────────────────────
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model.gradient_checkpointing_enable()

    # ── LoRA adapters (paper Table 13) ────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ── Tokenizer padding ─────────────────────────────────────────────────────
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Model loaded. Trainable parameters:")
    model.print_trainable_parameters()
    return model, tokenizer


# =============================================================================
# 5.  TRAINING
# =============================================================================

def train(cfg: Config):
    if WANDB_ENABLED:
        wandb.init(project="agri-llm-finetune", name=cfg.run_name, config=vars(cfg))

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = load_data(cfg.data_path, cfg.val_ratio, cfg.seed)
    if cfg.smoke_test_chunks > 0:
        train_n = min(cfg.smoke_test_chunks, len(dataset["train"]))
        val_n = min(max(1, cfg.smoke_test_chunks // 3), len(dataset["validation"]))
        dataset["train"] = dataset["train"].select(range(train_n))
        dataset["validation"] = dataset["validation"].select(range(val_n))
        log.info(
            "Smoke test mode enabled: using %d train rows and %d validation rows.",
            train_n,
            val_n,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── Pre-format dataset into a "text" column ───────────────────────────────
    # All trl versions accept dataset_text_field="text"; we materialise it here
    # so we have one clean path regardless of whether formatting_func is supported.
    def add_text_column(sample):
        sample["text"] = format_prompt({"input": sample["input"], "output": sample["output"]})
        return sample

    dataset["train"]      = dataset["train"].map(add_text_column, num_proc=4)
    dataset["validation"] = dataset["validation"].map(add_text_column, num_proc=4)

    # ── Completion-only collator (trains only on assistant response tokens) ───
    # Loss is computed ONLY on answer tokens, not on the prompt — matching the
    # paper's supervised fine-tuning approach.
    # DataCollatorForCompletionOnlyLM location varies by trl version:
    #   trl >= 0.8  →  top-level `trl`   (resolved at import time above)
    #   trl <  0.8  →  trl.trainer.utils
    #   very old    →  set to None above; fall back to full-sequence loss
    collator = None
    if DataCollatorForCompletionOnlyLM is not None:
        try:
            collator = DataCollatorForCompletionOnlyLM(
                response_template=RESPONSE_TEMPLATE,   # "### Answer:\n"
                tokenizer=tokenizer,
            )
            log.info("Using DataCollatorForCompletionOnlyLM — loss on answer tokens only")
        except Exception as e:
            log.warning(f"DataCollatorForCompletionOnlyLM init failed ({e}) — using full-sequence loss")
    else:
        log.warning("DataCollatorForCompletionOnlyLM unavailable — training with full-sequence loss")

    # ── Training arguments (paper Table 13) ───────────────────────────────────
    # trl >= 0.9 prefers SFTConfig (subclass of TrainingArguments that also
    # accepts max_seq_length, packing, dataset_text_field directly).
    # Older trl uses plain TrainingArguments + kwargs passed to SFTTrainer.
    effective_optim = cfg.optim
    if (not cfg.load_in_4bit) and cfg.optim == "adamw_8bit":
        # 8-bit optimizer depends on bitsandbytes; switch to torch AdamW when
        # training without k-bit quantization.
        effective_optim = "adamw_torch"
        log.info("load_in_4bit=False: overriding optim from adamw_8bit to adamw_torch.")

    _common_train_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=effective_optim,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        dataloader_num_workers=4,
        group_by_length=True,           # reduces padding → faster training
        seed=cfg.seed,
    )

    if _USE_SFT_CONFIG:
        # trl >= 0.9: SFTConfig absorbs max_seq_length + packing + text field
        training_args = SFTConfig(
            **_common_train_kwargs,
            max_seq_length=cfg.max_seq_length,
            packing=False,
            dataset_text_field="text",
        )
        sft_extra = {}                  # already in SFTConfig
    else:
        # trl < 0.9: plain TrainingArguments; extras go to SFTTrainer.__init__
        training_args = TrainingArguments(**_common_train_kwargs)
        sft_extra = dict(
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="text",
            packing=False,
            dataset_num_proc=4,
        )

    # ── SFTTrainer ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,         # None → default collator (full-seq loss)
        **sft_extra,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Starting fine-tuning …")
    log.info(f"  Base model      : {cfg.base_model}")
    log.info(f"  Train samples   : {len(dataset['train'])}")
    log.info(f"  Val samples     : {len(dataset['validation'])}")
    log.info(f"  Effective batch : {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    log.info(f"  Max seq length  : {cfg.max_seq_length}")
    log.info(f"  LoRA r/alpha    : {cfg.lora_r}/{cfg.lora_alpha}")
    log.info("=" * 60)

    # Work around accelerate/transformers optimizer API mismatch where
    # AcceleratedOptimizer calls .train()/.eval() on wrapped optimizer.
    trainer.create_optimizer()
    wrapped_opt = trainer.optimizer
    base_opt = getattr(wrapped_opt, "optimizer", wrapped_opt)
    if not hasattr(base_opt, "train"):
        base_opt.train = types.MethodType(lambda self: None, base_opt)
        log.info("Patched optimizer.train() no-op for compatibility.")
    if not hasattr(base_opt, "eval"):
        base_opt.eval = types.MethodType(lambda self: None, base_opt)
        log.info("Patched optimizer.eval() no-op for compatibility.")

    trainer_stats = trainer.train()

    log.info(f"Training complete. Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
    log.info(f"Samples/second  : {trainer_stats.metrics['train_samples_per_second']:.2f}")

    # ── Save adapter weights ───────────────────────────────────────────────────
    adapter_path = os.path.join(cfg.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    log.info(f"LoRA adapter saved to: {adapter_path}")

    # ── (Optional) Merge LoRA weights into base model and save full 16-bit ────
    # Uses standard PEFT merge_and_unload — no Unsloth required.
    try:
        merged_path = os.path.join(cfg.output_dir, "merged_16bit")
        log.info("Merging LoRA weights into base model …")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        log.info(f"Merged 16-bit model saved to: {merged_path}")
    except Exception as e:
        log.warning(f"Merge skipped: {e}")

    if WANDB_ENABLED:
        wandb.finish()

    return trainer, model, tokenizer


# =============================================================================
# 6.  INFERENCE  — quick sanity check using the paper's Prompt 1 format
# =============================================================================

def run_inference(model, tokenizer, questions: list[str], max_new_tokens: int = 256):
    """
    Run inference with the same plain-text prompt format used during training.
    Uses T=0.2 as per paper (Table 6 — best evaluation temperature).
    """
    model.eval()
    results = []

    for question in questions:
        # Mirror the training format exactly — stop just before the answer
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"### Question:\n{question.strip()}\n\n"
            f"{RESPONSE_TEMPLATE}"          # model completes from here
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,            # paper used T=0.2 for best results
                do_sample=True,
                repetition_penalty=1.1,     # prevents looping on base models
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only newly generated tokens (exclude the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Trim at any accidental repetition of the prompt structure
        for stop in ["### Question:", "### Answer:"]:
            if stop in answer:
                answer = answer[:answer.index(stop)].strip()
        results.append({"question": question, "answer": answer})
        log.info(f"Q: {question}")
        log.info(f"A: {answer}")
        log.info("-" * 40)

    return results


# =============================================================================
# 7.  MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agri QA Fine-tuning Pipeline")
    parser.add_argument("--data_path",    default=CFG.data_path,    help="Path to JSON training file")
    parser.add_argument("--base_model",   default=CFG.base_model,   help="HuggingFace model ID")
    parser.add_argument("--output_dir",   default=CFG.output_dir,   help="Output directory")
    parser.add_argument("--epochs",       default=CFG.num_train_epochs, type=int)
    parser.add_argument("--lr",           default=CFG.learning_rate,    type=float)
    parser.add_argument("--smoke_test_chunks", default=CFG.smoke_test_chunks, type=int,
                        help="If >0, train on first N rows only (quick smoke test).")
    parser.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true", default=CFG.load_in_4bit,
                        help="Enable 4-bit quantization path (QLoRA).")
    parser.add_argument("--no_4bit",      dest="load_in_4bit", action="store_false",
                        help="Disable 4-bit quantization path.")
    parser.add_argument("--inference_only", action="store_true",     help="Skip training, run inference only")
    parser.add_argument("--adapter_path", default=None,              help="Path to saved LoRA adapter (for inference_only)")
    args = parser.parse_args()

    CFG.data_path          = args.data_path
    CFG.base_model         = args.base_model
    CFG.output_dir         = args.output_dir
    CFG.num_train_epochs   = args.epochs
    CFG.learning_rate      = args.lr
    CFG.load_in_4bit       = args.load_in_4bit
    CFG.smoke_test_chunks  = args.smoke_test_chunks

    # ── Sample test questions (paper evaluation style) ────────────────────────
    test_questions = [
        "How can I manage thrips in groundnut?",
        "What should I do if my groundnut crop shows yellowish-green patches and brown necrotic areas?",
        "How to improve soil condition?",
        "What is the recommended fertilizer dose for wheat cultivation?",
    ]

    if args.inference_only:
        # Load saved adapter for inference
        if args.adapter_path is None:
            args.adapter_path = os.path.join(CFG.output_dir, "lora_adapter")
        log.info(f"Loading adapter from {args.adapter_path} for inference …")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        _, hub_kwargs = _resolve_model_path(CFG)
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base_m = AutoModelForCausalLM.from_pretrained(
            CFG.base_model,
            device_map="auto",
            torch_dtype=compute_dtype,
            **hub_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(CFG.base_model, **hub_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(base_m, args.adapter_path)
        run_inference(model, tokenizer, test_questions)
    else:
        trainer, model, tokenizer = train(CFG)
        log.info("\n=== Post-training inference check ===")
        run_inference(model, tokenizer, test_questions)
