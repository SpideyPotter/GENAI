"""
QA Synthesis Script using DeepSeek-R1-Distill-Llama-8B
Target: DGX H200 | 16 CPU cores | 35 GB VRAM
Input : chunked data dir (*.txt, *.json, *.jsonl, *.md)
        + merged exports: *.txt with lines "### CHUNK N" and optional "SOURCE: ..."
          (e.g. green agriculture corpus chunked at ~300 words / 50 overlap)
Output: greenagriculture_QA/ by default (or --output_dir)

CLI: --dtype fp16|bf16|4bit (full-precision paths vs NF4), --batch_size, paths,
     --no_resume / --fresh (checkpoint control under output_dir/checkpoints/).
"""

import json
import os
import re
import time
import logging
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from tqdm import tqdm

# ─────────────────────────── Configuration ───────────────────────────────────

MODEL_ID        = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
INPUT_DIR       = Path("greenagriculture_chunked")
OUTPUT_DIR      = Path("greenagriculture_QA")

# Hardware budget
NUM_CPU_WORKERS = 16          # matches allotted CPU cores
MAX_VRAM_GB     = 35          # stay within 35 GB VRAM
BATCH_SIZE      = 8           # default for fp16/bf16 on ~35GB; lower with --dtype 4bit or if OOM
MAX_NEW_TOKENS  = 512         # max tokens per QA output
MAX_CTX_CHARS   = 2500        # truncate context to keep prompt manageable

# Supported chunk file extensions
SUPPORTED_EXT = {".txt", ".json", ".jsonl", ".md"}

# Merged chunk files: "### CHUNK 1" + optional "SOURCE: path" + blank line + body
_MERGED_CHUNK_HEADER = re.compile(
    r"^###\s+CHUNK\s+(\d+)\s*\r?\n(?:SOURCE:\s*(.+?)\s*\r?\n)?(?:\s*\r?\n)?",
    re.MULTILINE,
)


def parse_merged_chunk_txt(text: str, filepath: Path) -> list[dict] | None:
    """
    Parse a single .txt that concatenates many logical chunks (300w/50 overlap style).
    Returns None if the file does not look like this format (caller falls back to 1 chunk = whole file).
    """
    sample = text[:8000] if len(text) > 8000 else text
    if "### CHUNK" not in sample:
        return None
    matches = list(_MERGED_CHUNK_HEADER.finditer(text))
    if not matches or matches[0].start() > 500:
        return None
    chunks: list[dict] = []
    for i, m in enumerate(matches):
        num = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        chunks.append({
            "source": filepath.name,
            "chunk_id": f"{filepath.stem}_chunk_{num}",
            "text": body,
        })
    return chunks if chunks else None

# ─────────────────────────── Prompt Template ─────────────────────────────────

SYSTEM_PROMPT = "You are an expert in agriculture."

USER_TEMPLATE = """\
Given the following context, generate high-quality,
diverse, and non-redundant question-answer pairs.
Rules:
- Questions must be clear and useful for real users
- Answers must be concise and strictly grounded in the context
- Do NOT add any external knowledge
- Cover different aspects (causes, symptoms, solutions, definitions)
- Avoid repeating similar questions
Context:
{context}
Output format:
Question: ...
Answer: ..."""

# ─────────────────────────── Logging Setup ───────────────────────────────────

def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "qa_synthesis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

logger = logging.getLogger(__name__)


def log_cuda_vram_vs_budget() -> None:
    """Log per-device VRAM; warn if any visible GPU is below MAX_VRAM_GB (GiB)."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available — running on CPU if the model allows it (very slow).")
        return
    for idx in range(torch.cuda.device_count()):
        total_bytes = torch.cuda.get_device_properties(idx).total_memory
        total_gib = total_bytes / (1024**3)
        name = torch.cuda.get_device_properties(idx).name
        logger.info("GPU %s: %s — %.1f GiB total", idx, name, total_gib)
        if total_gib + 1e-6 < MAX_VRAM_GB:
            logger.warning(
                "GPU %s has %.1f GiB total, below configured budget of %s GiB — risk of OOM; try smaller --batch_size.",
                idx,
                total_gib,
                MAX_VRAM_GB,
            )


# ─────────────────────────── File Loaders ────────────────────────────────────

def load_chunks_from_file(filepath: Path) -> list[dict]:
    """
    Returns a list of dicts: {"source": str, "chunk_id": str, "text": str}
    Handles .txt, .md, .json, .jsonl
    """
    chunks = []
    suffix = filepath.suffix.lower()

    try:
        if suffix in {".txt", ".md"}:
            text = filepath.read_text(encoding="utf-8").strip()
            if not text:
                return chunks
            merged = parse_merged_chunk_txt(text, filepath)
            if merged is not None:
                chunks.extend(merged)
            else:
                chunks.append({
                    "source": filepath.name,
                    "chunk_id": filepath.stem,
                    "text": text,
                })

        elif suffix == ".json":
            data = json.loads(filepath.read_text(encoding="utf-8"))
            # Accept list-of-dicts or single dict or plain list of strings
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or item.get("chunk") or ""
                        cid  = item.get("id") or item.get("chunk_id") or f"{filepath.stem}_{idx}"
                    else:
                        text = str(item)
                        cid  = f"{filepath.stem}_{idx}"
                    if text.strip():
                        chunks.append({"source": filepath.name, "chunk_id": str(cid), "text": text})
            elif isinstance(data, dict):
                text = data.get("text") or data.get("content") or ""
                if text.strip():
                    chunks.append({"source": filepath.name, "chunk_id": filepath.stem, "text": text})

        elif suffix == ".jsonl":
            with filepath.open(encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as err:
                        logger.warning("Skipping invalid JSONL line %s in %s: %s", idx, filepath.name, err)
                        continue
                    if not isinstance(item, dict):
                        text = str(item)
                        cid = f"{filepath.stem}_{idx}"
                    else:
                        text = item.get("text") or item.get("content") or item.get("chunk") or str(item)
                        cid = item.get("id") or item.get("chunk_id") or f"{filepath.stem}_{idx}"
                    if text.strip():
                        chunks.append({"source": filepath.name, "chunk_id": str(cid), "text": text})

    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")

    return chunks


def collect_all_chunks(input_dir: Path) -> list[dict]:
    if not input_dir.exists():
        logger.error(
            "Input directory does not exist (or is not visible to this process): %s",
            input_dir.resolve(),
        )
        return []
    if not input_dir.is_dir():
        logger.error("Input path is not a directory: %s", input_dir.resolve())
        return []

    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXT]
    logger.info(f"Found {len(files)} chunk files in {input_dir}")

    all_chunks = []
    with ThreadPoolExecutor(max_workers=min(NUM_CPU_WORKERS, len(files) or 1)) as exe:
        futures = {exe.submit(load_chunks_from_file, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading files"):
            all_chunks.extend(future.result())

    logger.info(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks

# ─────────────────────────── Model Loading ───────────────────────────────────

def load_model(dtype: str) -> tuple:
    """
    dtype:
      fp16 / bf16 — full weights (higher quality, more VRAM; good default on 35GB+).
      4bit       — NF4 quantised weights (less VRAM, slightly lower fidelity).
    """
    logger.info("Loading model: %s (dtype=%s)", MODEL_ID, dtype)
    log_cuda_vram_vs_budget()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if dtype == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    elif dtype == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    elif dtype == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unknown dtype: {dtype!r} (use fp16, bf16, or 4bit)")

    model.eval()
    logger.info("Model loaded successfully.")
    return tokenizer, model

# ─────────────────────────── QA Generation ───────────────────────────────────

def build_prompt(context: str, tokenizer) -> str:
    context = context[:MAX_CTX_CHARS]   # hard-truncate to avoid OOM
    user_msg = USER_TEMPLATE.format(context=context)
    # Prefer tokenizer chat template (matches HF model card, avoids BOS duplication issues)
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback: manual format for DeepSeek-R1–style token strings
    return (
        f"<｜begin▁of▁sentence｜>"
        f"<｜User｜>{SYSTEM_PROMPT}\n\n{user_msg}<｜Assistant｜>"
    )


def parse_qa_pairs(raw_text: str) -> list[dict]:
    """Extract Q/A pairs from model output."""
    pairs = []
    # Split on 'Question:' boundaries
    blocks = re.split(r"(?=Question\s*:)", raw_text, flags=re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q_match = re.search(r"Question\s*:\s*(.+?)(?=Answer\s*:|$)", block, re.IGNORECASE | re.DOTALL)
        a_match = re.search(r"Answer\s*:\s*(.+?)(?=Question\s*:|$)",  block, re.IGNORECASE | re.DOTALL)
        if q_match and a_match:
            q = q_match.group(1).strip()
            a = a_match.group(1).strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
    return pairs


def generate_qa_batch(
    batch_chunks: list[dict],
    tokenizer,
    model,
) -> list[dict]:
    prompts = [build_prompt(c["text"], tokenizer) for c in batch_chunks]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072,
    )
    # device_map="auto" can leave model.device unset or misleading; use first parameter device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Greedy-only: set sampling params to neutral values so merged model defaults
    # (e.g. temperature/top_p) do not trigger transformers verbosity warnings.
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_cfg)

    results = []
    input_len = inputs["input_ids"].shape[1]
    for i, chunk in enumerate(batch_chunks):
        generated_ids = outputs[i][input_len:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pairs = parse_qa_pairs(raw)
        results.append({
            "source":   chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "context":  chunk["text"][:500],   # store snippet for reference
            "qa_pairs": pairs,
            "raw_output": raw,                 # keep for debugging
        })
    return results

# ─────────────────────────── Checkpoints (resumable runs) ────────────────────

def chunk_key(chunk: dict) -> str:
    """Stable id for a chunk within an input corpus (source file name + chunk id)."""
    return f'{chunk["source"]}\x1f{chunk["chunk_id"]}'


def checkpoint_paths(output_dir: Path) -> tuple[Path, Path]:
    base = output_dir / "checkpoints"
    return base, base / "completed.jsonl"


def load_completed_checkpoint(output_dir: Path) -> tuple[list[dict], set[str]]:
    """Load prior results and processed keys from completed.jsonl (if present)."""
    _, jsonl_path = checkpoint_paths(output_dir)
    if not jsonl_path.is_file():
        return [], set()
    results: list[dict] = []
    keys: set[str] = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping bad checkpoint line in %s: %s", jsonl_path.name, e)
                continue
            if not isinstance(rec, dict) or "source" not in rec or "chunk_id" not in rec:
                logger.warning("Skipping malformed checkpoint record in %s", jsonl_path.name)
                continue
            results.append(rec)
            keys.add(chunk_key(rec))
    return results, keys


def append_checkpoint_jsonl(output_dir: Path, new_rows: list[dict]) -> None:
    """Append finished chunk records and sync to disk so a crash can resume."""
    if not new_rows:
        return
    ckpt_dir, jsonl_path = checkpoint_paths(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def clear_checkpoints(output_dir: Path) -> None:
    ckpt_dir, _ = checkpoint_paths(output_dir)
    if ckpt_dir.is_dir():
        shutil.rmtree(ckpt_dir)


# ─────────────────────────── Output Writers ──────────────────────────────────

def save_results(
    all_results: list[dict],
    output_dir: Path,
    *,
    dtype: str | None = None,
    batch_size: int | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. One JSON file per source file ─────────────────────────────────────
    grouped: dict[str, list] = {}
    for r in all_results:
        grouped.setdefault(r["source"], []).append(r)

    for source, records in grouped.items():
        stem = Path(source).stem
        out_path = output_dir / f"{stem}_qa.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    # ── 2. Flat JSONL with every QA pair (easy for fine-tuning) ──────────────
    flat_path = output_dir / "all_qa_pairs.jsonl"
    with open(flat_path, "w", encoding="utf-8") as f:
        for r in all_results:
            for pair in r["qa_pairs"]:
                record = {
                    "source":   r["source"],
                    "chunk_id": r["chunk_id"],
                    "question": pair["question"],
                    "answer":   pair["answer"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 3. Summary stats ─────────────────────────────────────────────────────
    total_pairs = sum(len(r["qa_pairs"]) for r in all_results)
    empty_chunks = sum(1 for r in all_results if not r["qa_pairs"])
    stats = {
        "total_chunks_processed": len(all_results),
        "total_qa_pairs_generated": total_pairs,
        "chunks_with_no_output": empty_chunks,
        "output_files": len(grouped),
    }
    if dtype is not None:
        stats["dtype"] = dtype
    if batch_size is not None:
        stats["batch_size"] = batch_size
    stats_path = output_dir / "synthesis_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved {total_pairs} QA pairs across {len(all_results)} chunks → {output_dir}")
    logger.info(f"Stats: {stats}")

# ─────────────────────────── Main ────────────────────────────────────────────

def main(args):
    setup_logging()
    logger.info("=" * 60)
    logger.info("QA Synthesis Pipeline — DeepSeek-R1-Distill-Llama-8B")
    logger.info("=" * 60)

    if args.fresh:
        clear_checkpoints(OUTPUT_DIR)
        logger.info("Removed prior checkpoints under %s (--fresh).", OUTPUT_DIR / "checkpoints")

    # 1. Load chunks
    chunks = collect_all_chunks(INPUT_DIR)
    if not chunks:
        logger.error(f"No chunks found in {INPUT_DIR}. Exiting.")
        return
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
        logger.info("Limiting run to first %s chunks (--max_chunks)", len(chunks))

    all_results: list[dict] = []
    processed_keys: set[str] = set()
    if args.resume:
        all_results, processed_keys = load_completed_checkpoint(OUTPUT_DIR)
        if processed_keys:
            logger.info(
                "Resuming: loaded %s completed chunks from checkpoint → %s",
                len(processed_keys),
                checkpoint_paths(OUTPUT_DIR)[1],
            )
        before = len(chunks)
        chunks = [c for c in chunks if chunk_key(c) not in processed_keys]
        skipped = before - len(chunks)
        if skipped:
            logger.info("Skipping %s chunks already present in checkpoint.", skipped)
    else:
        _, jsonl_path = checkpoint_paths(OUTPUT_DIR)
        if jsonl_path.is_file():
            jsonl_path.unlink()
            logger.info("Removed %s to start a new checkpoint log (--no_resume).", jsonl_path)
    if not chunks:
        logger.info("Nothing left to generate; writing final outputs from checkpoint only.")
        save_results(all_results, OUTPUT_DIR, dtype=args.dtype, batch_size=BATCH_SIZE)
        logger.info("Pipeline finished.")
        return

    # 2. Load model
    tokenizer, model = load_model(args.dtype)
    logger.info("Inference settings: dtype=%s batch_size=%s", args.dtype, BATCH_SIZE)

    # 3. Batched inference (append to checkpoint after each successful batch / chunk)
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    for batch_idx in tqdm(range(0, len(chunks), BATCH_SIZE), total=total_batches, desc="Generating QA"):
        batch = chunks[batch_idx : batch_idx + BATCH_SIZE]
        try:
            results = generate_qa_batch(batch, tokenizer, model)
            all_results.extend(results)
            append_checkpoint_jsonl(OUTPUT_DIR, results)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM on batch {batch_idx//BATCH_SIZE}. Processing one-by-one.")
            torch.cuda.empty_cache()
            for single in batch:
                try:
                    r = generate_qa_batch([single], tokenizer, model)
                    all_results.extend(r)
                    append_checkpoint_jsonl(OUTPUT_DIR, r)
                except Exception as e:
                    logger.error(f"Skipping chunk {single['chunk_id']}: {e}")
        except Exception as e:
            logger.error(f"Batch {batch_idx//BATCH_SIZE} failed: {e}")
            for c in batch:
                logger.error("  chunk not saved for this run: %s", c.get("chunk_id", "?"))

        bn = batch_idx // BATCH_SIZE + 1
        if bn % 50 == 0:
            elapsed = time.time() - start_time
            pairs_so_far = sum(len(r["qa_pairs"]) for r in all_results)
            logger.info(
                "progress: batch %s/%s | %.1f new chunks/min | %s QA pairs in session",
                bn,
                total_batches,
                ((batch_idx + len(batch)) / max(elapsed / 60, 1e-6)),
                pairs_so_far,
            )

    elapsed = time.time() - start_time
    logger.info(f"Inference complete in {elapsed/60:.1f} min")

    # 4. Save
    save_results(all_results, OUTPUT_DIR, dtype=args.dtype, batch_size=BATCH_SIZE)
    logger.info("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA Synthesis from chunked data")
    parser.add_argument("--input_dir",  default=str(INPUT_DIR),  help="Path to chunked data folder")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Path to save QA outputs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Inference batch size")
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "4bit"),
        default="fp16",
        help="fp16/bf16: full weights (quality). 4bit: NF4 quant (less VRAM).",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="Process only the first N chunks (0 = all). Use e.g. 80 for a quick health check.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore checkpoints: process all chunks and replace output_dir/checkpoints/completed.jsonl on first write.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete output_dir/checkpoints/ before running, then start an empty checkpoint log.",
    )
    args = parser.parse_args()
    args.resume = not args.no_resume

    INPUT_DIR   = Path(args.input_dir)
    OUTPUT_DIR  = Path(args.output_dir)
    BATCH_SIZE  = args.batch_size

    main(args)
