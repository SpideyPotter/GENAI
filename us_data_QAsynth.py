"""
QA Synthesis wrapper for US dataset.

This script reuses `synthesis.py` exactly (same model id, prompt, generation
config, batching, and checkpointing). It only overrides the default input and
output directories so you generate QA into `QA_us_data/` from
`us_web_chunks/us_data_chunks.jsonl` (by passing --input_dir us_web_chunks).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _import_synthesis_module():
    # Ensure we can import `synthesis.py` when executed as a script.
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    import synthesis  # type: ignore

    return synthesis


def main():
    synthesis = _import_synthesis_module()

    parser = argparse.ArgumentParser(description="US QA Synthesis wrapper")
    parser.add_argument(
        "--input_dir",
        default="us_web_chunks",
        help="Path to chunked data folder (default: us_web_chunks)",
    )
    parser.add_argument(
        "--output_dir",
        default="QA_us_data",
        help="Path to save QA outputs (default: QA_us_data)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=synthesis.BATCH_SIZE,
        help="Inference batch size (default: synthesis.py default)",
    )
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
        help="Process only the first N chunks (0 = all). Use for quick health checks.",
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

    # Mirror `synthesis.py`'s variable overrides done in its __main__ block.
    synthesis.INPUT_DIR = Path(args.input_dir)
    synthesis.OUTPUT_DIR = Path(args.output_dir)
    synthesis.BATCH_SIZE = args.batch_size

    synthesis.main(args)


if __name__ == "__main__":
    # Avoid torch multithreading contention on some systems; synthesis is CPU-threaded already.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

