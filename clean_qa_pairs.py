#!/usr/bin/env python3
"""
Post-process synthesis QA JSONL (e.g. QAdata/all_qa_pairs.jsonl).

Fixes common artifacts from tokenizer / model output:
  - U+0120 (Ġ) → space (SentencePiece-style word boundary)
  - U+010A (Ċ) → space (newline marker in flat Q/A text)
  - Collapse repeated whitespace
  - Strip redundant ** markdown wrappers

Optional filters: minimum answer/question length, dedupe per chunk or globally (first row wins).

Usage:
  python clean_qa_pairs.py
  python clean_qa_pairs.py --input QAdata/all_qa_pairs.jsonl --output QAdata/all_qa_pairs.cleaned.jsonl
  python clean_qa_pairs.py --input QAdata/all_qa_pairs.jsonl --output QAdata/all_qa_pairs.cleaned.jsonl --min-answer-len 20 --dedupe
  python clean_qa_pairs.py --input QAdata/all_qa_pairs.cleaned.jsonl --output finetune_QA_stage1/all_qa_pairs.jsonl --dedupe-global --min-question-len 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Decoder / BPE surface forms seen in DeepSeek–Llama synthesis exports
_U_WORD = "\u0120"  # often rendered as Ġ
_U_LINE = "\u010a"  # often rendered as Ċ


def strip_outer_markdown_bold(s: str) -> str:
    """Remove leading/trailing ** pairs; repeat until stable."""
    t = s.strip()
    while len(t) > 4 and t.startswith("**") and t.endswith("**"):
        inner = t[2:-2].strip()
        if inner == t:
            break
        t = inner
    return t


def normalize_qa_text(s: str) -> str:
    if not s:
        return s
    t = s.replace(_U_WORD, " ").replace(_U_LINE, " ")
    t = strip_outer_markdown_bold(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n+", " ", t)
    return t.strip()


def norm_question_key(q: str) -> str:
    """Stable key for dedupe (casefold + whitespace collapse)."""
    return re.sub(r"\s+", " ", q.strip().casefold())


def looks_truncated(answer: str, min_len: int = 40) -> bool:
    """Heuristic: long answer ending mid-clause without terminal punctuation."""
    a = answer.strip()
    if len(a) < min_len:
        return False
    if re.search(r"\*\*[^*]*$", a):  # unclosed bold
        return True
    if a.endswith(("...", "…")):
        return True
    last = a[-1]
    if last in ".!?;:\"'）】」":
        return False
    if last.isalnum() and not re.search(r"\b(etc|al|vs)\.?$", a, re.I):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean QA JSONL from synthesis.py exports.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("QAdata/all_qa_pairs.jsonl"),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("QAdata/all_qa_pairs.cleaned.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--min-answer-len",
        type=int,
        default=0,
        help="Drop rows whose cleaned answer is shorter than this (0 = keep all)",
    )
    parser.add_argument(
        "--min-question-len",
        type=int,
        default=0,
        help="Drop rows whose cleaned question is shorter than this (0 = keep all)",
    )
    parser.add_argument(
        "--drop-truncated",
        action="store_true",
        help="Drop rows whose cleaned answer looks cut off mid-sentence",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Keep first row per (chunk_id, normalized question)",
    )
    parser.add_argument(
        "--dedupe-global",
        action="store_true",
        help="Keep first row per normalized question (across all chunks; best for SFT without context)",
    )
    args = parser.parse_args()

    if args.dedupe and args.dedupe_global:
        sys.stderr.write("Use only one of --dedupe or --dedupe-global.\n")
        sys.exit(2)

    if not args.input.is_file():
        sys.stderr.write(f"Input not found: {args.input.resolve()}\n")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    dropped_short = 0
    dropped_trunc = 0
    dropped_dup = 0
    seen_chunk: set[tuple[str, str]] = set()
    seen_global: set[str] = set()

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Skip bad JSON line {total_in}: {e}\n")
                continue

            q = normalize_qa_text(str(row.get("question", "")))
            a = normalize_qa_text(str(row.get("answer", "")))
            if not q or not a:
                dropped_short += 1
                continue
            if args.min_question_len and len(q) < args.min_question_len:
                dropped_short += 1
                continue
            if args.min_answer_len and len(a) < args.min_answer_len:
                dropped_short += 1
                continue
            if args.drop_truncated and looks_truncated(a):
                dropped_trunc += 1
                continue
            if args.dedupe_global:
                gkey = norm_question_key(q)
                if gkey in seen_global:
                    dropped_dup += 1
                    continue
                seen_global.add(gkey)
            elif args.dedupe:
                ckey = (str(row.get("chunk_id", "")), norm_question_key(q))
                if ckey in seen_chunk:
                    dropped_dup += 1
                    continue
                seen_chunk.add(ckey)

            out = {
                "source": row.get("source", ""),
                "chunk_id": row.get("chunk_id", ""),
                "question": q,
                "answer": a,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            total_out += 1

    print(
        f"Read {total_in} rows → wrote {total_out} rows → {args.output.resolve()}\n"
        f"  dropped (empty/short): {dropped_short}\n"
        f"  dropped (truncated):   {dropped_trunc}\n"
        f"  dropped (dedupe):      {dropped_dup}",
        flush=True,
    )


if __name__ == "__main__":
    main()
