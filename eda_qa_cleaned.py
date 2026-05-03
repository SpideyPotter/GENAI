#!/usr/bin/env python3
"""
EDA + lightweight quality checks for cleaned QA JSONL (e.g. all_qa_pairs.cleaned.jsonl).

Usage:
  python eda_qa_cleaned.py
  python eda_qa_cleaned.py --input QAdata/all_qa_pairs.cleaned.jsonl --json-out QAdata/eda_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s, flags=re.UNICODE))


def norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().casefold())


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA and quality checks for cleaned QA JSONL.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("QAdata/all_qa_pairs.cleaned.jsonl"),
        help="Path to cleaned JSONL",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write numeric summary as JSON",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        sys.stderr.write(f"Missing input: {args.input.resolve()}\n")
        sys.exit(1)

    rows: list[dict] = []
    bad = 0
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1

    n = len(rows)
    if n == 0:
        print("No rows loaded.")
        sys.exit(1)

    q_lens = [len(r.get("question", "")) for r in rows]
    a_lens = [len(r.get("answer", "")) for r in rows]
    q_words = [word_count(r.get("question", "")) for r in rows]
    a_words = [word_count(r.get("answer", "")) for r in rows]

    q_sorted = sorted(q_lens)
    a_sorted = sorted(a_lens)

    sources = Counter(str(r.get("source", "")) for r in rows)
    chunk_ids = [str(r.get("chunk_id", "")) for r in rows]
    unique_chunks = len(set(chunk_ids))
    per_chunk = Counter(chunk_ids)

    questions_raw = [str(r.get("question", "")).strip() for r in rows]
    ends_question_mark = sum(1 for q in questions_raw if q.endswith("?"))
    starts_what_how = sum(
        1 for q in questions_raw
        if re.match(r"^(what|how|why|when|where|which|who)\b", q, re.I)
    )

    # Duplicate questions (normalized)
    q_norm = [norm_q(q) for q in questions_raw]
    q_norm_counts = Counter(q_norm)
    dup_q = sum(1 for c in q_norm_counts.values() if c > 1)
    rows_in_dup_groups = sum(c for c in q_norm_counts.values() if c > 1)

    # Residual tokenizer junk
    junk_pat = re.compile(r"[\u0120\u010a]|Ġ|Ċ")
    junk_q = sum(1 for q in questions_raw if junk_pat.search(q))
    junk_a = sum(1 for r in rows if junk_pat.search(str(r.get("answer", ""))))

    # Heuristic quality flags
    very_short_q = sum(1 for L in q_lens if L < 15)
    very_short_a = sum(1 for L in a_lens if L < 20)
    long_a = sum(1 for L in a_lens if L > 800)

    # Answer mostly repeats question tokens (cheap overlap)
    weak_answer = 0
    for r in rows:
        q, a = str(r.get("question", "")), str(r.get("answer", ""))
        qw = set(re.findall(r"\b\w{4,}\b", q.lower()))
        aw = set(re.findall(r"\b\w{4,}\b", a.lower()))
        if len(aw) >= 3 and qw:
            inter = len(qw & aw)
            if inter / max(len(aw), 1) > 0.85:
                weak_answer += 1

    # Numbered list debris in answer (model enumeration)
    enum_debris = sum(
        1 for r in rows
        if re.search(r"^\s*\d+\.\s*\*\*", str(r.get("answer", "")), re.M)
        or re.search(r"\b\d+\.\s*\*\*", str(r.get("answer", "")))
    )

    # Quality score 0–100 (heuristic)
    issues = 0
    issues += min(30, dup_q * 2)  # duplicate question templates
    issues += min(20, very_short_a * 0.5)
    issues += min(15, junk_q + junk_a)
    issues += min(15, enum_debris * 0.3)
    issues += min(10, weak_answer * 0.2)
    score = max(0, 100 - issues)

    def band(s: float) -> str:
        if s >= 80:
            return "good"
        if s >= 60:
            return "fair"
        return "weak"

    # Pairs per chunk distribution
    ppc = sorted(per_chunk.values())
    ppc_p50 = _percentile(ppc, 50)
    ppc_p90 = _percentile(ppc, 90)
    ppc_max = max(per_chunk.values()) if per_chunk else 0

    report_lines = [
        f"=== EDA: {args.input} ===",
        f"rows:              {n}  (json_errors: {bad})",
        f"unique chunk_id:   {unique_chunks}",
        f"unique sources:    {len(sources)}  {dict(sources)}",
        "",
        "--- Length (characters) ---",
        f"question  min/med/p90/max: {min(q_lens)} / {statistics.median(q_sorted):.0f} / {_percentile(q_sorted, 90):.0f} / {max(q_lens)}",
        f"answer    min/med/p90/max: {min(a_lens)} / {statistics.median(a_sorted):.0f} / {_percentile(a_sorted, 90):.0f} / {max(a_lens)}",
        "",
        "--- Length (words, approx) ---",
        f"question  mean: {statistics.mean(q_words):.1f}  median: {statistics.median(q_words):.0f}",
        f"answer    mean: {statistics.mean(a_words):.1f}  median: {statistics.median(a_words):.0f}",
        "",
        "--- Shape / style ---",
        f"questions ending with '?': {ends_question_mark} ({100 * ends_question_mark / n:.1f}%)",
        f"questions starting what/how/...: {starts_what_how} ({100 * starts_what_how / n:.1f}%)",
        f"pairs per chunk: median={ppc_p50:.0f}  p90={ppc_p90:.0f}  max={ppc_max}",
        "",
        "--- Duplicates ---",
        f"unique normalized questions: {len(q_norm_counts)}",
        f"duplicate question strings (norm, count>1): {dup_q} distinct  ({rows_in_dup_groups} total rows in those groups)",
        "",
        "--- Residual artifacts ---",
        f"rows with U+0120/U+010a/Ġ/Ċ in question: {junk_q}",
        f"rows with same in answer:                {junk_a}",
        "",
        "--- Heuristic flags ---",
        f"very short question (<15 chars): {very_short_q}",
        f"very short answer (<20 chars):   {very_short_a}",
        f"long answer (>800 chars):        {long_a}",
        f"answers with enumerated ** (1. **...): {enum_debris}",
        f"answers mostly overlapping question tokens (>85% 4+char overlap): {weak_answer}",
        "",
        "--- Composite quality (heuristic, not human eval) ---",
        f"score: {score:.0f}/100  ({band(score)})",
        "  (penalizes: dup questions, short answers, tokenizer junk, enum debris, near-copy answers)",
    ]

    print("\n".join(report_lines))

    # Top duplicate questions
    dup_items = [(q, c) for q, c in q_norm_counts.items() if c > 1]
    dup_items.sort(key=lambda x: -x[1])
    if dup_items[:5]:
        print("\n--- Top 5 duplicated questions (normalized) ---")
        for q, c in dup_items[:5]:
            disp = q[:100] + ("…" if len(q) > 100 else "")
            print(f"  [{c}x] {disp}")

    summary = {
        "input": str(args.input.resolve()),
        "rows": n,
        "json_errors": bad,
        "unique_chunks": unique_chunks,
        "sources": dict(sources),
        "question_chars": {
            "min": min(q_lens),
            "median": statistics.median(q_sorted),
            "p90": _percentile(q_sorted, 90),
            "max": max(q_lens),
        },
        "answer_chars": {
            "min": min(a_lens),
            "median": statistics.median(a_sorted),
            "p90": _percentile(a_sorted, 90),
            "max": max(a_lens),
        },
        "ends_with_question_mark_pct": round(100 * ends_question_mark / n, 2),
        "duplicate_question_templates": dup_q,
        "rows_in_duplicate_question_groups": rows_in_dup_groups,
        "residual_tokenizer_junk_q": junk_q,
        "residual_tokenizer_junk_a": junk_a,
        "very_short_answer": very_short_a,
        "heuristic_quality_score": round(score, 1),
        "heuristic_quality_band": band(score),
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
        print(f"\nWrote {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
