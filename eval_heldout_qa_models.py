#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean

import torch
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "As an experienced agronomist proficient in farming techniques, crop management, "
    "and disease-resistant crop cultivation, answer only from given knowledge."
)


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = str(rec.get("question", "")).strip()
            a = str(rec.get("answer", "")).strip()
            if q and a:
                rows.append({"question": q, "answer": a})
            if limit and len(rows) >= limit:
                break
    return rows


def build_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n### Question:\n{question}\n\n### Answer:\n"


@torch.inference_mode()
def generate_answers(model, tokenizer, rows: list[dict], max_new_tokens: int) -> list[str]:
    outputs: list[str] = []
    for r in rows:
        prompt = build_prompt(r["question"])
        batch = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen_ids[0][batch["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


@torch.inference_mode()
def compute_perplexities(model, tokenizer, rows: list[dict]) -> list[float]:
    ppl: list[float] = []
    for r in rows:
        prompt = build_prompt(r["question"])
        full = prompt + r["answer"]
        batch = tokenizer(full, return_tensors="pt").to(model.device)
        out = model(**batch, labels=batch["input_ids"])
        loss = float(out.loss.detach().cpu())
        ppl.append(float(math.exp(loss)))
    return ppl


def randomization_pvalue(diff: list[float], n_iter: int = 2000, seed: int = 42) -> float:
    rng = random.Random(seed)
    obs = abs(mean(diff))
    count = 0
    for _ in range(n_iter):
        signed = [d if rng.random() < 0.5 else -d for d in diff]
        if abs(mean(signed)) >= obs:
            count += 1
    return (count + 1) / (n_iter + 1)


def evaluate_model(
    name: str, path: str, rows: list[dict], max_new_tokens: int
) -> dict:
    print(f"\n[model] Loading {name}: {path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    preds = generate_answers(model, tokenizer, rows, max_new_tokens=max_new_tokens)
    refs = [r["answer"] for r in rows]
    _, _, f1 = bert_score(
        preds,
        refs,
        model_type="sentence-transformers/all-MiniLM-L6-v2",
        num_layers=6,
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    bert_f1 = [float(v) for v in f1.cpu().tolist()]
    ppls = compute_perplexities(model, tokenizer, rows)

    del model
    torch.cuda.empty_cache()
    return {
        "name": name,
        "preds": preds,
        "bert_f1": bert_f1,
        "ppl": ppls,
        "mean_bert_f1": float(mean(bert_f1)),
        "mean_ppl": float(mean(ppls)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="QA_us_data/heldout_qa.jsonl")
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--out", default="QA_us_data/heldout_qa_eval_report.json")
    args = p.parse_args()

    rows = load_rows(Path(args.data), args.limit)
    if not rows:
        raise RuntimeError("No eval rows loaded.")
    print(f"Loaded {len(rows)} rows from {args.data}", flush=True)

    models = {
        "base": "/workspace/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "web_finetune": "/workspace/FineTunedmodels/Finetuned_WebsiteData/llama3_agri_finetuned/merged_16bit",
        "youtube_finetune": "/workspace/FineTunedmodels/Finetuned_YoutubeData/merged_16bit",
        "web_youtube_finetune": "/workspace/FineTunedmodels/Finetuned_WebsiteandYoutubeData/merged_16bit",
    }

    results = {
        name: evaluate_model(name, path, rows, args.max_new_tokens) for name, path in models.items()
    }

    yt_minus_web_bert = [
        y - w for y, w in zip(results["web_youtube_finetune"]["bert_f1"], results["web_finetune"]["bert_f1"])
    ]
    yt_minus_web_ppl = [
        w - y for y, w in zip(results["web_youtube_finetune"]["ppl"], results["web_finetune"]["ppl"])
    ]
    yt_only_minus_base_bert = [
        y - b for y, b in zip(results["youtube_finetune"]["bert_f1"], results["base"]["bert_f1"])
    ]

    significance = {
        "web_youtube_vs_web": {
            "delta_mean_bert_f1": float(mean(yt_minus_web_bert)),
            "p_value_bert_f1": float(randomization_pvalue(yt_minus_web_bert)),
            "delta_mean_ppl_web_minus_web_youtube": float(mean(yt_minus_web_ppl)),
            "p_value_ppl": float(randomization_pvalue(yt_minus_web_ppl)),
        },
        "youtube_vs_base": {
            "delta_mean_bert_f1": float(mean(yt_only_minus_base_bert)),
            "p_value_bert_f1": float(randomization_pvalue(yt_only_minus_base_bert)),
        },
    }

    idxs = list(range(len(rows)))
    random.Random(7).shuffle(idxs)
    keep = idxs[: min(args.samples, len(rows))]
    samples = []
    for i in keep:
        item = {
            "question": rows[i]["question"],
            "reference_answer": rows[i]["answer"],
        }
        for name in models:
            item[name] = results[name]["preds"][i]
        samples.append(item)

    payload = {
        "config": {
            "data": args.data,
            "rows_used": len(rows),
            "max_new_tokens": args.max_new_tokens,
        },
        "aggregate": {
            k: {"mean_bert_f1": v["mean_bert_f1"], "mean_ppl": v["mean_ppl"]} for k, v in results.items()
        },
        "significance": significance,
        "samples": samples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report: {out_path}", flush=True)


if __name__ == "__main__":
    main()
