# Results — US QA pipeline, model comparison, and YouTube ablation

This document summarizes **quantitative statistics by stage** from the agriculture QA workflow (synthesis → cleaning → held-out set → four-way model evaluation) and closes with the **ablation-style analysis** of YouTube transcript fine-tuning versus web-only and base models.

**Artifact sources:** `QA_us_data/heldout_qa.jsonl`, `QA_us_data/heldout_qa_eval_report.json`, and the cleaning run logged when building the held-out file.

---

## Stage 1 — Raw US QA corpus (`QA_us_data/all_qa_pairs.jsonl`)

| Statistic | Value |
|-----------|------:|
| Total JSONL rows (input) | **971** |
| Source | US web chunks → synthesized QA (`us_data_chunks.jsonl` lineage) |
| Typical artifacts | SentencePiece-style `Ġ` / `Ċ`, markdown `**`, occasional truncation |

---

## Stage 2 — Cleaning and held-out construction (`clean_qa_pairs.py`)

**Command (conceptual):**  
`--input QA_us_data/all_qa_pairs.jsonl` → `--output QA_us_data/heldout_qa.jsonl` with `--drop-truncated` and `--dedupe-global`.

| Statistic | Value |
|-----------|------:|
| Rows read | **971** |
| Rows written (held-out file) | **618** |
| Dropped — empty / short | **0** |
| Dropped — truncated (heuristic) | **301** |
| Dropped — global dedupe (same normalized question) | **52** |
| Net retention | **63.6%** (618 / 971) |

**Interpretation:** The held-out file is **cleaner and de-duplicated**; it is suitable for comparable generation + scoring, but **does not use the full 971-row raw set** for metrics.

---

## Stage 3 — Model variants (ablation factors)

Four checkpoints were compared on the same prompt template (agronomist system + `### Question` / `### Answer` structure, aligned with `Finetune_Youtube.py` / `Finetune_YoutubeData.py` training format).

| Label | Role in ablation | Typical path (local / pod) |
|-------|------------------|----------------------------|
| **Base** | No domain fine-tune | `Meta-Llama-3.1-8B` snapshot under HF hub cache |
| **Web** | Website QA SFT only | `FineTunedmodels/Finetuned_WebsiteData/.../merged_16bit` |
| **YouTube** | YouTube QA SFT from **base** | `FineTunedmodels/Finetuned_YoutubeData/merged_16bit` |
| **Web + YouTube** | YouTube continuation from **web** model | `FineTunedmodels/Finetuned_WebsiteandYoutubeData/merged_16bit` |

This design supports:

- **YouTube vs base:** effect of YouTube-only data starting from the pretrained base.
- **Web + YouTube vs web:** marginal effect of **adding YouTube** on top of web fine-tuning (primary ablation for “value of YouTube transcripts” given web is already applied).

---

## Stage 4 — Evaluation protocol (`eval_heldout_qa_models.py`)

| Setting | Value |
|---------|--------|
| Scoring data | `QA_us_data/heldout_qa.jsonl` |
| Rows used for reported metrics | **120** (first chunk of file; subsample of 618) |
| Generation | Greedy, `max_new_tokens` = **96** |
| Semantic metric | **BERTScore F1** via `bert_score`, model `sentence-transformers/all-MiniLM-L6-v2`, `num_layers=6` (safetensors-friendly; avoids older `torch` + non-safetensors load issues on some runtimes) |
| Perplexity | Teacher-forced NLL on **prompt + reference answer**, reported as **exp(loss)** (lower is better) |
| Significance | **Paired randomization** on per-example paired differences (2000 permutations, seed 42); p-values are one-sided in the implementation as reported in JSON |

**Note:** A first evaluation attempt using a different BERTScore backbone failed on the pod due to **`torch.load` policy / checkpoint format**; the reported numbers use the MiniLM configuration above. Re-running with another scorer after upgrading `torch` (≥ 2.6) or using safetensors-only models would be needed for direct comparison to published DeBERTa BERTScore tables.

---

## Stage 5 — Aggregate metrics (held-out subsample, *n* = 120)

| Model | Mean BERTScore F1 ↑ | Mean perplexity ↓ |
|-------|---------------------:|------------------:|
| Base | 0.5655 | 30.12 |
| Web | **0.5704** | 39.94 |
| YouTube | 0.4761 | 32.76 |
| Web + YouTube | 0.4885 | 33.77 |

**Rankings (this slice):**

- **Best BERTScore F1:** Web fine-tune.  
- **Best perplexity (lowest):** Base (30.12). Among fine-tunes, YouTube-only has the lowest mean PPL (32.76), then Web+YouTube (33.77), then Web (39.94).

**Caveat:** Web’s **higher** perplexity despite better BERTScore can reflect **distribution shift** (answers closer to references in embedding space but less probable under the web-tuned LM on this prompt+gold text), or sensitivity to `max_new_tokens` and decoding. Full 618-row evaluation is recommended for stable conclusions.

---

## Ablation study — YouTube transcript data

### A. YouTube-only vs base (same pretrain start)

| Comparison | Δ mean BERTScore F1 | p-value (randomization) |
|------------|---------------------:|------------------------:|
| YouTube − Base | **−0.0894** | **≈ 5.0 × 10⁻⁴** |

**Conclusion:** On this **120-question** slice and metric setup, **YouTube-only fine-tuning significantly lowers** mean BERTScore F1 relative to the **base** model (paired test as implemented). That suggests **harm or strong mismatch** on US held-out QA **under this training recipe**, not a simple quality win on this benchmark.

### B. Web + YouTube vs web (marginal effect of YouTube after web SFT)

| Metric | Δ (definition) | Value | p-value |
|--------|----------------|------:|--------:|
| BERTScore F1 | (Web + YouTube) − Web | **−0.0818** | **≈ 5.0 × 10⁻⁴** |
| Perplexity | Web − (Web + YouTube) | **+6.17** | **≈ 5.0 × 10⁻⁴** |

**Interpretation:**

- **BERTScore:** Adding YouTube on top of web is associated with a **statistically significant drop** in mean F1 vs web-only (same paired test).  
- **Perplexity:** **Web − (Web+YouTube) > 0** means **Web + YouTube assigns lower perplexity** to the gold answers (conditional on the prompt) than web-only in this setup—i.e. **better fit to reference text** by this measure, **in tension** with BERTScore on generations.

Together, this ablation indicates **YouTube data is not a uniform win**: it may change the model’s **token-level likelihood** on references while **hurting** **generation–reference similarity** (BERTScore) on greedy short outputs for this held-out set.

### C. Qualitative patterns (from sampled generations in `heldout_qa_eval_report.json`)

Reported side-by-side answers often show:

- **YouTube** and **Web + YouTube** drifting into **numbered follow-up questions** or list-like continuations.  
- **Web** sometimes more **on-topic** but **repetitive** or overly generic.  
- **Base** sometimes verbose or off-reference but with moderate BERTScore.

These patterns align with **format leakage** from training data or **instruction-following** quirks affecting BERTScore more than perplexity.

---

## Reproducibility checklist

1. Regenerate `heldout_qa.jsonl` with `clean_qa_pairs.py` (same flags).  
2. Run `eval_heldout_qa_models.py` **without** `--limit` for full **618** rows (or fix a seed/split for train vs held-out if you change protocol).  
3. Archive `heldout_qa_eval_report.json` with **git commit hash**, **torch / transformers / bert-score versions**, and **GPU type**.  
4. If reporting “official” BERTScore, align scorer model and `torch` version with paper norms after upgrading the runtime.

---

## File reference

| File | Role |
|------|------|
| `QA_us_data/all_qa_pairs.jsonl` | Raw synthesis QA |
| `QA_us_data/heldout_qa.jsonl` | Cleaned + filtered eval set |
| `QA_us_data/heldout_qa_eval_report.json` | Machine-readable metrics + samples |
| `eval_heldout_qa_models.py` | Evaluation driver |
| `clean_qa_pairs.py` | QA normalization / filters |

---

*Generated to consolidate pipeline stats and ablation outcomes for team review. Extend with full-data runs and additional metrics (exact match, human eval, toxicity) as those become available.*
