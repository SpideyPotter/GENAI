# GenAI agriculture workspace — team guide

This repository holds fine-tuned Llama models, US QA data, evaluation scripts, and a static **AgriChat** web UI for demo and integration. Use this document to reproduce setup, run evaluations, serve the UI locally, and understand cluster access limits.

**Git:** Large **datasets**, **chunk/QA folders**, **fine-tuned model trees**, and **HF `.cache`** are listed in `.gitignore` so they are not committed. Clone the repo, then restore data and weights from your team’s storage or regenerate them locally.

---

## What lives here (high level)

| Area | Location | Purpose |
|------|----------|---------|
| Base model (local HF cache) | `workspace/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/.../snapshots/<hash>` | Unmerged Llama 3.1 8B weights |
| Web fine-tune (merged) | `FineTunedmodels/Finetuned_WebsiteData/llama3_agri_finetuned/merged_16bit` | Website-agriculture SFT |
| YouTube fine-tune (merged) | `FineTunedmodels/Finetuned_YoutubeData/merged_16bit` | YouTube-transcript SFT from base |
| Web + YouTube (merged) | `FineTunedmodels/Finetuned_WebsiteandYoutubeData/merged_16bit` | Continued training after web model |
| US QA source | `QA_us_data/all_qa_pairs.jsonl` | Raw synthesis QA (may contain `Ġ` / `Ċ` artifacts) |
| Cleaned held-out QA | `QA_us_data/heldout_qa.jsonl` | Cleaned + filtered JSONL for eval |
| Held-out eval report | `QA_us_data/heldout_qa_eval_report.json` | BERTScore + perplexity + samples (when run) |
| AgriChat UI | `web/` | Static chat shell; optional backend API URL |
| QA cleaning script | `clean_qa_pairs.py` | Normalizes synthesis artifacts, optional dedupe/truncation |
| Multi-model eval script | `eval_heldout_qa_models.py` | Compares four models on held-out JSONL |

Training entrypoints (for reference) include `Finetune_Youtube.py`, `Finetune_YoutubeData.py`, `finetune_stage1_llama31_8b_base.py`.

---

## 1. Clean QA data → `heldout_qa.jsonl`

Synthesis exports often contain SentencePiece-style characters (`Ġ`, `Ċ`) and truncated rows. The cleaner fixes common artifacts and can drop bad rows.

From the repo root (paths below assume host checkout at `.../genai`):

```bash
python3 clean_qa_pairs.py \
  --input QA_us_data/all_qa_pairs.jsonl \
  --output QA_us_data/heldout_qa.jsonl \
  --drop-truncated \
  --dedupe-global
```

Tuning flags:

- `--dedupe` — dedupe per `(chunk_id, question)` instead of globally.
- `--min-answer-len N` / `--min-question-len N` — drop short rows.

---

## 2. Evaluate four models (BERTScore + perplexity + significance)

Script: `eval_heldout_qa_models.py` (intended to run in an environment with **GPU**, **torch**, **transformers**, **bert-score**, and merged model paths available—often the same machine/pod used for training).

It loads each merged checkpoint, generates answers with the same agronomy-style prompt as fine-tuning, scores with **BERTScore** (MiniLM-based setup in script), computes **perplexity** on gold answers, and writes a JSON report including paired significance summaries.

Example:

```bash
python3 eval_heldout_qa_models.py \
  --data QA_us_data/heldout_qa.jsonl \
  --limit 120 \
  --max_new_tokens 96 \
  --samples 6 \
  --out QA_us_data/heldout_qa_eval_report.json
```

Omit `--limit` to use the full cleaned file.

**Torch / BERTScore note:** Some clusters ship an older `torch` where loading certain non–safetensors BERT checkpoints fails. The evaluator uses a **safetensors-friendly** scoring configuration (see script). If you change the scorer model, verify compatibility with your `torch` version.

---

## 3. AgriChat web UI (`web/`)

The UI is **static HTML/CSS/JS**. It works offline with **demo replies** until you point it at an inference API.

### Files

- `web/index.html` — layout and copy.
- `web/styles.css` — agriculture green theme.
- `web/app.js` — chat logic, `localStorage` history, `fetch` to your backend.

### Serve locally (headnode or laptop)

Ports **8080** and **8765** are often busy on shared hosts. Use the helper script so it **binds `127.0.0.1`** and **tries several ports** until one is free:

```bash
cd web
chmod +x serve.sh   # once
./serve.sh
```

The script prints the URL (for example `http://127.0.0.1:8766/`). To force a port:

```bash
./serve.sh 9123
```

Manual equivalent:

```bash
python3 -m http.server 9123 --bind 127.0.0.1 --directory /path/to/genai/web
```

If you see **`OSError: [Errno 98] Address already in use`**, another process owns that port—pick a different port or use `./serve.sh` without arguments.

### Backend API (optional)

Set **Inference API URL** in the sidebar. The UI sends:

`POST` JSON body:

```json
{
  "message": "user text",
  "model": "web_finetune",
  "system": "optional system string",
  "history": [{ "role": "user", "content": "..." }, ...]
}
```

Expected response shapes (any one):

- `{ "reply": "..." }`
- `{ "message": "..." }`
- `{ "response": "..." }`
- OpenAI-style `{ "choices": [{ "message": { "content": "..." } }] }`

`model` values match the four checkpoints conceptually: `base`, `web_finetune`, `youtube_finetune`, `web_youtube_finetune`. Your server must map these to on-disk paths or services.

---

## 4. Opening the UI from your MacBook (cluster / headnode)

Kubernetes and network layout vary by site. The following is accurate for **restricted RBAC** setups like this project’s namespace.

### 4.1 You cannot list all namespaces

`kubectl get pods -A` may return **Forbidden**. Use **only your allowed namespace**:

```bash
kubectl get pods -n dgx-s-bmu-soet-230557-restricted
```

If your context already defaults to that namespace, `kubectl get pods` is enough.

### 4.2 No pod ⇒ no `kubectl port-forward`

If `kubectl get pods` shows **no resources**, the `genai` pod **does not exist right now** (never applied, deleted, or running elsewhere). You cannot port-forward to `pod/genai` until a pod with that name exists **and** serves HTTP on the target port.

Options:

1. **Apply** your workload (e.g. `kubectl apply -f genai.yaml`) if policy allows, **or** ask an admin to deploy the pod/service.
2. **Skip Kubernetes** for the static UI: run `./serve.sh` on the **headnode** (where your repo is mounted) and use **SSH local forwarding** from the Mac (section 4.4).

### 4.3 Port-forward (when a pod exists and listens)

Replace pod name and ports with real values:

```bash
kubectl port-forward -n dgx-s-bmu-soet-230557-restricted pod/<pod-name> 8766:<container-port>
```

On the Mac browser: `http://127.0.0.1:8766/`  
`kubectl` on the Mac must use a kubeconfig that reaches the API server (often VPN). **Same Wi‑Fi alone does not guarantee** reaching cluster node IPs; the API path via `kubectl` is the reliable approach.

### 4.4 SSH tunnel (no pod / static server on headnode)

If the HTTP server runs on the headnode at `127.0.0.1:<port>`:

```bash
ssh -L 8766:127.0.0.1:8766 <user>@<headnode-hostname-or-ip>
```

Then on the Mac: `http://127.0.0.1:8766/`

---

## 5. Troubleshooting quick reference

| Symptom | Likely cause | What to try |
|--------|----------------|-------------|
| `pods is forbidden` … `at the cluster scope` | No RBAC for `-A` | Omit `-A`; use `-n dgx-s-bmu-soet-230557-restricted` |
| `No resources found` in namespace | No workloads deployed | Apply manifest or use SSH + `serve.sh` on headnode |
| `pods "genai" not found` | Pod missing or different name | `kubectl get pods -n …` and use real pod name |
| `Address already in use` | Port taken | `./serve.sh` or `./serve.sh 9123` |
| UI works but “Could not reach the API” | Backend down or wrong URL | Fix URL / CORS / HTTPS mixed content |
| Eval script import errors on login node | No GPU / no torch | Run inside GPU pod or conda env with deps |

---

## 6. Related cluster manifest

`genai.yaml` defines an example pod mounting this workspace; default container command is idle (`sleep`). Serving the web UI **inside** the cluster requires either changing the command to run `serve.sh`/`http.server` or running the server via `kubectl exec`—coordinate with your cluster policy.

---

## 7. Contact / ownership

Document here who maintains inference endpoints, namespace quotas, and VPN/kubeconfig access for external laptops.
