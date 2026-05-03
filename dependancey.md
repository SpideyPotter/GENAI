# Dependency Notes: Hugging Face CLI + Typer compatibility

## Problem pattern

You may see this after upgrading packages:

- `TypeError: Typer.__init__() got an unexpected keyword argument 'suggest_commands'`

Root cause:

- `transformers` 5.x pulls `huggingface_hub` 1.x.
- The installed `hf` CLI path and related packages in this environment can break with mixed `huggingface_hub`/`typer` combinations.
- Downgrading only one package creates a new conflict unless all related packages are aligned.

## Known-good stack for this project

Use this stack for synthesis/fine-tuning scripts in this repo:

- `transformers==4.45.2`
- `huggingface_hub==0.23.5`
- `typer==0.12.3`
- `accelerate==0.33.0`
- `tokenizers==0.20.1`
- `sentencepiece==0.2.0`
- `tqdm==4.67.1`
- `regex==2024.9.11`

Notes:

- Keep `transformers` on 4.x with `huggingface_hub` 0.23.5.
- Avoid `transformers` 5.x in this environment unless you fully migrate the stack.
- `bitsandbytes` is optional and only needed for 4-bit mode.

## One-shot recovery (inside container)

```bash
python -m pip uninstall -y transformers huggingface-hub huggingface_hub hf-xet tokenizers accelerate typer
python -m pip install --no-cache-dir \
  "transformers==4.45.2" \
  "huggingface_hub==0.23.5" \
  "typer==0.12.3" \
  "accelerate==0.33.0" \
  "tokenizers==0.20.1" \
  "sentencepiece==0.2.0" \
  "tqdm==4.67.1" \
  "regex==2024.9.11"
```

Optional (only if running 4-bit quantization):

```bash
python -m pip install --no-cache-dir "bitsandbytes>=0.43.0"
```

## CLI command note (important)

With `huggingface_hub==0.23.5`, the login command is:

```bash
huggingface-cli login
```

The `hf` command is from newer hub CLI packaging and may not exist in this pinned stack.

## Verify after install

```bash
python - <<'PY'
import transformers, huggingface_hub, typer, tokenizers, accelerate
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("typer", typer.__version__)
print("tokenizers", tokenizers.__version__)
print("accelerate", accelerate.__version__)
PY

huggingface-cli --version
huggingface-cli login
```

## Recommended practice

- Use a dedicated virtual environment in the pod (`python -m venv .venv`) to avoid breaking system packages.
- Do not run broad `pip install -U ...` without version pins in this image.
