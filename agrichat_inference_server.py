#!/usr/bin/env python3
"""
HTTP inference API for the AgriChat static UI (web/app.js).

Request:  POST JSON { "message", "model", "system"?, "history"? }
Response: { "reply": "..." }

Run inside the genai pod (GPU), bind 0.0.0.0 so kubectl port-forward works:

  pip install fastapi uvicorn
  python3 agrichat_inference_server.py --host 0.0.0.0 --port 8000

On your Mac (second terminal, while UI forward is running):

  kubectl port-forward -n dgx-s-bmu-soet-230557-restricted pod/genai 8000:8000

In AgriChat sidebar set Inference API URL to: http://127.0.0.1:8000/chat

Only one model is kept in VRAM at a time; switching the UI model unloads the previous one.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default paths under /workspace (genai pod mount). Override with env if needed.
_DEFAULT_ROOT = os.environ.get("AGRICHAT_WORKSPACE", "/workspace")

DEFAULT_MODEL_PATHS: dict[str, str] = {
    "base": os.environ.get(
        "AGRICHAT_BASE",
        f"{_DEFAULT_ROOT}/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/"
        "snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    ),
    "web_finetune": os.environ.get(
        "AGRICHAT_WEB",
        f"{_DEFAULT_ROOT}/FineTunedmodels/Finetuned_WebsiteData/llama3_agri_finetuned/merged_16bit",
    ),
    "youtube_finetune": os.environ.get(
        "AGRICHAT_YT",
        f"{_DEFAULT_ROOT}/FineTunedmodels/Finetuned_YoutubeData/merged_16bit",
    ),
    "web_youtube_finetune": os.environ.get(
        "AGRICHAT_WEB_YT",
        f"{_DEFAULT_ROOT}/FineTunedmodels/Finetuned_WebsiteandYoutubeData/merged_16bit",
    ),
}

DEFAULT_SYSTEM = (
    "You are an experienced agronomist assistant. You help with farming techniques, "
    "crop management, soil health, pests, sustainable practices, and regional agriculture "
    "guidance. Be clear, practical, and safety-conscious when discussing chemicals or livestock."
)


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1)
    model: str = "web_finetune"
    system: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


_loaded_key: str | None = None
_model: Any = None
_tokenizer: Any = None


def _unload() -> None:
    global _model, _tokenizer, _loaded_key
    if _model is not None:
        del _model
    _model = None
    _tokenizer = None
    _loaded_key = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_model(model_key: str, paths: dict[str, str]) -> None:
    global _model, _tokenizer, _loaded_key
    if model_key not in paths:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    path = paths[model_key]
    if not os.path.isdir(path):
        raise HTTPException(
            status_code=503,
            detail=f"Model path not found on disk: {path}. Set AGRICHAT_* env or sync weights.",
        )
    if _loaded_key == model_key and _model is not None:
        return
    _unload()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    m.eval()
    _tokenizer = tok
    _model = m
    _loaded_key = model_key


def _build_prompt(tokenizer: Any, system: str, history: list[dict[str, Any]], message: str) -> str:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    for h in history:
        role = str(h.get("role", "user"))
        content = str(h.get("content", ""))
        if role not in ("user", "assistant"):
            role = "user"
        if content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback: plain concatenation
    parts = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    parts.append("assistant:")
    return "\n\n".join(parts)


def create_app(paths: dict[str, str], max_new_tokens: int) -> FastAPI:
    app = FastAPI(title="AgriChat inference")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "loaded": _loaded_key or ""}

    @app.post("/chat")
    def chat(body: ChatBody) -> dict[str, str]:
        _ensure_model(body.model, paths)
        assert _tokenizer is not None and _model is not None
        system = (body.system or DEFAULT_SYSTEM).strip() or DEFAULT_SYSTEM
        prompt = _build_prompt(_tokenizer, system, body.history, body.message)
        batch = _tokenizer(prompt, return_tensors="pt").to(_model.device)
        with torch.inference_mode():
            gen_ids = _model.generate(
                **batch,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=max_new_tokens,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )
        new_tokens = gen_ids[0][batch["input_ids"].shape[1] :]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {"reply": text}

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="AgriChat inference API for web/app.js")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    paths = dict(DEFAULT_MODEL_PATHS)
    app = create_app(paths, max_new_tokens=args.max_new_tokens)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
