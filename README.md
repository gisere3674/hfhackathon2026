---
title: Misadventure Master
emoji: 🐉
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

# Misadventure Master

A tiny tabletop RPG where the Game Master is a small LLM, the dice are real, and the dungeon is legally allowed to roast your choices.

## Hugging Face inference notes

The app uses Hugging Face's current Inference Providers router via `huggingface_hub` for hosted text and image generation. If a token, provider, or network lookup fails, gameplay now continues with local text and SVG scene fallbacks instead of surfacing a Gradio error.

Optional environment variables:

- `HF_TOKEN`: Hugging Face token for hosted inference.
- `HF_TEXT_PROVIDER`: text provider passed to `InferenceClient` (defaults to `auto`).
- `HF_IMAGE_PROVIDER`: image provider passed to `InferenceClient` (defaults to `black-forest-labs`).
- `TEXT_MODEL`: text model id (defaults to `Qwen/Qwen2.5-7B-Instruct`).
- `IMAGE_MODEL`: image model id (defaults to `black-forest-labs/FLUX.1-schnell`).
- `MODAL_TEXT_ENDPOINT` / `MODAL_IMAGE_ENDPOINT`: optional custom endpoints, used before Hugging Face.
