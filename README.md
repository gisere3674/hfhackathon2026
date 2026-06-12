# Troll Dungeon Master

Troll Dungeon Master is a Gradio-powered tabletop RPG prototype for the Hugging Face Build Small Hackathon. The player explores a tiny fantasy dungeon while a small language model acts as a chaotic but fair Game Master that constantly roasts fictional choices.

## Hackathon fit

- **Format:** Hugging Face Space using Gradio.
- **Core idea:** An online DnD-style mini adventure where the Game Master is an LLM.
- **Tone:** Everything in the app is in English, playful, and safe-for-demo.
- **Parameter budget:** 16B total, under the 32B limit.

| Model | Role | Parameters |
| --- | --- | ---: |
| `Qwen/Qwen3-4B` | Game Master text generation | 4B |
| `black-forest-labs/FLUX.1-schnell` | Scene image generation | 12B |
| **Total** |  | **16B / 32B** |

## Features

- Deterministic game state with HP, gold, shame, curses, rooms, inventory, and seed.
- Real d20 rolls with normal, advantage, and disadvantage modes.
- A JSON-constrained Game Master prompt for reliable UI updates.
- Optional Modal endpoints for hosted text and image generation.
- Built-in fallback narrator and SVG scene generator so the Space remains playable without GPU credits.
- Custom Gradio styling for a parchment-and-dungeon feel.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Optional environment variables

The app works without these variables by using local fallbacks. Set them when you want live model inference.

```bash
export HF_TOKEN="hf_..."
export MODAL_TEXT_ENDPOINT="https://your-modal-text-endpoint"
export MODAL_IMAGE_ENDPOINT="https://your-modal-image-endpoint"
```

`MODAL_TEXT_ENDPOINT` should accept JSON with `system`, `user`, and `model`, then return either `{ "text": "...json..." }` or the JSON response directly.

`MODAL_IMAGE_ENDPOINT` should accept JSON with `prompt`, `model`, `width`, and `height`, then return either `{ "image_url": "..." }` or `{ "image_base64": "..." }`.

## Demo script

1. Start with seed `banana-lich-042` and hero `Sir Buttonmash`.
2. Set troll level to `4`.
3. Type: `I quietly open the suspicious door while looking extremely professional.`
4. Click **Roll and act**.
5. Use one of the generated choice buttons for the next turn.

## Safety notes

The Game Master is instructed to mock fictional situations, not real users. It should avoid slurs, identity-based insults, sexual humiliation, threats, and hateful content. The fallback narrator follows the same playful tone.
