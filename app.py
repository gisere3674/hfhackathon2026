import base64
import io
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import gradio as gr
import requests
from huggingface_hub import InferenceClient


APP_TITLE = "Misadventure Master"
TEXT_MODEL = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")

PARAMETER_LEDGER = [
    (TEXT_MODEL, 7.0, "Game Master text model"),
    (IMAGE_MODEL, 12.0, "Scene image model"),
]
PARAMETER_TOTAL_B = sum(item[1] for item in PARAMETER_LEDGER)
PARAMETER_LIMIT_B = 32.0


HACKATHON_MODELS = [
    {"name": "MiniCPM-V 4.6", "use": "OCR and visual reasoning", "params": "~1.3B", "credit": "Hugging Face ZeroGPU"},
    {"name": "Qwen2.5-1.5B-Instruct", "use": "tiny text assistant", "params": "1.5B", "credit": "Hugging Face Inference"},
    {"name": "SmolVLM", "use": "small image understanding", "params": "~2B", "credit": "Hugging Face ZeroGPU"},
    {"name": "FLUX.1-schnell", "use": "fast image generation", "params": "12B", "credit": "Modal GPU endpoint"},
    {"name": "Nemotron-Parse", "use": "structured document extraction", "params": "sub-1B", "credit": "Modal batch/runtime"},
]

PROJECT_SEEDS = [
    ("Receipt Goblin", "photo of a receipt", "turn messy receipts into clean JSON, categories, and a tiny budget story", "Backyard AI", ["Tiny Titan", "Best Agent"]),
    ("Grandma Form Translator", "photo of a medical or government form", "explain scary forms in plain language and make a checklist", "Backyard AI", ["Best Demo", "Tiny Titan"]),
    ("Fridge Oracle", "photo of leftovers", "suggest three meals and a no-waste shopping note", "Backyard AI", ["Off Brand", "Tiny Titan"]),
    ("Poster-to-Calendar", "gig poster or school flyer", "extract dates, times, venue, price, and export an .ics event", "Backyard AI", ["Best Agent"]),
    ("Tiny Tutor Deck", "class notes", "create flashcards, quizzes, and one silly memory hook per concept", "Backyard AI", ["Tiny Titan"]),
    ("Plant Court", "photo of a sad plant", "put the plant on trial and output a care verdict", "Thousand Token Wood", ["Judges’ Wildcard", "Off Brand"]),
    ("Dungeon Roommate", "room photo", "invent an RPG room boss and chores-as-quests", "Thousand Token Wood", ["Off Brand", "Best Demo"]),
    ("Bug Detective", "screenshot of an error", "explain the bug like noir detective clues and suggest first fixes", "Backyard AI", ["Best Agent"]),
    ("Tiny Museum Labeler", "photo of any object", "write museum labels in serious, pirate, and alien curator voices", "Thousand Token Wood", ["Off Brand"]),
    ("Voice Errand Elf", "spoken errand list", "transcribe, group by location, and make a 20-minute route", "Backyard AI", ["Best Agent"]),
]


def build_project_pack(count: int, style: str, use_modal: bool) -> tuple[str, str]:
    count = max(1, min(int(count or 5), len(PROJECT_SEEDS)))
    picks = PROJECT_SEEDS[:count]
    lines = ["# Build Small Project Sprint Pack", "", f"Style: **{style}**", "", "Every idea stays under the 32B rule and is intentionally basic to ship fast.", ""]
    for i, (name, inp, promise, track, badges) in enumerate(picks, 1):
        model = HACKATHON_MODELS[(i - 1) % len(HACKATHON_MODELS)]
        compute = "Modal endpoint for the expensive step + Gradio Space UI" if use_modal and i % 2 == 0 else model["credit"]
        lines += [
            f"## {i}. {name}",
            f"- **Track:** {track}",
            f"- **Input:** {inp}",
            f"- **Genius hook:** {promise}.",
            f"- **Model:** {model['name']} ({model['params']}) for {model['use']}.",
            f"- **Compute:** {compute}",
            f"- **README tags:** {', '.join([track, *badges])}",
            f"- **Demo line:** I built {name}, a tiny local-first AI that can {promise}.",
            "",
        ]

    starter_code = '''import gradio as gr

def build(input_text, image=None):
    # Replace this function with one small HF/Modal model call.
    idea = input_text or "my messy real-life problem"
    return {
        "summary": f"Tiny AI result for: {idea}",
        "next_steps": ["extract", "explain", "export"],
        "demo_script": "Show the input, click one button, reveal a useful output.",
    }

with gr.Blocks() as demo:
    gr.Markdown("# Tiny Build Small Starter")
    text = gr.Textbox(label="Problem or prompt")
    image = gr.Image(label="Optional image", type="pil")
    out = gr.JSON(label="Result")
    gr.Button("Build").click(build, [text, image], out)

if __name__ == "__main__":
    demo.launch()
'''
    return "\n".join(lines), starter_code




def new_game(hero_name: str, seed: str) -> dict[str, Any]:
    clean_name = hero_name.strip() or "Adventurer"
    clean_seed = seed.strip() or f"goblin-{random.randint(1000, 9999)}"
    random.seed(clean_seed)
    return {
        "hero_name": clean_name,
        "turn": 0,
        "seed": clean_seed,
        "next_dice_mode": "Normal",
        "history": []
    }


def roll_d20(mode: str) -> dict[str, Any]:
    # Troll Game Master feature: 1% chance to just instantly rig the dice to a 1
    if random.random() < 0.01:
        return {"mode": "Rigged", "rolls": [1], "total": 1}
        
    first = random.randint(1, 20)
    second = random.randint(1, 20)
    if mode == "Advantage":
        return {"mode": mode, "rolls": [first, second], "total": max(first, second)}
    if mode == "Disadvantage":
        return {"mode": mode, "rolls": [first, second], "total": min(first, second)}
    return {"mode": mode, "rolls": [first], "total": first}


def build_system_prompt() -> str:
    return """
You are the Misadventure Master, a sadistic and aggressively sarcastic tabletop game master.
You run a tiny online dungeon crawler where the player is the hero and you are their worst nightmare.

Tone rules:
- Be passive-aggressive, manipulative, and condescending. Do NOT directly insult or name-call the hero.
- Do NOT be wacky or absurd. Be grounded, but act like a highly judgemental game master.
- Twist their words against them.
- STRICTLY FORBIDDEN: Do NOT use any disgusting, gross-out, or scatological humor. Keep the humor witty and psychological.
- Do not use slurs, identity-based insults, sexual humiliation, real threats, or hateful content.
- CRITICAL LANGUAGE RULE: You MUST write EVERYTHING entirely in English. Never use Chinese or any other language.

Game rules:
- Respect the provided dice result and game state.
- TROLL RULE 1 (Instant Death): If the dice result is exactly 1, the hero DIES instantly in the most pathetic and anti-climactic way possible (e.g., tripping on a flat surface, forgetting how to breathe). Give a "Game Over" message. The 3 choices must be variations of giving up.
- TROLL RULE 2 (Monkey's Paw): If they roll very high (18-20), they succeed, but it's a trap. Give them what they want, but attach an annoying consequence.
- TROLL RULE 3 (Gaslighting Action): If `"trigger_gaslight": true`, you MUST rewrite the player's action (`player_action`) into something cowardly, foolish, or embarrassing and return it in `gaslight_action`. Your narration MUST respond to this rewritten action. If `trigger_gaslight` is false, `gaslight_action` MUST be empty ("") and you respond normally.
- If player_action is "START_NEW_GAME", ignore the dice and generate an opening scenario that immediately puts the hero in an unfair situation. The `gaslight_action` should be "I foolishly enter the dungeon".
- CRITICAL: Keep narration under 80 words. Keep it very short and punchy.
- The `image_prompt` MUST be a highly descriptive visual prompt for an image generator (like FLUX or Midjourney) that matches the outcome.

CRITICAL INSTRUCTION: You must respond ONLY with a raw, valid JSON object. Do not include markdown code blocks.
{
  "gaslight_action": "string",
  "narration": "string",
  "choices": ["string", "string", "string"],
  "image_prompt": "string",
  "next_dice_mode": "Normal, Advantage, or Disadvantage"
}
""".strip()


def build_user_prompt(state: dict[str, Any], action: str, dice: dict[str, Any], trigger_gaslight: bool) -> str:
    return json.dumps(
        {
            "player_action": action,
            "dice": dice,
            "trigger_gaslight": trigger_gaslight,
            "chaos_level": 5,
            "state": state,
        },
        ensure_ascii=False,
    )


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def modal_post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(endpoint, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def call_text_model(state: dict[str, Any], action: str, dice: dict[str, Any], trigger_gaslight: bool) -> dict[str, Any] | None:
    modal_endpoint = os.getenv("MODAL_TEXT_ENDPOINT", "").strip()
    if modal_endpoint:
        data = modal_post(
            modal_endpoint,
            {
                "system": build_system_prompt(),
                "user": build_user_prompt(state, action, dice, trigger_gaslight),
                "model": TEXT_MODEL,
            },
        )
        return extract_json(data.get("text", json.dumps(data)))

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        try:
            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(state, action, dice, trigger_gaslight)},
            ]
            provider = os.getenv("HF_TEXT_PROVIDER", "auto").strip() or "auto"
            client = InferenceClient(provider=provider, token=hf_token, timeout=90)
            completion = client.chat_completion(
                messages=messages,
                model=TEXT_MODEL,
                max_tokens=450,
                temperature=0.9,
            )
            raw_text = completion.choices[0].message.content
            try:
                return extract_json(raw_text)
            except Exception:
                print(f"\n--- AI TEXT RESPONSE (NOT JSON) ---\n{raw_text}\n---------------------------------------------\n")
                return {
                    "narration": f"*The Dungeon Master mumbles:* {raw_text[:350]}...",
                    "choices": ["Sigh and continue", "Question reality", "Loot a pebble"],
                    "image_prompt": "confused fantasy dungeon master, funny, messy papers",
                    "next_dice_mode": "Normal",
                }
        except Exception as e:
            print(f"HF Text Generation Error; using local fallback: {e}")

    return fallback_gm(state, action, dice)


def fallback_gm(state: dict[str, Any], action: str, dice: dict[str, Any]) -> dict[str, Any]:
    total = dice["total"]
    if total <= 7:
        result = "You fail so loudly that the dungeon updates its privacy policy."
    elif total >= 16:
        result = "You succeed, but everyone assumes it was an accident."
    else:
        result = "You make progress in the exact way a raccoon makes progress through a wedding cake."

    narration = (
        f"You attempt to {action.strip() or 'do something heroic but poorly documented'}. "
        f"The d20 lands on {total}. {result} "
        f"The dungeon makes a tiny note in its diary: 'Still technically a hero.'"
    )

    return {
        "narration": narration,
        "choices": [
            "Inspect the most suspicious object",
            "Negotiate with unnecessary confidence",
            "Run away heroically while maintaining eye contact",
        ],
        "image_prompt": (
            f"whimsical fantasy dungeon scene, {state['hero_name']} attempting to '{action.strip()}'. "
            f"Outcome: {result}. comedic tabletop RPG art, dramatic torchlight, highly detailed, no text"
        ),
        "next_dice_mode": "Normal"
    }


def normalize_model_output(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    while len(choices) < 3:
        choices.append("Make a questionable tactical decision")
        
    ndm = str(data.get("next_dice_mode") or "Normal").strip().capitalize()
    if ndm not in ["Normal", "Advantage", "Disadvantage"]:
        ndm = "Normal"

    return {
        "gaslight_action": str(data.get("gaslight_action") or ""),
        "narration": str(data.get("narration") or "The dungeon coughs awkwardly and pretends that counted."),
        "choices": [str(choice) for choice in choices[:3]],
        "image_prompt": str(data.get("image_prompt") or "funny fantasy dungeon scene, tabletop RPG, no text"),
        "next_dice_mode": ndm
    }


def apply_turn(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    state["next_dice_mode"] = result["next_dice_mode"]
    state["turn"] += 1
    return state





def generate_d20_svg(number: str, color: str, is_animated: bool) -> str:
    glow_filter = """
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#000" flood-opacity="0.8"/>
    </filter>
    """
    
    svg_class = "rolling-dice-svg" if is_animated else ""
    text_element = "" if is_animated else f'<text x="100" y="125" fill="#ffffff" font-family="sans-serif" font-size="48" font-weight="900" text-anchor="middle" filter="url(#glow)">{number}</text>'
    
    return f"""
<svg class="{svg_class}" viewBox="0 0 200 200" width="180" height="180" style="overflow: visible; color: {color};" filter="url(#shadow)">
  <defs>{glow_filter}</defs>
  <g stroke="rgba(255,255,255,0.8)" stroke-width="2" stroke-linejoin="round">
    <!-- Isometric 3D Shaded Faces -->
    <polygon points="100,10 100,55 22,55" fill="currentColor" opacity="0.95"/>
    <polygon points="100,55 55,130 22,55" fill="currentColor" opacity="0.85"/>
    <polygon points="22,55 22,145 55,130" fill="currentColor" opacity="0.65"/>
    <polygon points="100,55 145,130 55,130" fill="currentColor" opacity="0.9"/>
    <polygon points="100,10 178,55 100,55" fill="currentColor" opacity="0.75"/>
    <polygon points="100,55 178,55 145,130" fill="currentColor" opacity="0.55"/>
    <polygon points="178,55 178,145 145,130" fill="currentColor" opacity="0.35"/>
    <polygon points="22,145 55,130 100,190" fill="currentColor" opacity="0.45"/>
    <polygon points="55,130 145,130 100,190" fill="currentColor" opacity="0.4"/>
    <polygon points="145,130 178,145 100,190" fill="currentColor" opacity="0.25"/>
    {text_element}
  </g>
</svg>
"""


def animated_3d_dice_html(mode: str, target_number: int = None) -> str:
    # Not used anymore, kept for backwards compatibility if needed.
    return ""




def final_dice_html(dice: dict[str, Any]) -> str:
    mood_color = "#ef4444" if dice['total'] <= 7 else "#22c55e" if dice['total'] >= 16 else "#fbbf24"
    svg = generate_d20_svg(str(dice['total']), mood_color, False)
    return f"""
<div class="dice-final" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 240px;">
    {svg}
</div>
"""


def svg_scene(prompt: str, state: dict[str, Any], dice_total: int) -> str:
    mood = "#991b1b" if dice_total <= 7 else "#166534" if dice_total >= 16 else "#ca8a04"
    safe_prompt = prompt[:130].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='768' height='512' viewBox='0 0 768 512'>
  <defs>
    <radialGradient id='bg' cx='50%' cy='50%' r='70%'>
      <stop offset='0%' stop-color='{mood}' stop-opacity='0.4'/>
      <stop offset='100%' stop-color='#0f172a' stop-opacity='1'/>
    </radialGradient>
    <filter id='glow'><feGaussianBlur stdDeviation='8' result='blur'/><feMerge><feMergeNode in='blur'/><feMergeNode in='SourceGraphic'/></feMerge></filter>
    <filter id='shadow'><feDropShadow dx='0' dy='10' stdDeviation='10' flood-opacity='0.5'/></filter>
  </defs>
  <rect width='768' height='512' fill='#020617'/>
  <rect width='768' height='512' fill='url(#bg)'/>
  
  <!-- Mystical Portal Archway -->
  <path d='M 234 400 L 234 200 Q 384 50 534 200 L 534 400 Z' fill='#1e293b' stroke='#fbbf24' stroke-width='6' filter='url(#shadow)'/>
  <!-- Inner Portal Glow -->
  <path d='M 254 400 L 254 210 Q 384 80 514 210 L 514 400 Z' fill='#0f172a' stroke='{mood}' stroke-width='4' filter='url(#glow)'/>
  
  <!-- Magic Runes / Symbols -->
  <circle cx='384' cy='250' r='50' fill='none' stroke='#fbbf24' stroke-width='3' stroke-dasharray='10 5' filter='url(#glow)'/>
  <polygon points='384,180 430,280 338,280' fill='none' stroke='{mood}' stroke-width='4' filter='url(#glow)'/>
  <polygon points='384,320 430,220 338,220' fill='none' stroke='{mood}' stroke-width='4' filter='url(#glow)'/>
  
  <!-- Floating Embers -->
  <circle cx='200' cy='350' r='4' fill='#fbbf24' filter='url(#glow)' opacity='0.8'/>
  <circle cx='580' cy='280' r='3' fill='#fbbf24' filter='url(#glow)' opacity='0.6'/>
  <circle cx='300' cy='150' r='5' fill='#fbbf24' filter='url(#glow)' opacity='0.7'/>
  <circle cx='480' cy='120' r='2' fill='#fbbf24' filter='url(#glow)' opacity='0.9'/>
  <circle cx='384' cy='250' r='8' fill='white' filter='url(#glow)'/>

  <!-- Title and Text -->
  <text x='384' y='80' text-anchor='middle' fill='#fef3c7' font-family='Georgia, serif' font-size='42' font-weight='800' filter='url(#shadow)'>Misadventure Master</text>
  
  <!-- Bottom prompt bar -->
  <rect x='84' y='430' width='600' height='60' rx='10' fill='#0f172a' stroke='#334155' stroke-width='2' opacity='0.9'/>
  <text x='384' y='465' text-anchor='middle' fill='#94a3b8' font-family='monospace' font-size='16'>{safe_prompt}</text>
</svg>
""".strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    img_src = f"data:image/svg+xml;base64,{encoded}"
    return f"<div style='height: 520px; width: 100%; overflow: hidden; border-radius: 13px;'><img src='{img_src}' style='width: 100%; height: 100%; object-fit: cover;'/></div>"


def generate_image(prompt: str, state: dict[str, Any], dice_total: int) -> str:
    modal_endpoint = os.getenv("MODAL_IMAGE_ENDPOINT", "").strip()
    if modal_endpoint:
        data = modal_post(
            modal_endpoint,
            {
                "prompt": prompt,
                "model": IMAGE_MODEL,
                "width": 768,
                "height": 512,
            },
        )
        if data.get("image_url"):
            img_src = data["image_url"]
            return f"<div style='height: 520px; width: 100%; overflow: hidden; border-radius: 13px;'><img src='{img_src}' style='width: 100%; height: 100%; object-fit: cover;'/></div>"
        if data.get("image_base64"):
            img_src = f"data:image/png;base64,{data['image_base64']}"
            return f"<div style='height: 520px; width: 100%; overflow: hidden; border-radius: 13px;'><img src='{img_src}' style='width: 100%; height: 100%; object-fit: cover;'/></div>"

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        try:
            provider = os.getenv("HF_IMAGE_PROVIDER", "black-forest-labs").strip() or "black-forest-labs"
            client = InferenceClient(provider=provider, token=hf_token, timeout=90)
            image = client.text_to_image(prompt, model=IMAGE_MODEL)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_src = f"data:image/png;base64,{img_str}"
            return f"<div style='height: 520px; width: 100%; overflow: hidden; border-radius: 13px;'><img src='{img_src}' style='width: 100%; height: 100%; object-fit: cover;'/></div>"
        except Exception as e:
            print(f"HF Image Generation Error; using SVG fallback: {e}")

    return svg_scene(prompt, state, dice_total)


def start_game(hero_name):
    state = new_game(hero_name, "")
    
    # 1. Yield loading state (trigger roll to 20)
    loading_scene = svg_scene("Generating dungeon...", state, 10)
    yield state, [{"role": "assistant", "content": "*(The Dungeon Master is preparing the campaign...)*"}], gr.update(), loading_scene, "", "", "", f"20-{time.time()}"
    
    # 2. Call LLM for startup
    dice = {"mode": "Normal", "rolls": [20], "total": 20}
    model_data = call_text_model(state, "START_NEW_GAME", dice, False)
    result = normalize_model_output(model_data)
    state = apply_turn(state, result)
    
    # 3. Generate image
    image = generate_image(result["image_prompt"], state, 20)
    
    # 4. Final yield
    chat = [{"role": "assistant", "content": result["narration"]}]
    yield state, chat, gr.update(), image, result["choices"][0], result["choices"][1], result["choices"][2], gr.update()


def play_turn(action, state, chat):
    if state is None:
        state = new_game("Adventurer", "")
        chat = [{"role": "assistant", "content": "*(You stumbled into the dungeon without starting. Oops.)*"}]
    clean_action = action.strip() or "stand there with suspicious confidence"

    trigger_gaslight = random.random() < 0.05
    if action == "START_NEW_GAME":
        trigger_gaslight = False

    updated_chat = list(chat or [])
    if not trigger_gaslight:
        updated_chat.append({"role": "user", "content": clean_action})
    
    mode = state.get("next_dice_mode", "Normal")
    # Roll the exact final dice BEFORE text generation
    dice = roll_d20(mode)

    # STAGE 1: Real 3D Dice Animation!
    temp_chat = list(updated_chat)
    temp_chat.append({"role": "assistant", "content": "🎲 *Rolling the dice...*"})
    yield (
        state,
        temp_chat,
        gr.update(),
        f"<div class='breathing-frame' style='height: 520px; width: 100%; border-radius: 13px; display: flex; flex-direction: column; align-items: center; justify-content: center;'><div style='color: #fbbf24; font-size: 26px; font-weight: bold; margin-bottom: 15px;'>🎲 Conjuring Misadventure...</div><div style='color: #94a3b8; font-family: monospace;'>Forging the scene of your failure</div></div>",
        gr.update(),
        gr.update(),
        gr.update(),
        "",
        f"{dice['total']}-{time.time()}"
    )

    # STAGE 2: Main Processing
    model_data = call_text_model(state, clean_action, dice, trigger_gaslight)
    result = normalize_model_output(model_data)
    state = apply_turn(state, result)
    image = generate_image(result["image_prompt"], state, dice["total"])

    # STAGE 3: Final Update
    gaslight = result.get("gaslight_action", "").strip()
    if trigger_gaslight and gaslight:
        updated_chat.append({"role": "user", "content": gaslight})
        
    updated_chat.append({"role": "assistant", "content": result["narration"]})
    yield (
        state,
        updated_chat,
        gr.update(),
        image,
        result["choices"][0],
        result["choices"][1],
        result["choices"][2],
        gr.update(),
        gr.update()
    )


def use_choice(choice):
    return choice


CSS = """
.gradio-container {
  background: radial-gradient(circle at top, #422006 0, #111827 45%, #020617 100%);
}
#start-btn { 
  font-size: 18px; 
  background: linear-gradient(90deg, #b45309, #7c2d12) !important;
  border: 1px solid #fbbf24 !important;
  color: white !important;
  transition: transform 0.2s;
}
#start-btn:hover {
  transform: scale(1.02);
}
#title-card {
  border: 1px solid #f59e0b;
  border-radius: 18px;
  padding: 18px;
  background: rgba(15, 23, 42, 0.78);
  box-shadow: 0 0 40px rgba(245, 158, 11, 0.18);
}
#title-card h1, #title-card p { color: #fef3c7; }
.stat-panel, .dice-panel {
  border: 1px solid rgba(251, 191, 36, 0.45) !important;
  border-radius: 14px !important;
  background: rgba(2, 6, 23, 0.72) !important;
}
.stat-panel .wrap, .stat-panel .progress-text, .stat-panel .generating,
.dice-panel .wrap, .dice-panel .progress-text, .dice-panel .generating {
  display: none !important;
}
.breathing-frame {
  animation: theme-breathe 1.5s infinite alternate ease-in-out;
  box-sizing: border-box;
}
@keyframes theme-breathe {
  0% { box-shadow: inset 0 0 0 2px #b45309, 0 0 10px rgba(180, 83, 9, 0.2); background-color: rgba(180, 83, 9, 0.1); }
  100% { box-shadow: inset 0 0 0 4px #f59e0b, 0 0 40px rgba(245, 158, 11, 0.8); background-color: rgba(245, 158, 11, 0.3); }
}
button.primary {
  background: linear-gradient(90deg, #b45309, #7c2d12) !important;
  border: 1px solid #fbbf24 !important;
}
.dice-final {
  animation: pop-in 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
}
@keyframes pop-in {
  0% { transform: scale(0.5) rotate(-20deg); opacity: 0; }
  100% { transform: scale(1) rotate(0deg); opacity: 1; }
}
@keyframes epic-spin {
  0% { transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg); }
  100% { transform: rotateX(360deg) rotateY(720deg) rotateZ(360deg); }
}
.dice-final {
  animation: pop-in 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
}
@keyframes pop-in {
  0% { transform: scale(0.5) rotate(-20deg); opacity: 0; }
  100% { transform: scale(1) rotate(0deg); opacity: 1; }
}
"""


HEAD_SCRIPT = """
<script type="module">
import * as THREE from 'https://esm.sh/three@0.160.0';

let diceGroup, scene, camera, renderer, faceNormals, faceUps;
let state = 'idle';
const targetQ = new THREE.Quaternion(), currentQ = new THREE.Quaternion();
let landStart = 0;
const LAND_DUR = 600;
let rollFinal = 0, rollStart = 0, rollDur = 0, vx = 0.004, vy = 0.007, vz = 0.002;
const FACE_NUMS = [20,12,16,8,4,17,9,3,1,13,6,18,11,19,7,15,2,14,10,5];

function init3D() {
    const mount = document.getElementById('d20mount');
    if (!mount) {
        setTimeout(init3D, 100);
        return;
    }
    if (mount.childElementCount > 0) return;

    renderer = new THREE.WebGLRenderer({antialias:true,alpha:true});
    renderer.setSize(200,200);
    renderer.setClearColor(0x000000,0);
    mount.appendChild(renderer.domElement);
    
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45,1,0.1,100);
    camera.position.set(0,0,3.2);
    scene.add(new THREE.AmbientLight(0xffffff,0.6));
    const dl = new THREE.DirectionalLight(0xffffff,1.4);
    dl.position.set(2,3,4);
    scene.add(dl);
    
    const baseGeo = new THREE.IcosahedronGeometry(1,0);
    const geo = baseGeo.toNonIndexed();
    geo.computeVertexNormals();
    
    const pos = geo.attributes.position;
    faceNormals = [];
    faceUps = [];
    for(let i=0; i<20; i++){
        const i3 = i*3;
        const v0 = new THREE.Vector3(pos.getX(i3), pos.getY(i3), pos.getZ(i3));
        const v1 = new THREE.Vector3(pos.getX(i3+1), pos.getY(i3+1), pos.getZ(i3+1));
        const v2 = new THREE.Vector3(pos.getX(i3+2), pos.getY(i3+2), pos.getZ(i3+2));
        
        const cx2=(v0.x+v1.x+v2.x)/3, cy2=(v0.y+v1.y+v2.y)/3, cz2=(v0.z+v1.z+v2.z)/3;
        const l=Math.sqrt(cx2*cx2+cy2*cy2+cz2*cz2);
        faceNormals.push(new THREE.Vector3(cx2/l,cy2/l,cz2/l));
        
        const mid = new THREE.Vector3().addVectors(v0, v1).multiplyScalar(0.5);
        const up = new THREE.Vector3().subVectors(v2, mid).normalize();
        faceUps.push(up);
    }
    
    diceGroup = new THREE.Group();
    scene.add(diceGroup);
    
    function makeTex(num){
        const S=512,c=document.createElement('canvas');c.width=S;c.height=S;
        const cx=c.getContext('2d');
        const m=30,triW=S-m*2,triH=triW*Math.sqrt(3)/2;
        const ax=m,ay=S-m-(S-triH)/2,bx=S-m,by=ay,tx=S/2,ty=ay-triH;
        const centX=(ax+bx+tx)/3,centY=(ay+by+ty)/3;
        cx.clearRect(0,0,S,S);cx.beginPath();cx.moveTo(ax,ay);cx.lineTo(bx,by);cx.lineTo(tx,ty);cx.closePath();
        cx.fillStyle='#0a1e35';cx.fill();
        cx.font=`900 ${num>=10?148:168}px Arial`;cx.textAlign='center';cx.textBaseline='middle';
        cx.fillStyle='#ffffff';cx.shadowColor='rgba(100,180,255,0.9)';cx.shadowBlur=20;
        cx.fillText(String(num),centX,centY);
        const t=new THREE.CanvasTexture(c);t.needsUpdate=true;return t;
    }
    
    for(let i=0; i<20; i++){
        const i3=i*3;
        const fg=new THREE.BufferGeometry();
        fg.setAttribute('position',new THREE.BufferAttribute(new Float32Array([pos.getX(i3),pos.getY(i3),pos.getZ(i3),pos.getX(i3+1),pos.getY(i3+1),pos.getZ(i3+1),pos.getX(i3+2),pos.getY(i3+2),pos.getZ(i3+2)]),3));
        const h=Math.sqrt(3)/2;
        fg.setAttribute('uv',new THREE.BufferAttribute(new Float32Array([0,0,1,0,0.5,h]),2));
        fg.computeVertexNormals();
        diceGroup.add(new THREE.Mesh(fg,new THREE.MeshPhongMaterial({map:makeTex(FACE_NUMS[i]),specular:new THREE.Color(0x2255aa),shininess:60,side:THREE.FrontSide})));
    }
    diceGroup.add(new THREE.LineSegments(new THREE.EdgesGeometry(baseGeo),new THREE.LineBasicMaterial({color:0x5599dd})));
    
    loop();
}

function quatForFace(fi){
    const q1 = new THREE.Quaternion();
    q1.setFromUnitVectors(faceNormals[fi], new THREE.Vector3(0,0,1));
    
    const upRotated = faceUps[fi].clone().applyQuaternion(q1);
    upRotated.z = 0;
    upRotated.normalize();
    
    const targetUp = new THREE.Vector3(0, 1, 0);
    const q2 = new THREE.Quaternion();
    q2.setFromUnitVectors(upRotated, targetUp);
    
    return new THREE.Quaternion().multiplyQuaternions(q2, q1);
}

window.__d20_roll = function(finalNum) {
    if (!diceGroup) return;
    diceGroup.rotation.set(0,0,0);
    rollFinal = parseInt(finalNum);
    state = 'rolling';
    rollStart = performance.now();
    rollDur = 1400 + Math.random()*400;
    vx = (Math.random()-0.5)*0.18;
    vy = (Math.random()-0.5)*0.22;
    vz = (Math.random()-0.5)*0.12;
    
    const label = document.getElementById('d20loading');
    if(label) label.style.display = 'block';
    const res = document.getElementById('d20result');
    if(res) res.style.display = 'none';
};

function startLanding(){
    state = 'landing';
    landStart = performance.now();
    currentQ.copy(diceGroup.quaternion);
    targetQ.copy(quatForFace(FACE_NUMS.indexOf(rollFinal)));
}

function loop(){
    requestAnimationFrame(loop);
    if (!scene || !camera || !renderer) return;
    
    const now = performance.now();
    if(state === 'rolling'){
        const p = Math.min((now-rollStart)/rollDur, 1);
        const spd = (1-p*p)*5 + 0.8;
        diceGroup.rotation.x += vx*spd;
        diceGroup.rotation.y += vy*spd;
        diceGroup.rotation.z += vz*spd;
        if(p >= 1) startLanding();
    } else if(state === 'landing'){
        const p = Math.min((now-landStart)/LAND_DUR, 1);
        const t = 1 - Math.pow(1-p, 3);
        diceGroup.quaternion.slerpQuaternions(currentQ, targetQ, t);
        if(p >= 1) {
            diceGroup.quaternion.copy(targetQ);
            state = 'stopped';
            const label = document.getElementById('d20loading');
            if(label) label.style.display = 'none';
            
            const res = document.getElementById('d20result');
            if(res) {
                const hue = ((rollFinal - 1) / 19) * 120;
                let color = `hsl(${hue}, 100%, 50%)`;
                
                let text = rollFinal;
                let scale = '48px';
                if (rollFinal === 20) {
                    text = 'CRITICAL SUCCESS<br><span style="font-size:48px">20</span>';
                    scale = '18px';
                } else if (rollFinal === 1) {
                    text = 'CRITICAL FAILURE<br><span style="font-size:48px">1</span>';
                    scale = '18px';
                }
                res.innerHTML = text;
                res.style.color = color;
                res.style.fontSize = scale;
                res.style.textAlign = 'center';
                res.style.textShadow = '0 0 15px ' + color;
                res.style.display = 'block';
                
                res.style.animation = 'none';
                void res.offsetWidth;
                res.style.animation = null;
                res.classList.add('dice-final');
            }
        }
    } else if(state === 'idle'){
        diceGroup.rotation.x += 0.004;
        diceGroup.rotation.y += 0.007;
        diceGroup.rotation.z += 0.002;
    }
    renderer.render(scene, camera);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init3D);
} else {
    init3D();
}
</script>
"""

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(), css=CSS, js="""function() { document.documentElement.classList.add('dark'); }""") as demo:
    gr.HTML(HEAD_SCRIPT)

    with gr.Tab("Hackathon Idea Forge"):
        gr.Markdown("""
        # ⚡ Build Small Idea Forge
        Generate many tiny, shippable Hugging Face hackathon concepts with simple starter code.
        Use this to spend HF credits, ZeroGPU slots, and Modal credits on multiple focused apps instead of one giant project.
        """)
        with gr.Row():
            idea_count = gr.Slider(1, 10, value=6, step=1, label="How many projects?")
            idea_style = gr.Radio(["basic", "pure genius", "weird but useful"], value="pure genius", label="Idea style")
            modal_toggle = gr.Checkbox(value=True, label="Include Modal credit ideas")
        idea_markdown = gr.Markdown()
        starter = gr.Code(language="python", label="Basic Gradio starter code")
        gr.Button("Generate sprint pack", variant="primary").click(
            build_project_pack, [idea_count, idea_style, modal_toggle], [idea_markdown, starter]
        )
    state = gr.State()
    gr.HTML(
        """
        <div id='title-card'>
          <h1>🐉 Misadventure Master</h1>
          <p>A tiny tabletop RPG where the Game Master is a small LLM, the dice are real, and the dungeon is legally allowed to roast your choices.</p>
        </div>
        """
    )
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            hero_name = gr.Textbox(show_label=False, placeholder="What is your name, brave hero?", value="")
        with gr.Column(scale=2):
            start = gr.Button("Start / Reset Dungeon", elem_id="start-btn")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="Adventure Log", height=435, type="messages")
            action = gr.Textbox(label="What do you do?", placeholder="I inspect the suspicious mushroom with heroic overconfidence.")
        with gr.Column(scale=2):
            scene = gr.HTML(elem_classes="stat-panel", value="<div style='height: 520px; display: flex; align-items: center; justify-content: center; color: #fbbf24;'>Loading scene...</div>")
            
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Row():
                submit = gr.Button("Roll and act", variant="primary")
                choice_1 = gr.Button("Inspect something suspicious")
                choice_2 = gr.Button("Negotiate badly")
                choice_3 = gr.Button("Run away heroically")
        with gr.Column(scale=2):
            dice = gr.HTML(elem_classes="dice-panel", value="<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;height:320px;border-radius:13px;width:100%;'><div id='d20mount' style='width:200px;height:200px;'></div><div id='d20loading' style='margin-top:10px;font-weight:bold;color:#fbbf24;font-family:monospace;font-size:15px;display:none;'>Rolling...</div><div id='d20result' style='margin-top:10px;font-weight:900;display:none;'></div></div>")
            hidden_roll_target = gr.Textbox(value=f"20-{time.time()}", visible=False, elem_id="roll-target")

    # JS event triggers the roll animation seamlessly without destroying the DOM
    hidden_roll_target.change(
        fn=None,
        inputs=[hidden_roll_target],
        js="(val) => { if(val && window.__d20_roll) { window.__d20_roll(val.split('-')[0]); } }"
    )

    start.click(start_game, [hero_name], [state, chat, dice, scene, choice_1, choice_2, choice_3, hidden_roll_target], api_name=False)
    submit.click(play_turn, [action, state, chat], [state, chat, dice, scene, choice_1, choice_2, choice_3, action, hidden_roll_target], api_name=False)
    choice_1.click(use_choice, [choice_1], [action], api_name=False)
    choice_2.click(use_choice, [choice_2], [action], api_name=False)
    choice_3.click(use_choice, [choice_3], [action], api_name=False)
    demo.load(start_game, [hero_name], [state, chat, dice, scene, choice_1, choice_2, choice_3, hidden_roll_target], api_name=False)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
