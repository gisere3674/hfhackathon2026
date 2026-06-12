import base64
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


APP_TITLE = "Troll Dungeon Master"
PARAMETER_LEDGER = [
    ("Qwen/Qwen3-4B", 4.0, "Game Master text model"),
    ("black-forest-labs/FLUX.1-schnell", 12.0, "Scene image model"),
]
PARAMETER_TOTAL_B = sum(item[1] for item in PARAMETER_LEDGER)
PARAMETER_LIMIT_B = 32.0

FAILURES = [
    "You fail so loudly that the dungeon updates its privacy policy.",
    "A goblin starts live-commentating your mistake with a tiny brass megaphone.",
    "Your weapon files a workplace complaint against your grip.",
    "The door remains closed, but your confidence takes visible splash damage.",
    "A nearby skeleton stops being dead just long enough to sigh at you.",
]

SUCCESSES_WITH_A_COST = [
    "You succeed, but everyone assumes it was an accident.",
    "It works perfectly, which makes the dungeon suspicious and emotionally distant.",
    "You achieve your goal while looking exactly like a confused intern.",
    "The plan succeeds, but a rat now thinks it is your manager.",
]

CURSES = [
    "All doors judge you in silence.",
    "Every NPC calls you Captain Moistboots.",
    "Your backpack is haunted by a passive-aggressive spoon.",
    "The dungeon narrator spoils one tiny part of every plan.",
    "Your shadow keeps rolling its eyes before you act.",
]

ROOMS = [
    "The Lobby of Questionable Decisions",
    "The Hallway of Wet Socks",
    "The Goblin HR Department",
    "The Mimic Cafeteria",
    "The Bridge of Mild Inconvenience",
    "The Treasury of Almost Valuable Objects",
]

LOOT = [
    "Rubber Sword of Emotional Damage",
    "Potion of Slightly Better Posture",
    "Coupon for One Free Goblin Apology",
    "Cursed Spoon of Feedback",
    "Helmet of Unwanted Confidence",
    "Map with Only Your Mistakes Marked",
]


@dataclass
class GameState:
    hero_name: str = "Adventurer"
    room: str = "The Lobby of Questionable Decisions"
    hp: int = 12
    gold: int = 3
    shame: int = 0
    turn: int = 0
    curse: str = "None yet. Suspicious."
    inventory: list[str] = field(default_factory=lambda: ["Rusty Dagger", "Snack of Unknown Origin"])
    quest: str = "Find the Crown of Reasonable Expectations and immediately misuse it."
    seed: str = field(default_factory=lambda: f"goblin-{random.randint(1000, 9999)}")

    def clamp(self) -> None:
        self.hp = max(0, min(20, self.hp))
        self.gold = max(0, min(999, self.gold))
        self.shame = max(0, min(999, self.shame))


def new_game(hero_name: str, seed: str) -> GameState:
    clean_name = hero_name.strip() or "Adventurer"
    clean_seed = seed.strip() or f"goblin-{random.randint(1000, 9999)}"
    random.seed(clean_seed)
    return GameState(hero_name=clean_name, room=random.choice(ROOMS), seed=clean_seed)


def roll_d20(mode: str) -> dict[str, Any]:
    first = random.randint(1, 20)
    second = random.randint(1, 20)
    if mode == "Advantage":
        return {"mode": mode, "rolls": [first, second], "total": max(first, second)}
    if mode == "Disadvantage":
        return {"mode": mode, "rolls": [first, second], "total": min(first, second)}
    return {"mode": mode, "rolls": [first], "total": first}


def build_system_prompt() -> str:
    return """
You are The Troll Dungeon Master, a chaotic but fair fantasy tabletop game master.
You run a tiny online dungeon crawler where the player is the hero and you are the narrator.

Tone rules:
- Be funny, absurd, theatrical, and sarcastic.
- Mock the fictional situation, not the real user.
- Do not use slurs, identity-based insults, sexual humiliation, real threats, or hateful content.
- Keep stakes cartoonish and playful.

Game rules:
- Respect the provided dice result and game state.
- A low roll should create a comedic setback.
- A high roll should succeed, but can still be silly.
- Keep the story moving and offer exactly three next choices.
- Keep narration under 170 words.

Return only valid JSON with this schema:
{
  "narration": "string",
  "stat_changes": {"hp": integer, "gold": integer, "shame": integer},
  "inventory_add": ["string"],
  "inventory_remove": ["string"],
  "new_room": "string or empty string",
  "new_curse": "string or empty string",
  "choices": ["string", "string", "string"],
  "image_prompt": "string"
}
""".strip()


def build_user_prompt(state: GameState, action: str, dice: dict[str, Any], chaos: int) -> str:
    return json.dumps(
        {
            "player_action": action,
            "dice": dice,
            "chaos_level_1_to_5": chaos,
            "state": asdict(state),
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


def call_text_model(state: GameState, action: str, dice: dict[str, Any], chaos: int) -> dict[str, Any] | None:
    modal_endpoint = os.getenv("MODAL_TEXT_ENDPOINT", "").strip()
    if modal_endpoint:
        data = modal_post(
            modal_endpoint,
            {
                "system": build_system_prompt(),
                "user": build_user_prompt(state, action, dice, chaos),
                "model": "Qwen/Qwen3-4B",
            },
        )
        return extract_json(data.get("text", json.dumps(data)))

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        client = InferenceClient(model="Qwen/Qwen3-4B", token=hf_token)
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(state, action, dice, chaos)},
        ]
        completion = client.chat_completion(messages=messages, max_tokens=450, temperature=0.9)
        return extract_json(completion.choices[0].message.content)

    return None


def fallback_gm(state: GameState, action: str, dice: dict[str, Any], chaos: int) -> dict[str, Any]:
    total = dice["total"]
    random.seed(f"{state.seed}-{state.turn}-{action}-{total}-{chaos}")
    if total <= 7:
        result = random.choice(FAILURES)
        hp_delta = -1 if total <= 3 else 0
        shame_delta = random.randint(1, 2 + chaos)
        gold_delta = -1 if state.gold > 0 and total <= 5 else 0
    elif total >= 16:
        result = random.choice(SUCCESSES_WITH_A_COST)
        hp_delta = 0
        shame_delta = random.randint(0, 1)
        gold_delta = random.randint(1, 3)
    else:
        result = "You make progress in the exact way a raccoon makes progress through a wedding cake."
        hp_delta = 0
        shame_delta = random.randint(0, 2)
        gold_delta = 0

    new_item = random.choice(LOOT) if total >= 14 and random.random() < 0.45 else None
    new_curse = random.choice(CURSES) if (state.turn + 1) % max(2, 6 - chaos) == 0 else ""
    new_room = random.choice(ROOMS) if total >= 12 and random.random() < 0.5 else ""

    narration = (
        f"You attempt to {action.strip() or 'do something heroic but poorly documented'}. "
        f"The d20 lands on {total}. {result} "
        f"The dungeon makes a tiny note in its diary: 'Still technically a hero.'"
    )
    if new_item:
        narration += f" You acquire the {new_item}, which may or may not be evidence."
    if new_curse:
        narration += f" New curse applied: {new_curse}"

    return {
        "narration": narration,
        "stat_changes": {"hp": hp_delta, "gold": gold_delta, "shame": shame_delta},
        "inventory_add": [new_item] if new_item else [],
        "inventory_remove": [],
        "new_room": new_room,
        "new_curse": new_curse,
        "choices": [
            "Inspect the most suspicious object in the room",
            "Negotiate with unnecessary confidence",
            "Run away heroically while maintaining eye contact",
        ],
        "image_prompt": (
            f"whimsical fantasy dungeon scene, {state.hero_name} in {new_room or state.room}, "
            f"comedic tabletop RPG art, goblins judging, cursed props, dramatic torchlight, no text"
        ),
    }


def normalize_model_output(data: dict[str, Any]) -> dict[str, Any]:
    changes = data.get("stat_changes") or {}
    choices = data.get("choices") or []
    while len(choices) < 3:
        choices.append("Make a questionable tactical decision")
    return {
        "narration": str(data.get("narration") or "The dungeon coughs awkwardly and pretends that counted."),
        "stat_changes": {
            "hp": int(changes.get("hp", 0)),
            "gold": int(changes.get("gold", 0)),
            "shame": int(changes.get("shame", 0)),
        },
        "inventory_add": [str(item) for item in data.get("inventory_add") or [] if item],
        "inventory_remove": [str(item) for item in data.get("inventory_remove") or [] if item],
        "new_room": str(data.get("new_room") or ""),
        "new_curse": str(data.get("new_curse") or ""),
        "choices": [str(choice) for choice in choices[:3]],
        "image_prompt": str(data.get("image_prompt") or "funny fantasy dungeon scene, tabletop RPG, no text"),
    }


def apply_turn(state: GameState, result: dict[str, Any]) -> GameState:
    changes = result["stat_changes"]
    state.hp += changes["hp"]
    state.gold += changes["gold"]
    state.shame += changes["shame"]
    for item in result["inventory_remove"]:
        if item in state.inventory:
            state.inventory.remove(item)
    for item in result["inventory_add"]:
        if item not in state.inventory:
            state.inventory.append(item)
    if result["new_room"]:
        state.room = result["new_room"]
    if result["new_curse"]:
        state.curse = result["new_curse"]
    state.turn += 1
    state.clamp()
    return state


def status_markdown(state: GameState) -> str:
    inventory = "\n".join(f"- {item}" for item in state.inventory) or "- Nothing but vibes"
    ledger = "\n".join(f"- {name}: {params:.1f}B — {role}" for name, params, role in PARAMETER_LEDGER)
    return f"""
### Hero Sheet
- **Name:** {state.hero_name}
- **Room:** {state.room}
- **HP:** {state.hp}/20
- **Gold:** {state.gold}
- **Shame:** {state.shame}
- **Curse:** {state.curse}
- **Quest:** {state.quest}
- **Seed:** `{state.seed}`

### Inventory
{inventory}

### Parameter Budget
{ledger}

**Total:** {PARAMETER_TOTAL_B:.1f}B / {PARAMETER_LIMIT_B:.0f}B
""".strip()


def dice_markdown(dice: dict[str, Any]) -> str:
    rolls = ", ".join(str(roll) for roll in dice["rolls"])
    return f"### 🎲 d20 Roll\n**Mode:** {dice['mode']}  \n**Rolls:** {rolls}  \n**Result:** {dice['total']}"


def svg_scene(prompt: str, state: GameState, dice_total: int) -> str:
    mood = "#7c2d12" if dice_total <= 7 else "#166534" if dice_total >= 16 else "#854d0e"
    safe_prompt = prompt[:130].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='768' height='512' viewBox='0 0 768 512'>
  <defs>
    <linearGradient id='bg' x1='0' x2='1' y1='0' y2='1'>
      <stop offset='0%' stop-color='#111827'/>
      <stop offset='60%' stop-color='{mood}'/>
      <stop offset='100%' stop-color='#020617'/>
    </linearGradient>
    <filter id='glow'><feGaussianBlur stdDeviation='5' result='blur'/><feMerge><feMergeNode in='blur'/><feMergeNode in='SourceGraphic'/></feMerge></filter>
  </defs>
  <rect width='768' height='512' fill='url(#bg)'/>
  <path d='M0 392 C120 330 200 450 330 370 C460 290 560 420 768 335 L768 512 L0 512 Z' fill='#0f172a' opacity='0.88'/>
  <circle cx='625' cy='92' r='42' fill='#facc15' opacity='0.85' filter='url(#glow)'/>
  <rect x='120' y='170' width='130' height='210' rx='18' fill='#292524' stroke='#fbbf24' stroke-width='5'/>
  <circle cx='218' cy='278' r='9' fill='#fbbf24'/>
  <path d='M430 190 l48 80 l-96 0 z' fill='#16a34a' stroke='#dcfce7' stroke-width='4'/>
  <circle cx='407' cy='230' r='8' fill='white'/><circle cx='453' cy='230' r='8' fill='white'/>
  <path d='M400 263 Q430 286 462 263' stroke='white' stroke-width='5' fill='none'/>
  <text x='384' y='64' text-anchor='middle' fill='#fef3c7' font-family='Georgia, serif' font-size='34' font-weight='700'>Troll Dungeon Master</text>
  <text x='384' y='112' text-anchor='middle' fill='#fde68a' font-family='monospace' font-size='22'>Room: {state.room[:45]}</text>
  <text x='384' y='438' text-anchor='middle' fill='#f8fafc' font-family='monospace' font-size='18'>{safe_prompt}</text>
</svg>
""".strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def generate_image(prompt: str, state: GameState, dice_total: int) -> str:
    modal_endpoint = os.getenv("MODAL_IMAGE_ENDPOINT", "").strip()
    if modal_endpoint:
        data = modal_post(
            modal_endpoint,
            {
                "prompt": prompt,
                "model": "black-forest-labs/FLUX.1-schnell",
                "width": 768,
                "height": 512,
            },
        )
        if data.get("image_url"):
            return data["image_url"]
        if data.get("image_base64"):
            return f"data:image/png;base64,{data['image_base64']}"
    return svg_scene(prompt, state, dice_total)


def boot_message(state: GameState) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                f"Welcome, {state.hero_name}. You awaken in **{state.room}**. "
                "A goblin wearing a tiny headset announces that your competence is under review. "
                "What do you do?"
            ),
        }
    ]


def start_game(hero_name: str, seed: str):
    state = new_game(hero_name, seed)
    image = svg_scene("A doomed hero enters a sarcastic fantasy dungeon, no text", state, 10)
    return state, boot_message(state), status_markdown(state), dice_markdown({"mode": "Normal", "rolls": [10], "total": 10}), image, "", "", ""


def play_turn(action: str, mode: str, chaos: int, state: GameState, chat: list[dict[str, str]]):
    if state is None:
        state = new_game("Adventurer", "")
        chat = boot_message(state)
    clean_action = action.strip() or "stand there with suspicious confidence"
    random.seed(f"{state.seed}-{state.turn}-{time.time_ns()}")
    dice = roll_d20(mode)

    model_data = call_text_model(state, clean_action, dice, chaos)
    result = normalize_model_output(model_data or fallback_gm(state, clean_action, dice, chaos))
    state = apply_turn(state, result)
    image = generate_image(result["image_prompt"], state, dice["total"])

    updated_chat = list(chat or [])
    updated_chat.append({"role": "user", "content": clean_action})
    updated_chat.append({"role": "assistant", "content": result["narration"]})

    return (
        state,
        updated_chat,
        status_markdown(state),
        dice_markdown(dice),
        image,
        result["choices"][0],
        result["choices"][1],
        result["choices"][2],
        "",
    )


def use_choice(choice: str) -> str:
    return choice


CSS = """
.gradio-container {
  background: radial-gradient(circle at top, #422006 0, #111827 45%, #020617 100%);
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
button.primary {
  background: linear-gradient(90deg, #b45309, #7c2d12) !important;
  border: 1px solid #fbbf24 !important;
}
"""


with gr.Blocks(title=APP_TITLE) as demo:
    state = gr.State()
    gr.HTML(
        """
        <div id='title-card'>
          <h1>🐉 Troll Dungeon Master</h1>
          <p>A tiny tabletop RPG where the Game Master is a small LLM, the dice are real, and the dungeon is legally allowed to roast your choices.</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            hero_name = gr.Textbox(label="Hero name", value="Sir Buttonmash")
            seed = gr.Textbox(label="Dungeon seed", value="banana-lich-042")
            chaos = gr.Slider(1, 5, value=3, step=1, label="Troll level")
            mode = gr.Radio(["Normal", "Advantage", "Disadvantage"], value="Normal", label="Dice mode")
            start = gr.Button("Start / Reset Dungeon", variant="primary")
            status = gr.Markdown(elem_classes="stat-panel")
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Adventure Log", height=520)
            action = gr.Textbox(label="What do you do?", placeholder="I inspect the suspicious mushroom with heroic overconfidence.")
            with gr.Row():
                submit = gr.Button("Roll and act", variant="primary")
                choice_1 = gr.Button("Inspect something suspicious")
                choice_2 = gr.Button("Negotiate badly")
                choice_3 = gr.Button("Run away heroically")
        with gr.Column(scale=1):
            scene = gr.Image(label="Current scene", height=360)
            dice = gr.Markdown(elem_classes="dice-panel")
            gr.Markdown(
                """
### How it works
1. Python rolls the d20 and updates the game state.
2. Qwen3-4B acts as the troll Game Master when configured.
3. FLUX.1-schnell creates scenes when a Modal image endpoint is configured.
4. Built-in fallbacks keep the demo playable without paid GPUs.
"""
            )

    start.click(start_game, [hero_name, seed], [state, chat, status, dice, scene, choice_1, choice_2, choice_3])
    submit.click(play_turn, [action, mode, chaos, state, chat], [state, chat, status, dice, scene, choice_1, choice_2, choice_3, action])
    choice_1.click(use_choice, [choice_1], [action])
    choice_2.click(use_choice, [choice_2], [action])
    choice_3.click(use_choice, [choice_3], [action])
    demo.load(start_game, [hero_name, seed], [state, chat, status, dice, scene, choice_1, choice_2, choice_3])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
