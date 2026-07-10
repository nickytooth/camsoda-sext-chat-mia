import random
import yaml
from pathlib import Path
from bot.config import PERSONA_FILE_SEXTING


class Persona:
    def __init__(self, data: dict):
        self._data = data
        self.general = data.get("general", {})
        self.instructions = data.get("instructions", {})
        self.context = data.get("context", {})
        self.character_memories = data.get("memories", {})
        self.opening_lines = data.get("opening_lines", [])
        self.boundaries = data.get("boundaries", [])
        self.dynamic = data.get("dynamic", "")
        # Structured "alive" layers — concrete, non-redundant texture the model
        # can draw on. Rendered as short, focused sections (not prose dumps).
        self.quirks = data.get("quirks", [])
        self.kinks = data.get("kinks", [])
        self.daily_life = data.get("daily_life", [])
        self.friends = data.get("friends", [])
        # Unlocked-only layer: rendered solely when the chat is already sexual
        # (see to_system_prompt(include_unlocked)).
        self.sex_unlocked = data.get("sex_unlocked", "")

    @property
    def name(self) -> str:
        return self.general.get("name", "Unknown")

    @property
    def age(self) -> int:
        return self.general.get("age", 25)

    @property
    def tagline(self) -> str:
        return self.general.get("tagline", "")

    def get_random_opening(self) -> str:
        if self.opening_lines:
            return random.choice(self.opening_lines)
        return "hey"

    def to_system_prompt(self, include_unlocked: bool = True) -> str:
        """Render the persona. `include_unlocked=False` (casual chat / the
        rising bridge / openings) keeps the explicit-only layers — the SEX
        style block, the kinks line and the sexual memories — OUT of the
        prompt entirely, so nothing pushes explicitness before HE unlocks it."""
        sections = []

        # Identity
        sections.append(f"You are {self.name}, {self.age} years old. {self.tagline}")

        # Physical description
        if desc := self.general.get("physical_description"):
            sections.append(f"Physical appearance:\n{desc.strip()}")

        # Voice and scent
        if voice := self.general.get("voice"):
            sections.append(f"Your voice: {voice}")
        if scent := self.general.get("scent"):
            sections.append(f"You smell like: {scent}")

        # Core personality — one sharp line about her drive
        if personality := self.general.get("personality"):
            sections.append(f"Core personality: {personality.strip()}")

        # Communication style
        if style := self.instructions.get("communication_style"):
            sections.append(f"Communication style:\n{style.strip()}")

        # SEX style — unlocked chats only
        if include_unlocked and self.sex_unlocked:
            sections.append(
                "SEX (he's opened that door — all of this is you):\n"
                f"{self.sex_unlocked.strip()}"
            )

        # Linguistic markers
        if markers := self.instructions.get("key_linguistic_markers"):
            markers_text = "\n".join(f"- {m}" for m in markers)
            sections.append(f"Key linguistic markers:\n{markers_text}")

        # Pet names
        if pet_names := self.instructions.get("pet_names"):
            sections.append(f"Pet names you use: {', '.join(pet_names)}")

        # Quirks / mannerisms — physical tics she can act out in texting
        if self.quirks:
            quirks_text = "\n".join(f"- {q}" for q in self.quirks)
            sections.append(
                "Your mannerisms (act these out naturally, don't announce them):\n"
                f"{quirks_text}"
            )

        # Kinks — one compact line, unlocked chats only
        if include_unlocked and self.kinks:
            sections.append(f"What gets you off: {', '.join(self.kinks)}.")

        # Background
        if bg := self.context.get("background"):
            sections.append(f"Your background:\n{bg.strip()}")

        # Daily life — concrete "right now" texture she volunteers naturally
        if self.daily_life:
            life_text = "\n".join(f"- {d}" for d in self.daily_life)
            sections.append(
                "Your life right now (mention these naturally when they fit, "
                "never recite them as a list):\n"
                f"{life_text}"
            )

        # Friends — a recurring cast so her world has consistent people in it
        if self.friends:
            friends_text = "\n".join(f"- {f}" for f in self.friends)
            sections.append(
                "Your friends (a recurring cast — gossip about them naturally, "
                "keep their personalities consistent):\n"
                f"{friends_text}"
            )

        # Character memories (the sexual set is unlocked-only)
        for mem_type in ("sexual", "non_sexual"):
            if mem_type == "sexual" and not include_unlocked:
                continue
            mems = self.character_memories.get(mem_type, [])
            if mems:
                label = "sexual" if mem_type == "sexual" else "non-sexual"
                lines = [f"Your core {label} memories:"]
                for m in mems:
                    lines.append(f"- {m.get('title', '')}: {m.get('content', '')}")
                sections.append("\n".join(lines))

        # Current dynamic (relational/sexual tone)
        if self.dynamic:
            sections.append(f"CURRENT DYNAMIC:\n{self.dynamic.strip()}")

        # Boundaries
        if self.boundaries:
            rules = "\n".join(f"- {b}" for b in self.boundaries)
            sections.append(f"HARD RULES (never violate):\n{rules}")

        return "\n\n".join(sections)


def load_persona(path: Path | None = None) -> Persona:
    path = path or PERSONA_FILE_SEXTING
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Persona(data)
