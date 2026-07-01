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

    def to_system_prompt(self) -> str:
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

        # Communication style
        if style := self.instructions.get("communication_style"):
            sections.append(f"Communication style:\n{style.strip()}")

        # Linguistic markers
        if markers := self.instructions.get("key_linguistic_markers"):
            markers_text = "\n".join(f"- {m}" for m in markers)
            sections.append(f"Key linguistic markers:\n{markers_text}")

        # Pet names
        if pet_names := self.instructions.get("pet_names"):
            sections.append(f"Pet names you use: {', '.join(pet_names)}")

        # Background
        if bg := self.context.get("background"):
            sections.append(f"Your background:\n{bg.strip()}")

        # Character memories
        for mem_type in ("sexual", "non_sexual"):
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
