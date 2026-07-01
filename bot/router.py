import re
import logging

logger = logging.getLogger(__name__)

NSFW_KEYWORDS = {
    "sex", "fuck", "cock", "dick", "pussy", "ass", "tits", "boobs", "nipple",
    "cum", "orgasm", "horny", "naked", "nude", "blowjob", "bj", "handjob",
    "masturbat", "jerk off", "finger", "moan", "wet", "hard for", "suck",
    "lick", "ride", "doggy", "missionary", "anal", "deepthroat", "strip",
    "lingerie", "bdsm", "spank", "choke", "dominate", "submissive", "kinky",
    "fetish", "dildo", "vibrator", "threesome", "oral", "erotic", "seduc",
    "touch yourself", "touch me", "feel you", "inside me", "inside you",
    "make love", "make me cum", "want you", "need you bad",
}

NSFW_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in NSFW_KEYWORDS) + r")",
    re.IGNORECASE,
)


def classify_fast(message: str) -> str | None:
    if NSFW_PATTERN.search(message):
        return "nsfw"
    return None
