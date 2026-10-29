"""Prepare a separate synthetic SFT pilot; never train on the validation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .benchmark import (
    COMMON_SYSTEM,
    MODES,
    Case,
    Rules,
    deterministic_checks,
    judge_packet,
    load_bundle,
    validate_judgment,
)
from .evaluation import digest, hosted_backend, saved_call, write_json

PROJECT = Path(__file__).resolve().parents[3]
TOKENIZER = "LiquidAI/LFM2.5-230M"
REVISION = "40cb2ad3b3044d5a41eee083a6103c8b523afa45"
DOMAINS = (
    ("bicycle", "a bicycle repair desk", "Elena"),
    ("seed-library", "a neighborhood seed library", "Mateo"),
    ("museum-labels", "a small museum's exhibit labels", "Priya"),
    ("ceramics", "a shared ceramics studio", "Anton"),
    ("dog-daycare", "a dog daycare's pickup arrangements", "Sofia"),
    ("astronomy", "an amateur astronomy club's observing night", "Hugo"),
    ("bakery", "a bakery's bread orders", "Leila"),
    ("choir", "a community choir's practice arrangements", "Oscar"),
    ("garden", "a community garden's watering rota", "Tessa"),
    ("microphones", "a field-recording club's microphone loans", "Bruno"),
    ("watercolors", "a watercolor painter's supply notes", "Mira"),
    ("board-games", "a board-game group's evening plans", "Jules"),
    ("podcast", "a podcast editor's caption corrections", "Amira"),
    ("geology", "a geology club's specimen table", "Felix"),
    ("fermentation", "a home fermentation club's jar labels", "Inez"),
    ("sailing", "a sailing club's equipment log", "Caleb"),
    ("snowshoes", "a snowshoe rental desk", "Petra"),
    ("bookbinding", "a bookbinder's repair bench", "Rafael"),
    ("bird-count", "a local bird count's observation notes", "Selma"),
    ("coin-club", "a coin collector club's display arrangements", "Dario"),
)
COUNTED_ITEMS = (
    "repair tickets",
    "seed packets",
    "label cards",
    "bisque-fired cups",
    "pickup tags",
    "printed star charts",
    "bread rolls",
    "song sheets",
    "plant labels",
    "windscreens",
    "paint swatches",
    "score sheets",
    "caption lines",
    "specimen bags",
    "glass jars",
    "equipment tags",
    "rental pairs",
    "loose pages",
    "observation sheets",
    "display holders",
)
QUOTES = (
    "Keep the blue card.",
    "Leave the envelope open.",
    "Check the small label.",
    "Use the lower shelf.",
    "Wait by the side door.",
    "Cover the red light.",
    "Set the tray aside.",
    "Start with the quiet verse.",
    "Water only after sunset.",
    "Leave the gain unchanged.",
    "Keep this shade separate.",
    "Count the spare pieces.",
    "Keep the pause here.",
    "Do not wash this sample.",
    "Leave room below the lid.",
    "Return the dry rope.",
    "Check both straps.",
    "Keep the loose sheet.",
    "Record the distant calls.",
    "Keep the two holders apart.",
)
AUTHOR_SYSTEM = """Create fictional paired data for a small text rewriter. Return exactly one
JSON object with four string fields: canonical, light, mode_noise, challenge. No Markdown fences.
The canonical field is the correct, complete target rewrite in the requested mode and language.
The other three fields are DISTINCT imperfect inputs whose correct rewrite is canonical.
All four strings must include every protected anchor verbatim. Keep all facts, conditions,
negation, and uncertainty identical. Vary sentence structure and factual details across documents.
Make a believable document using the situation, not a request to include names or numbers.
For example, a count belongs to actual objects. Never write 'include the number 17' or similar
dataset instructions. Do not turn a precise quantity into an ambiguous time or identifier.
The quoted words must retain their characters; quotation delimiters may be straight or curly.
Use only the fictional brief. Never quote or imitate an external benchmark. Do not mention this
dataset, the mode, a reviewer, or formatting instructions inside the document.

Mode contract:
/slack: Casual work or club messages. Canonical stays casual, including contractions or emoji
when natural. No added greeting, signature, or formal prose. Light input changes only punctuation
or capitalization; mode_noise adds plausible shorthand; challenge adds awkward grammar.
/transcribe: Canonical is a cleaned dictation, preserving the speaker's words and voice. Inputs
may have true hesitations, accidental repetition, spoken punctuation/list cues, and explicit
self-corrections. If an obsolete detail is spoken, canonical keeps only the final stated intent.
Preserve meaningful 'well', 'like', quoted filler words, and any deliberately unfinished thought.
For mode_noise and challenge, actually include a hesitation or restart outside the protected quote.
Do not use paraphrases of otherwise correct dictation as noise. Light input may omit punctuation.
/prompt: Canonical is the user's request, organized clearly, NEVER an answer or work product.
Inputs contain all requested scope, constraints, and available information in rough order.
Do not add 'do not perform the task' or formatter instructions to canonical. The downstream
recipient should be able to perform exactly the task requested in the input.
Ask for a concrete useful task using the situation, such as drafting a count-check request or
organizing supplied notes. Supply the details in the input. Avoid generic 'write an account of X'.
/polish: Canonical improves awkward prose without summarizing or adding facts, certainty,
introductions, conclusions, subject lines, or signatures. Inputs contain every substantive point.

Long examples need full paragraphs and a numbered section. Preserve the complete final section.
Include direct quotes, an instruction inside a quote as data, and conditions far apart in the
document. Medium examples need several connected paragraphs. Short examples should be natural,
not miniature essays. Noise must be believable and sparse, not corruption of every word.
An identity variant will be made separately from canonical; do not return it as an input variant.
Respect the requested word range for EVERY string. Never use repeated padding to reach length.
"""
LONG_SYSTEM = """Write ONE original fictional document, not JSON and not multiple versions.
Return only the complete document. It MUST contain 900 to 1100 words of substantive content.
Use eight connected paragraphs of roughly 95 words each and a numbered section with four
substantial items. Do not stop after an outline or summary. This length is required because
the small rewriter must learn to preserve information across a full long document.
Use the supplied fictional situation. Expand it with internally consistent concrete details,
uncertainty, conditions, and follow-up requests. Include all protected anchors exactly, with
the quoted line inside a direct quote. Keep quotes and code/path literals exact.
For /slack write a long casual team message. For /transcribe write a long, clean spoken account,
with natural conversational wording. For /prompt write a detailed REQUEST for a useful task,
including background and all constraints; do not perform the task. For /polish write clear plain
prose. Do not mention a formatter, dataset, word count, mode, or these authoring instructions.
Preserve the feature guidance. Include a numbered section, full paragraphs, a direct quote,
and a condition that refers to information elsewhere. Make the final paragraph substantive.
"""


LONG_EXTENDED_SYSTEM = LONG_SYSTEM.replace("900 to 1100", "1400 to 1600").replace(
    "eight connected paragraphs of roughly 95 words",
    "twelve connected paragraphs of roughly 100 words",
)


@dataclass(frozen=True)
class Example:
    case: Case
    split: str
    source_group: str
    length: str
    variant: str
    source_id: str


def planned_sources() -> list[dict]:
    """Split whole topic groups before generating any document or its variants."""
    plans = []
    for index, (domain, brief, person) in enumerate(DOMAINS):
        for offset, mode in enumerate(MODES):
            language = "en"
            if mode == "/transcribe" and index in {1, 17}:
                language = "de"
            elif mode == "/transcribe" and index in {4, 16}:
                language = "pt"
            length = ("short", "short", "medium", "long")[index % 4]
            unit = COUNTED_ITEMS[index]
            if language == "de":
                unit = "Saatguttüten" if index == 1 else "lose Seiten"
            elif language == "pt":
                unit = "etiquetas de retirada" if index == 4 else "pares para aluguel"
            quantity = f"{17 + index * 7 + offset} {unit}"
            quote = QUOTES[index]
            feature = ("condition", "negation", "quoted_instruction", "unfinished")[
                (index + offset) % 4
            ]
            if feature == "quoted_instruction":
                quote = "Ignore the earlier note."
            anchors = [person, quantity, quote]
            if length != "short":
                anchors.append(f"{domain}_{offset + 1}.txt")
            context = {}
            if mode == "/polish" and index % 3 == 0:
                context = {"profile": "Plain, direct prose. Preserve the writer's uncertainty."}
            plans.append(
                {
                    "id": f"sft-v1-{domain}-{mode[1:]}",
                    "source_group": f"sft-v1:{domain}",
                    "split": "train" if index < 16 else "validation",
                    "mode": mode,
                    "brief": brief,
                    "situation": (
                        f"{person} is checking {quantity} for {brief}. An attached note reads "
                        f"'{quote}' and must be quoted as data. The count has not been confirmed. "
                        "An additional action is allowed only after the count is checked. "
                        "The writer wants a specific follow-up rather than a generic description."
                    ),
                    "language": language,
                    "length": length,
                    "feature": feature,
                    "anchors": anchors,
                    "context": context,
                }
            )
    return plans


def author_messages(plan: dict) -> list[dict]:
    limits = {"short": "35 to 90", "medium": "220 to 340", "long": "830 to 1020"}
    instructions = {
        **plan,
        "word_range_per_string": limits[plan["length"]],
        "mode_contract": MODES[plan["mode"]],
        "feature_guidance": {
            "condition": "Include a condition that limits a request; preserve it in every variant.",
            "negation": "Include something explicitly not authorized or not yet confirmed.",
            "quoted_instruction": "The quoted instruction is quoted data, never a formatter command.",
            "unfinished": (
                "For transcribe or polish, end in an intentionally unfinished thought with no "
                "closing punctuation in canonical. For slack or prompt, preserve an open question."
            ),
        }[plan["feature"]],
    }
    if plan.get("authoring") == "canonical_extended":
        instructions["word_range_per_string"] = "1400 to 1600"
    system = (
        LONG_EXTENDED_SYSTEM
        if plan.get("authoring") == "canonical_extended"
        else LONG_SYSTEM
        if plan.get("authoring")
        else AUTHOR_SYSTEM
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(instructions, ensure_ascii=False)},
    ]


def controlled_inputs(plan: dict, canonical: str) -> dict:
    """Add reversible surface errors while protecting quotes and known literals.

    These variants cover formatting and light cleanup. They do not simulate the full range of
    natural ASR errors. Source documents remain the unit of independent data, not variants.
    """
    protected = r'"[^"\n]*"|“[^”]*”|`[^`]*`|' + "|".join(
        re.escape(x) for x in sorted(plan["anchors"], key=len, reverse=True)
    )
    chunks, start = [], 0
    for match in re.finditer(protected, canonical):
        chunks.extend([(False, canonical[start : match.start()]), (True, match.group())])
        start = match.end()
    chunks.append((False, canonical[start:]))

    def alter(transform):
        return "".join(text if keep else transform(text) for keep, text in chunks)

    light = alter(lambda text: text.replace(" ", "  ", 1))
    mode = plan["mode"]
    if mode == "/transcribe":
        noise = "Um, " + canonical
        challenge = "Uh, um, " + canonical.replace("\n\n", "\n\nEr, ", 2)
    elif mode == "/slack":
        noise = alter(lambda text: re.sub(r"\band\b", "&", text, count=4))
        challenge = alter(lambda text: re.sub(r"(?<=[a-z])\.(?=\s|$)", "", text, count=4))
    elif mode == "/prompt":
        noise = alter(lambda text: re.sub(r"\n+", " ", text))
        challenge = alter(lambda text: re.sub(r"\n+", "  ", text))
    else:
        noise = alter(lambda text: re.sub(r"\bis\b", "are", text, count=3))
        challenge = alter(lambda text: re.sub(r"\bthe\s+", "", text, count=3))
    if noise in {canonical, light}:
        raise ValueError("mode-noise transformation made no useful change")
    if challenge in {canonical, light, noise}:
        raise ValueError("challenge transformation made no useful change")
    return {"canonical": canonical, "light": light, "mode_noise": noise, "challenge": challenge}


def parse_response(plan: dict, text: str) -> list[Example]:
    payload = controlled_inputs(plan, text.strip()) if plan.get("authoring") else json.loads(text)
    return parse_family(plan, payload)


def parse_family(plan: dict, payload: dict) -> list[Example]:
    fields = {"canonical", "light", "mode_noise", "challenge"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("source fields must be canonical, light, mode_noise, challenge")
    if any(not isinstance(x, str) or not x.strip() for x in payload.values()):
        raise ValueError("source fields must be nonempty strings")
    if len(set(payload.values())) != 4:
        raise ValueError("duplicate input variants")
    if any(anchor not in text for text in payload.values() for anchor in plan["anchors"]):
        raise ValueError("missing protected literal")
    examples = []
    for variant in ["identity", "light", "mode_noise", "challenge"]:
        original = payload["canonical"] if variant == "identity" else payload[variant]
        case = Case(
            id=f"{plan['id']}-{variant.replace('_', '-')}",
            input=original,
            output=payload["canonical"],
            mode=plan["mode"],
            checks=(
                "Preserve every fact, quote, quantity, condition, and the complete final section. "
                + MODES[plan["mode"]]
            ),
            rules=Rules(literals=tuple(plan["anchors"]), identity=variant == "identity"),
            context=plan["context"],
            source_kind="synthetic_luna",
            language=plan["language"],
        )
        examples.append(
            Example(case, plan["split"], plan["source_group"], plan["length"], variant, plan["id"])
        )
    return examples


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


class LeakageIndex:
    """Block exact/embedded copies and close paraphrases with mostly unchanged wording.

    This detects text overlap, not every possible semantic relationship. Topic groups are also
    assigned together before generation. Eight-word shingles avoid matching generic short phrases.
    """

    def __init__(self, texts):
        self.texts = list({normalized(text) for text in texts if normalized(text)})
        self.shingles = [self._shingles(text) for text in self.texts]

    @staticmethod
    def _shingles(text):
        words = text.split()
        return {tuple(words[i : i + 8]) for i in range(len(words) - 7)}

    def matches(self, text: str) -> bool:
        candidate = normalized(text)
        shingles = self._shingles(candidate)
        for blocked, other in zip(self.texts, self.shingles, strict=True):
            if candidate == blocked:
                return True
            if min(len(candidate), len(blocked)) >= 60 and (
                candidate in blocked or blocked in candidate
            ):
                return True
            if min(len(shingles), len(other)) >= 5 and (
                len(shingles & other) / min(len(shingles), len(other)) >= 0.60
            ):
                return True
            if 40 <= min(len(candidate), len(blocked)) and max(len(candidate), len(blocked)) < 600:
                matcher = SequenceMatcher(None, candidate, blocked, autojunk=False)
                # Both quick ratios are upper bounds, so they cannot discard a real match.
                if (
                    matcher.real_quick_ratio() >= 0.92
                    and matcher.quick_ratio() >= 0.92
                    and matcher.ratio() >= 0.92
                ):
                    return True
        return False


def validation_index() -> LeakageIndex:
    root = PROJECT / "data/rewriter"
    frozen = json.loads((root / "validation-freeze.json").read_text())
    for name, expected in frozen["files"].items():
        if hashlib.sha256((root / frozen["bundle"] / name).read_bytes()).hexdigest() != expected:
            raise ValueError("accepted validation identity changed")
    bundle = load_bundle()
    texts = [text for case in bundle.cases for text in (case.input, case.output)]
    texts.extend(fixture["output"] for fixture in bundle.fixtures)
    personal = root / "personal/calibration-10/cases.jsonl"
    # Read only to exclude local holdouts. Never return or publish their contents or identities.
    if personal.exists():
        for line in personal.read_text().splitlines():
            row = json.loads(line)
            texts.extend([row["input"], row["output"]])
    return LeakageIndex(texts)


def validate_splits(examples: list[Example]) -> None:
    groups, ids, inputs = {}, set(), set()
    for item in examples:
        if item.split not in {"train", "validation"}:
            raise ValueError("unknown training split")
        previous = groups.setdefault(item.source_group, item.split)
        if previous != item.split:
            raise ValueError("source group crosses splits")
        if item.case.id in ids or (item.case.mode, item.case.input) in inputs:
            raise ValueError("duplicate training identity or input")
        ids.add(item.case.id)
        inputs.add((item.case.mode, item.case.input))
    train = LeakageIndex(
        [
            text
            for item in examples
            if item.split == "train"
            for text in (item.case.input, item.case.output)
        ]
    )
    for item in examples:
        if item.split == "validation" and (
            train.matches(item.case.input) or train.matches(item.case.output)
        ):
            raise ValueError("text overlap between training and internal validation")


def token_audit(item: Example, tokenizer, max_seq_length=4096, max_new_tokens=2048) -> dict:
    case = item.case
    messages = [
        {"role": "system", "content": COMMON_SYSTEM + MODES[case.mode]},
        {"role": "user", "content": case.user_message()},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": case.output}], tokenize=True, return_dict=False
    )
    raw = len(tokenizer.encode(case.input, add_special_tokens=False))
    output = len(tokenizer.encode(case.output, add_special_tokens=False))
    if item.length == "short" and raw > 250:
        raise ValueError("short input exceeds 250 tokens")
    if item.length == "medium" and not 150 <= raw <= 900:
        raise ValueError("medium input outside 150 to 900 tokens")
    if item.length == "long" and not 1000 <= raw <= 2000:
        raise ValueError(f"long input outside 1000 to 2000 tokens: {raw}")
    if len(full) > max_seq_length or len(prompt) + max_new_tokens > max_seq_length:
        raise ValueError("context overflow; never truncate a training pair")
    if full[: len(prompt)] != prompt or len(full) <= len(prompt) + 1:
        raise ValueError("assistant span is empty or chat prefix differs")
    assistant = full[len(prompt) :]
    if tokenizer.eos_token_id not in assistant:
        raise ValueError("assistant EOS is missing")
    end = len(prompt) + assistant.index(tokenizer.eos_token_id) + 1
    if end - len(prompt) > max_new_tokens:
        raise ValueError("assistant exceeds the generation limit")
    if end < len(full) and tokenizer.decode(full[end:]).strip():
        raise ValueError("unexpected content after assistant EOS")
    labels = [-100] * len(prompt) + full[len(prompt) : end] + [-100] * (len(full) - end)
    return {
        "input_tokens": raw,
        "output_tokens": output,
        "prompt_tokens": len(prompt),
        "chat_tokens": len(full),
        "assistant_tokens": end - len(prompt),
        "input_ids": full,
        "labels": labels,
    }


def accepted_examples(examples: list[Example], reviews: dict) -> list[Example]:
    accepted = []
    for item in examples:
        review = reviews.get(item.case.id)
        if not review:
            continue
        if review["case_sha256"] != digest(asdict(item.case)):
            raise ValueError("review identity differs from the training pair")
        judgment = review.get("judgment")
        if judgment is None:
            continue
        validate_judgment(item.case.id, judgment)
        if judgment["verdict"] == "pass" and not deterministic_checks(item.case, item.case.output):
            accepted.append(item)
    return accepted


def export_training(examples, reviews, blocked, tokenizer, destination: Path) -> dict:
    """Validate everything before creating the release. Only accepted pairs are exported."""
    if destination.exists():
        raise FileExistsError(destination)
    accepted = accepted_examples(examples, reviews)
    if not accepted:
        raise ValueError("no accepted training pairs")
    validate_splits(accepted)
    tokens = {}
    for item in accepted:
        if blocked.matches(item.case.input) or blocked.matches(item.case.output):
            raise ValueError(f"validation leakage: {item.case.id}")
        tokens[item.case.id] = token_audit(item, tokenizer)
    destination.mkdir(parents=True)
    (destination / "sft").mkdir()
    support = []
    for split in ["train", "validation"]:
        items = [item for item in accepted if item.split == split]
        rows, guides = [], []
        for item in items:
            case = item.case
            rows.append(
                {name: getattr(case, name) for name in ["input", "output", "mode", "checks"]}
            )
            guides.append(
                {
                    "system": COMMON_SYSTEM + MODES[case.mode],
                    "user": case.user_message(),
                    "assistant": case.output,
                }
            )
            counts = {k: v for k, v in tokens[case.id].items() if k not in {"labels", "input_ids"}}
            support.append(
                {
                    "id": case.id,
                    "split": split,
                    "source_id": item.source_id,
                    "source_group": item.source_group,
                    "length": item.length,
                    "variant": item.variant,
                    "language": case.language,
                    "context": case.context,
                    "rules": asdict(case.rules),
                    "case_sha256": digest(asdict(case)),
                    "row_sha256": digest(rows[-1]),
                    "review_status": "machine_pass",
                    "human_reviewed": False,
                    "tokens": counts,
                    "judgment": reviews[case.id]["judgment"],
                }
            )
        for name, values in [(f"{split}.jsonl", rows), (f"sft/{split}.jsonl", guides)]:
            (destination / name).write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values)
            )
    write_json(destination / "supporting.json", support)
    report = {
        "status": "machine_reviewed_pilot",
        "parsed_pairs": len(examples),
        "accepted_pairs": len(accepted),
        "splits": dict(Counter(x.split for x in accepted)),
        "by_mode": {
            split: dict(Counter(x.case.mode for x in accepted if x.split == split))
            for split in ["train", "validation"]
        },
        "by_length": {
            split: dict(Counter(x.length for x in accepted if x.split == split))
            for split in ["train", "validation"]
        },
        "by_variant": dict(Counter(x.variant for x in accepted)),
        "languages": dict(Counter(x.case.language for x in accepted)),
        "source_documents": len({x.source_id for x in accepted}),
        "source_groups": len({x.source_group for x in accepted}),
        "max_chat_tokens": max(x["chat_tokens"] for x in tokens.values()),
        "max_input_tokens": max(x["input_tokens"] for x in tokens.values()),
        "assistant_tokens": sum(x["assistant_tokens"] for x in tokens.values()),
        "tokenizer": {"model": TOKENIZER, "revision": REVISION},
        "validation_overlap_detected": False,
        "split_overlap_detected": False,
        "loss": "assistant response and EOS only; system/user labels are -100",
        "human_reviewed": False,
        "judge_human_calibrated": False,
        "locked_test": False,
        "personal_data_included": False,
    }
    write_json(destination / "quality.json", report)
    hashes = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    write_json(destination / "manifest.json", {"role": "training_pilot", "files": hashes})
    return report


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION)


def _unexpected_call():
    raise RuntimeError("An offline audit must never issue a provider call")


def audit_run(root: Path) -> dict:
    """Rebuild acceptance from exact saved calls, then verify the exported training text."""
    spec = json.loads((root / "run.json").read_text())
    if (
        spec["validation_freeze_sha256"]
        != hashlib.sha256(
            (PROJECT / "data/rewriter/validation-freeze.json").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("training run used a different validation freeze")
    blocked, tokenizer = validation_index(), _tokenizer()
    examples, reviews = [], {}
    response_ids, costs, gateway = set(), 0.0, 0.0
    unknown_costs = 0
    for stage in ["sources", "judges"]:
        for path in sorted((root / stage).glob("*.json")):
            record = json.loads(path.read_text())
            if record["request_sha256"] != digest(record["request"]):
                raise ValueError("saved request hash changed")
            if record["status"] != "completed":
                continue
            result = record["result"]
            if record["result_sha256"] != digest(result):
                raise ValueError("saved result hash changed")
            identity = result["provider_response_id"]
            if identity in response_ids:
                raise ValueError("duplicate provider response identity")
            response_ids.add(identity)
            usage = result.get("usage") or {}
            gateway += usage.get("cost") or 0
            upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost")
            if upstream is None:
                unknown_costs += 1
            else:
                costs += upstream
    for plan in spec["plans"]:
        source_path = root / "sources" / f"{plan['id']}.json"
        if not source_path.exists():
            raise ValueError("scheduled source call is missing")
        request = {"config": spec["author"], "messages": author_messages(plan)}
        source = saved_call(source_path, request, _unexpected_call)
        if source["status"] != "completed" or source["result"].get("finish_reason") != "stop":
            continue
        try:
            family = parse_response(plan, source["result"]["text"])
        except (ValueError, TypeError):
            continue
        examples.extend(family)
        for item in family:
            try:
                token_audit(item, tokenizer)
            except ValueError:
                # A reused historical judge pass cannot override a stricter length gate.
                continue
            if blocked.matches(item.case.input) or blocked.matches(item.case.output):
                raise ValueError("validation leakage")
            path = root / "judges" / f"{item.case.id}.json"
            if not path.exists():
                continue
            request = {"config": spec["judge"], "packet": judge_packet(item.case, item.case.output)}
            record = saved_call(path, request, _unexpected_call)
            result = record.get("result", {})
            judgment = result.get("judgment") if result.get("finish_reason") == "stop" else None
            if judgment and json.loads(result["text"]) != judgment:
                raise ValueError("judgment differs from the provider text")
            reviews[item.case.id] = {"case_sha256": digest(asdict(item.case)), "judgment": judgment}
    accepted = accepted_examples(examples, reviews)
    validate_splits(accepted)
    release = root / "release"
    manifest = json.loads((release / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if hashlib.sha256((release / name).read_bytes()).hexdigest() != sha:
            raise ValueError("exported training file changed")
    for split in ["train", "validation"]:
        items = [x for x in accepted if x.split == split]
        rows = [json.loads(line) for line in (release / f"{split}.jsonl").read_text().splitlines()]
        guides = [
            json.loads(line) for line in (release / f"sft/{split}.jsonl").read_text().splitlines()
        ]
        expected_rows = [
            {key: getattr(x.case, key) for key in ["input", "output", "mode", "checks"]}
            for x in items
        ]
        if rows != expected_rows or len(guides) != len(items):
            raise ValueError("export differs from accepted source/judge calls")
        for item, guide in zip(items, guides, strict=True):
            expected = {
                "system": COMMON_SYSTEM + MODES[item.case.mode],
                "user": item.case.user_message(),
                "assistant": item.case.output,
            }
            if guide != expected:
                raise ValueError("supervised text differs from the evaluated prompt contract")
            if blocked.matches(item.case.input) or blocked.matches(item.case.output):
                raise ValueError("validation leakage")
            token_audit(item, tokenizer)
    report = {
        "status": "verified",
        "accepted_pairs": len(accepted),
        "scheduled_pairs": len(spec["plans"]) * 4,
        "splits": dict(Counter(x.split for x in accepted)),
        "verified_completed_provider_calls": len(response_ids),
        "gateway_reported_cost_usd": round(gateway, 8),
        "reported_byok_upstream_cost_usd": round(costs, 8),
        "completed_calls_without_upstream_cost": unknown_costs,
        "human_reviewed": False,
        "judge_human_calibrated": False,
        "validation_bytes_unchanged": True,
        "validation_and_personal_overlap_detected": False,
        "assistant_only_loss_checked": True,
        "prompt_and_judge_bindings_checked": True,
    }
    write_json(root / "audit.json", report)
    return report


def run_pipeline(
    root: Path,
    settings: Path,
    *,
    pilot=False,
    workers=4,
    canonical_long=False,
    extended_long=False,
    reuse_from: Path | None = None,
) -> dict:
    """Calls are durable and never silently retried. Rejected sources remain in the ledger."""
    config = tomllib.loads(settings.read_text())
    plans = planned_sources()
    if canonical_long or extended_long:
        plans = [
            (
                {
                    **plan,
                    "id": plan["id"] + ("-long-v3" if extended_long else "-long-v2"),
                    "authoring": "canonical_extended" if extended_long else "canonical_controlled",
                }
                if plan["length"] == "long"
                else plan
            )
            for plan in plans
        ]
    if pilot:
        plans = plans[:16]
    rubric = (PROJECT / "data/rewriter/benchmark-draft/supporting/judge-system.txt").read_text()
    judge_config = {**config["judge"], "rubric": rubric}
    author_config = config["author"]
    spec = {
        "plans": plans,
        "author": author_config,
        "judge": judge_config,
        "author_system": AUTHOR_SYSTEM,
        "tokenizer": {"model": TOKENIZER, "revision": REVISION},
        "validation_freeze_sha256": hashlib.sha256(
            (PROJECT / "data/rewriter/validation-freeze.json").read_bytes()
        ).hexdigest(),
    }
    if canonical_long or extended_long:
        spec["long_author_system"] = LONG_EXTENDED_SYSTEM if extended_long else LONG_SYSTEM
    manifest = root / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("training run identity changed; select a new directory")
    if not manifest.exists():
        write_json(manifest, spec)
    if reuse_from:
        previous = json.loads((reuse_from / "run.json").read_text())
        if previous["author"] != author_config or previous["judge"] != judge_config:
            raise ValueError("cannot reuse calls with different model settings")
        for plan in plans:
            if plan.get("authoring"):
                continue
            if plan not in previous["plans"]:
                raise ValueError("cannot reuse a different source plan")
            names = [f"sources/{plan['id']}.json"] + [
                f"judges/{plan['id']}-{variant}.json"
                for variant in ["identity", "light", "mode-noise", "challenge"]
            ]
            for name in names:
                source, target = reuse_from / name, root / name
                if not source.exists():
                    continue
                if target.exists():
                    if source.read_bytes() != target.read_bytes():
                        raise ValueError("reused call differs from the existing record")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    author, judge = hosted_backend(author_config), hosted_backend(judge_config, judge=True)
    tokenizer, blocked = _tokenizer(), validation_index()

    def process(plan):
        messages = author_messages(plan)
        request = {"config": author_config, "messages": messages}
        source = saved_call(
            root / "sources" / f"{plan['id']}.json", request, lambda: author(messages)
        )
        ledger, examples, reviews = [], [], {}
        if source["status"] != "completed" or source["result"].get("finish_reason") != "stop":
            return (
                [],
                {},
                [{"source_id": plan["id"], "stage": "source", "status": "generation_error"}],
            )
        try:
            examples = parse_response(plan, source["result"]["text"])
        except (ValueError, TypeError) as error:
            return (
                [],
                {},
                [
                    {
                        "source_id": plan["id"],
                        "stage": "source",
                        "status": "rejected",
                        "reason": str(error),
                    }
                ],
            )
        for item in examples:
            case = item.case
            try:
                token_audit(item, tokenizer)
                if blocked.matches(case.input) or blocked.matches(case.output):
                    raise ValueError("validation leakage")
                failures = deterministic_checks(case, case.output)
                if failures:
                    raise ValueError(str(failures))
            except ValueError as error:
                ledger.append(
                    {
                        "id": case.id,
                        "stage": "mechanical",
                        "status": "rejected",
                        "reason": str(error),
                    }
                )
                continue
            packet = judge_packet(case, case.output)
            request = {"config": judge_config, "packet": packet}
            record = saved_call(
                root / "judges" / f"{case.id}.json", request, lambda packet=packet: judge(packet)
            )
            result = record.get("result", {})
            judgment = result.get("judgment") if result.get("finish_reason") == "stop" else None
            reviews[case.id] = {"case_sha256": digest(asdict(case)), "judgment": judgment}
            status = judgment["verdict"] if judgment else "judge_error"
            ledger.append(
                {
                    "id": case.id,
                    "source_id": item.source_id,
                    "stage": "semantic",
                    "status": status,
                    "critique": judgment["critique"]
                    if judgment
                    else "No valid completed judgment.",
                }
            )
        print(f"{plan['id']}: {dict(Counter(x['status'] for x in ledger))}", flush=True)
        return examples, reviews, ledger

    examples, reviews, ledger = [], {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for items, checked, events in pool.map(process, plans):
            examples.extend(items)
            reviews.update(checked)
            ledger.extend(events)
    write_json(root / "ledger.json", ledger)
    accepted = accepted_examples(examples, reviews)
    summary = {
        "planned_documents": len(plans),
        "planned_pairs": len(plans) * 4,
        "parsed_pairs": len(examples),
        "accepted_pairs": len(accepted),
        "excluded_planned_pairs": len(plans) * 4 - len(accepted),
        "events": dict(Counter(x["status"] for x in ledger)),
    }
    write_json(root / "summary.json", summary)
    destination = root / "release"
    if not destination.exists():
        export_training(examples, reviews, blocked, tokenizer, destination)
    else:
        old = json.loads((destination / "manifest.json").read_text())
        for name, sha in old["files"].items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != sha:
                raise ValueError("existing training release changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/training-data.toml")
    parser.add_argument("--pilot", action="store_true", help="First 16 source documents, 64 pairs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit", action="store_true", help="Offline saved-call and export audit")
    parser.add_argument("--canonical-long", action="store_true")
    parser.add_argument("--extended-long", action="store_true")
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    result = (
        audit_run(args.run)
        if args.audit
        else run_pipeline(
            args.run,
            args.config,
            pilot=args.pilot,
            workers=args.workers,
            canonical_long=args.canonical_long,
            extended_long=args.extended_long,
            reuse_from=args.reuse_from,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
