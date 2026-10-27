"""Training-data failure cases, specified before implementing the pipeline.

Failures: validation leakage with a prefix or small edits; related sources across splits;
missing literals; wrong mode/variant; context overflow; empty assistant loss; absent, uncertain,
or mismatched reviews; metadata in supervised text; silent truncation or overwrite.
"""

from dataclasses import replace

import pytest

from finetuning_labs.rewriter.benchmark import Case, Rules
from finetuning_labs.rewriter.evaluation import digest
from finetuning_labs.rewriter.training import (
    Example,
    LeakageIndex,
    accepted_examples,
    export_training,
    parse_family,
    planned_sources,
    token_audit,
    validate_splits,
)


def example(**changes):
    case = Case(
        id="training-one",
        input="ask Elena for 17 blue cards",
        output="Ask Elena for 17 blue cards.",
        mode="/slack",
        checks="Preserve the recipient and quantity.",
        rules=Rules(literals=("Elena", "17")),
    )
    return Example(case, "train", "bicycle", "short", "light", "source-one", **changes)


def passing_review(item):
    return {
        "case_sha256": digest(item.case.__dict__ | {"rules": item.case.rules.__dict__}),
        "judgment": {
            "case_id": item.case.id,
            "verdict": "pass",
            "critique": "Preserves the requested action and all facts.",
            "critical_errors": [],
            "criteria": dict.fromkeys(
                ["meaning", "terms_and_literals", "mode", "readability", "output_discipline"],
                "pass",
            ),
        },
    }


def test_plan_keeps_domains_together_and_covers_modes_lengths_languages():
    plans = planned_sources()
    assert len(plans) == 80
    assert {x["mode"] for x in plans} == {"/slack", "/transcribe", "/prompt", "/polish"}
    assert {x["length"] for x in plans} == {"short", "medium", "long"}
    assert {x["language"] for x in plans} >= {"en", "de", "pt"}
    by_group = {}
    for plan in plans:
        by_group.setdefault(plan["source_group"], set()).add(plan["split"])
    assert all(len(splits) == 1 for splits in by_group.values())


def test_validation_copy_prefix_and_near_copy_are_blocked():
    text = (
        "Ask Elena to inspect the green bicycle before Tuesday. Keep the rear wheel dry and "
        "record the pressure on the blue card. Do not replace the chain without permission."
    )
    blocked = LeakageIndex([text])
    assert blocked.matches(text)
    assert blocked.matches("Please format this:\n" + text)
    assert blocked.matches(text.replace("Tuesday", "Wednesday"))
    assert not blocked.matches("The choir will meet on Saturday at the west entrance.")


def test_related_sources_must_not_cross_splits():
    one = example()
    two = replace(one, case=replace(one.case, id="two"), split="validation")
    with pytest.raises(ValueError, match="source group"):
        validate_splits([one, two])


def test_family_rejects_missing_anchor_and_duplicate_variants():
    plan = planned_sources()[0]
    canonical = " ".join(plan["anchors"]) + " Please check the cards."
    payload = {
        "canonical": canonical,
        "light": "hey " + canonical,
        "mode_noise": "um " + canonical,
        "challenge": canonical + " uh",
    }
    parsed = parse_family(plan, payload)
    assert len(parsed) == 4
    assert parsed[0].variant == "identity" and parsed[0].case.input == parsed[0].case.output
    with pytest.raises(ValueError, match="literal"):
        parse_family(plan, payload | {"canonical": "No original facts remain."})
    with pytest.raises(ValueError, match="duplicate"):
        parse_family(plan, payload | {"light": canonical})
    with pytest.raises(ValueError, match="fields"):
        parse_family(plan, payload | {"metadata": "must not enter the text"})


def test_missing_uncertain_or_unbound_review_cannot_enter_training():
    item = example()
    assert accepted_examples([item], {}) == []
    review = passing_review(item)
    assert accepted_examples([item], {item.case.id: review}) == [item]
    uncertain = review | {
        "judgment": review["judgment"]
        | {
            "verdict": "uncertain",
            "criteria": review["judgment"]["criteria"] | {"meaning": "uncertain"},
        }
    }
    assert accepted_examples([item], {item.case.id: uncertain}) == []
    with pytest.raises(ValueError, match="identity"):
        accepted_examples(
            [replace(item, case=replace(item.case, output="Changed answer."))],
            {item.case.id: review},
        )


class Tokenizer:
    """A chat prefix and assistant span, with explicit EOS, independent of dataset code."""

    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
        prefix = [1, 2, 3]
        if add_generation_prompt:
            return prefix
        return prefix + [4] * len(messages[-1]["content"].split()) + [self.eos_token_id]


def test_full_chat_limits_and_assistant_only_labels():
    item = example()
    report = token_audit(item, Tokenizer())
    assert report["labels"][:3] == [-100, -100, -100]
    assert report["labels"][3:] == report["input_ids"][3:]
    assert report["labels"][-1] == Tokenizer.eos_token_id
    with pytest.raises(ValueError, match="context"):
        token_audit(item, Tokenizer(), max_seq_length=5)
    with pytest.raises(ValueError, match="long"):
        token_audit(replace(item, length="long"), Tokenizer())
    broken = Tokenizer()
    broken.apply_chat_template = lambda *args, **kwargs: [1, 2, 3]
    with pytest.raises(ValueError, match="assistant"):
        token_audit(item, broken)


def test_export_is_simple_checked_and_refuses_overwrite(tmp_path):
    item = example()
    reviews = {item.case.id: passing_review(item)}
    out = tmp_path / "release"
    export_training([item], reviews, LeakageIndex([]), Tokenizer(), out)
    import json

    row = json.loads((out / "train.jsonl").read_text())
    assert set(row) == {"input", "output", "mode", "checks"}
    guide = json.loads((out / "sft/train.jsonl").read_text())
    assert set(guide) == {"system", "user", "assistant"}
    assert guide["assistant"] == item.case.output
    assert "source-one" not in str(guide) and "Preserve the recipient" not in str(guide)
    with pytest.raises(FileExistsError):
        export_training([item], reviews, LeakageIndex([]), Tokenizer(), out)
    with pytest.raises(ValueError, match="leakage"):
        export_training(
            [item], reviews, LeakageIndex([item.case.output]), Tokenizer(), tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists()
