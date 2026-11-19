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


def test_overlap_screen_matches_original_rules_for_close_and_unrelated_text():
    from difflib import SequenceMatcher

    from finetuning_labs.rewriter.training import LeakageIndex, normalized

    sources = [
        "Please return the sealed parcel to Amaya before the workshop closes on Thursday.",
        "Keep the original documents in the east drawer and send the scanned copies to Milo.",
        "Do not rename the archive files until the review is complete.",
        "The invoice is 63.40, not 36.40, and the recipient is Tess.",
    ]
    index = LeakageIndex(sources)
    candidates = (
        sources
        + [s.replace("Thursday", "Friday").replace("Milo", "Mila") for s in sources]
        + ["A different short message.", "A substantially longer unrelated report. " * 20]
    )
    for raw in candidates:
        text = normalized(raw)
        shingles = index._shingles(text)
        expected = any(
            text == other
            or min(len(text), len(other)) >= 60
            and (text in other or other in text)
            or min(len(shingles), len(grams)) >= 5
            and len(shingles & grams) / min(len(shingles), len(grams)) >= 0.60
            or 40 <= min(len(text), len(other))
            and max(len(text), len(other)) < 600
            and SequenceMatcher(None, text, other, autojunk=False).ratio() >= 0.92
            for other, grams in zip(index.texts, index.shingles, strict=True)
        )
        assert index.matches(raw) == expected


def test_training_pipeline_rejects_nonpositive_workers_and_cli_handles_errors(
    monkeypatch, tmp_path
):
    from finetuning_labs.rewriter import training
    from finetuning_labs.rewriter.training import main, run_pipeline

    with pytest.raises(ValueError, match="workers"):
        run_pipeline(tmp_path / "run", tmp_path / "missing.toml", workers=0)

    bad_cfg = tmp_path / "bad.toml"
    bad_cfg.write_text("[author]\nmodel = 'test'\n")
    with pytest.raises(ValueError, match=r"Missing \[judge\] section"):
        run_pipeline(tmp_path / "run", bad_cfg, workers=1)

    monkeypatch.setattr(
        "sys.argv",
        ["training", "--run", str(tmp_path / "missing"), "--audit"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    monkeypatch.setattr(
        training, "audit_run", lambda *_, **__: (_ for _ in ()).throw(TypeError("bug"))
    )
    with pytest.raises(TypeError, match="bug"):
        main()


def test_audit_run_requires_explicit_opt_in_for_historical_v1_validation_freeze(
    tmp_path, monkeypatch
):
    import json

    from finetuning_labs.rewriter import training
    from finetuning_labs.rewriter.training import HISTORICAL_V1_FREEZE_SHA256, audit_run

    bad_run = tmp_path / "bad_run"
    bad_run.mkdir()
    (bad_run / "run.json").write_text(json.dumps({"validation_freeze_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="different validation freeze"):
        audit_run(bad_run)
    with pytest.raises(ValueError, match="different validation freeze"):
        audit_run(bad_run, allow_historical_freeze=True)

    hist_run = tmp_path / "hist_run"
    hist_run.mkdir()
    (hist_run / "run.json").write_text(
        json.dumps(
            {
                "validation_freeze_sha256": HISTORICAL_V1_FREEZE_SHA256,
                "plans": [{"id": "missing-plan"}],
            }
        )
    )
    with pytest.raises(ValueError, match="different validation freeze"):
        audit_run(hist_run)

    monkeypatch.setattr(training, "validation_index", lambda: LeakageIndex([]))
    monkeypatch.setattr(training, "_tokenizer", lambda: Tokenizer())
    with pytest.raises(ValueError, match="scheduled source call is missing"):
        audit_run(hist_run, allow_historical_freeze=True)

    # Full minimal historical audit verifies validation_freeze_version and does not write audit.json
    plan = planned_sources()[0]
    canonical = " ".join(plan["anchors"]) + " Please check the cards."
    payload = {
        "canonical": canonical,
        "light": "hey " + canonical,
        "mode_noise": "um " + canonical,
        "challenge": canonical + " uh",
    }
    parsed = parse_family(plan, payload)
    reviews = {item.case.id: passing_review(item) for item in parsed}
    export_training(parsed, reviews, LeakageIndex([]), Tokenizer(), hist_run / "release")
    author_cfg = {"model": "author"}
    judge_cfg = {"model": "judge"}
    (hist_run / "run.json").write_text(
        json.dumps(
            {
                "validation_freeze_sha256": HISTORICAL_V1_FREEZE_SHA256,
                "plans": [plan],
                "author": author_cfg,
                "judge": judge_cfg,
            }
        )
    )
    from finetuning_labs.rewriter.benchmark import judge_packet
    from finetuning_labs.rewriter.evaluation import saved_call
    from finetuning_labs.rewriter.training import author_messages

    saved_call(
        hist_run / "sources" / f"{plan['id']}.json",
        {"config": author_cfg, "messages": author_messages(plan)},
        lambda: {
            "text": json.dumps(payload),
            "finish_reason": "stop",
            "provider_response_id": "src-1",
            "usage": {"cost": 0.0, "cost_details": {"upstream_inference_cost": 0.01}},
        },
    )
    for idx, item in enumerate(parsed):
        j = reviews[item.case.id]["judgment"]
        saved_call(
            hist_run / "judges" / f"{item.case.id}.json",
            {"config": judge_cfg, "packet": judge_packet(item.case, item.case.output)},
            lambda j=j, idx=idx: {
                "text": json.dumps(j),
                "judgment": j,
                "finish_reason": "stop",
                "provider_response_id": f"jdg-{idx}",
                "usage": {"cost": 0.0, "cost_details": {"upstream_inference_cost": 0.01}},
            },
        )
    report = audit_run(hist_run, allow_historical_freeze=True)
    assert report["validation_freeze_version"] == "historical_v1"
    assert not (hist_run / "audit.json").exists()

    (hist_run / "audit.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "validation_freeze_version"})
    )
    before_bytes = (hist_run / "audit.json").read_bytes()
    assert (
        audit_run(hist_run, allow_historical_freeze=True)["validation_freeze_version"]
        == "historical_v1"
    )
    assert (hist_run / "audit.json").read_bytes() == before_bytes

    (hist_run / "audit.json").write_text(json.dumps(report | {"accepted_pairs": 999}))
    with pytest.raises(ValueError, match="saved audit report differs"):
        audit_run(hist_run, allow_historical_freeze=True)
