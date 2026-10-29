"""Reject reversals, invented code and paraphrases before paid pair verification."""

import pytest

from finetuning_labs.rewriter.english_critical import QUOTAS, critical_issues, plans


def test_new_sources_are_english_only_independent_and_cover_critical_failures():
    items = plans(multiplier=1)
    assert len(items) == 500 == sum(sum(c.values()) for c in QUOTAS.values())
    assert {p["language"] for p in items} == {"en"}
    assert {p["split"] for p in items} == {"train"}
    assert len({p["id"] for p in items}) == 500
    assert sum(p["identity"] for p in items) == 125
    assert sum(p["length"] == "long" for p in items) == 100
    assert {p["focus"] for p in items} >= {
        "recipient_exclusion",
        "amount_exclusion",
        "negation",
        "correction_context",
        "code_exactly_once",
        "conditional_fallback",
        "unfinished",
        "long_completeness",
    }


def test_authoring_constraints_do_not_contradict_the_future_explainer_or_unfinished_speech():
    for plan in plans(multiplier=1):
        if plan["identity"]:
            continue
        if plan["focus"] == "code_exactly_once":
            assert "an answer" not in plan["brief"]
        if plan["focus"] == "unfinished":
            assert plan["critical_checks"]["verbatim_words"] is True


@pytest.mark.parametrize(
    "checks,raw,good,bad,issue",
    [
        (
            {"phrases": ["Mara not Quentin"]},
            "send to Mara not Quentin",
            "Send to Mara, not Quentin.",
            "Send to Quentin.",
            "required_relation_changed",
        ),
        (
            {"phrases": ["63.40 not 36.40"]},
            "$63.40 not $36.40",
            "$63.40, not $36.40.",
            "$36.40, not $36.40.",
            "required_relation_changed",
        ),
        (
            {"phrases": ["do not rename files"]},
            "don't rename files",
            "Do not rename files.",
            "Rename files.",
            "required_relation_changed",
        ),
        (
            {"keep": ["B8", "Tess"], "forbid": ["A4"]},
            "put it in A4 sorry B8 and keep Tess",
            "Put it in B8 and keep Tess.",
            "Put it in B8.",
            "required_detail_missing",
        ),
        (
            {"verbatim_words": True, "leading_filler": "um"},
            "um uh-huh that is useful",
            "Uh-huh, that is useful.",
            "Yeah, that is useful.",
            "transcribe_words_changed",
        ),
        (
            {"ending": "we could"},
            "when it arrives we could",
            "When it arrives, we could...",
            "When it arrives, we could proceed.",
            "unfinished_thought_completed",
        ),
    ],
)
def test_critical_rules_keep_the_final_intent(checks, raw, good, bad, issue):
    meta = {"critical_checks": checks}
    assert not critical_issues(meta, {"input": raw, "output": good})
    assert issue in critical_issues(meta, {"input": raw, "output": bad})


def test_code_must_be_verbatim_exactly_once_with_no_added_block():
    snippet = "items = [4, 7, 4]\nunique = list(dict.fromkeys(items))"
    source = "explain this\n```python\n" + snippet + "\n```"
    meta = {"critical_checks": {"code": snippet}}
    assert not critical_issues(meta, {"input": source, "output": source})
    for target in [
        source + "\n```python\npass\n```",
        source + "\n" + source,
        source.replace("items =", "def wrap():\n    items ="),
    ]:
        assert "supplied_code_changed_or_duplicated" in critical_issues(
            meta, {"input": source, "output": target}
        )


@pytest.mark.parametrize("focus", ["amount_exclusion", "correction_context", "recipient_exclusion"])
def test_reverse_sources_swap_roles_without_editing_an_existing_pair(focus):
    import copy

    from finetuning_labs.rewriter.english_critical import reverse_source

    source = next(p for p in plans(multiplier=1) if p["focus"] == focus and not p["identity"])
    before = copy.deepcopy(source)
    reverse = reverse_source(source, 1)
    assert source == before and reverse["id"] != source["id"]
    assert reverse["source_group"] == reverse["id"] and reverse["language"] == "en"
    if focus in {"amount_exclusion", "recipient_exclusion"}:
        left, right = before["critical_checks"]["phrases"][0].split(" not ")
        assert reverse["critical_checks"]["phrases"] == [right + " not " + left]
    else:
        assert reverse["critical_checks"]["keep"][0] == before["critical_checks"]["forbid"][0]
        assert reverse["critical_checks"]["forbid"][0] == before["critical_checks"]["keep"][0]


def test_larger_reverse_backfill_batches_do_not_reuse_cached_source_ids():
    from finetuning_labs.rewriter.english_critical import reverse_candidates, reverse_source

    source = next(
        p
        for p in plans()
        if p["mode"] == "/prompt"
        and p["length"] == "long"
        and p["focus"] == "recipient_exclusion"
        and not p["identity"]
    )
    legacy = {reverse_source(source, n)["id"] for n in range(12)}
    first = {p["id"] for p in reverse_candidates(source, 3)}
    second = {p["id"] for p in reverse_candidates(source, 4)}
    assert first and second and not first & second and not (first | second) & legacy
    fresh = reverse_candidates(source, 5)
    assert len({p["domain"] for p in fresh}) > 1
    assert all(p["critical_checks"] == fresh[0]["critical_checks"] for p in fresh)
    assert all(p["domain"] in p["brief"] for p in fresh)


def test_resume_replays_the_saved_batch_when_quota_counts_change(tmp_path):
    import json

    from finetuning_labs.rewriter.english_critical import saved_batch_plans

    saved = [{"id": "already-seeded-long"}, {"id": "already-seeded-short"}]
    path = tmp_path / "generation/source-plan.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"plans": saved}))
    assert saved_batch_plans(tmp_path, [{"id": "already-seeded-short"}]) == saved


def test_completion_pool_uses_only_untried_open_cells_and_identity_capacity():
    from finetuning_labs.rewriter.english_critical import completion_pool

    source = plans(round_number=1)
    tried = {p["id"] for p in source[:10]}
    counts = {(m, length): n for m, cells in QUOTAS.items() for length, n in cells.items()}
    counts["/prompt", "long"] -= 4
    pool = completion_pool(source, tried, counts, {True: 125, False: 371})
    assert pool
    assert all(p["mode"] == "/prompt" and p["length"] == "long" for p in pool)
    assert not any(p["identity"] or p["id"] in tried for p in pool)


def test_named_completion_pool_cannot_escape_the_run_directory(tmp_path):
    from finetuning_labs.rewriter.english_critical import complete_run

    with pytest.raises(ValueError, match="pool name"):
        complete_run(tmp_path, pool_name="../outside")


def test_fidelity_topup_uses_fresh_sources_with_stronger_word_preservation():
    from finetuning_labs.rewriter.english_critical import fidelity_long_plans

    pool = fidelity_long_plans()
    assert len(pool) == 96
    assert not ({p["id"] for p in pool} & {p["id"] for p in plans()})
    assert all(p["mode"] == "/prompt" and p["length"] == "long" for p in pool)
    assert all(not p["identity"] and p["critical_checks"]["verbatim_words"] for p in pool)


def test_direction_balance_is_bound_to_rows_and_checks_both_directions():
    from finetuning_labs.rewriter.english_critical import verify_direction_balance
    from finetuning_labs.rewriter.evaluation import digest

    entries = [
        {
            "row": {"input": str(i), "output": str(i)},
            "meta": {"identity": False, "focus": focus, "critical_checks": checks},
        }
        for i, (focus, checks) in enumerate(
            [
                ("amount_exclusion", {"phrases": ["63.40 not 105.20"]}),
                ("amount_exclusion", {"phrases": ["135.20 not 53.40"]}),
                ("correction_context", {"keep": ["A120"], "forbid": ["B24"]}),
                ("correction_context", {"keep": ["B18"], "forbid": ["A149"]}),
                ("recipient_exclusion", {"phrases": ["Amaya not Bram"]}),
                ("recipient_exclusion", {"phrases": ["Bram not Amaya"]}),
            ]
        )
    ]
    entries[-1]["meta"]["contrast_direction"] = "reverse"
    report = {"selected_rows_sha256": digest([e["row"] for e in entries])}
    verify_direction_balance(entries, report)
    entries[1]["meta"]["critical_checks"] = {"phrases": ["13.20 not 153.40"]}
    with pytest.raises(ValueError, match="direction"):
        verify_direction_balance(entries, report)
