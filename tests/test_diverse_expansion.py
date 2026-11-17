"""Unit tests for the diverse dataset expansion, schema extensions, and benchmark bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetuning_labs.rewriter.arka_pipeline import candidate_issues, header, parse_candidate
from finetuning_labs.rewriter.benchmark import (
    Case,
    Rules,
    audit,
    load_bundle,
)
from finetuning_labs.rewriter.diverse_expansion import (
    SCENARIO_FAMILIES,
    build_coverage_schedule,
    build_multilevel_contrast_group,
    build_scenario_plan,
    check_chatbot_preamble,
    check_missing_at_tags,
    check_retained_retractions,
    check_unconverted_file_extensions,
    expansion_leakage_index,
    is_candidate_admitted,
    make_generation_config,
    run_expansion_pipeline,
    stage_expansion_dataset,
    validate_diverse_candidate,
)
from finetuning_labs.rewriter.evaluation import digest
from finetuning_labs.rewriter.training import Example, _tokenizer, token_audit

EXPANSION_BUNDLE_DIR = Path(__file__).resolve().parents[1] / "data/rewriter/expansion-benchmark-v1"


class TestSchemaAndPromptExtensions:
    def test_backward_compatibility_user_message(self):
        """When only profile and dictionary are present, Case.user_message matches legacy format."""
        case = Case(
            id="legacy-test",
            input="hello world",
            output="Hello world.",
            mode="/slack",
            checks="Test legacy format",
            context={"profile": "developer", "dictionary": ["FastAPI", "Uvicorn"]},
        )
        msg = case.user_message()
        assert msg == (
            "MODE: /slack\n"
            "LANGUAGE: en\n"
            "PROFILE: developer\n"
            "DICTIONARY: FastAPI, Uvicorn\n"
            "INPUT:\n"
            "hello world"
        )

    def test_user_message_with_level_files_instructions(self):
        """Case.user_message formats LEVEL, FILES, and INSTRUCTIONS in exact required order."""
        case = Case(
            id="expansion-test",
            input="update the modal and run tests",
            output="Update the modal and run tests.",
            mode="/prompt",
            checks="Test new headers",
            context={
                "level": "custom",
                "profile": "frontend",
                "dictionary": ["ModalDialog"],
                "files": ["src/components/Modal.tsx", "src/types.ts"],
                "instructions": "Format as a Jira issue with acceptance criteria",
            },
        )
        msg = case.user_message()
        expected = (
            "MODE: /prompt\n"
            "LANGUAGE: en\n"
            "LEVEL: custom\n"
            "PROFILE: frontend\n"
            "DICTIONARY: ModalDialog\n"
            "FILES: src/components/Modal.tsx, src/types.ts\n"
            "INSTRUCTIONS: Format as a Jira issue with acceptance criteria\n"
            "INPUT:\n"
            "update the modal and run tests"
        )
        assert msg == expected

    def test_level_ordering_immediately_after_language(self):
        """LEVEL appears immediately after LANGUAGE even without other context fields."""
        case = Case(
            id="level-only",
            input="quick check",
            output="Quick check.",
            mode="/transcribe",
            checks="Check level position",
            context={"level": "low"},
        )
        lines = case.user_message().splitlines()
        assert lines[0] == "MODE: /transcribe"
        assert lines[1] == "LANGUAGE: en"
        assert lines[2] == "LEVEL: low"
        assert lines[3] == "INPUT:"
        assert lines[4] == "quick check"

    def test_load_bundle_context_validation(self, tmp_path):
        """load_bundle validates level, files, and instructions, rejecting invalid values."""
        row = {
            "mode": "/slack",
            "input": "test input",
            "output": "Test input.",
            "checks": "Pass checks",
        }
        (tmp_path / "cases.jsonl").write_text(json.dumps(row) + "\n")
        (tmp_path / "supporting").mkdir()
        (tmp_path / "supporting/fixtures.jsonl").write_text("")

        def write_support(context_dict):
            from finetuning_labs.rewriter.benchmark import _row_hash

            meta = [
                {
                    "id": "case-01",
                    "row_sha256": _row_hash(row),
                    "rules": {
                        "required_any": [],
                        "forbidden": [],
                        "literals": [],
                        "identity": False,
                    },
                    "context": context_dict,
                    "source_kind": "test",
                    "language": "en",
                }
            ]
            (tmp_path / "supporting/cases.json").write_text(json.dumps(meta))
            import hashlib

            manifest = {
                "sources": [],
                "files": {
                    "cases.jsonl": hashlib.sha256(
                        (tmp_path / "cases.jsonl").read_bytes()
                    ).hexdigest(),
                    "supporting/cases.json": hashlib.sha256(
                        (tmp_path / "supporting/cases.json").read_bytes()
                    ).hexdigest(),
                    "supporting/fixtures.jsonl": hashlib.sha256(
                        (tmp_path / "supporting/fixtures.jsonl").read_bytes()
                    ).hexdigest(),
                },
            }
            (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        # Valid expansion context
        write_support(
            {
                "level": "medium",
                "files": ["src/main.py"],
                "instructions": "Be concise",
            }
        )
        bundle = load_bundle(tmp_path)
        assert len(bundle.cases) == 1
        assert bundle.cases[0].context["level"] == "medium"

        # Invalid level value raises ValueError
        write_support({"level": "extreme"})
        with pytest.raises(ValueError, match="level must be one of low, medium, high, custom"):
            load_bundle(tmp_path)

        # Invalid files type raises ValueError
        write_support({"files": "not a list"})
        with pytest.raises(ValueError, match="files must contain text"):
            load_bundle(tmp_path)

        # Unsupported context key raises ValueError
        write_support({"unsupported_key": "val"})
        with pytest.raises(ValueError, match="unsupported model context"):
            load_bundle(tmp_path)


class TestExpansionBenchmarkBundle:
    def test_expansion_bundle_exists_and_loads(self):
        assert EXPANSION_BUNDLE_DIR.exists()
        bundle = load_bundle(EXPANSION_BUNDLE_DIR)
        assert len(bundle.cases) == 60
        assert len(bundle.fixtures) >= 80

    def test_expansion_bundle_passes_audit(self):
        bundle = load_bundle(EXPANSION_BUNDLE_DIR)
        report = audit(bundle)
        assert report["case_count"] == 60
        assert report["mechanical_failures"] == []
        assert report["modes"] == {
            "/slack": 15,
            "/transcribe": 15,
            "/prompt": 15,
            "/polish": 15,
        }
        assert report["semantic_only_attack_count"] >= 4

    def test_category_distribution_and_coverage(self):
        bundle = load_bundle(EXPANSION_BUNDLE_DIR)

        # Area 1: Coding Filenames & Symbols (8 cases)
        code_cases = [c for c in bundle.cases if c.id.startswith("exp-code-")]
        assert len(code_cases) == 8

        # Area 2: Workspace Context & @file Tagging (8 cases)
        file_cases = [c for c in bundle.cases if c.id.startswith("exp-file-")]
        assert len(file_cases) == 8
        tagged_cases = [c for c in file_cases if "@" in c.output]
        assert len(tagged_cases) >= 5

        # Area 3: Spoken List Formatting & Structure (8 cases)
        list_cases = [c for c in bundle.cases if c.id.startswith("exp-list-")]
        assert len(list_cases) == 8

        # Area 4: Same-Text Multi-Level Contrasts (12 cases: 3 transcripts x 4 modes)
        contrast_cases = [c for c in bundle.cases if c.id.startswith("exp-contrast-")]
        assert len(contrast_cases) == 12

        # Verify the 3 transcript groups have identical raw inputs
        t1_cases = [c for c in contrast_cases if "-t1-" in c.id]
        t2_cases = [c for c in contrast_cases if "-t2-" in c.id]
        t3_cases = [c for c in contrast_cases if "-t3-" in c.id]
        assert len(t1_cases) == 4
        assert len(t2_cases) == 4
        assert len(t3_cases) == 4
        assert len({c.input for c in t1_cases}) == 1
        assert len({c.input for c in t2_cases}) == 1
        assert len({c.input for c in t3_cases}) == 1
        assert {c.mode for c in t1_cases} == {"/transcribe", "/slack", "/polish", "/prompt"}

        # Area 5: Speech Repairs & Isolation (4 cases)
        repair_cases = [
            c
            for c in bundle.cases
            if c.id.startswith("exp-repair-") or c.id.startswith("exp-isolation-")
        ]
        assert len(repair_cases) == 4

        # Scaled Cases (20 cases: 8 large medium + 12 long)
        scaled_cases = [c for c in bundle.cases if c.id.startswith("exp-bench-")]
        assert len(scaled_cases) == 20

    def test_all_reference_cases_pass_mechanical_validators(self):
        """All 60 reference cases in expansion-benchmark-v1 must pass validate_diverse_candidate with 0 false positives."""
        bundle = load_bundle(EXPANSION_BUNDLE_DIR)
        assert len(bundle.cases) == 60
        supporting_meta = json.loads(
            (EXPANSION_BUNDLE_DIR / "supporting/cases.json").read_text(encoding="utf-8")
        )
        meta_by_id = {m["id"]: m for m in supporting_meta}

        failures = []
        for case in bundle.cases:
            meta = meta_by_id.get(case.id, {})
            issues = validate_diverse_candidate(
                case.input, case.output, context=case.context, metadata=meta
            )
            if issues:
                failures.append(f"{case.id}: {issues}")

        assert failures == [], f"Reference cases failed diverse mechanical validation: {failures}"


class TestMechanicalValidators:
    def test_unconverted_file_extensions(self):
        # Unconverted spoken extension flagged
        bad = "Please look at main dot py and type dot ts to verify."
        issues = check_unconverted_file_extensions(bad)
        assert len(issues) == 2
        assert any("dot py" in i for i in issues)
        assert any("dot ts" in i for i in issues)

        # Unconverted at start of string
        bad_start = "dot py is our entrypoint file."
        issues_start = check_unconverted_file_extensions(bad_start)
        assert len(issues_start) >= 1

        # Properly converted files pass cleanly
        good = "Please look at main.py and type.ts to verify."
        assert check_unconverted_file_extensions(good) == []

        # Matches inside quotes when input explicitly quoted that literal phrase are ignored
        raw_quoted = 'the prompt contains the phrase "dot py" literally'
        out_quoted = 'The prompt contains the phrase "dot py" literally.'
        assert check_unconverted_file_extensions(out_quoted, input_text=raw_quoted) == []

    def test_missing_at_tags(self):
        raw = "tag main dot py and add error logging"
        context = {"files": ["src/main.py", "src/types.ts"]}

        # Output with proper tag passes
        good_output = "Tag @src/main.py and add error logging."
        assert check_missing_at_tags(raw, good_output, context) == []

        # Output missing @ tag flagged
        bad_output = "Tag src/main.py and add error logging."
        issues = check_missing_at_tags(raw, bad_output, context)
        assert len(issues) >= 1
        assert any("missing_at_tag" in i for i in issues)

        # Negative control: ordinary prepositions not flagged
        norm_raw = "let's meet at noon to discuss the plan"
        norm_output = "Let's meet at noon to discuss the plan."
        assert check_missing_at_tags(norm_raw, norm_output, context) == []

        # Negative control: 'meet at 3 pm' or 'tag the release as stable' never trigger
        assert check_missing_at_tags("meet at 3 pm for standup", "Meet at 3 PM for standup.") == []
        assert (
            check_missing_at_tags("tag the release as stable", "Tag the release as stable.") == []
        )

        # Rejects trailing suffixes like @main.py.bak
        bad_suffix = "Tag @src/main.py.bak and add error logging."
        issues_suffix = check_missing_at_tags(raw, bad_suffix, context)
        assert len(issues_suffix) >= 1

        # Exact stem/basename match, not loose substring matching where main.py matches src/domain.py
        context_domain = {"files": ["src/domain.py"]}
        raw_main = "tag main dot py and verify"
        issues_domain = check_missing_at_tags(
            raw_main, "Tag @domain.py and verify.", context_domain
        )
        assert len(issues_domain) >= 1

        # Speech repair: tag X scratch that Y -> Y inherits tag trigger, X retracted
        repair_raw = "tag main dot py scratch that helper dot ts and verify"
        repair_ctx = {"files": ["src/main.py", "src/helper.ts"]}
        assert check_missing_at_tags(repair_raw, "Tag @src/helper.ts and verify.", repair_ctx) == []
        issues_repair_bad = check_missing_at_tags(
            repair_raw, "Tag @src/main.py and verify.", repair_ctx
        )
        assert any("missing_at_tag: @src/helper.ts" in i for i in issues_repair_bad)

        # Speech repair repeating tag cue: tag X scratch that tag Y -> requires @Y, not @tag_Y
        repair_rep = "tag main dot py scratch that tag helper dot ts and verify"
        assert check_missing_at_tags(repair_rep, "Tag @src/helper.ts and verify.", repair_ctx) == []
        issues_rep_bad = check_missing_at_tags(
            repair_rep, "Tag @tag_helper.ts and verify.", repair_ctx
        )
        assert any("missing_at_tag: @src/helper.ts" in i for i in issues_rep_bad)

    def test_retained_retractions(self):
        raw = "we need five scratch that ten instances running"
        # Output keeping superseded 'five' flagged
        bad = "We need five and ten instances running."
        issues = check_retained_retractions(raw, bad)
        assert len(issues) >= 1
        assert any("five" in i for i in issues)

        # Numeric equivalent '5' also flagged
        bad_num = "We need 5 and ten instances running."
        issues_num = check_retained_retractions(raw, bad_num)
        assert len(issues_num) >= 1

        # Output keeping only final intent 'ten' passes
        good = "We need 10 instances running."
        assert check_retained_retractions(raw, good, known_retractions=["five"]) == []

        # Heuristic does NOT reject 'I actually like this'
        raw_like = "I actually like this proposal"
        out_like = "I actually like this proposal."
        assert check_retained_retractions(raw_like, out_like) == []

        # Heuristic does NOT reject 'backup under any circumstances'
        raw_backup = "do not delete the production database backup under any circumstances no wait archive it instead"
        out_backup = "Do not delete the production database backup under any circumstances; archive it instead."
        assert check_retained_retractions(raw_backup, out_backup) == []

        # Heuristic does NOT reject reintroduced list items that appear in surviving text
        raw_reintro = "item two run performance benchmarks scratch that item two run security scans and item three run performance benchmarks"
        out_reintro = "1. Run security scans\n2. Run performance benchmarks"
        assert check_retained_retractions(raw_reintro, out_reintro) == []

        # Conversational adverb before verb: 'The server actually supports retries.'
        raw_actually = "The server actually supports retries."
        out_actually = "The server actually supports retries."
        assert check_retained_retractions(raw_actually, out_actually) == []

        # Retracted file extraction stripping preceding verbs like 'change'
        raw_change = "change auth service dot ts scratch that user service dot ts"
        bad_change = "Change auth_service.ts and user_service.ts."
        good_change = "Change user_service.ts."
        issues_change = check_retained_retractions(raw_change, bad_change)
        assert any("auth_service.ts" in i or "auth service dot ts" in i for i in issues_change)
        assert check_retained_retractions(raw_change, good_change) == []

        # identity=True exempts retractions
        assert (
            check_retained_retractions(
                "five scratch that ten", "five scratch that ten", identity=True
            )
            == []
        )

    def test_chatbot_preamble(self):
        # Conversational assistant preambles flagged
        assert check_chatbot_preamble("Sure! Here is the rewritten text:\nHello world.") != []
        assert check_chatbot_preamble("Certainly, I have organized your prompt below.") != []
        assert check_chatbot_preamble("Here is the updated configuration file.") != []
        assert check_chatbot_preamble("Here’s the updated configuration file.") != []
        assert check_chatbot_preamble("sure, here is the rewritten text:") != []

        # Input-aware: if input itself starts with 'Sure', do not reject output for preserving user words
        raw_sure = "Sure we can meet at noon"
        out_sure = "Sure, we can meet at noon."
        assert check_chatbot_preamble(out_sure, input_text=raw_sure) == []

        # Input-aware does NOT exempt output when chatbot meta-wrapper is introduced
        bad_sure_wrapper = "Sure! Here is the rewritten text: we can meet at noon."
        assert check_chatbot_preamble(bad_sure_wrapper, input_text=raw_sure) != []

        # Clean text passes
        assert check_chatbot_preamble("Update the database schema with the new timestamp.") == []
        assert check_chatbot_preamble("Can you review this pull request?") == []

    def test_rules_enforcement_required_and_forbidden(self):
        plan = {
            "id": "rules-test-01",
            "mode": "/slack",
            "required_any": [["8080"]],
            "forbidden": ["9090"],
            "literals": ["8080"],
            "identity": False,
        }
        # Output containing forbidden '9090' and missing '8080' must be rejected
        bad_output = "Server listening on port 9090."
        issues = validate_diverse_candidate("listen on 8080", bad_output, metadata=plan)
        assert any("forbidden: 9090" in i for i in issues)
        assert any("required: 8080" in i for i in issues)

        # Output satisfying rules passes cleanly
        good_output = "Server listening on port 8080."
        issues_good = validate_diverse_candidate("listen on 8080", good_output, metadata=plan)
        assert issues_good == []

    def test_aggregate_validator(self):
        raw = "tag main dot py scratch that helper dot ts"
        context = {"files": ["src/main.py", "src/helper.ts"]}

        # Output with chatbot preamble and unconverted extension
        bad = "Sure! Here is helper dot ts."
        issues = validate_diverse_candidate(raw, bad, context)
        assert len(issues) >= 2

        # Clean output
        good = "Inspect @src/helper.ts."
        assert validate_diverse_candidate(raw, good, context) == []


class TestArkaScenarioGeneratorSetup:
    def test_scenario_families_and_plans(self):
        assert "developer_symbols" in SCENARIO_FAMILIES
        assert "workspace_context" in SCENARIO_FAMILIES
        assert "spoken_lists" in SCENARIO_FAMILIES
        assert "multilevel_contrasts" in SCENARIO_FAMILIES
        assert "speech_repairs_and_isolation" in SCENARIO_FAMILIES

        plan = build_scenario_plan(
            family="developer_symbols",
            scenario_id="arka-dev-01",
            mode="/slack",
            length="short",
            brief="can you check user interface dot TS to make sure the profile is updated",
            context={"files": ["src/user_interface.ts"]},
            identity=False,
        )
        assert plan["id"] == "arka-dev-01"
        assert plan["mode"] == "/slack"
        assert plan["context"]["files"] == ["src/user_interface.ts"]
        assert plan["identity"] is False
        assert "seed_instruction" in plan
        assert "source_group" in plan
        assert plan["source_group"] == "expansion-arka-dev-01"

    def test_multilevel_contrast_group_builder(self):
        group = build_multilevel_contrast_group(
            "test-group",
            "um hey team please fix account service dot ts before 3 pm",
            files=["src/services/account_service.ts"],
            instructions="Format as a bug ticket",
        )
        assert len(group) == 4
        # All 4 share the exact same source_group
        assert len({p["source_group"] for p in group}) == 1
        assert group[0]["source_group"] == "contrast-test-group"
        # All 4 share the exact same brief/input
        assert len({p["brief"] for p in group}) == 1
        # Modes cover /transcribe, /slack, /polish, /prompt
        assert {p["mode"] for p in group} == {"/transcribe", "/slack", "/polish", "/prompt"}

    def test_coverage_schedule_spans_categories_and_lengths(self, tmp_path):
        schedule = build_coverage_schedule()
        assert len(schedule) >= 60
        families = {p["family"] for p in schedule}
        assert families == set(SCENARIO_FAMILIES.keys())
        lengths = {p["length"] for p in schedule}
        assert lengths == {"short", "medium", "long"}

        # Verify zero leakage with all benchmark bundles and validation holdouts
        blocked = expansion_leakage_index()
        for p in schedule:
            assert not blocked.matches(p["brief"]), f"Leakage detected in brief for {p['id']}"
            if p.get("target"):
                assert not blocked.matches(p["target"]), f"Leakage detected in target for {p['id']}"

        # Verify all training plans pass dry-run pipeline verification
        summary = run_expansion_pipeline(tmp_path, schedule, dry_run=True)
        assert summary["planned_scenarios"] == len(schedule)
        assert summary["validated_candidates"] == len(schedule)
        assert summary["mechanically_clean"] == len(schedule)

    def test_offline_pipeline_execution_and_training_audit(self, tmp_path):
        """End-to-end integration test exercising plan generation, candidate parsing, Rules(identity=...), and token_audit."""
        plans = [
            build_scenario_plan(
                family="developer_symbols",
                scenario_id="integ-01",
                mode="/transcribe",
                length="short",
                brief="we need to run main dot py with the flag",
                context={"level": "low"},
                identity=False,
            ),
            build_scenario_plan(
                family="developer_symbols",
                scenario_id="integ-02-identity",
                mode="/transcribe",
                length="short",
                brief="Already clean input.",
                context={"level": "low"},
                identity=True,
            ),
        ]

        summary = run_expansion_pipeline(tmp_path, plans)
        assert summary["planned_scenarios"] == 2
        assert summary["validated_candidates"] == 2

        # Check candidate parsing directly with Rules(identity=plan["identity"])
        plan = plans[0]
        hdr = header(plan)
        instr = f"{hdr}we need to run main dot py with the flag"
        resp = "We need to run main.py with the flag."
        parsed = parse_candidate(plan, instr, resp)
        assert parsed["input"] == "we need to run main dot py with the flag"
        assert parsed["output"] == resp

        # Verify candidate_issues directly with parsed output
        issues = candidate_issues(plan, parsed)
        assert issues == []

        # Construct Case and verify token_audit works without KeyError
        case = Case(
            id=plan["id"],
            **parsed,
            context=plan["context"],
            language=plan["language"],
            rules=Rules(identity=plan["identity"]),
        )
        assert case.rules.identity is False

        tokenizer = _tokenizer()
        ex = Example(case, "train", plan["source_group"], plan["length"], "independent", plan["id"])
        tokens = token_audit(ex, tokenizer)
        assert "input_tokens" in tokens
        assert tokens["input_tokens"] > 0

    def test_generation_config_structure(self):
        config = make_generation_config("seeds.jsonl", 10)
        assert config["version"] == "1"
        assert any(stage["type"] == "prompt_based_generator" for stage in config["pipeline"])
        assert any(stage["type"] == "decision_panel" for stage in config["pipeline"])
        assert any(stage["type"] == "labeling_engine" for stage in config["pipeline"])

    def test_live_pipeline_path_and_sol_review(self, tmp_path, monkeypatch):
        """Mock Arka runner & OpenRouter to test live path (dry_run=False) and review_candidates path."""
        from unittest.mock import MagicMock

        from finetuning_labs.rewriter.arka_pipeline import DIMENSIONS

        plan = build_scenario_plan(
            family="developer_symbols",
            scenario_id="mock-dev-01",
            mode="/slack",
            length="short",
            brief="can you check user interface dot ts to make sure the profile is updated",
            target="Can you check user_interface.ts to make sure the profile is updated?",
            context={"files": ["src/user_interface.ts"], "level": "medium"},
            identity=False,
            literals=["user_interface.ts"],
            required_any=[["user_interface.ts"]],
        )

        # Set fake OpenRouter API key so review_candidates triggers
        monkeypatch.setenv("OPENROUTER_API_KEY", "mock-openrouter-key")

        # Mock PipelineRunner
        mock_result = MagicMock()
        mock_result.final_count = 1

        class MockRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, *args, **kwargs):
                return mock_result

        # Mock ConfigLoader and StageBuilder
        mock_cfg_loader = MagicMock()
        mock_cfg_loader.load.return_value = MagicMock()

        mock_stage_builder = MagicMock()
        mock_stage_builder.build.return_value = []

        # Mock records for OutputWriter
        class MockRecord:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        mock_source = MockRecord(id="src-001")
        mock_gen = MockRecord(
            id="gen-001",
            lineage=MockRecord(root_id="src-001"),
            payload=MockRecord(
                instruction=header(plan) + plan["brief"],
                response=plan["target"],
            ),
        )

        class MockDecisionVal:
            def __init__(self, score, confidence):
                self.score = score
                self.confidence = confidence

            def model_dump(self, mode="json"):
                return {"score": self.score, "confidence": self.confidence}

        decisions_dict = {f"labeling_engine.{k}": MockDecisionVal(0.95, 0.9) for k in DIMENSIONS}
        mock_scored = MockRecord(
            id="gen-001",
            scores=MockRecord(decisions=decisions_dict),
        )

        class MockWriter:
            def read_parquet(self, path):
                p_str = str(path)
                if "01_source" in p_str:
                    return [mock_source]
                elif "02_generate" in p_str:
                    return [mock_gen]
                elif "03_label_quality" in p_str:
                    return [mock_scored]
                return []

        # Mock OpenRouter client in scale_data
        row_dict = parse_candidate(plan, header(plan) + plan["brief"], plan["target"])
        row_sha = digest(row_dict)

        def mock_build_openai_client(config):
            mock_client = MagicMock()
            completion_data = {
                "model": "openai/gpt-6.1-sol",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "reviews": [
                                        {
                                            "id": plan["id"],
                                            "row_sha256": row_sha,
                                            "input_verdict": "pass",
                                            "output_verdict": "pass",
                                            "pair_verdict": "pass",
                                            "verdict": "pass",
                                            "critique": "Candidate correctly converts dot ts to filename.",
                                        }
                                    ]
                                }
                            )
                        },
                    }
                ],
            }
            mock_comp = MagicMock()
            mock_comp.model_dump.return_value = completion_data
            mock_client.chat.completions.create.return_value = mock_comp
            return mock_client

        monkeypatch.setattr("arka.pipeline.runner.PipelineRunner", MockRunner)
        monkeypatch.setattr("arka.config.loader.ConfigLoader", lambda: mock_cfg_loader)
        monkeypatch.setattr(
            "arka.pipeline.stage_builder.StageBuilder", lambda *a, **kw: mock_stage_builder
        )
        monkeypatch.setattr("arka.pipeline.output.OutputWriter", MockWriter)
        monkeypatch.setattr("arka.llm.openai_client.build_openai_client", mock_build_openai_client)

        # Create stage parquet marker so path.exists() succeeds
        stage_dir = tmp_path / "runs/author/stages/03_label_quality"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "data.parquet").touch()

        summary = run_expansion_pipeline(tmp_path, [plan], dry_run=False)

        assert summary["planned_scenarios"] == 1
        assert summary["generated_candidates"] == 1
        assert summary["validated_candidates"] == 1
        assert summary["mechanically_clean"] == 1
        assert summary["sol_reviewed"] is True
        assert summary["sol_summary"] is not None
        assert summary["sol_summary"]["reviewed"] == 1
        assert summary["sol_summary"]["verdicts"] == {"pass": 1}

        # Verify output paths expected by scale_data.review_candidates exist
        gen_dir = tmp_path / "generation"
        assert (gen_dir / "candidates.jsonl").exists()
        assert (gen_dir / "supporting.json").exists()
        assert (tmp_path / "candidates.jsonl").exists()
        assert (tmp_path / "supporting.json").exists()
        assert (tmp_path / "candidate-supporting.json").exists()
        assert (tmp_path / "sol-reviews.json").exists()

        # Test stage_expansion_dataset
        stage_dir = tmp_path / "staged"
        staged = stage_expansion_dataset(tmp_path, stage_dir)
        assert staged["staged_pairs"] == 1
        assert (stage_dir / "viewer" / "train.jsonl").exists()
        assert (stage_dir / "sft" / "train.jsonl").exists()
        assert (stage_dir / "manifest.json").exists()
        assert (stage_dir / "quality.json").exists()


class TestDiverseExpansionFixes:
    def test_all_seed_instructions_under_15000(self):
        """Every single scenario in build_coverage_schedule must have seed_instruction < 15,000 chars."""
        plans = build_coverage_schedule()
        assert len(plans) >= 60
        for plan in plans:
            seed_inst = plan.get("seed_instruction", "")
            if seed_inst:
                assert len(seed_inst) < 15000, (
                    f"Seed instruction for {plan['id']} exceeds 15,000 chars: {len(seed_inst)}"
                )

            # Also check compact serialized format for seeds.jsonl
            compact = {
                "id": plan["id"],
                "mode": plan["mode"],
                "language": plan["language"],
                "context": plan["context"],
                "length": plan["length"],
                "header": header(plan),
                "mode_contract": plan.get("mode_contract", ""),
                "brief": plan["brief"],
            }
            compact_str = json.dumps(compact, ensure_ascii=False)
            assert len(compact_str) < 15000, (
                f"Compact seed for {plan['id']} exceeds 15,000 chars: {len(compact_str)}"
            )

    def test_stage_release_gate_missing_sol_reviews_raises_error(self, tmp_path):
        """In live mode (dry_run=False), missing sol-reviews.json must raise ValueError."""
        (tmp_path / "candidates.jsonl").write_text(
            json.dumps({"input": "test in", "output": "test out", "mode": "/slack"}) + "\n"
        )
        (tmp_path / "supporting.json").write_text(
            json.dumps([{"id": "c-01", "mechanical_issues": []}])
        )
        stage_dir = tmp_path / "stage"

        with pytest.raises(ValueError, match="sol-reviews.json is missing in live mode"):
            stage_expansion_dataset(tmp_path, stage_dir, dry_run=False)

    def test_stage_release_gate_empty_sol_reviews_raises_error(self, tmp_path):
        """In live mode (dry_run=False), empty sol-reviews.json must raise ValueError."""
        (tmp_path / "candidates.jsonl").write_text(
            json.dumps({"input": "test in", "output": "test out", "mode": "/slack"}) + "\n"
        )
        (tmp_path / "supporting.json").write_text(
            json.dumps([{"id": "c-01", "mechanical_issues": []}])
        )
        (tmp_path / "sol-reviews.json").write_text(json.dumps({"reviews": []}))
        stage_dir = tmp_path / "stage"

        with pytest.raises(ValueError, match="sol-reviews.json is empty in live mode"):
            stage_expansion_dataset(tmp_path, stage_dir, dry_run=False)

    def test_stage_release_gate_mismatched_row_sha256_blocks_staging(self, tmp_path):
        """A review with mismatched row_sha256 must block staging (0 staged pairs)."""
        row = {
            "input": "test in",
            "output": "test out",
            "mode": "/slack",
            "checks": "Preserve every fact.",
        }
        row_sha = digest(row)
        from finetuning_labs.rewriter.arka_pipeline import DIMENSIONS

        decisions = {f"labeling_engine.{k}": {"score": 0.95, "confidence": 0.9} for k in DIMENSIONS}
        meta = {
            "id": "c-01",
            "row_sha256": row_sha,
            "mechanical_issues": [],
            "decisions": decisions,
            "native_accepted": True,
        }
        (tmp_path / "candidates.jsonl").write_text(json.dumps(row) + "\n")
        (tmp_path / "supporting.json").write_text(json.dumps([meta]))
        # Write review with mismatched row_sha256
        (tmp_path / "sol-reviews.json").write_text(
            json.dumps(
                {
                    "reviews": [
                        {
                            "id": "c-01",
                            "row_sha256": "wrong_sha256_hash",
                            "input_verdict": "pass",
                            "output_verdict": "pass",
                            "pair_verdict": "pass",
                            "verdict": "pass",
                            "critique": "Looks fine but hash is wrong",
                        }
                    ]
                }
            )
        )
        stage_dir = tmp_path / "stage"
        staged = stage_expansion_dataset(tmp_path, stage_dir, dry_run=False)
        assert staged["staged_pairs"] == 0

    def test_stage_contrast_group_requires_identical_inputs(self, tmp_path):
        """A contrast group where variants have different input strings is dropped completely."""
        from finetuning_labs.rewriter.arka_pipeline import DIMENSIONS

        decisions = {f"labeling_engine.{k}": {"score": 0.95, "confidence": 0.9} for k in DIMENSIONS}
        group_id = "test-group"
        levels = ["low", "medium", "high", "custom"]
        modes = ["/transcribe", "/slack", "/polish", "/prompt"]

        rows = []
        metas = []
        reviews = []
        for i, (lvl, md) in enumerate(zip(levels, modes, strict=True)):
            cid = f"{group_id}-{lvl}"
            # Variant 3 has different input!
            raw_in = "identical input" if i != 3 else "different input"
            r = {"input": raw_in, "output": f"output {lvl}", "mode": md, "checks": "Checks."}
            r_sha = digest(r)
            m = {
                "id": cid,
                "source_group": f"contrast-{group_id}",
                "family": "multilevel_contrasts",
                "row_sha256": r_sha,
                "context": {"level": lvl},
                "mechanical_issues": [],
                "decisions": decisions,
                "native_accepted": True,
            }
            rev = {
                "id": cid,
                "row_sha256": r_sha,
                "input_verdict": "pass",
                "output_verdict": "pass",
                "pair_verdict": "pass",
                "verdict": "pass",
                "critique": "Pass.",
            }
            rows.append(r)
            metas.append(m)
            reviews.append(rev)

        (tmp_path / "candidates.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        (tmp_path / "supporting.json").write_text(json.dumps(metas), encoding="utf-8")
        (tmp_path / "sol-reviews.json").write_text(
            json.dumps({"reviews": reviews}), encoding="utf-8"
        )

        stage_dir = tmp_path / "stage"
        staged = stage_expansion_dataset(tmp_path, stage_dir, dry_run=False)
        assert staged["staged_pairs"] == 0

    def test_stage_contrast_group_requires_all_four_variants(self, tmp_path):
        """A contrast group with only 3 variants is dropped; complete 4-variant group is staged."""
        from finetuning_labs.rewriter.arka_pipeline import DIMENSIONS

        decisions = {f"labeling_engine.{k}": {"score": 0.95, "confidence": 0.9} for k in DIMENSIONS}
        group_id = "test-group"
        levels = ["low", "medium", "high", "custom"]
        modes = ["/transcribe", "/slack", "/polish", "/prompt"]

        # Only 3 variants provided
        rows = []
        metas = []
        reviews = []
        for lvl, md in zip(levels[:3], modes[:3], strict=True):
            cid = f"{group_id}-{lvl}"
            r = {
                "input": "shared input",
                "output": f"output {lvl}",
                "mode": md,
                "checks": "Checks.",
            }
            r_sha = digest(r)
            m = {
                "id": cid,
                "source_group": f"contrast-{group_id}",
                "family": "multilevel_contrasts",
                "row_sha256": r_sha,
                "context": {"level": lvl},
                "mechanical_issues": [],
                "decisions": decisions,
                "native_accepted": True,
            }
            rev = {
                "id": cid,
                "row_sha256": r_sha,
                "input_verdict": "pass",
                "output_verdict": "pass",
                "pair_verdict": "pass",
                "verdict": "pass",
                "critique": "Pass.",
            }
            rows.append(r)
            metas.append(m)
            reviews.append(rev)

        (tmp_path / "candidates.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        (tmp_path / "supporting.json").write_text(json.dumps(metas), encoding="utf-8")
        (tmp_path / "sol-reviews.json").write_text(
            json.dumps({"reviews": reviews}), encoding="utf-8"
        )

        stage_dir = tmp_path / "stage"
        staged = stage_expansion_dataset(tmp_path, stage_dir, dry_run=False)
        assert staged["staged_pairs"] == 0

        # Now add the 4th variant (custom)
        cid = f"{group_id}-custom"
        r4 = {
            "input": "shared input",
            "output": "output custom",
            "mode": "/prompt",
            "checks": "Checks.",
        }
        r4_sha = digest(r4)
        m4 = {
            "id": cid,
            "source_group": f"contrast-{group_id}",
            "family": "multilevel_contrasts",
            "row_sha256": r4_sha,
            "context": {"level": "custom"},
            "mechanical_issues": [],
            "decisions": decisions,
            "native_accepted": True,
        }
        rev4 = {
            "id": cid,
            "row_sha256": r4_sha,
            "input_verdict": "pass",
            "output_verdict": "pass",
            "pair_verdict": "pass",
            "verdict": "pass",
            "critique": "Pass.",
        }
        rows.append(r4)
        metas.append(m4)
        reviews.append(rev4)

        (tmp_path / "candidates.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        (tmp_path / "supporting.json").write_text(json.dumps(metas), encoding="utf-8")
        (tmp_path / "sol-reviews.json").write_text(
            json.dumps({"reviews": reviews}), encoding="utf-8"
        )

        stage_dir2 = tmp_path / "stage2"
        staged2 = stage_expansion_dataset(tmp_path, stage_dir2, dry_run=False)
        assert staged2["staged_pairs"] == 4

    def test_admission_policy_two_tier(self):
        """Test two-tier admission policy: native strong + 1 sol vs uncertain + 2 sol vs veto."""
        from finetuning_labs.rewriter.arka_pipeline import DIMENSIONS

        clean_decisions = {
            f"labeling_engine.{k}": {"score": 0.85, "confidence": 0.5} for k in DIMENSIONS
        }
        veto_decisions = dict(clean_decisions)
        veto_decisions[f"labeling_engine.{next(iter(DIMENSIONS))}"] = {
            "score": 0.05,
            "confidence": 0.9,
        }

        row = {"input": "in", "output": "out", "mode": "/slack", "checks": "checks"}
        r_sha = digest(row)
        sol_pass = {
            "id": "c-01",
            "row_sha256": r_sha,
            "critique": "Round 1 passed cleanly.",
            "input_verdict": "pass",
            "output_verdict": "pass",
            "pair_verdict": "pass",
            "verdict": "pass",
            "call_id": "call-001",
            "request_sha256": "req-sha-001",
            "response_sha256": "resp-sha-001",
            "round": 1,
        }

        # 1. Native veto blocks admission even if sol passed
        meta_veto = {
            "id": "c-01",
            "row_sha256": r_sha,
            "mechanical_issues": [],
            "decisions": veto_decisions,
            "native_accepted": True,
        }
        assert not is_candidate_admitted(row, meta_veto, sol_pass)

        # 2. Native strong (native_accepted=True) passes with 1 sol pass
        meta_strong = {
            "id": "c-01",
            "row_sha256": r_sha,
            "mechanical_issues": [],
            "decisions": clean_decisions,
            "native_accepted": True,
        }
        assert is_candidate_admitted(row, meta_strong, sol_pass)

        # 3. Non-native-strong requires round 2 sol pass
        uncertain_decisions = {
            f"labeling_engine.{k}": {"score": 0.70, "confidence": 0.5} for k in DIMENSIONS
        }
        meta_uncertain = {
            "id": "c-01",
            "row_sha256": r_sha,
            "mechanical_issues": [],
            "decisions": uncertain_decisions,
            "native_accepted": False,
        }
        # Fails with only 1 sol pass
        assert not is_candidate_admitted(row, meta_uncertain, sol_pass, sol_review_round2=None)
        # Rejects identical review object (r1 is r2)
        assert not is_candidate_admitted(row, meta_uncertain, sol_pass, sol_review_round2=sol_pass)
        # Rejects identical dictionary copies (dict(sol_pass))
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass, sol_review_round2=dict(sol_pass)
        )
        # Rejects identical dictionary copies omitting critique/call_id
        r_bare = {
            "id": "c-01",
            "row_sha256": r_sha,
            "input_verdict": "pass",
            "output_verdict": "pass",
            "pair_verdict": "pass",
            "verdict": "pass",
        }
        assert not is_candidate_admitted(
            row, meta_uncertain, r_bare, sol_review_round2=dict(r_bare)
        )
        # Rejects two reviews with different critiques but no distinct call_id/response_sha256/round
        r_diff_crit1 = dict(r_bare, critique="First distinct critique text.")
        r_diff_crit2 = dict(r_bare, critique="Second distinct critique text.")
        assert not is_candidate_admitted(
            row, meta_uncertain, r_diff_crit1, sol_review_round2=r_diff_crit2
        )
        # Rejects two reviews with round=1 and round=2 and different critiques, but NO call_id, response_sha256, or request_sha256
        r_diff_rounds_only_1 = dict(r_diff_crit1, round=1)
        r_diff_rounds_only_2 = dict(r_diff_crit2, round=2)
        assert not is_candidate_admitted(
            row, meta_uncertain, r_diff_rounds_only_1, sol_review_round2=r_diff_rounds_only_2
        )
        # Rejects two reviews where call_id="" (empty string) even when another field differs
        sol_pass_empty_call_1 = dict(
            sol_pass,
            call_id="",
            response_sha256="sha256-001",
            request_sha256="req-001",
            critique="Critique 1",
        )
        sol_pass_empty_call_2 = dict(
            sol_pass,
            call_id="call-002",
            response_sha256="sha256-002",
            request_sha256="req-002",
            critique="Critique 2",
            round=2,
        )
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass_empty_call_1, sol_review_round2=sol_pass_empty_call_2
        )
        # Rejects identical call_id
        sol_pass_same_call_1 = dict(
            sol_pass,
            call_id="call-001",
            critique="Critique 1",
            request_sha256="req-001",
            response_sha256="resp-001",
        )
        sol_pass_same_call_2 = dict(
            sol_pass,
            call_id="call-001",
            critique="Critique 2",
            request_sha256="req-002",
            response_sha256="resp-002",
            round=2,
        )
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass_same_call_1, sol_review_round2=sol_pass_same_call_2
        )
        # Rejects identical critique
        sol_pass_same_critique = dict(
            sol_pass,
            call_id="call-002",
            request_sha256="req-002",
            response_sha256="resp-002",
            round=2,
        )
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass, sol_review_round2=sol_pass_same_critique
        )
        # Passes with 2 distinct sol passes with distinct critique and distinct call_id, request_sha256, response_sha256, and round
        sol_pass_round2 = dict(
            sol_pass,
            critique="Distinct round 2 verification critique",
            call_id="call-003",
            request_sha256="req-sha-002",
            response_sha256="resp-sha-002",
            round=2,
        )
        assert is_candidate_admitted(
            row, meta_uncertain, sol_pass, sol_review_round2=sol_pass_round2
        )

        # Finding 3 tests:
        # 1. request_sha256 is identical between r1 and r2
        sol_pass_same_req = dict(sol_pass_round2, request_sha256="req-sha-001")
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass, sol_review_round2=sol_pass_same_req
        )

        # 2. request_sha256 is missing on either review while call_id and response_sha256 are present
        sol_pass_missing_req_1 = dict(sol_pass)
        del sol_pass_missing_req_1["request_sha256"]
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass_missing_req_1, sol_review_round2=sol_pass_round2
        )
        sol_pass_missing_req_2 = dict(sol_pass_round2)
        del sol_pass_missing_req_2["request_sha256"]
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass, sol_review_round2=sol_pass_missing_req_2
        )

        # 3. call_id is missing on either review while response_sha256 is distinct
        sol_pass_missing_call_1 = dict(sol_pass)
        del sol_pass_missing_call_1["call_id"]
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass_missing_call_1, sol_review_round2=sol_pass_round2
        )
        sol_pass_missing_call_2 = dict(sol_pass_round2)
        del sol_pass_missing_call_2["call_id"]
        assert not is_candidate_admitted(
            row, meta_uncertain, sol_pass, sol_review_round2=sol_pass_missing_call_2
        )

    def test_native_arka_provenance_and_state_db_cache(self, monkeypatch):
        """Verify:
        1. Every lineage.root_id in 06_decision_panel/data.parquet exists in 01_source/data.parquet (id column), and all seeds < 15,000 chars.
        2. Every staged row's content_hash in 06_decision_panel/data.parquet matches Arka's hash of its (instruction, response) payload.
        3. Every staged row (38/38) has matching cache entries in state.db across all 6 dimensions for its exact released (instruction, response) text.
        """
        import sqlite3

        from arka.config.loader import ConfigLoader
        from arka.decisions.client import DecisionClient
        from arka.pipeline.decision_stages import decision_questions, record_state
        from arka.pipeline.output import OutputWriter
        from arka.records.identity import content_hash

        from finetuning_labs.rewriter.diverse_expansion import load_openrouter_api_key

        monkeypatch.setenv("OPENROUTER_API_KEY", "offline-cache-audit-key")
        load_openrouter_api_key()
        run_dir = Path(__file__).resolve().parents[1] / "data/rewriter/expansions/diverse-v1-run"
        writer = OutputWriter()

        s1_records = writer.read_parquet(run_dir / "runs/author/stages/01_source/data.parquet")
        s6_records = writer.read_parquet(
            run_dir / "runs/author/stages/06_decision_panel/data.parquet"
        )

        # 1. Provenance & Seed Length
        s1_ids = {r.id for r in s1_records}
        for r in s1_records:
            assert len(r.payload.instruction) < 15000, f"Seed {r.id} exceeds 15,000 chars"
        for r in s6_records:
            assert r.lineage.root_id in s1_ids, f"root_id {r.lineage.root_id} not in 01_source ids"

        # 2. Content Hash Match
        staged_viewer_path = (
            Path(__file__).resolve().parents[1]
            / "data/rewriter/expansions/diverse-v1/viewer/train.jsonl"
        )
        staged_viewer = [
            json.loads(x) for x in staged_viewer_path.read_text(encoding="utf-8").splitlines()
        ]
        staged_pairs = {(r["input"], r["output"]) for r in staged_viewer}
        candidates = [
            json.loads(x)
            for x in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        staged_indices = [
            i for i, c in enumerate(candidates) if (c["input"], c["output"]) in staged_pairs
        ]
        assert len(staged_indices) == 38, (
            f"Expected 38 staged candidates, found {len(staged_indices)}"
        )

        staged_supp = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "data/rewriter/expansions/diverse-v1/supporting.json"
            ).read_text(encoding="utf-8")
        )
        run_supp = json.loads((run_dir / "supporting.json").read_text(encoding="utf-8"))

        for idx in staged_indices:
            rec = s6_records[idx]
            expected_hash = content_hash(rec.payload)
            assert rec.content_hash == expected_hash, (
                f"Row {idx} hash mismatch: {rec.content_hash} != {expected_hash}"
            )
            assert staged_supp[idx]["record_id"] == rec.id, (
                f"Row {idx} record_id mismatch in staged supporting.json: {staged_supp[idx]['record_id']} != {rec.id}"
            )
            assert run_supp[idx]["record_id"] == rec.id, (
                f"Row {idx} record_id mismatch in run supporting.json: {run_supp[idx]['record_id']} != {rec.id}"
            )

        # 3. 100% genuine state.db decisions cache entries across all 6 dimensions
        cfg_loader = ConfigLoader()
        config = cfg_loader.load(run_dir / "pipeline.yaml")
        client = DecisionClient(config.decisions)
        label_stage = next(s for s in config.pipeline if s.type == "labeling_engine")

        con = sqlite3.connect(run_dir / "state.db")
        cur = con.cursor()
        db_keys = {r[0] for r in cur.execute("SELECT cache_key FROM decisions_cache").fetchall()}
        con.close()

        for idx in staged_indices:
            rec = s6_records[idx]
            questions = decision_questions(label_stage, rec, run_dir)
            assert len(questions) == 6, f"Expected 6 questions, got {len(questions)}"
            state = record_state(rec)
            keys = [client._key(state, q) for q in questions]
            hits = sum(k in db_keys for k in keys)
            assert hits == 6, f"Row {idx} ({rec.id}) has only {hits}/6 cache hits in state.db"

    def test_diverse_v2_and_validation_v1_decision_cache_and_parquet_sync(self, monkeypatch):
        """Verify 100% of retained rows in diverse-v2 (115 rows) and validation-v1 (60 rows)

        have exact cache hits in state.db (decisions_cache) across all 6 dimensions,
        matching content_hash, and byte-for-byte identical 02p and 06 decision panels.
        """
        import sqlite3

        from arka.config.loader import ConfigLoader
        from arka.decisions.client import DecisionClient
        from arka.decisions.factory import resolve_decision_config
        from arka.labeling.rubric import RubricLoader
        from arka.pipeline.decision_stages import record_state, rubric_questions
        from arka.pipeline.output import OutputWriter
        from arka.records.identity import content_hash

        from finetuning_labs.rewriter.diverse_expansion import load_openrouter_api_key

        monkeypatch.setenv("OPENROUTER_API_KEY", "offline-cache-audit-key")
        load_openrouter_api_key()
        writer = OutputWriter()

        test_cases = [
            ("diverse-v2-run", "diverse-v2", 115),
            ("validation-v1-run", "validation-v1", 60),
        ]

        for run_name, staged_name, expected_count in test_cases:
            run_dir = Path(__file__).resolve().parents[1] / f"data/rewriter/expansions/{run_name}"
            staged_dir = (
                Path(__file__).resolve().parents[1] / f"data/rewriter/expansions/{staged_name}"
            )

            # 1. Byte-for-byte identity of 02p and 06
            p02 = run_dir / "runs/author/stages/02p_decision_panel/data.parquet"
            p06 = run_dir / "runs/author/stages/06_decision_panel/data.parquet"
            assert p02.read_bytes() == p06.read_bytes(), f"02p and 06 differ in {run_name}"

            # 2. 01_source seeds < 15,000 chars
            s1 = writer.read_parquet(run_dir / "runs/author/stages/01_source/data.parquet")
            for r in s1:
                assert len(r.payload.instruction) < 15000, f"Seed {r.id} >= 15,000 chars"
                assert r.content_hash == content_hash(r.payload)

            # 3. 100% of retained rows
            s2 = writer.read_parquet(run_dir / "runs/author/stages/02_generate/data.parquet")
            s6 = writer.read_parquet(p06)
            staged_viewer = [
                json.loads(line)
                for line in (staged_dir / "viewer/train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            assert len(staged_viewer) == expected_count

            candidates = [
                json.loads(line)
                for line in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            staged_pairs = {(r["input"], r["output"]) for r in staged_viewer}
            staged_indices = [
                i for i, c in enumerate(candidates) if (c["input"], c["output"]) in staged_pairs
            ]
            assert len(staged_indices) == expected_count

            cfg = ConfigLoader().load(run_dir / "pipeline.yaml")
            rubric = RubricLoader().load(run_dir / "rubric.yaml")
            questions = rubric_questions(rubric)
            assert len(questions) == 6

            dec_cfg = resolve_decision_config(cfg.decisions, None)
            client = DecisionClient(dec_cfg)

            con = sqlite3.connect(run_dir / "state.db")
            cur = con.cursor()
            db_keys = {
                r[0] for r in cur.execute("SELECT cache_key FROM decisions_cache").fetchall()
            }
            con.close()

            for idx in staged_indices:
                cand = candidates[idx]
                rec = s6[idx]
                grec = s2[idx]

                expected_hash = content_hash(rec.payload)
                assert rec.content_hash == expected_hash
                assert grec.content_hash == expected_hash
                assert rec.payload.instruction.endswith(cand["input"])
                assert rec.payload.response == cand["output"]

                s_rec = record_state(rec)
                for q in questions:
                    k = client._key(s_rec, q)
                    assert k in db_keys, f"Missing cache key in {run_name} for {rec.id} {q.name}"

    def test_load_openrouter_api_key_reads_only_from_environment(self, monkeypatch):
        """load_openrouter_api_key must read only from os.environ and never inspect personal dotfiles."""
        import inspect

        from finetuning_labs.rewriter import diverse_expansion

        source = inspect.getsource(diverse_expansion.load_openrouter_api_key)
        assert "/home/" not in source
        assert "zshrc" not in source

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert diverse_expansion.load_openrouter_api_key() is None

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-env-key")
        assert diverse_expansion.load_openrouter_api_key() == "test-env-key"
