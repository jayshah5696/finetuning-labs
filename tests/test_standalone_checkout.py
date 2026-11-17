"""Check Git operations from a standalone project checkout."""

import hashlib
import runpy
import subprocess
from pathlib import Path

from finetuning_labs.rewriter import boost_data, evaluation

PROJECT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root):
    git(root, "init", "--quiet")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        "Fixture",
    )
    return git(root, "rev-parse", "HEAD")


def test_provenance_uses_project_checkout(tmp_path, monkeypatch):
    root = tmp_path / "standalone"
    runner = root / "src/finetuning_labs/rewriter/evaluation.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# Fixture runner\n")
    expected = commit(root)
    monkeypatch.setattr(evaluation, "__file__", str(runner))
    monkeypatch.delenv("RUN_CODE_COMMIT", raising=False)

    result = evaluation.code_provenance()
    assert result["code_commit"] == expected
    assert result["code_dirty"] is False
    assert result["runner_sha256"] == hashlib.sha256(runner.read_bytes()).hexdigest()

    (root / "untracked.txt").write_text("Local change\n")
    assert evaluation.code_provenance()["code_dirty"] is True


def test_recovery_reads_project_relative_git_baseline(tmp_path, monkeypatch):
    root = tmp_path / "standalone"
    source = root / "data/rewriter/current/data/train.jsonl"
    source.parent.mkdir(parents=True)
    original = b'{"input": "original"}\n'
    source.write_bytes(original)
    baseline = commit(root)
    source.write_text('{"input": "changed"}\n')
    monkeypatch.setattr(boost_data, "PROJECT", root)
    script = runpy.run_path(str(PROJECT / "scripts/backfill_english_quality.py"))

    restored = script["restore_baseline"](baseline, tmp_path / "restored")
    assert (restored / "data/train.jsonl").read_bytes() == original
    assert source.read_bytes() != original


def test_justfile_replaces_makefile_and_runs_list():
    justfile = PROJECT / "justfile"
    assert justfile.exists()
    assert not (PROJECT / "Makefile").exists()
    output = subprocess.check_output(["just", "--list"], cwd=PROJECT, text=True)
    for recipe in [
        "default",
        "sync",
        "test",
        "check",
        "build",
        "review",
        "export",
        "tokens",
        "training-audit",
        "reviewed-audit",
        "notebook",
        "notebook-check",
    ]:
        assert recipe in output


def test_every_script_has_docstring_stating_current_or_historical():
    import ast

    scripts_dir = PROJECT / "scripts"
    readme = (scripts_dir / "README.md").read_text()
    assert "make reviewed-audit" not in readme
    assert "just reviewed-audit" in readme
    for path in sorted(scripts_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        doc = ast.get_docstring(tree)
        assert doc, f"{path.name} is missing a top docstring"
        assert "current" in doc.lower() or "historical" in doc.lower(), (
            f"{path.name} docstring must state whether it is current or historical"
        )
        assert path.name in readme, f"{path.name} must be listed in scripts/README.md"


def test_check_english_release_accepts_output_dir_and_current_split_counts(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["check_english_release.py", "--output-dir", str(tmp_path)])
    runpy.run_path(str(PROJECT / "scripts/check_english_release.py"), run_name="__main__")
    report_path = tmp_path / "english-expansion-check.json"
    assert report_path.exists()
    import json

    report = json.loads(report_path.read_text())
    assert report["rows"] == 500
    assert report["exact_pair_verification_passes"] is True


def test_verify_state_db_and_parquets_has_no_import_side_effects():
    source = (PROJECT / "scripts/verify_state_db_and_parquets.py").read_text()
    assert "\nload_openrouter_api_key()" not in source.split("def verify_run_and_staged")[0]
    sync_source = (PROJECT / "scripts/sync_arka_decisions_and_stages.py").read_text()
    assert "from scripts." not in sync_source
