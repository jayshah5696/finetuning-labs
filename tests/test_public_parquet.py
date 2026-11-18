"""Specify the clean Hub layout and row-preservation failures before implementation."""

import io

import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from finetuning_labs.rewriter.hf_data import build_dataset, cleanup_paths


def example():
    row = {
        "input": 'hello\n1. keep "cache.enabled"',
        "output": 'Hello.\n1. Keep "cache.enabled".',
        "mode": "/transcribe",
        "checks": "Preserve the literal and both lines.",
    }
    guide = {
        "system": "Rewrite only.",
        "user": "MODE: /transcribe\nLANGUAGE: en\nINPUT:\n" + row["input"],
        "assistant": row["output"],
    }
    return row, guide


def build(row=None, guide=None):
    expected_row, expected_guide = example()
    return build_dataset(
        {"train": [row or expected_row]},
        {"train": [guide or expected_guide]},
        title="Rewriter",
        description="A small test.",
        license_text=b"Source notices.",
        source_license=b"GPL source notice.",
    )


def test_parquet_roundtrip_keeps_visible_and_supervised_fields_exactly():
    files = build()
    row, guide = example()
    assert pq.read_table(io.BytesIO(files["data/train.parquet"])).to_pylist() == [row]
    assert pq.read_table(io.BytesIO(files["sft/train.parquet"])).to_pylist() == [guide]
    assert set(files) == {
        "README.md",
        "DATA_LICENSE.md",
        "licenses/TYPEWHISPER-LICENSE.txt",
        "data/train.parquet",
        "sft/train.parquet",
    }
    assert b"default: true" in files["README.md"]


@pytest.mark.parametrize("change", ["metadata", "wrong_target", "missing_context"])
def test_extra_metadata_or_changed_supervision_is_rejected(change):
    row, guide = example()
    if change == "metadata":
        row["review"] = "pass"
    elif change == "wrong_target":
        guide["assistant"] = "An unrelated answer."
    else:
        guide["user"] = "Different input."
    with pytest.raises(ValueError):
        build(row, guide)


def test_english_export_uses_verified_language_metadata_and_retains_order():
    from finetuning_labs.rewriter.hf_data import english_splits

    row, guide = example()
    german = row | {"input": "LANGUAGE: en is quoted text", "output": "German example"}
    de_guide = guide | {
        "user": "MODE: /transcribe\nLANGUAGE: de\nINPUT:\n" + german["input"],
        "assistant": german["output"],
    }
    rows, guides = english_splits(
        {"train": [row, german]},
        {"train": [guide, de_guide]},
        [{"split": "train", "language": "en"}, {"split": "train", "language": "de"}],
    )
    assert rows == {"train": [row]} and guides == {"train": [guide]}


def test_cleanup_removes_code_reports_and_old_data_paths():
    files = build()
    existing = {
        ".gitattributes",
        "README.md",
        "runtime/train.py",
        "notebooks/train.ipynb",
        "audit/reviews.json",
        "old/train.jsonl",
        "data/train.parquet",
    }
    assert cleanup_paths(existing, files) == [
        "audit/reviews.json",
        "notebooks/train.ipynb",
        "old/train.jsonl",
        "runtime/train.py",
    ]
    with pytest.raises(ValueError, match="data"):
        cleanup_paths(existing, {**files, "runtime/train.py": b"code"})


def test_build_dataset_uses_configured_repo_id():
    from finetuning_labs.rewriter.hf_data import BENCHMARK_REPO, DEFAULT_HF_OWNER, TRAIN_REPO

    assert DEFAULT_HF_OWNER == "jayshah5696"
    assert TRAIN_REPO == "jayshah5696/finetuning-labs-rewriter-training"
    assert BENCHMARK_REPO == "jayshah5696/finetuning-labs-rewriter"
    row, guide = example()
    files = build_dataset(
        {"train": [row]},
        {"train": [guide]},
        title="Rewriter",
        description="Test",
        license_text=b"License",
        source_license=b"Source",
        repo_id="custom-owner/custom-repo",
    )
    assert b'load_dataset("custom-owner/custom-repo", "sft")' in files["README.md"]


def test_hub_and_hf_data_cli_handle_errors_cleanly(monkeypatch, tmp_path):
    from finetuning_labs.rewriter import hf_data, hub

    monkeypatch.setattr(
        "sys.argv",
        [
            "hub",
            "upload",
            "--package",
            str(tmp_path),
            "--repo-id",
            "owner/repo",
            "--receipt",
            str(tmp_path / "r.json"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        hub.main()
    assert exc_info.value.code == 1

    monkeypatch.setattr(hf_data, "_sources", lambda **_: (_ for _ in ()).throw(ValueError("bad")))
    monkeypatch.setattr("sys.argv", ["hf_data", "--output", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc_info:
        hf_data.main()
    assert exc_info.value.code == 1


def test_default_hf_owner_is_pure_constant_and_env_var_only_affects_cli_default(
    monkeypatch, tmp_path
):
    import importlib

    from finetuning_labs.rewriter import hf_data, hub

    monkeypatch.setenv("FINETUNING_LABS_HF_OWNER", "env-fork-owner")
    importlib.reload(hf_data)
    importlib.reload(hub)
    assert hf_data.DEFAULT_HF_OWNER == "jayshah5696"
    assert hf_data.TRAIN_REPO == "jayshah5696/finetuning-labs-rewriter-training"
    assert hf_data.BENCHMARK_REPO == "jayshah5696/finetuning-labs-rewriter"
    assert hub.DEFAULT_HF_OWNER == "jayshah5696"

    observed_owners = []
    row, guide = example()

    def fake_sources(*, owner=hf_data.DEFAULT_HF_OWNER):
        observed_owners.append(owner)
        return (
            [
                (
                    f"{owner}/finetuning-labs-rewriter-training",
                    {"train": [row]},
                    {"train": [guide]},
                    "Title",
                    "Desc",
                )
            ],
            b"License",
            b"Source",
        )

    monkeypatch.setattr(hf_data, "_sources", fake_sources)
    monkeypatch.setattr("sys.argv", ["hf_data", "--output", str(tmp_path / "out")])
    hf_data.main()
    assert observed_owners == ["env-fork-owner"]


def test_hf_data_publish_with_non_default_owner_writes_receipt_to_output_dir(monkeypatch, tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace

    import huggingface_hub

    from finetuning_labs.rewriter import hf_data

    tracked_receipt = hf_data.PROJECT / "data/rewriter/releases/public-parquet.json"
    tracked_before = tracked_receipt.read_bytes()
    row, guide = example()

    def fake_sources(*, owner=hf_data.DEFAULT_HF_OWNER):
        return (
            [
                (
                    f"{owner}/finetuning-labs-rewriter-training",
                    {"train": [row]},
                    {"train": [guide]},
                    "Title",
                    "Desc",
                )
            ],
            b"License",
            b"Source",
        )

    uploaded = {}

    class FakeApi:
        def repo_info(self, repo, repo_type=None):
            return SimpleNamespace(sha="oldsha", private=False)

        def list_repo_files(self, repo, repo_type=None, revision=None):
            return sorted(uploaded.get(repo, ["README.md"]))

        def create_commit(self, repo_id, repo_type, parent_commit, commit_message, operations):
            uploaded[repo_id] = {}
            for op in operations:
                if hasattr(op, "path_or_fileobj"):
                    uploaded[repo_id][op.path_in_repo] = op.path_or_fileobj
            return SimpleNamespace(oid="newsha")

        def update_repo_settings(self, repo_id, repo_type, private):
            pass

    def fake_download(repo, name, repo_type=None, revision=None):
        p = tmp_path / "dl" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(uploaded[repo][name])
        return str(p)

    monkeypatch.setattr(hf_data, "_sources", fake_sources)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    out_dir = tmp_path / "custom-out"
    monkeypatch.setattr(
        "sys.argv",
        ["hf_data", "--owner", "custom-owner", "--output", str(out_dir), "--publish"],
    )
    hf_data.main()

    assert tracked_receipt.read_bytes() == tracked_before
    custom_receipt_path = out_dir / "public-parquet.json"
    assert custom_receipt_path.exists()
    data = json.loads(custom_receipt_path.read_text())
    assert data["repositories"][0]["repo_id"] == "custom-owner/finetuning-labs-rewriter-training"
    assert (
        data["repositories"][0]["files"]["DATA_LICENSE.md"]
        == hashlib.sha256(b"License").hexdigest()
    )


def test_hub_package_dataset_formats_card_inside_package_dataset(tmp_path):
    from finetuning_labs.rewriter import hub

    source = hub.Path(__file__).resolve().parents[1] / "data/rewriter/benchmark-draft"
    default_out = tmp_path / "default-pkg"
    custom_out = tmp_path / "custom-pkg"
    hub.package_dataset(source, default_out, code_commit="abc")
    hub.package_dataset(source, custom_out, code_commit="abc", owner="custom-owner")
    default_readme = (default_out / "README.md").read_text()
    custom_readme = (custom_out / "README.md").read_text()
    assert (
        "https://huggingface.co/datasets/jayshah5696/finetuning-labs-rewriter/blob/main/DATA_LICENSE.md"
        in default_readme
    )
    assert (
        "https://huggingface.co/datasets/custom-owner/finetuning-labs-rewriter/blob/main/DATA_LICENSE.md"
        in custom_readme
    )
