"""Specify assistant loss and adapter-only upload before writing the LoRA helpers."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetuning_labs.rewriter.lora_training import adapter_files, encode_sft, push_adapter


@pytest.mark.parametrize("train_count", [1000, 1500, 1864])
def test_public_data_cell_runs_without_auth_environment(monkeypatch, train_count):
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks/rewriter_train_l4.ipynb").read_text()
    )
    source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "data")
    calls = []

    class Rows(list):
        column_names = ("system", "user", "assistant")

    class Dataset(dict):
        def map(self, *args, **kwargs):
            return self

    raw = Dataset(
        {
            s: Rows(
                [
                    {
                        "system": "Rewrite.",
                        "user": "MODE: /transcribe\nLANGUAGE: en\nINPUT:\nhello",
                        "assistant": "Hello.",
                    }
                ]
                * n
            )
            for s, n in {"train": train_count, "dev": 100, "validation": 84}.items()
        }
    )

    def public_load(*args, **kwargs):
        assert kwargs["token"] is False
        calls.append(args)
        return raw

    def public_tokenizer(*args, **kwargs):
        assert kwargs["token"] is False
        return SimpleNamespace(pad_token_id=0)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=public_load))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=public_tokenizer)),
    )
    scope = {
        "CONFIG": {
            "dataset_id": "public/data",
            "dataset_config": "sft",
            "dataset_revision": "data-rev",
            "model_id": "public/base",
            "model_revision": "base-rev",
            "max_length": 4096,
            "expected_split_counts": {"train": train_count, "dev": 100, "validation": 84},
        }
    }
    exec(compile(source, "notebook:data", "exec"), scope)  # noqa: S102 -- exercise trusted notebook cell without secrets
    assert calls == [("public/data", "sft")]
    assert scope["encoded"] is raw


@pytest.mark.parametrize("size", ["230m", "350m"])
def test_model_choice_switches_base_revision_and_destination_together(size):
    builder = runpy.run_path(str(Path(__file__).parents[1] / "scripts/build_lora_notebook.py"))
    counts = {"train": 1864, "dev": 89, "validation": 81}
    scope = {}
    exec(builder["configuration_source"]("b" * 40, counts, size), scope)  # noqa: S102 -- execute the trusted config cell
    cfg = scope["CONFIG"]
    assert cfg["model_id"] == f"LiquidAI/LFM2.5-{size.upper()}"
    assert cfg["hub_model_id"] == f"jayshah5696/lfm2.5-{size}-rewriter-lora"
    assert len(cfg["model_revision"]) == 40 and cfg["expected_split_counts"] == counts
    assert cfg["language"] == "en"


class Tokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3] if len(messages) == 2 else [1, 2, 3, 4, 9, 7]

    def decode(self, ids):
        return "\n" if ids == [7] else "text"


def test_prompt_is_masked_and_assistant_eos_is_supervised():
    row = {"system": "Rewrite.", "user": "hi", "assistant": "Hi."}
    result = encode_sft(row, Tokenizer(), max_length=10)
    assert result["labels"] == [-100, -100, -100, 4, 9, -100]
    with pytest.raises(ValueError, match="length"):
        encode_sft(row, Tokenizer(), max_length=5)


def adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "README.md").write_text("An adapter.")
    return tmp_path


def test_full_weights_or_source_code_cannot_enter_adapter_upload(tmp_path):
    path = adapter(tmp_path)
    assert set(adapter_files(path)) == {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    (path / "model.safetensors").write_bytes(b"base")
    with pytest.raises(ValueError, match="adapter"):
        adapter_files(path)
    (path / "model.safetensors").unlink()
    (path / "train.py").write_text("code")
    with pytest.raises(ValueError, match="adapter"):
        adapter_files(path)


def test_upload_targets_model_repo_with_only_adapter_artifacts(tmp_path):
    path = adapter(tmp_path)

    class Hub:
        def create_repo(self, **kwargs):
            assert kwargs == {
                "repo_id": "owner/rewriter-lora",
                "repo_type": "model",
                "private": False,
                "exist_ok": True,
            }

        def upload_folder(self, **kwargs):
            assert kwargs["repo_id"] == "owner/rewriter-lora"
            assert kwargs["repo_type"] == "model"
            assert set(kwargs["allow_patterns"]) == set(adapter_files(path))
            return "uploaded"

    assert push_adapter(path, "owner/rewriter-lora", private=False, api=Hub()) == "uploaded"
