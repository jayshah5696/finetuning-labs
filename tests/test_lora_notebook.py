"""Specify assistant loss and adapter-only upload before writing the LoRA helpers."""

import json

import pytest

from finetuning_labs.rewriter.lora_training import adapter_files, encode_sft, push_adapter


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
