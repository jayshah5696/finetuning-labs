"""Training failures specified before implementation: loss on prompts/padding,
missing EOS, context truncation, reference leakage, and mismatched resume data.
"""

import pytest

from finetuning_labs.rewriter.colab_training import (
    bind_run,
    encode_example,
    inference_messages,
    pad_batch,
)


class Tokenizer:
    eos_token_id = 9

    def __init__(self, full=None):
        self.full = full or [1, 2, 3, 4, 9, 7]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3] if len(messages) == 2 else self.full

    def decode(self, ids, **kwargs):
        return "\n" if ids == [7] else "content"


GUIDE = {"system": "Rewrite only.", "user": "INPUT:\nhello", "assistant": "Hello."}


def test_only_assistant_and_its_eos_receive_loss():
    encoded = encode_example(GUIDE, Tokenizer())
    assert encoded["labels"] == [-100, -100, -100, 4, 9, -100]
    assert encoded["attention_mask"] == [1] * 6


@pytest.mark.parametrize("full", [[1, 2, 3, 4, 7], [1, 8, 3, 4, 9], [1, 2, 3, 9, 7]])
def test_missing_eos_changed_prefix_or_empty_assistant_fails(full):
    with pytest.raises(ValueError):
        encode_example(GUIDE, Tokenizer(full))


def test_context_is_rejected_instead_of_truncated():
    with pytest.raises(ValueError, match="context"):
        encode_example(GUIDE, Tokenizer(), max_length=5, max_new_tokens=2)


def test_padding_does_not_remove_the_real_eos_label():
    short = {"input_ids": [1, 9], "labels": [-100, 9], "attention_mask": [1, 1]}
    long = {"input_ids": [1, 2, 9], "labels": [-100, 2, 9], "attention_mask": [1, 1, 1]}
    batch = pad_batch([short, long], pad_id=9)
    assert batch["input_ids"][0] == [1, 9, 9]
    assert batch["labels"][0] == [-100, 9, -100]
    assert batch["attention_mask"][0] == [1, 1, 0]


def test_generation_never_receives_reference_answer():
    messages = inference_messages(GUIDE)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert GUIDE["assistant"] not in [m["content"] for m in messages]


def test_resume_refuses_a_changed_dataset_or_training_configuration(tmp_path):
    bind_run(tmp_path, {"revision": "one", "lr": 1e-5}, resume=False)
    with pytest.raises(ValueError):
        bind_run(tmp_path, {"revision": "two", "lr": 1e-5}, resume=True)
    with pytest.raises(FileExistsError):
        bind_run(tmp_path, {"revision": "one", "lr": 1e-5}, resume=False)
    bind_run(tmp_path, {"revision": "one", "lr": 1e-5}, resume=True)
