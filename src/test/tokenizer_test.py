from types import SimpleNamespace

import pytest

from olmo_eval.tokenizer import HFTokenizer


def test_hf_tokenizer_from_hub():
    HFTokenizer(
        "allenai/dolma2-tokenizer",
        pad_token_id=0,
        eos_token_id=0,
    )


def test_hf_tokenizer_from_package_file():
    tokenizer = HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )
    inputs = ["hello world", "a second sentence"]
    encoded = tokenizer.encode_batch(inputs, add_special_tokens=False)
    assert encoded == [tokenizer.encode(text, add_special_tokens=False) for text in inputs]
    assert tokenizer.decode_batch(encoded) == inputs
    assert tokenizer.encode_batch(inputs) == [ids + [tokenizer.eos_token_id] for ids in encoded]


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_hf_tokenizer_encoding_matches_upstream(fast, add_special_tokens, tmp_path):
    tokenizer = HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=0,
    )
    base = tokenizer._base_tokenizer
    path = tmp_path / "tokenizer.json"
    base.save(str(path))
    local = HFTokenizer(str(path), pad_token_id=0, eos_token_id=0, bos_token_id=0)
    if not fast:
        local._base_tokenizer = SimpleNamespace(encode_batch=base.encode_batch)
    inputs = ["", "hello world", " café 中文 😀", "\n\t spaces ", "<|endoftext|>"]
    expected = [
        encoding.ids
        for encoding in base.encode_batch(inputs, add_special_tokens=add_special_tokens)
    ]
    if add_special_tokens:
        expected = [tokenizer._add_special_tokens(ids) for ids in expected]
    assert local.encode_batch(inputs, add_special_tokens=add_special_tokens) == expected
    assert local.encode_batch([], add_special_tokens=add_special_tokens) == []
