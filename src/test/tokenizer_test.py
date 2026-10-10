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
