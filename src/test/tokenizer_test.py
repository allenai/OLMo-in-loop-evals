from olmo_eval.tokenizer import HFTokenizer


def test_hf_tokenizer_from_hub():
    HFTokenizer(
        "allenai/dolma2-tokenizer",
        pad_token_id=0,
        eos_token_id=0,
    )


def test_hf_tokenizer_from_package_file():
    HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )
