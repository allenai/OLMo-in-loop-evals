import pytest

from olmo_eval.tasks import build_task, list_tasks
from olmo_eval.tokenizer import HFTokenizer


@pytest.mark.parametrize("label", list_tasks())
def test_build_task(label: str):
    tokenizer = HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )
    task = build_task(label, tokenizer)
    assert len(task) >= 2
    assert task.max_sequence_length > 0
    instance1, instance2 = task[0], task[1]
    batch = task.collate_fn([instance1, instance2])
    assert isinstance(batch, dict)
    assert "input_ids" in batch
