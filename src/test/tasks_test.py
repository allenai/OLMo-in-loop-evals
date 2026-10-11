import pytest
import torch
from datasets import Dataset

from olmo_eval.metrics import ICLMetric
from olmo_eval.tasks import (
    ArcEasyCELoss,
    OEEvalTask,
    WinoGrande,
    build_task,
    list_tasks,
)
from olmo_eval.tokenizer import HFTokenizer


@pytest.mark.parametrize("answer_key", ["B", "2"])
def test_arc_loss_scores_the_gold_answer(answer_key):
    task = ArcEasyCELoss.__new__(ArcEasyCELoss)
    doc = {"choices": {"text": ["wrong", "correct"]}, "answerKey": answer_key}
    assert task.doc_to_continuations(doc) == [" correct"]
    assert task.doc_to_label(doc) == 0


def test_winogrande_scores_truncated_contexts_and_repeats_preparation(monkeypatch):
    doc = {
        "sentence": "A " * 32 + "_ said so.",
        "option1": "Alice",
        "option2": "Bob",
        "answer": "1",
    }
    monkeypatch.setattr("olmo_eval.tasks.load_hf_dataset", lambda *args: Dataset.from_list([doc]))
    tokenizer = HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )
    task = WinoGrande(tokenizer, model_ctx_len=8)
    for _ in range(2):
        assert len(task) == 2
        for sample in task.samples:
            assert sample["ctx_len"] == len(sample["query"]) - sample["cont_len"] + 1 > 0
        batch = task.collate_fn(task.samples)
        metric = ICLMetric(task.metric_type)
        metric.update(batch, lm_logits=torch.zeros(*batch["input_ids"].shape, tokenizer.vocab_size))
        assert all(torch.isfinite(value) for value in metric.compute().values())
        task.prep_examples()


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


@pytest.mark.parametrize("fast_mc", [False, True])
@pytest.mark.parametrize("infer_metric", [False, True])
def test_oe_task_preserves_multiple_datasets(monkeypatch, fast_mc, infer_metric):
    requests = [
        {
            "doc": {},
            "doc_id": doc_id,
            "idx": choice,
            "label": 0,
            "request_type": "loglikelihood",
            "request": {"context": "Question:", "continuation": continuation},
        }
        for doc_id in [0, 2]
        for choice, continuation in enumerate([" A", " B"])
    ]
    monkeypatch.setattr(
        "olmo_eval.tasks.load_oe_eval_requests",
        lambda path, name, split: (
            {"task_config": {"primary_metric": "acc_raw"}},
            [] if name == "empty" else requests,
        ),
    )
    tokenizer = HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )
    task = OEEvalTask(
        tokenizer,
        "test",
        dataset_name=["first", "empty", "second"],
        metric_type=None if infer_metric else "acc",
        fast_mc=fast_mc,
    )
    assert task.metric_type == "acc"
    if infer_metric:
        monkeypatch.setattr("olmo_eval.tasks.load_oe_eval_requests", lambda *args: (None, []))
        with pytest.raises(ValueError, match="pass metric_type explicitly"):
            OEEvalTask(tokenizer, "test")
    for _ in range(2):
        assert sorted(task[index]["doc_id"] for index in range(len(task))) == (
            [0, 2, 3, 5] if fast_mc else [0, 0, 2, 2, 3, 3, 5, 5]
        )
        assert task.collate_fn([task[0], task[-1]])["doc_id"].tolist() == [0, 5]
        task.prep_examples()
