from typing import Any

import pytest
import torch
from datasets import Dataset

from olmo_eval.metrics import ICLMetric
from olmo_eval.tasks import (
    ArcEasy,
    ArcEasyCELoss,
    OEEvalTask,
    WinoGrande,
    build_task,
    list_tasks,
)
from olmo_eval.tokenizer import HFTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )


@pytest.mark.parametrize("task_class", [ArcEasy, OEEvalTask])
@pytest.mark.parametrize("metric_type", ["acc", "len_norm", "pmi_dc"])
def test_fast_choices_match_ordinary_scoring(monkeypatch, tokenizer, task_class, metric_type):
    docs: list[dict[str, Any]] = [
        {"question": "Q", "choices": {"text": ["cat", "café"]}, "answerKey": "B"},
        {"question": "Q", "choices": {"text": ["cat"]}, "answerKey": "A"},
    ]
    requests = [
        {
            "doc": doc,
            "doc_id": doc_id,
            "idx": choice,
            "label": 1 if doc_id == 0 else 0,
            "request_type": "loglikelihood",
            "request": {"context": "Question: Q\nAnswer:", "continuation": " " + text},
        }
        for doc_id, doc in enumerate(docs)
        for choice, text in enumerate(doc["choices"]["text"])
    ]
    monkeypatch.setattr("olmo_eval.tasks.load_hf_dataset", lambda *args: Dataset.from_list(docs))
    monkeypatch.setattr("olmo_eval.tasks.load_oe_eval_requests", lambda *args: (None, requests))
    results = []
    for fast_mc in [False, True]:
        task = task_class(tokenizer, "test", fast_mc=fast_mc, metric_type=metric_type)
        batch = task.collate_fn(task.samples)
        assert batch["cont_str_len"].ndim == 1
        assert all(isinstance(value, torch.Tensor) for value in batch.values())
        logits = torch.full((*batch["input_ids"].shape, tokenizer.vocab_size), -1e8)
        cat, cafe = [task.token_encode(text)[0] for text in [" cat", " café"]]
        logits[:, :, cat] = torch.tensor(0.52).log()
        logits[:, :, cafe] = torch.tensor(0.48).log()
        dc_logits = torch.full((*batch["dc_input_ids"].shape, tokenizer.vocab_size), -1e8)
        dc_logits[:, :, [cat, cafe]] = torch.tensor(0.5).log()
        metric = ICLMetric(metric_type)
        metric.update(batch, lm_logits=logits, dc_lm_logits=dc_logits)
        results.append(metric.compute())
    assert results[0].keys() == results[1].keys()
    assert all(torch.allclose(value, results[1][key]) for key, value in results[0].items())


@pytest.mark.parametrize("source", ["hf", "oe", "wino"])
@pytest.mark.parametrize("bos_token_id", [None, 7])
def test_empty_context_uses_a_prefix_token(monkeypatch, source, bos_token_id):
    class EmptyContextTask(ArcEasy):
        def doc_to_text(self, doc):
            return ""

    doc = {
        "question": "Q",
        "choices": {"text": ["cat"]},
        "answerKey": "A",
        "sentence": "_ cat",
        "option1": "",
        "option2": "",
        "answer": "1",
    }
    request = {
        "doc": doc,
        "doc_id": 0,
        "idx": 0,
        "label": 0,
        "request_type": "loglikelihood",
        "request": {"context": "", "continuation": " cat"},
    }
    monkeypatch.setattr("olmo_eval.tasks.load_hf_dataset", lambda *args: Dataset.from_list([doc]))
    monkeypatch.setattr("olmo_eval.tasks.load_oe_eval_requests", lambda *args: (None, [request]))
    tokenizer = HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=bos_token_id,
    )
    task_class = {"hf": EmptyContextTask, "oe": OEEvalTask, "wino": WinoGrande}[source]
    task = task_class(tokenizer, "test", metric_type="acc")
    assert all(sample["ctx"] == [7 if bos_token_id is not None else 0] for sample in task.samples)
    batch = task.collate_fn(task.samples)
    metric = ICLMetric("acc")
    metric.update(batch, lm_logits=torch.zeros(*batch["input_ids"].shape, tokenizer.vocab_size))
    assert all(torch.isfinite(value) for value in metric.compute().values())


@pytest.mark.parametrize("answer_key", ["B", "2"])
def test_arc_loss_scores_the_gold_answer(answer_key):
    task = ArcEasyCELoss.__new__(ArcEasyCELoss)
    doc = {"choices": {"text": ["wrong", "correct"]}, "answerKey": answer_key}
    assert task.doc_to_continuations(doc) == [" correct"]
    assert task.doc_to_label(doc) == 0


def test_winogrande_scores_truncated_contexts_and_repeats_preparation(monkeypatch, tokenizer):
    doc = {
        "sentence": "A " * 32 + "_ said so.",
        "option1": "Alice",
        "option2": "Bob",
        "answer": "1",
    }
    monkeypatch.setattr("olmo_eval.tasks.load_hf_dataset", lambda *args: Dataset.from_list([doc]))
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
def test_build_task(label: str, tokenizer):
    task = build_task(label, tokenizer)
    assert len(task) >= 2
    assert task.max_sequence_length > 0
    instance1, instance2 = task[0], task[1]
    batch = task.collate_fn([instance1, instance2])
    assert isinstance(batch, dict)
    assert "input_ids" in batch


@pytest.mark.parametrize("fast_mc", [False, True])
@pytest.mark.parametrize("infer_metric", [False, True])
def test_oe_task_preserves_multiple_datasets(monkeypatch, fast_mc, infer_metric, tokenizer):
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

        class DefaultMetricTask(OEEvalTask):
            metric_type = "acc"

        assert DefaultMetricTask(tokenizer, "test").metric_type == "acc"
        assert DefaultMetricTask(tokenizer, "test", metric_type="bpb").metric_type == "bpb"
    for _ in range(2):
        assert sorted(task[index]["doc_id"] for index in range(len(task))) == (
            [0, 2, 3, 5] if fast_mc else [0, 0, 2, 2, 3, 3, 5, 5]
        )
        assert task.collate_fn([task[0], task[-1]])["doc_id"].tolist() == [0, 5]
        if fast_mc:
            assert all(
                sample["choices"] == [task.token_encode(text)[0] for text in [" A", " B"]]
                for sample in task.samples
            )
        task.prep_examples()
