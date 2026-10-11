import pytest
import torch

from olmo_eval.metrics import ICLMetric


@pytest.mark.parametrize("fast_mc", [False, True])
def test_pmi_uses_log_probability_difference_and_raw_losses(fast_mc):
    batch = {
        key: torch.tensor([value, value])
        for key, value in {
            "doc_id": 0,
            "label_id": 0,
            "ctx_len": 1,
            "dc_len": 1,
            "cont_len": 1,
            "cont_str_len": 2,
            "cont_byte_len": 2,
            "cont_str_len_no_leading_space": 1,
            "cont_byte_len_no_leading_space": 1,
        }.items()
    }
    batch["cont_id"] = torch.tensor([0, 1])
    batch["continuation"] = torch.tensor([[1], [2]])
    logits = torch.tensor([[[0.2, 0.6, 0.2]]] * 2).log()
    dc_logits = torch.tensor([[[0.89, 0.01, 0.1]]] * 2).log()
    if fast_mc:
        batch = {key: value[:1] for key, value in batch.items()}
        batch["choice_ids"] = torch.tensor([[1, 2, -1]])
        logits, dc_logits = logits[:1], dc_logits[:1]
    metric = ICLMetric("pmi_dc")
    metric.update(batch, lm_logits=logits, dc_lm_logits=dc_logits)
    results = metric.compute()
    assert results["pmi_dc_v1"].item() == results["pmi_dc_v2"].item() == 1.0
    assert results["ce_loss_v1"].item() == pytest.approx(-torch.tensor(0.6).log().item())
    assert results["ce_loss_v2"].item() == pytest.approx(results["ce_loss_v1"].item() / 2)
    assert results["bpb_v2"].item() == pytest.approx(-torch.tensor(0.6).log2().item() / 2)


@pytest.mark.parametrize("metric_type", ["acc", "len_norm", "pmi_dc"])
def test_metric_accuracy_versions_use_their_own_predictions(metric_type):
    metric = ICLMetric(metric_type)
    metric.update({}, lm_logits=None)
    metric.labels = [(0, 0, 0)]
    for name in metric.metric_state:
        if name != "labels":
            score = 1.0 if name.endswith("_no_leading_space") else -1.0
            setattr(metric, name, [(0, 0, 0.0), (0, 1, score)])
    results = metric.compute()
    assert results[f"{metric_type}_v1"].item() == 0.0
    assert results[f"{metric_type}_v2"].item() == 1.0


@pytest.mark.parametrize("metric_type", ["acc", "f1", "len_norm", "pmi_dc", "ce_loss", "bpb"])
@pytest.mark.parametrize("fast_mc", [False, True])
def test_metric_reset_clears_cached_results(metric_type, fast_mc):
    batch = {
        "doc_id": torch.tensor([0, 0]),
        "cont_id": torch.tensor([0, 1]),
        "continuation": torch.tensor([[1], [2]]),
        "ctx_len": torch.tensor([1, 1]),
        "dc_len": torch.tensor([1, 1]),
        "cont_len": torch.tensor([1, 1]),
        "cont_str_len": torch.tensor([2, 2]),
        "cont_byte_len": torch.tensor([2, 2]),
        "cont_str_len_no_leading_space": torch.tensor([1, 1]),
        "cont_byte_len_no_leading_space": torch.tensor([1, 1]),
        "label_id": torch.tensor([0, 0]),
    }
    logits = torch.zeros(2, 1, 3)
    logits[0, 0, 1] = 10
    if fast_mc:
        batch = {key: value[:1] for key, value in batch.items()}
        batch["choice_ids"] = torch.tensor([[1, 2, -1]])
        logits = logits[:1]
    metric = ICLMetric(metric_type)
    metric.update({}, lm_logits=None)
    metric.update(batch, lm_logits=logits, dc_lm_logits=torch.zeros_like(logits))
    before_reset = metric.compute()
    assert all(torch.isfinite(value) for value in before_reset.values())

    metric.reset()
    assert metric.update_count == 0
    assert all(not state for state in metric.metric_state.values())
    logits[0, 0, 1] = -10
    metric.update(batch, lm_logits=logits, dc_lm_logits=torch.zeros_like(logits))
    after_reset = metric.compute()
    assert metric.update_count == 1
    assert any(not torch.equal(value, after_reset[key]) for key, value in before_reset.items())
    metric.reset()
    forwarded = metric(batch, lm_logits=logits, dc_lm_logits=torch.zeros_like(logits))
    assert all(torch.equal(value, forwarded[key]) for key, value in after_reset.items())
