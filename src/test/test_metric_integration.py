"""
Metric Integration Tests (Mock-Based)

Full-scale integration tests for evaluation task metrics using mocked logits
(no real model required). Tests the complete pipeline: task loading → batching
→ metric computation without expensive model inference.

Task Categories:
- Multiple Choice (MC/RC) Tasks: Tasks with multiple answer options per question.
  These compute accuracy metrics (acc, len_norm) and BPB for the gold answer.
  Example: arc_challenge (has options A/B/C/D, computes accuracy + gold BPB)

- Generative BPB Tasks: Tasks with open-ended generation where only the gold
  answer continuation is evaluated. These compute only BPB metrics.
  Example: minerva_math_500 (free-form math answers, only BPB)

NOTE: These tests are marked as 'slow' and excluded from regular CI.
Run them manually with: pytest src/test/test_metric_integration.py -v
Or include slow tests in CI with: pytest -m "slow" src/test/
"""

from typing import Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

from olmo_eval import ICLMetric, build_task, list_tasks
from olmo_eval.tokenizer import HFTokenizer

# Mark all tests in this module as slow (excluded from CI by default)
pytestmark = pytest.mark.slow

# =====================================================
# TASK CONFIGURATION
# =====================================================

# Representative tasks for integration testing (limited set for speed)
# Full task coverage can be achieved by expanding these lists.

# -----------------------------------------------------
# Multiple Choice / Reading Comprehension Tasks
# These tasks have multiple answer options and compute accuracy metrics
# -----------------------------------------------------

# MC tasks with len_norm metric (length-normalized accuracy)
MC_LEN_NORM_TASKS = [
    "arc_challenge",
]

# MC tasks with acc metric (raw accuracy)
MC_ACC_TASKS = [
    "medmcqa_rc",
]

# All multiple choice tasks
MC_TASKS = MC_LEN_NORM_TASKS + MC_ACC_TASKS

# -----------------------------------------------------
# Generative BPB Tasks
# These tasks have only one continuation (gold answer) and compute BPB only
# -----------------------------------------------------

# BPB tasks from QA datasets (gold answer completion)
BPB_QA_TASKS = [
    "squad_bpb_5shot",
]

# BPB tasks from language modeling
BPB_LM_TASKS = [
    "lambada_bpb",
]

# BPB tasks from math/code generation
BPB_GENERATION_TASKS = [
    "minerva_math_500_gold_bpb_0shot",
]

# All generative BPB tasks
BPB_TASKS = BPB_QA_TASKS + BPB_LM_TASKS + BPB_GENERATION_TASKS

# -----------------------------------------------------
# Expected Metric Configuration
# -----------------------------------------------------

METRIC_TYPE_BY_TASK = {
    # MC len_norm tasks
    "arc_challenge": "len_norm",
    # MC acc tasks
    "medmcqa_rc": "acc",
    # BPB tasks
    "squad_bpb_5shot": "bpb",
    "lambada_bpb": "bpb",
    "minerva_math_500_gold_bpb_0shot": "bpb",
}

# Expected metric keys by metric type
EXPECTED_METRIC_KEYS = {
    "bpb": {"bpb_v1", "bpb_v2"},
    "acc": {
        "acc_v1",
        "acc_v2",
        "bpb_v1",
        "bpb_v2",
        "ce_loss_v1",
        "ce_loss_v2",
        "soft_v1",
        "soft_v2",
        "soft_log_v1",
        "soft_log_v2",
    },
    "len_norm": {
        "len_norm_v1",
        "len_norm_v2",
        "bpb_v1",
        "bpb_v2",
        "ce_loss_v1",
        "ce_loss_v2",
        "soft_v1",
        "soft_v2",
        "soft_log_v1",
        "soft_log_v2",
    },
}


# =====================================================
# FIXTURES
# =====================================================


@pytest.fixture(scope="module")
def tokenizer():
    """Create a tokenizer for testing."""
    return HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )


@pytest.fixture(scope="module")
def vocab_size(tokenizer):
    """Get vocabulary size."""
    return tokenizer.vocab_size


# =====================================================
# MOCK LOGIT GENERATION FUNCTIONS
# =====================================================


def make_perfect_logits(batch: Dict[str, torch.Tensor], vocab_size: int) -> torch.Tensor:
    """
    Generate logits that perfectly predict the continuation tokens for ALL samples.
    This should result in BPB ≈ 0.

    For accuracy tasks, this means all answers have the same log-likelihood,
    so accuracy won't necessarily be 1.0. Use make_correct_answer_logits for
    testing accuracy = 1.0.
    """
    batch_size, seq_len = batch["input_ids"].shape
    logits = torch.full((batch_size, seq_len, vocab_size), -100.0)

    for i in range(batch_size):
        ctx_len = batch["ctx_len"][i].item()
        cont_len = batch["cont_len"][i].item()
        cont_tokens = batch["continuation"][i][:cont_len]

        for j in range(cont_len):
            logit_pos = ctx_len - 1 + j
            if logit_pos < seq_len:
                target_token = cont_tokens[j].item()
                logits[i, logit_pos, target_token] = 0.0

    return logits


def make_correct_answer_logits(batch: Dict[str, torch.Tensor], vocab_size: int) -> torch.Tensor:
    """
    Generate logits that give high probability to the CORRECT answer's continuation
    and low probability to wrong answers. This should result in accuracy = 1.0.

    For accuracy tasks, the metric uses argmax over log-likelihoods to pick the answer.
    We give the correct answer a large positive boost so it wins the argmax.
    """
    batch_size, seq_len = batch["input_ids"].shape
    logits = torch.zeros((batch_size, seq_len, vocab_size))

    for i in range(batch_size):
        cont_id = batch["cont_id"][i].item()
        label_id = batch["label_id"][i].item()
        ctx_len = batch["ctx_len"][i].item()
        cont_len = batch["cont_len"][i].item()
        cont_tokens = batch["continuation"][i][:cont_len]

        if cont_id == label_id:
            for j in range(cont_len):
                logit_pos = ctx_len - 1 + j
                if logit_pos < seq_len:
                    target_token = cont_tokens[j].item()
                    logits[i, logit_pos, target_token] = 100.0
        else:
            for j in range(cont_len):
                logit_pos = ctx_len - 1 + j
                if logit_pos < seq_len:
                    target_token = cont_tokens[j].item()
                    logits[i, logit_pos, target_token] = -100.0

    return logits


def make_random_logits(batch: Dict[str, torch.Tensor], vocab_size: int) -> torch.Tensor:
    """
    Generate uniform random logits.
    This should result in BPB ≈ log2(vocab_size) and accuracy ≈ 1/num_choices.
    """
    batch_size, seq_len = batch["input_ids"].shape
    return torch.zeros((batch_size, seq_len, vocab_size))


def make_noisy_logits(
    batch: Dict[str, torch.Tensor], vocab_size: int, correct_boost: float = 5.0
) -> torch.Tensor:
    """
    Generate partially correct logits with noise.
    The correct continuation token gets a boost but isn't perfectly predicted.
    """
    batch_size, seq_len = batch["input_ids"].shape
    logits = torch.randn((batch_size, seq_len, vocab_size))

    for i in range(batch_size):
        ctx_len = batch["ctx_len"][i].item()
        cont_len = batch["cont_len"][i].item()
        cont_tokens = batch["continuation"][i][:cont_len]

        for j in range(cont_len):
            logit_pos = ctx_len - 1 + j
            if logit_pos < seq_len:
                target_token = cont_tokens[j].item()
                logits[i, logit_pos, target_token] += correct_boost

    return logits


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def validate_metrics_finite(results: Dict[str, torch.Tensor], task_name: str):
    """Validate that all metric values are finite (no NaN or inf)."""
    for key, value in results.items():
        assert torch.isfinite(value), f"Task {task_name}: {key} is not finite: {value}"


def validate_bpb_range(
    bpb: float, task_name: str, scenario: str, min_val: float = 0.0, max_val: float = 20.0
):
    """Validate BPB is in expected range."""
    assert bpb >= min_val, f"Task {task_name} ({scenario}): BPB {bpb} < {min_val}"
    assert bpb <= max_val, f"Task {task_name} ({scenario}): BPB {bpb} > {max_val}"


def validate_accuracy_range(acc: float, task_name: str, scenario: str, expected_acc: float):
    """Validate accuracy is close to expected value."""
    assert abs(acc - expected_acc) < 0.01, (
        f"Task {task_name} ({scenario}): accuracy {acc} != expected {expected_acc}"
    )


def get_complete_document_samples(task, num_docs: int = 5):
    """
    Get samples for complete documents (all continuations included).

    For multiple-choice tasks, each document has multiple continuations
    and the metric needs ALL of them to compute accuracy properly.
    """
    from collections import defaultdict

    docs: Dict[int, List] = defaultdict(list)
    for sample in task.samples:
        docs[sample["doc_id"]].append(sample)

    complete_samples = []
    for doc_id in sorted(docs.keys())[:num_docs]:
        complete_samples.extend(docs[doc_id])

    return complete_samples


# =====================================================
# TEST 1: Task Loading & Structure
# =====================================================


class TestTaskLoadingAndStructure:
    """Test that all tasks can be loaded and have the expected structure."""

    @pytest.mark.parametrize("task_name", MC_TASKS + BPB_TASKS)
    def test_task_exists_in_registry(self, task_name: str):
        """Verify all tasks exist in the registry."""
        all_tasks = list_tasks()
        assert task_name in all_tasks, f"Task {task_name} not found in registry"

    @pytest.mark.parametrize("task_name", MC_TASKS + BPB_TASKS)
    def test_task_has_correct_metric_type(self, task_name: str, tokenizer):
        """Verify tasks have the expected metric type."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        expected = METRIC_TYPE_BY_TASK.get(task_name)
        assert task.metric_type == expected, (
            f"Task {task_name}: metric_type {task.metric_type} != expected {expected}"
        )

    @pytest.mark.parametrize("task_name", MC_TASKS + BPB_TASKS)
    def test_task_sample_structure(self, task_name: str, tokenizer):
        """Verify task samples have required keys."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        assert len(task) > 0, f"Task {task_name} has no samples"

        required_keys = {
            "doc_id",
            "cont_id",
            "ctx",
            "continuation",
            "ctx_len",
            "cont_len",
            "cont_byte_len",
            "label_id",
            "query",
        }
        sample = task[0]
        for key in required_keys:
            assert key in sample, f"Task {task_name}: sample missing key '{key}'"

    @pytest.mark.parametrize("task_name", MC_TASKS + BPB_TASKS)
    def test_task_collate_fn(self, task_name: str, tokenizer):
        """Verify collate_fn produces valid batches."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        batch = task.collate_fn([task[0], task[1]] if len(task) > 1 else [task[0]])

        assert "input_ids" in batch, f"Task {task_name}: batch missing 'input_ids'"
        assert "doc_id" in batch, f"Task {task_name}: batch missing 'doc_id'"
        assert "cont_id" in batch, f"Task {task_name}: batch missing 'cont_id'"
        assert "label_id" in batch, f"Task {task_name}: batch missing 'label_id'"
        assert "continuation" in batch, f"Task {task_name}: batch missing 'continuation'"
        assert "cont_byte_len" in batch, f"Task {task_name}: batch missing 'cont_byte_len'"


# =====================================================
# TEST 2: Generative BPB Tasks - Metric Computation
# =====================================================


class TestGenerativeBPBTasks:
    """Test BPB metric computation for generative (non-MC) tasks."""

    @pytest.mark.parametrize("task_name", BPB_TASKS)
    def test_perfect_logits_produce_low_bpb(self, task_name: str, tokenizer, vocab_size):
        """Perfect logits should produce BPB ≈ 0."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="bpb")

        dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            logits = make_perfect_logits(batch, vocab_size)
            metric.update(batch, lm_logits=logits)

        results = metric.compute()

        validate_metrics_finite(results, task_name)
        assert "bpb_v1" in results, f"Task {task_name}: missing bpb_v1"
        assert "bpb_v2" in results, f"Task {task_name}: missing bpb_v2"

        bpb_v1 = results["bpb_v1"].item()
        bpb_v2 = results["bpb_v2"].item()

        validate_bpb_range(bpb_v1, task_name, "perfect", min_val=0.0, max_val=0.1)
        validate_bpb_range(bpb_v2, task_name, "perfect", min_val=0.0, max_val=0.1)

    @pytest.mark.parametrize("task_name", BPB_TASKS)
    def test_random_logits_produce_high_bpb(self, task_name: str, tokenizer, vocab_size):
        """Random logits should produce high BPB (much higher than perfect logits)."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="bpb")

        dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            logits = make_random_logits(batch, vocab_size)
            metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        bpb_v1 = results["bpb_v1"].item()
        bpb_v2 = results["bpb_v2"].item()

        validate_bpb_range(bpb_v1, task_name, "random", min_val=1.0, max_val=20.0)
        validate_bpb_range(bpb_v2, task_name, "random", min_val=1.0, max_val=20.0)

    @pytest.mark.parametrize("task_name", BPB_TASKS)
    def test_noisy_logits_produce_middle_bpb(self, task_name: str, tokenizer, vocab_size):
        """Noisy logits should produce BPB in middle range."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="bpb")

        dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            logits = make_noisy_logits(batch, vocab_size, correct_boost=5.0)
            metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        bpb_v1 = results["bpb_v1"].item()
        bpb_v2 = results["bpb_v2"].item()

        validate_bpb_range(bpb_v1, task_name, "noisy", min_val=0.5, max_val=15.0)
        validate_bpb_range(bpb_v2, task_name, "noisy", min_val=0.5, max_val=15.0)

    @pytest.mark.parametrize("task_name", BPB_TASKS)
    def test_bpb_task_returns_expected_keys(self, task_name: str, tokenizer, vocab_size):
        """BPB tasks should return all expected metric keys."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="bpb")

        dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
        batch = next(iter(dataloader))
        logits = make_perfect_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()

        expected_keys = EXPECTED_METRIC_KEYS["bpb"]
        actual_keys = set(results.keys())
        missing_keys = expected_keys - actual_keys
        assert not missing_keys, f"Task {task_name}: missing metric keys {missing_keys}"


# =====================================================
# TEST 3: Multiple Choice Tasks - Accuracy Metrics
# =====================================================


class TestMultipleChoiceAccuracyTasks:
    """Test accuracy metric computation for MC tasks with acc metric type."""

    @pytest.mark.parametrize("task_name", MC_ACC_TASKS)
    def test_correct_answer_logits_produce_full_accuracy(
        self, task_name: str, tokenizer, vocab_size
    ):
        """Logits favoring correct answer should produce accuracy = 1.0."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="acc")

        samples = get_complete_document_samples(task, num_docs=5)
        batch = task.collate_fn(samples)
        logits = make_correct_answer_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        assert "acc_v1" in results, f"Task {task_name}: missing acc_v1"
        assert "acc_v2" in results, f"Task {task_name}: missing acc_v2"

        acc_v1 = results["acc_v1"].item()
        acc_v2 = results["acc_v2"].item()

        validate_accuracy_range(acc_v1, task_name, "correct_answer", expected_acc=1.0)
        validate_accuracy_range(acc_v2, task_name, "correct_answer", expected_acc=1.0)

    @pytest.mark.parametrize("task_name", MC_ACC_TASKS)
    def test_random_logits_produce_valid_accuracy(self, task_name: str, tokenizer, vocab_size):
        """Random logits should produce accuracy in valid range [0, 1]."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="acc")

        samples = get_complete_document_samples(task, num_docs=10)
        batch = task.collate_fn(samples)
        logits = make_random_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        acc_v1 = results["acc_v1"].item()
        assert 0.0 <= acc_v1 <= 1.0, f"Task {task_name}: accuracy {acc_v1} out of bounds"

    @pytest.mark.parametrize("task_name", MC_ACC_TASKS)
    def test_acc_task_returns_expected_keys(self, task_name: str, tokenizer, vocab_size):
        """Acc tasks should return all expected metric keys."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="acc")

        samples = get_complete_document_samples(task, num_docs=3)
        batch = task.collate_fn(samples)
        logits = make_correct_answer_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()

        expected_keys = EXPECTED_METRIC_KEYS["acc"]
        actual_keys = set(results.keys())
        missing_keys = expected_keys - actual_keys
        assert not missing_keys, f"Task {task_name}: missing metric keys {missing_keys}"


# =====================================================
# TEST 4: Multiple Choice Tasks - Len-Norm Metrics
# =====================================================


class TestMultipleChoiceLenNormTasks:
    """Test len_norm metric computation for MC tasks."""

    @pytest.mark.parametrize("task_name", MC_LEN_NORM_TASKS)
    def test_correct_answer_logits_produce_full_accuracy(
        self, task_name: str, tokenizer, vocab_size
    ):
        """Logits favoring correct answer should produce len_norm accuracy = 1.0."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="len_norm")

        samples = get_complete_document_samples(task, num_docs=5)
        batch = task.collate_fn(samples)
        logits = make_correct_answer_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        assert "len_norm_v1" in results, f"Task {task_name}: missing len_norm_v1"
        assert "len_norm_v2" in results, f"Task {task_name}: missing len_norm_v2"

        acc_v1 = results["len_norm_v1"].item()
        acc_v2 = results["len_norm_v2"].item()

        validate_accuracy_range(acc_v1, task_name, "correct_answer", expected_acc=1.0)
        validate_accuracy_range(acc_v2, task_name, "correct_answer", expected_acc=1.0)

    @pytest.mark.parametrize("task_name", MC_LEN_NORM_TASKS)
    def test_random_logits_produce_valid_accuracy(self, task_name: str, tokenizer, vocab_size):
        """Random logits should produce accuracy in valid range [0, 1]."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="len_norm")

        samples = get_complete_document_samples(task, num_docs=10)
        batch = task.collate_fn(samples)
        logits = make_random_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        acc_v1 = results["len_norm_v1"].item()
        assert 0.0 <= acc_v1 <= 1.0, f"Task {task_name}: accuracy {acc_v1} out of bounds"

    @pytest.mark.parametrize("task_name", MC_LEN_NORM_TASKS)
    def test_len_norm_task_returns_expected_keys(self, task_name: str, tokenizer, vocab_size):
        """Len-norm tasks should return all expected metric keys."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="len_norm")

        samples = get_complete_document_samples(task, num_docs=3)
        batch = task.collate_fn(samples)
        logits = make_correct_answer_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()

        expected_keys = EXPECTED_METRIC_KEYS["len_norm"]
        actual_keys = set(results.keys())
        missing_keys = expected_keys - actual_keys
        assert not missing_keys, f"Task {task_name}: missing metric keys {missing_keys}"


# =====================================================
# TEST 5: Sanity Checks (All Tasks)
# =====================================================


class TestSanityChecks:
    """Cross-cutting sanity checks for all tasks."""

    @pytest.mark.parametrize("task_name", MC_TASKS + BPB_TASKS)
    def test_no_nan_inf_in_perfect_scenario(self, task_name: str, tokenizer, vocab_size):
        """Perfect logits should never produce NaN or inf."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric_type = task.metric_type
        metric = ICLMetric(metric_type=metric_type)

        if metric_type in ["acc", "len_norm"]:
            samples = get_complete_document_samples(task, num_docs=5)
            batch = task.collate_fn(samples)
            logits = make_correct_answer_logits(batch, vocab_size)
        else:
            dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
            batch = next(iter(dataloader))
            logits = make_perfect_logits(batch, vocab_size)

        metric.update(batch, lm_logits=logits)
        results = metric.compute()
        validate_metrics_finite(results, task_name)

    @pytest.mark.parametrize("task_name", MC_TASKS + BPB_TASKS)
    def test_no_nan_inf_in_random_scenario(self, task_name: str, tokenizer, vocab_size):
        """Random logits should never produce NaN or inf."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric_type = task.metric_type
        metric = ICLMetric(metric_type=metric_type)

        if metric_type in ["acc", "len_norm"]:
            samples = get_complete_document_samples(task, num_docs=5)
            batch = task.collate_fn(samples)
        else:
            dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
            batch = next(iter(dataloader))

        logits = make_random_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)
        results = metric.compute()
        validate_metrics_finite(results, task_name)

    @pytest.mark.parametrize("task_name", MC_TASKS + BPB_TASKS)
    def test_bpb_is_non_negative(self, task_name: str, tokenizer, vocab_size):
        """BPB should always be non-negative."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric_type = task.metric_type
        metric = ICLMetric(metric_type=metric_type)

        if metric_type in ["acc", "len_norm"]:
            samples = get_complete_document_samples(task, num_docs=5)
            batch = task.collate_fn(samples)
        else:
            dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
            batch = next(iter(dataloader))

        logits = make_noisy_logits(batch, vocab_size, correct_boost=3.0)
        metric.update(batch, lm_logits=logits)
        results = metric.compute()

        if "bpb_v1" in results:
            assert results["bpb_v1"].item() >= 0, f"Task {task_name}: BPB v1 is negative"
        if "bpb_v2" in results:
            assert results["bpb_v2"].item() >= 0, f"Task {task_name}: BPB v2 is negative"


# =====================================================
# TEST 6: Task Structure Validation
# =====================================================


class TestTaskStructureValidation:
    """Validate task-specific structural requirements."""

    @pytest.mark.parametrize("task_name", BPB_QA_TASKS)
    def test_bpb_qa_tasks_have_one_sample_per_doc(self, task_name: str, tokenizer):
        """
        BPB QA tasks should have exactly one sample per doc_id
        (the gold answer continuation).
        """
        task = build_task(task_name, tokenizer, model_ctx_len=512)

        from collections import Counter

        doc_counts = Counter(s["doc_id"] for s in task.samples)

        for doc_id, count in list(doc_counts.items())[:20]:
            assert count == 1, f"Task {task_name}: doc_id={doc_id} has {count} samples, expected 1"

    @pytest.mark.parametrize("task_name", MC_TASKS)
    def test_mc_tasks_have_multiple_samples_per_doc(self, task_name: str, tokenizer):
        """
        MC tasks should have multiple samples per doc_id
        (one per answer option).
        """
        task = build_task(task_name, tokenizer, model_ctx_len=512)

        from collections import Counter

        doc_counts = Counter(s["doc_id"] for s in task.samples)

        # Check first few docs have > 1 sample (multiple choice)
        for doc_id in list(sorted(doc_counts.keys()))[:5]:
            count = doc_counts[doc_id]
            assert count > 1, (
                f"Task {task_name}: doc_id={doc_id} has only {count} sample, expected >1"
            )

    @pytest.mark.parametrize("task_name", BPB_QA_TASKS)
    def test_bpb_qa_tasks_have_normalized_ids(self, task_name: str, tokenizer):
        """BPB QA tasks should have label_id=0 and cont_id=0."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)

        for i in range(min(10, len(task))):
            sample = task[i]
            assert sample["label_id"] == 0, (
                f"Task {task_name}: sample {i} has label_id={sample['label_id']}, expected 0"
            )
            assert sample["cont_id"] == 0, (
                f"Task {task_name}: sample {i} has cont_id={sample['cont_id']}, expected 0"
            )


# =====================================================
# STANDALONE EXECUTION
# =====================================================


def run_verbose_tests():
    """Run tests with verbose output when executed as a script."""
    print("=" * 70)
    print("Metric Integration Tests - Mock-Based Evaluation Task Validation")
    print("=" * 70)

    tokenizer = HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    # Test Generative BPB Tasks
    print("\n" + "=" * 70)
    print("Generative BPB Tasks (no multiple choice)")
    print("=" * 70)

    for task_name in BPB_TASKS:
        print(f"\n--- {task_name} ---")
        try:
            task = build_task(task_name, tokenizer, model_ctx_len=512)
            print(f"  Samples: {len(task)}")
            print(f"  Metric type: {task.metric_type}")

            metric = ICLMetric(metric_type="bpb")
            dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)

            for i, batch in enumerate(dataloader):
                if i >= 3:
                    break
                logits = make_perfect_logits(batch, vocab_size)
                metric.update(batch, lm_logits=logits)

            results = metric.compute()
            print(f"  Perfect BPB v1: {results['bpb_v1'].item():.4f}")
            print(f"  Perfect BPB v2: {results['bpb_v2'].item():.4f}")

            metric.reset()
            for i, batch in enumerate(dataloader):
                if i >= 3:
                    break
                logits = make_random_logits(batch, vocab_size)
                metric.update(batch, lm_logits=logits)

            results = metric.compute()
            print(f"  Random BPB v1: {results['bpb_v1'].item():.4f}")
            print(f"  Random BPB v2: {results['bpb_v2'].item():.4f}")
            print("  [PASS]")

        except Exception as e:
            print(f"  [FAIL] {e}")

    # Test MC Acc Tasks
    print("\n" + "=" * 70)
    print("Multiple Choice Tasks (acc metric)")
    print("=" * 70)

    for task_name in MC_ACC_TASKS:
        print(f"\n--- {task_name} ---")
        try:
            task = build_task(task_name, tokenizer, model_ctx_len=512)
            print(f"  Samples: {len(task)}")
            print(f"  Metric type: {task.metric_type}")

            metric = ICLMetric(metric_type="acc")
            samples = get_complete_document_samples(task, num_docs=5)
            batch = task.collate_fn(samples)
            logits = make_correct_answer_logits(batch, vocab_size)
            metric.update(batch, lm_logits=logits)

            results = metric.compute()
            print(f"  Correct Answer Acc v1: {results['acc_v1'].item():.4f}")
            print(f"  Gold BPB v1: {results['bpb_v1'].item():.4f}")
            print("  [PASS]")

        except Exception as e:
            print(f"  [FAIL] {e}")

    # Test MC Len-Norm Tasks
    print("\n" + "=" * 70)
    print("Multiple Choice Tasks (len_norm metric)")
    print("=" * 70)

    for task_name in MC_LEN_NORM_TASKS:
        print(f"\n--- {task_name} ---")
        try:
            task = build_task(task_name, tokenizer, model_ctx_len=512)
            print(f"  Samples: {len(task)}")
            print(f"  Metric type: {task.metric_type}")

            metric = ICLMetric(metric_type="len_norm")
            samples = get_complete_document_samples(task, num_docs=5)
            batch = task.collate_fn(samples)
            logits = make_correct_answer_logits(batch, vocab_size)
            metric.update(batch, lm_logits=logits)

            results = metric.compute()
            print(f"  Correct Answer Len-Norm v1: {results['len_norm_v1'].item():.4f}")
            print(f"  Gold BPB v1: {results['bpb_v1'].item():.4f}")
            print("  [PASS]")

        except Exception as e:
            print(f"  [FAIL] {e}")

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_verbose_tests()
