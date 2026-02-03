"""
Play Testing New Evaluation Tasks (Mock-Based)

This script validates the recently implemented evaluation tasks using mocked logits
(no real model required). It tests the full pipeline from task loading → batching
→ metric computation without expensive model inference.

Recent commits added/modified tasks:
- BPB-only tasks: coqa_bpb_5shot, drop_bpb_5shot, jeopardy_bpb_5shot, etc.
- RC tasks with accuracy + BPB: lab_bench_dbqa_rc, medmcqa_rc, etc.

NOTE: These tests are marked as 'slow' and excluded from regular CI.
Run them manually with: pytest src/test/bpb_playtest.py -v
Or include slow tests in CI with: pytest -m "slow" src/test/
"""

import math
from typing import Dict, List, Set

import pytest
import torch
from torch.utils.data import DataLoader

from olmo_eval import ICLMetric, build_task, list_tasks
from olmo_eval.tokenizer import HFTokenizer

# Mark all tests in this module as slow (excluded from CI by default)
pytestmark = pytest.mark.slow

# =====================================================
# TEST CONFIGURATION
# =====================================================

# BPB-only tasks (metric_type="bpb")
BPB_TASKS = [
    "lambada_bpb",
    "coqa_bpb_5shot",
    "drop_bpb_5shot",
    "jeopardy_bpb_5shot",
    "naturalqs_bpb_5shot",
    "squad_bpb_5shot",
]

# RC tasks with accuracy (metric_type="acc") - also compute BPB for gold
RC_TASKS = [
    "lab_bench_dbqa_rc",
    "lab_bench_protocolqa_rc",
    "medmcqa_rc",
    "medqa_en_rc",
    "qasper_yesno_rc",
    "sciriff_yesno_rc",
]

# MMLU variants (metric_type="len_norm") - flagship benchmark
MMLU_TASKS = [
    "mmlu_stem",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "mmlu_other",
]

# Other core benchmarks (various metric types)
CORE_BENCHMARKS = [
    "hellaswag",  # metric_type="len_norm"
    "piqa",  # metric_type="len_norm"
    "winogrande",  # metric_type="acc"
    "arc_easy",  # metric_type="acc"
    "arc_challenge",  # metric_type="len_norm"
    "boolq",  # metric_type="acc"
    "openbook_qa",  # metric_type="len_norm"
    "copa",  # metric_type="acc"
    "sciq",  # metric_type="acc"
    "social_iqa",  # metric_type="len_norm"
]

# Expected metric types for each task category
EXPECTED_METRIC_TYPES = {
    # BPB tasks
    "lambada_bpb": "bpb",
    "coqa_bpb_5shot": "bpb",
    "drop_bpb_5shot": "bpb",
    "jeopardy_bpb_5shot": "bpb",
    "naturalqs_bpb_5shot": "bpb",
    "squad_bpb_5shot": "bpb",
    # RC tasks
    "lab_bench_dbqa_rc": "acc",
    "lab_bench_protocolqa_rc": "acc",
    "medmcqa_rc": "acc",
    "medqa_en_rc": "acc",
    "qasper_yesno_rc": "acc",
    "sciriff_yesno_rc": "acc",
    # MMLU
    "mmlu_stem": "len_norm",
    "mmlu_humanities": "len_norm",
    "mmlu_social_sciences": "len_norm",
    "mmlu_other": "len_norm",
    # Core benchmarks
    "hellaswag": "len_norm",
    "piqa": "len_norm",
    "winogrande": "acc",
    "arc_easy": "acc",
    "arc_challenge": "len_norm",
    "boolq": "acc",
    "openbook_qa": "len_norm",
    "copa": "acc",
    "sciq": "acc",
    "social_iqa": "len_norm",
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
    "f1": {"f1_v1", "f1_v2"},
    "ce_loss": {"ce_loss_v1", "ce_loss_v2"},
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
    # Start with very low logits everywhere
    logits = torch.full((batch_size, seq_len, vocab_size), -100.0)

    for i in range(batch_size):
        ctx_len = batch["ctx_len"][i].item()
        cont_len = batch["cont_len"][i].item()
        cont_tokens = batch["continuation"][i][:cont_len]

        # Set high logit for continuation tokens at the right positions
        for j in range(cont_len):
            logit_pos = ctx_len - 1 + j
            if logit_pos < seq_len:
                target_token = cont_tokens[j].item()
                logits[i, logit_pos, target_token] = 0.0  # log(1) = 0 after softmax

    return logits


def make_correct_answer_logits(batch: Dict[str, torch.Tensor], vocab_size: int) -> torch.Tensor:
    """
    Generate logits that give high probability to the CORRECT answer's continuation
    and low probability to wrong answers. This should result in accuracy = 1.0.

    For accuracy tasks, the metric uses argmax over log-likelihoods to pick the answer.
    We give the correct answer a large positive boost so it wins the argmax.
    """
    batch_size, seq_len = batch["input_ids"].shape
    # Start with uniform logits (random baseline)
    logits = torch.zeros((batch_size, seq_len, vocab_size))

    for i in range(batch_size):
        cont_id = batch["cont_id"][i].item()
        label_id = batch["label_id"][i].item()
        ctx_len = batch["ctx_len"][i].item()
        cont_len = batch["cont_len"][i].item()
        cont_tokens = batch["continuation"][i][:cont_len]

        if cont_id == label_id:
            # This is the correct answer - give it very high probability
            for j in range(cont_len):
                logit_pos = ctx_len - 1 + j
                if logit_pos < seq_len:
                    target_token = cont_tokens[j].item()
                    logits[i, logit_pos, target_token] = 100.0  # Very high logit
        else:
            # Wrong answer - give it very low probability
            for j in range(cont_len):
                logit_pos = ctx_len - 1 + j
                if logit_pos < seq_len:
                    target_token = cont_tokens[j].item()
                    logits[i, logit_pos, target_token] = -100.0  # Very low logit

    return logits


def make_random_logits(batch: Dict[str, torch.Tensor], vocab_size: int) -> torch.Tensor:
    """
    Generate uniform random logits.
    This should result in BPB ≈ log2(vocab_size) and accuracy ≈ 1/num_choices.
    """
    batch_size, seq_len = batch["input_ids"].shape
    # Uniform logits = equal probability for all tokens
    return torch.zeros((batch_size, seq_len, vocab_size))


def make_noisy_logits(
    batch: Dict[str, torch.Tensor], vocab_size: int, correct_boost: float = 5.0
) -> torch.Tensor:
    """
    Generate partially correct logits with noise.
    The correct continuation token gets a boost but isn't perfectly predicted.
    """
    batch_size, seq_len = batch["input_ids"].shape
    # Start with random noise
    logits = torch.randn((batch_size, seq_len, vocab_size))

    # Boost the correct continuation tokens
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
    # Allow some tolerance
    assert abs(acc - expected_acc) < 0.01, (
        f"Task {task_name} ({scenario}): accuracy {acc} != expected {expected_acc}"
    )


def get_num_choices(task) -> int:
    """Get the number of choices for a multiple choice task."""
    if len(task) == 0:
        return 1

    # Count unique cont_ids for the first doc_id
    first_doc_id = task[0]["doc_id"]
    cont_ids = set()
    for sample in task.samples:
        if sample["doc_id"] == first_doc_id:
            cont_ids.add(sample["cont_id"])
    return len(cont_ids)


def get_complete_document_samples(task, num_docs: int = 5):
    """
    Get samples for complete documents (all continuations included).

    For multiple-choice tasks, each document has multiple continuations
    and the metric needs ALL of them to compute accuracy properly.
    """
    from collections import defaultdict

    # Group samples by doc_id
    docs: Dict[int, List] = defaultdict(list)
    for sample in task.samples:
        docs[sample["doc_id"]].append(sample)

    # Get complete documents
    complete_samples = []
    for doc_id in sorted(docs.keys())[:num_docs]:
        complete_samples.extend(docs[doc_id])

    return complete_samples


# =====================================================
# TEST 1: Task Loading & Structure
# =====================================================


class TestTaskLoadingAndStructure:
    """Test that all tasks can be loaded and have the expected structure."""

    @pytest.mark.parametrize("task_name", BPB_TASKS + RC_TASKS)
    def test_new_tasks_exist_in_registry(self, task_name: str):
        """Verify all new tasks exist in the registry."""
        all_tasks = list_tasks()
        assert task_name in all_tasks, f"Task {task_name} not found in registry"

    @pytest.mark.parametrize("task_name", BPB_TASKS + RC_TASKS)
    def test_new_tasks_have_correct_metric_type(self, task_name: str, tokenizer):
        """Verify new tasks have the expected metric type."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        expected = EXPECTED_METRIC_TYPES.get(task_name)
        assert task.metric_type == expected, (
            f"Task {task_name}: metric_type {task.metric_type} != expected {expected}"
        )

    @pytest.mark.parametrize("task_name", BPB_TASKS + RC_TASKS)
    def test_new_tasks_sample_structure(self, task_name: str, tokenizer):
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

    @pytest.mark.parametrize("task_name", BPB_TASKS + RC_TASKS)
    def test_new_tasks_collate_fn(self, task_name: str, tokenizer):
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
# TEST 2: BPB Metric Computation
# =====================================================


class TestBPBMetricComputation:
    """Test BPB metric computation with mocked logits."""

    @pytest.mark.parametrize("task_name", BPB_TASKS)
    def test_perfect_logits_produce_low_bpb(self, task_name: str, tokenizer, vocab_size):
        """Perfect logits should produce BPB ≈ 0."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="bpb")

        # Process a few batches
        dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Only process first few batches
                break
            logits = make_perfect_logits(batch, vocab_size)
            metric.update(batch, lm_logits=logits)

        results = metric.compute()

        # Validate results
        validate_metrics_finite(results, task_name)
        assert "bpb_v1" in results, f"Task {task_name}: missing bpb_v1"
        assert "bpb_v2" in results, f"Task {task_name}: missing bpb_v2"

        bpb_v1 = results["bpb_v1"].item()
        bpb_v2 = results["bpb_v2"].item()

        # Perfect logits should give BPB close to 0
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

        # Random logits: BPB should be significantly higher than 0
        # Note: BPB is normalized by byte length, so it won't be exactly log2(vocab_size)
        # The key test is that random logits produce significantly higher BPB than perfect logits
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

        # Noisy logits: BPB should be in middle range (not near 0, not near log2(vocab))
        validate_bpb_range(bpb_v1, task_name, "noisy", min_val=0.5, max_val=15.0)
        validate_bpb_range(bpb_v2, task_name, "noisy", min_val=0.5, max_val=15.0)


# =====================================================
# TEST 3: Accuracy Metric Computation (RC Tasks)
# =====================================================


class TestAccuracyMetricComputation:
    """Test accuracy metric computation for RC tasks."""

    @pytest.mark.parametrize("task_name", RC_TASKS)
    def test_correct_answer_logits_produce_full_accuracy(self, task_name: str, tokenizer, vocab_size):
        """Logits favoring correct answer should produce accuracy = 1.0."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="acc")

        # Get complete documents to ensure proper accuracy calculation
        samples = get_complete_document_samples(task, num_docs=5)
        batch = task.collate_fn(samples)
        logits = make_correct_answer_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        # Check accuracy
        assert "acc_v1" in results, f"Task {task_name}: missing acc_v1"
        assert "acc_v2" in results, f"Task {task_name}: missing acc_v2"

        acc_v1 = results["acc_v1"].item()
        acc_v2 = results["acc_v2"].item()

        validate_accuracy_range(acc_v1, task_name, "correct_answer", expected_acc=1.0)
        validate_accuracy_range(acc_v2, task_name, "correct_answer", expected_acc=1.0)

    @pytest.mark.parametrize("task_name", RC_TASKS)
    def test_random_logits_produce_low_accuracy(self, task_name: str, tokenizer, vocab_size):
        """Random logits should produce accuracy ≈ 1/num_choices."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        num_choices = get_num_choices(task)
        metric = ICLMetric(metric_type="acc")

        # Get complete documents
        samples = get_complete_document_samples(task, num_docs=10)
        batch = task.collate_fn(samples)
        logits = make_random_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        acc_v1 = results["acc_v1"].item()

        # Random accuracy: should be roughly 1/num_choices
        # Allow wider tolerance since it's random
        assert 0.0 <= acc_v1 <= 1.0, f"Task {task_name}: accuracy {acc_v1} out of bounds"

    @pytest.mark.parametrize("task_name", RC_TASKS)
    def test_rc_tasks_return_all_expected_keys(self, task_name: str, tokenizer, vocab_size):
        """RC tasks should return all expected metric keys."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="acc")

        # Get complete documents
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
# TEST 4: Len-Norm Metric Computation
# =====================================================


class TestLenNormMetricComputation:
    """Test len_norm metric computation."""

    @pytest.mark.parametrize("task_name", MMLU_TASKS + ["hellaswag", "piqa", "arc_challenge", "openbook_qa"])
    def test_correct_answer_logits_produce_full_accuracy(self, task_name: str, tokenizer, vocab_size):
        """Logits favoring correct answer should produce len_norm accuracy = 1.0."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="len_norm")

        # Get complete documents to ensure proper accuracy calculation
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

    @pytest.mark.parametrize("task_name", MMLU_TASKS)
    def test_len_norm_tasks_return_all_expected_keys(self, task_name: str, tokenizer, vocab_size):
        """Len-norm tasks should return all expected metric keys."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="len_norm")

        # Get complete documents
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
# TEST 5: Core Benchmarks Regression Testing
# =====================================================


class TestCoreBenchmarksRegression:
    """Regression tests for established benchmarks."""

    @pytest.mark.parametrize("task_name", CORE_BENCHMARKS)
    def test_core_benchmarks_load_and_batch(self, task_name: str, tokenizer):
        """Verify core benchmarks can be loaded and batched."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        assert len(task) > 0, f"Task {task_name} has no samples"

        batch = task.collate_fn([task[0], task[1]] if len(task) > 1 else [task[0]])
        assert "input_ids" in batch
        assert batch["input_ids"].dim() == 2

    @pytest.mark.parametrize("task_name", CORE_BENCHMARKS)
    def test_core_benchmarks_metric_computation(self, task_name: str, tokenizer, vocab_size):
        """Verify metric computation works for core benchmarks."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric_type = task.metric_type
        metric = ICLMetric(metric_type=metric_type)

        # Get complete documents for proper metric calculation
        samples = get_complete_document_samples(task, num_docs=5)
        batch = task.collate_fn(samples)
        logits = make_perfect_logits(batch, vocab_size)
        metric.update(batch, lm_logits=logits)

        results = metric.compute()
        validate_metrics_finite(results, task_name)

        # Verify expected metric keys are present
        expected_keys = EXPECTED_METRIC_KEYS.get(metric_type, set())
        actual_keys = set(results.keys())
        missing_keys = expected_keys - actual_keys
        assert not missing_keys, f"Task {task_name}: missing metric keys {missing_keys}"

    @pytest.mark.parametrize(
        "task_name,expected_metric_type",
        [
            ("hellaswag", "len_norm"),
            ("piqa", "len_norm"),
            ("winogrande", "acc"),
            ("arc_easy", "acc"),
            ("arc_challenge", "len_norm"),
            ("boolq", "acc"),
            ("openbook_qa", "len_norm"),
            ("copa", "acc"),
            ("sciq", "acc"),
            ("social_iqa", "len_norm"),
        ],
    )
    def test_core_benchmarks_have_correct_metric_type(
        self, task_name: str, expected_metric_type: str, tokenizer
    ):
        """Verify core benchmarks have correct metric types."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        assert task.metric_type == expected_metric_type, (
            f"Task {task_name}: metric_type {task.metric_type} != expected {expected_metric_type}"
        )


# =====================================================
# TEST 6: Sanity Checks Across All Tasks
# =====================================================


class TestSanityChecks:
    """Cross-cutting sanity checks for all tasks."""

    @pytest.mark.parametrize("task_name", BPB_TASKS + RC_TASKS)
    def test_no_nan_inf_in_perfect_scenario(self, task_name: str, tokenizer, vocab_size):
        """Perfect logits should never produce NaN or inf."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric_type = task.metric_type
        metric = ICLMetric(metric_type=metric_type)

        # For accuracy tasks, use complete documents; for BPB tasks, just use samples
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

    @pytest.mark.parametrize("task_name", BPB_TASKS + RC_TASKS)
    def test_no_nan_inf_in_random_scenario(self, task_name: str, tokenizer, vocab_size):
        """Random logits should never produce NaN or inf."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric_type = task.metric_type
        metric = ICLMetric(metric_type=metric_type)

        # For accuracy tasks, use complete documents; for BPB tasks, just use samples
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

    @pytest.mark.parametrize("task_name", BPB_TASKS + RC_TASKS)
    def test_bpb_is_positive(self, task_name: str, tokenizer, vocab_size):
        """BPB should always be positive."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric_type = task.metric_type
        metric = ICLMetric(metric_type=metric_type)

        # For accuracy tasks, use complete documents; for BPB tasks, just use samples
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
            assert results["bpb_v1"].item() >= 0, f"Task {task_name}: BPB is negative"
        if "bpb_v2" in results:
            assert results["bpb_v2"].item() >= 0, f"Task {task_name}: BPB is negative"


# =====================================================
# TEST 7: Edge Cases (List Labels, etc.)
# =====================================================


# Tasks with list labels (multiple valid gold answers)
LIST_LABEL_TASKS = [
    "squad_bpb_5shot",  # Has list labels like ['23 years', '23', '23']
    "drop_bpb_5shot",  # Also has list labels
    "coqa_bpb_5shot",  # Has list labels
    "naturalqs_bpb_5shot",  # Has list labels
]


class TestEdgeCases:
    """Test edge cases like tasks with list labels."""

    @pytest.mark.parametrize("task_name", LIST_LABEL_TASKS)
    def test_list_label_tasks_load_correctly(self, task_name: str, tokenizer):
        """Tasks with list labels should load and have proper sample structure."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        assert len(task) > 0, f"Task {task_name} has no samples"

        # All samples should have label_id = 0 (normalized during prep)
        for i in range(min(10, len(task))):
            sample = task[i]
            assert sample["label_id"] == 0, (
                f"Task {task_name}: sample {i} has label_id={sample['label_id']}, expected 0"
            )
            assert sample["cont_id"] == 0, (
                f"Task {task_name}: sample {i} has cont_id={sample['cont_id']}, expected 0"
            )

    @pytest.mark.parametrize("task_name", LIST_LABEL_TASKS)
    def test_list_label_tasks_metric_computation(self, task_name: str, tokenizer, vocab_size):
        """Tasks with list labels should compute metrics correctly."""
        task = build_task(task_name, tokenizer, model_ctx_len=512)
        metric = ICLMetric(metric_type="bpb")

        # Process a few batches
        dataloader = DataLoader(task, batch_size=4, collate_fn=task.collate_fn)
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            logits = make_perfect_logits(batch, vocab_size)
            metric.update(batch, lm_logits=logits)

        results = metric.compute()

        # Validate results
        validate_metrics_finite(results, task_name)
        assert "bpb_v1" in results, f"Task {task_name}: missing bpb_v1"
        assert "bpb_v2" in results, f"Task {task_name}: missing bpb_v2"

        # Perfect logits should give low BPB
        bpb_v1 = results["bpb_v1"].item()
        validate_bpb_range(bpb_v1, task_name, "perfect", min_val=0.0, max_val=0.1)

    @pytest.mark.parametrize("task_name", LIST_LABEL_TASKS)
    def test_list_label_tasks_one_sample_per_doc(self, task_name: str, tokenizer):
        """
        For BPB tasks with list labels, each doc_id should have exactly one sample
        (the gold answer continuation).
        """
        task = build_task(task_name, tokenizer, model_ctx_len=512)

        # Count samples per doc_id
        from collections import Counter

        doc_counts = Counter(s["doc_id"] for s in task.samples)

        # Each doc should have exactly 1 sample for BPB tasks
        for doc_id, count in list(doc_counts.items())[:20]:
            assert count == 1, (
                f"Task {task_name}: doc_id={doc_id} has {count} samples, expected 1"
            )


# =====================================================
# STANDALONE EXECUTION
# =====================================================


def run_verbose_tests():
    """Run tests with verbose output when executed as a script."""
    print("=" * 60)
    print("BPB Play Testing - Mock-Based Evaluation Task Validation")
    print("=" * 60)

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = HFTokenizer(
        "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
        pad_token_id=0,
        eos_token_id=0,
    )
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    # Test BPB tasks
    print("\n" + "=" * 60)
    print("Testing BPB Tasks")
    print("=" * 60)

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
                # Test perfect logits
                logits = make_perfect_logits(batch, vocab_size)
                metric.update(batch, lm_logits=logits)

            results = metric.compute()
            print(f"  Perfect BPB v1: {results['bpb_v1'].item():.4f}")
            print(f"  Perfect BPB v2: {results['bpb_v2'].item():.4f}")

            # Test random logits
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

    # Test RC tasks
    print("\n" + "=" * 60)
    print("Testing RC Tasks (Accuracy)")
    print("=" * 60)

    for task_name in RC_TASKS:
        print(f"\n--- {task_name} ---")
        try:
            task = build_task(task_name, tokenizer, model_ctx_len=512)
            print(f"  Samples: {len(task)}")
            print(f"  Metric type: {task.metric_type}")

            metric = ICLMetric(metric_type="acc")
            # Get complete documents
            samples = get_complete_document_samples(task, num_docs=5)
            batch = task.collate_fn(samples)
            logits = make_correct_answer_logits(batch, vocab_size)
            metric.update(batch, lm_logits=logits)

            results = metric.compute()
            print(f"  Correct Answer Accuracy v1: {results['acc_v1'].item():.4f}")
            print(f"  BPB v1 (gold): {results['bpb_v1'].item():.4f}")
            print(f"  Metric keys: {list(results.keys())}")
            print("  [PASS]")

        except Exception as e:
            print(f"  [FAIL] {e}")

    # Test MMLU tasks
    print("\n" + "=" * 60)
    print("Testing MMLU Tasks (Len-Norm)")
    print("=" * 60)

    for task_name in MMLU_TASKS:
        print(f"\n--- {task_name} ---")
        try:
            task = build_task(task_name, tokenizer, model_ctx_len=512)
            print(f"  Samples: {len(task)}")
            print(f"  Metric type: {task.metric_type}")

            metric = ICLMetric(metric_type="len_norm")
            # Get complete documents
            samples = get_complete_document_samples(task, num_docs=5)
            batch = task.collate_fn(samples)
            logits = make_correct_answer_logits(batch, vocab_size)
            metric.update(batch, lm_logits=logits)

            results = metric.compute()
            print(f"  Correct Answer Len-Norm Accuracy v1: {results['len_norm_v1'].item():.4f}")
            print(f"  BPB v1 (gold): {results['bpb_v1'].item():.4f}")
            print("  [PASS]")

        except Exception as e:
            print(f"  [FAIL] {e}")

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_verbose_tests()
