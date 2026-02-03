"""
Test file documenting and validating the expected label formats for all oe-eval tasks.

This file serves as documentation for the different label formats used across tasks:

1. LIST LABELS (multiple gold continuations):
   - coqa, drop, naturalqs_open, squad
   - These are QA/reading comprehension tasks where multiple answer formulations are valid
   - The code in tasks.py handles these with: `isinstance(label_id, (str, list))`

2. STRING LABELS (single gold continuation):
   - jeopardy, gsm8k, minerva_math_*, codex_mbpp, mt_mbpp_*
   - These have exactly one correct answer as a string

3. INTEGER LABELS (multiple choice index):
   - RC (reading comprehension) tasks like arc_easy, arc_challenge, etc.
   - The label is the index of the correct choice (0, 1, 2, 3, ...)

4. NONE LABELS:
   - codex_humaneval
   - No explicit label provided
"""

import gzip
import json

import pytest

from olmo_eval.util import get_data_path


def load_first_request(dataset_path: str, dataset_name: str) -> dict:
    """Load the first request from an oe-eval task dataset."""
    rel_path = f"oe_eval_tasks/{dataset_path}"
    if dataset_name:
        rel_path = f"{rel_path}/{dataset_name}"

    with get_data_path(rel_path) as path:
        data_file = path / "requests.jsonl.gz"
        if not data_file.exists():
            data_file = path / "requests.jsonl"

        if data_file.suffix == ".gz":
            with gzip.open(data_file, "rt", encoding="utf-8") as f:
                return json.loads(f.readline().strip())
        else:
            with open(data_file, "r") as f:
                return json.loads(f.readline().strip())


def load_first_doc_requests(dataset_path: str, dataset_name: str) -> dict:
    """
    Load all requests for the first document (doc_id=0) from an oe-eval task dataset.

    Returns a dict with:
        - "label": the correct choice index
        - "correct": the request where idx == label (correct continuation)
        - "wrong": list of requests where idx != label (wrong continuations)
    """
    rel_path = f"oe_eval_tasks/{dataset_path}"
    if dataset_name:
        rel_path = f"{rel_path}/{dataset_name}"

    requests_for_doc_0 = []
    with get_data_path(rel_path) as path:
        data_file = path / "requests.jsonl.gz"
        if not data_file.exists():
            data_file = path / "requests.jsonl"

        if data_file.suffix == ".gz":
            with gzip.open(data_file, "rt", encoding="utf-8") as f:
                for line in f:
                    req = json.loads(line.strip())
                    if req["doc_id"] == 0:
                        requests_for_doc_0.append(req)
                    else:
                        break  # doc_id=0 requests are contiguous at the start
        else:
            with open(data_file, "r") as f:
                for line in f:
                    req = json.loads(line.strip())
                    if req["doc_id"] == 0:
                        requests_for_doc_0.append(req)
                    else:
                        break

    label = requests_for_doc_0[0]["label"]
    correct = None
    wrong = []

    for req in requests_for_doc_0:
        if req["idx"] == label:
            correct = req
        else:
            wrong.append(req)

    return {"label": label, "correct": correct, "wrong": wrong}


class TestListLabelTasks:
    """
    Tasks with LIST labels - these have multiple valid gold continuations.
    The label field is a list of acceptable answers.

    These tasks required the fix in tasks.py:
        if label_id != cont_id and not isinstance(label_id, (str, list)):
    """

    def test_coqa_bpb_5shot(self):
        """
        CoQA: Conversational QA with multiple annotator answers.

        First instance has 4 annotators who all answered "white".
        Label is a list of all annotator responses.
        """
        request = load_first_request("coqa", "bpb_5shot")

        # Label is a list of annotator answers (can have duplicates)
        assert request["label"] == ["white", "white", "white", "white"]

        # Continuation is the first answer with leading space
        assert request["request"]["continuation"] == " white"

    def test_drop_bpb_5shot(self):
        """
        DROP: Discrete Reasoning Over Paragraphs.

        First instance asks "Which ancestral group is larger: swedish or united states?"
        Label is a nested list structure: [[answer1], [answer2], ...]
        """
        request = load_first_request("drop", "bpb_5shot")

        # DROP has nested list structure for answers
        assert request["label"] == [["swedish"]]

        # Continuation format includes tuple-like representation
        assert request["request"]["continuation"] == " ('swedish',)"

    def test_naturalqs_open_bpb_5shot(self):
        """
        Natural Questions Open: Open-domain QA from Google search queries.

        First instance asks "who is playing halftime show super bowl 2018"
        Label is a list of valid answers (usually just one for NQ).
        """
        request = load_first_request("naturalqs_open", "bpb_5shot")

        # Label is list of valid answers
        assert request["label"] == ["Justin Timberlake"]

        # Continuation is the answer with leading space
        assert request["request"]["continuation"] == " Justin Timberlake"

    def test_squad_bpb_5shot(self):
        """
        SQuAD: Stanford Question Answering Dataset.

        First instance has multiple valid answer spans from annotators.
        The same answer may appear in different forms ("23 years" vs "23").
        """
        request = load_first_request("squad", "bpb_5shot")

        # Label has multiple valid answer formulations
        assert request["label"] == ["23 years", "23", "23"]

        # Continuation uses the first/canonical answer
        assert request["request"]["continuation"] == " 23 years"


class TestStringLabelTasks:
    """
    Tasks with STRING labels - these have exactly one gold continuation.
    The label field is a string representing the single correct answer.
    """

    def test_jeopardy_bpb_5shot(self):
        """
        Jeopardy: Trivia QA in Jeopardy format.

        First instance answer is "sound".
        """
        request = load_first_request("jeopardy", "bpb_5shot")

        assert request["label"] == "sound"
        assert request["request"]["continuation"] == " sound"

    def test_gsm8k_gold_bpb_5shot(self):
        """
        GSM8K: Grade School Math.

        First instance: Janet's ducks problem.
        Label is the final numeric answer as a string.
        Continuation includes the full chain-of-thought solution.
        """
        request = load_first_request("gsm8k", "gold_bpb_5shot")

        # Label is the final answer
        assert request["label"] == "18"

        # Continuation includes the reasoning chain
        continuation = request["request"]["continuation"]
        assert continuation.startswith(" Janet sells 16 - 3 - 4 = 9 duck eggs a day.")
        assert "$18" in continuation

    def test_codex_mbpp_gold_bpb_3shot(self):
        """
        MBPP: Mostly Basic Python Problems.

        First instance: remove_Occ function to remove first and last occurrence of a char.
        Label is the reference solution code.
        """
        request = load_first_request("codex_mbpp", "gold_bpb_3shot")

        # Label is the full reference solution
        assert request["label"].startswith("def remove_Occ(s,ch):")
        assert "break" in request["label"]

        # Continuation starts with the function definition
        assert request["request"]["continuation"].startswith("def remove_Occ(s,ch):")

    def test_minerva_math_500_gold_bpb_0shot(self):
        """
        Minerva Math 500: Subset of MATH benchmark.

        First instance has a LaTeX answer.
        """
        request = load_first_request("minerva_math_500", "gold_bpb_0shot")

        # Label is a LaTeX expression
        assert request["label"] == "\\le(3,\\frac{\\pi}{2}\\right)"

    def test_minerva_math_algebra_gold_bpb_0shot(self):
        """Minerva Math Algebra: First instance answer is "2"."""
        request = load_first_request("minerva_math_algebra", "gold_bpb_0shot")
        assert request["label"] == "2"

    def test_minerva_math_counting_and_probability_gold_bpb_0shot(self):
        """Minerva Math Counting/Probability: First instance is a fraction."""
        request = load_first_request("minerva_math_counting_and_probability", "gold_bpb_0shot")
        assert request["label"] == "\\frac{10}{11}"

    def test_minerva_math_geometry_gold_bpb_0shot(self):
        """Minerva Math Geometry: First instance is a fraction."""
        request = load_first_request("minerva_math_geometry", "gold_bpb_0shot")
        assert request["label"] == "\\frac{3}{4}"

    def test_minerva_math_intermediate_algebra_gold_bpb_0shot(self):
        """Minerva Math Intermediate Algebra: First instance is a square root."""
        request = load_first_request("minerva_math_intermediate_algebra", "gold_bpb_0shot")
        assert request["label"] == "\\sqrt{74}"

    def test_minerva_math_number_theory_gold_bpb_0shot(self):
        """Minerva Math Number Theory: First instance is a fraction."""
        request = load_first_request("minerva_math_number_theory", "gold_bpb_0shot")
        assert request["label"] == "\\frac{1}{11}"

    def test_minerva_math_prealgebra_gold_bpb_0shot(self):
        """Minerva Math Prealgebra: First instance answer is "4"."""
        request = load_first_request("minerva_math_prealgebra", "gold_bpb_0shot")
        assert request["label"] == "4"

    def test_minerva_math_precalculus_gold_bpb_0shot(self):
        """Minerva Math Precalculus: First instance answer is "0"."""
        request = load_first_request("minerva_math_precalculus", "gold_bpb_0shot")
        assert request["label"] == "0"

    @pytest.mark.parametrize(
        "lang,expected_func_name",
        [
            ("bash", "remove_Occ"),
            ("c", "remove_Oc"),  # truncated in label preview
            ("cpp", "remove_Occ"),
            ("csharp", "RemoveOcc"),
            ("go", "removeOcc"),
            ("haskell", "removeOcc"),
            ("java", "removeOcc"),
            ("javascript", "removeOcc"),
            ("matlab", "remove_Occ"),
            ("php", "remove_Occ"),
            ("python", "remove_Occ"),
            ("r", "remove_Occ"),
            ("ruby", "remove_occ"),
            ("rust", "remove_occ"),
            ("scala", "removeOcc"),
            ("swift", "removeOcc"),
            ("typescript", "removeOcc"),
        ],
    )
    def test_mt_mbpp_gold_bpb_3shot(self, lang: str, expected_func_name: str):
        """
        MultiPL-E MBPP translations: Same problems as MBPP but in different languages.

        First instance is always the remove_Occ problem translated to each language.
        """
        request = load_first_request(f"mt_mbpp_{lang}", "gold_bpb_3shot")

        assert isinstance(request["label"], str)
        assert expected_func_name in request["label"]


class TestIntegerLabelTasks:
    """
    Tasks with INTEGER labels - these are multiple choice tasks.
    The label field is an integer index (0-indexed) of the correct choice.

    For each document, there are multiple requests (one per choice).
    The label indicates which choice is correct.
    """

    def test_arc_easy_rc_5shot(self):
        """
        ARC-Easy: AI2 Reasoning Challenge (Easy set).

        First instance: question about communication technology.
        4 choices (labels 0-3).
        """
        doc = load_first_doc_requests("arc_easy", "rc_5shot")

        assert doc["label"] == 0
        assert doc["correct"]["request"]["continuation"] == " cellular telephone"
        assert doc["wrong"][0]["request"]["continuation"] == " television"

    def test_arc_challenge_rc_5shot(self):
        """
        ARC-Challenge: AI2 Reasoning Challenge (Challenge set).

        First instance: science question about investigation procedures.
        4 choices (labels 0-3).
        """
        doc = load_first_doc_requests("arc_challenge", "rc_5shot")

        assert doc["label"] == 3
        assert (
            doc["correct"]["request"]["continuation"] == " Record the details of the investigation."
        )
        assert doc["wrong"][0]["request"]["continuation"] == " Put the objects in groups."

    def test_hellaswag_rc_5shot(self):
        """
        HellaSwag: Commonsense NLI about grounded situations.

        First instance: completing a sentence about changing your name.
        4 choices (labels 0-3).
        """
        doc = load_first_doc_requests("hellaswag", "rc_5shot")

        assert doc["label"] == 3
        assert "don't have to be a resident" in doc["correct"]["request"]["continuation"]
        assert "may be able to change your name" in doc["wrong"][0]["request"]["continuation"]

    def test_piqa_rc_5shot(self):
        """
        PIQA: Physical Intuition QA.

        First instance: question about cooking sausages.
        2 choices (labels 0-1). Both continuations start similarly but differ.
        """
        doc = load_first_doc_requests("piqa", "rc_5shot")

        assert doc["label"] == 0
        assert "frying pan" in doc["correct"]["request"]["continuation"]
        # Both choices start with "In a frying pan" - they differ later in the text

    def test_winogrande_rc_5shot(self):
        """
        WinoGrande: Commonsense coreference resolution.

        First instance: sentence completion about cases.
        2 choices (labels 0-1). The continuation is the same but context differs.
        """
        doc = load_first_doc_requests("winogrande", "rc_5shot")

        assert doc["label"] == 1
        # WinoGrande has same continuation for both choices - context determines correct one
        assert doc["correct"]["request"]["continuation"] == " always got the easier cases."
        assert doc["wrong"][0]["request"]["continuation"] == " always got the easier cases."

    def test_boolq_rc_5shot(self):
        """
        BoolQ: Boolean questions (yes/no).

        First instance: answer is "yes".
        2 choices: 0=yes, 1=no.
        """
        doc = load_first_doc_requests("boolq", "rc_5shot")

        assert doc["label"] == 0
        assert doc["correct"]["request"]["continuation"] == " yes"
        assert doc["wrong"][0]["request"]["continuation"] == " no"

    def test_openbookqa_rc_5shot(self):
        """
        OpenBookQA: Science questions with open book.

        First instance: question about what lacks the ability to see light.
        4 choices (labels 0-3).
        """
        doc = load_first_doc_requests("openbookqa", "rc_5shot")

        assert doc["label"] == 0
        assert doc["correct"]["request"]["continuation"] == " Deep sea animals"
        assert doc["wrong"][0]["request"]["continuation"] == " fish"

    def test_csqa_rc_5shot(self):
        """
        CommonsenseQA: Commonsense question answering.

        First instance: question about where to keep money safe.
        5 choices (labels 0-4).
        """
        doc = load_first_doc_requests("csqa", "rc_5shot")

        assert doc["label"] == 0
        assert doc["correct"]["request"]["continuation"] == " bank"
        assert doc["wrong"][0]["request"]["continuation"] == " library"

    def test_socialiqa_rc_5shot(self):
        """
        SocialIQA: Social intelligence QA.

        First instance: question about how someone would be described.
        3 choices (labels 0-2).
        """
        doc = load_first_doc_requests("socialiqa", "rc_5shot")

        assert doc["label"] == 1
        assert doc["correct"]["request"]["continuation"] == " a bad child"
        assert doc["wrong"][0]["request"]["continuation"] == " cheating"

    def test_copa_rc_0shot(self):
        """
        COPA: Choice of Plausible Alternatives.

        First instance: cause/effect about turning on the faucet.
        2 choices (labels 0-1).
        """
        doc = load_first_doc_requests("copa", "rc_0shot")

        assert doc["label"] == 1
        assert doc["correct"]["request"]["continuation"] == " water flowed from the spout."
        assert doc["wrong"][0]["request"]["continuation"] == " the toilet filled with water."

    def test_sciq_rc_0shot(self):
        """
        SciQ: Science exam questions.

        First instance: question about who developed a classification system.
        4 choices (labels 0-3).
        """
        doc = load_first_doc_requests("sciq", "rc_0shot")

        assert doc["label"] == 3
        assert doc["correct"]["request"]["continuation"] == " darwin"
        assert doc["wrong"][0]["request"]["continuation"] == " Linnaeus"

    def test_lab_bench_dbqa_rc(self):
        """
        Lab Bench DBQA: Database QA for scientific literature.

        First instance: question about a gene name.
        Note: All instances in this dataset have label=3 (4th choice is always correct).
        """
        doc = load_first_doc_requests("lab_bench_dbqa", "rc")

        assert doc["label"] == 3
        assert doc["correct"]["request"]["continuation"] == " PCSK5"
        assert doc["wrong"][0]["request"]["continuation"] == " MNX1"

    def test_lab_bench_protocolqa_rc(self):
        """
        Lab Bench ProtocolQA: Protocol understanding for lab procedures.

        First instance: question about PBMC isolation protocol.
        Labels in this dataset range from 3-6 (varying number of choices).
        """
        doc = load_first_doc_requests("lab_bench_protocolqa", "rc")

        assert doc["label"] == 3
        assert (
            doc["correct"]["request"]["continuation"]
            == " Use 0.85g of NaCl in step 2 to avoid PBMC lysis."
        )
        assert (
            doc["wrong"][0]["request"]["continuation"]
            == " Do not use NaCl in step 2 to avoid PBMC lysis."
        )


class TestNoneLabelTasks:
    """
    Tasks with NONE labels - no explicit label provided.
    These tasks are evaluated by code execution rather than string matching.
    """

    def test_codex_humaneval_gold_bpb_0shot(self):
        """
        HumanEval: Code generation evaluated by execution.

        First instance: has_close_elements function.
        No label because correctness is determined by running test cases.
        """
        request = load_first_request("codex_humaneval", "gold_bpb_0shot")

        assert request["label"] is None

        # Continuation is the reference solution
        continuation = request["request"]["continuation"]
        assert "for idx, elem in enumerate(numbers):" in continuation

    def test_codex_humaneval_gold_bpb_3shot(self):
        """HumanEval 3-shot: Same as 0-shot, label is None."""
        request = load_first_request("codex_humaneval", "gold_bpb_3shot")
        assert request["label"] is None
