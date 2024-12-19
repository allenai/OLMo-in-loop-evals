import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchmetrics import Metric

from .util import all_gather_object

LOG_2_OF_E = 1.44269504089


log = logging.getLogger(__name__)


T = TypeVar("T")


def dist_combine_lists(x: List[T]) -> List[T]:
    all_lists = all_gather_object(x)
    return [item for sublist in all_lists for item in sublist]


class ICLMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(self, metric_type="acc") -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        super().__init__(sync_on_compute=True)

        self.metric_type = metric_type

        self.add_state("loglikelihoods", default=[], dist_reduce_fx=dist_combine_lists)
        self.add_state("celosses", default=[], dist_reduce_fx=dist_combine_lists)
        self.add_state("bpbs", default=[], dist_reduce_fx=dist_combine_lists)
        self.add_state("labels", default=[], dist_reduce_fx=dist_combine_lists)

    def reset(self):
        self.loglikelihoods: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []
        self.celosses: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []
        self.bpbs: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []
        self.labels: List[Tuple[Optional[int], Optional[int], Optional[int]]] = []

    def update(
        self,
        batch: Dict[str, Any],
        lm_logits: Optional[torch.Tensor] = None,
        dc_lm_logits: Optional[torch.Tensor] = None,
    ):
        # NOTE: `lm_logits` could be none for some ranks if using pipeline parallelism. We still
        # need to add something to these state lists though in order for them to get synchronized
        # for reasons not clear to me other than the fact that torchmetrics is jenky a.f.
        if lm_logits is None:
            self.loglikelihoods.append((None, None, None))
            self.celosses.append((None, None, None))
            self.bpbs.append((None, None, None))
            self.labels.append((None, None, None))
            return

        lm_logits = F.log_softmax(lm_logits, dim=-1)

        if self.metric_type == "pmi_dc":
            assert (
                dc_lm_logits is not None
            ), "PMI_DC acc type selected but no domain conditional logits provided"

        for idx, (doc_id, cont_id) in enumerate(zip(batch["doc_id"], batch["cont_id"])):
            doc_id = int(doc_id)
            cont_id = int(cont_id)

            # [cont_len]: continuation is padded for batching
            cont_tokens = batch["continuation"][idx][: batch["cont_len"][idx]]
            # get logits from LM for the continuation: [cont_len, vocab]
            # batch['input_ids'][idx] -> ctx + cont + padding
            # -1 in both indices: lm_logits will be left shited 1 pos as 0th pos in input generates next token in the 0th pos of lm_logits
            lm_cont_logits = lm_logits[idx][
                batch["ctx_len"][idx] - 1 : batch["ctx_len"][idx] + batch["cont_len"][idx] - 1
            ]

            log_likelihood: torch.Tensor
            celoss: torch.Tensor
            bpb: torch.Tensor
            if self.metric_type == "pmi_dc":
                assert dc_lm_logits is not None
                # get domain conditional continuation logits: [cont_len, vocab]
                dc_lm_cont_logits = dc_lm_logits[idx][
                    batch["dc_len"][idx] - 1 : batch["dc_len"][idx] + batch["cont_len"][idx] - 1
                ]

                # gather log-probs at continuation token indices but divide by domain conditional prob
                log_likelihood = (
                    torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / torch.gather(dc_lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                )
                celoss = -log_likelihood
                bpb = -log_likelihood  # the normalization factors cancel out
            elif self.metric_type == "acc" or self.metric_type == "f1":
                # gather log-probs at continuation token indices
                log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                celoss = (
                    -torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_str_len"][idx]
                )
                bpb = (
                    -torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_byte_len"][idx]
                    * LOG_2_OF_E
                )
            elif self.metric_type in ["len_norm", "ce_loss", "bpb"]:
                log_likelihood = (
                    torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_str_len"][idx]
                )
                celoss = (
                    -torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_str_len"][idx]
                )
                bpb = (
                    -torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    / batch["cont_byte_len"][idx]
                    * LOG_2_OF_E
                )
            else:
                raise ValueError(self.metric_type)

            self.loglikelihoods.append((doc_id, cont_id, float(log_likelihood)))
            self.labels.append((doc_id, cont_id, int(batch["label_id"][idx])))
            self.celosses.append((doc_id, cont_id, float(celoss)))
            self.bpbs.append((doc_id, cont_id, float(bpb)))

    def compute(self) -> Dict[str, torch.Tensor]:
        # Task "suffix" -> tensor

        # states should have been synced from all accelerators at this point
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        loglikelihood_dict: Dict[int, Dict[int, float]] = {}
        label_dict: Dict[int, int] = {}
        celoss_dict: Dict[int, Dict[int, float]] = {}
        bpb_dict: Dict[int, Dict[int, float]] = {}

        # collect labels
        for doc_id, cont_id, label_id in self.labels:
            if doc_id is None or cont_id is None or label_id is None:
                continue

            if doc_id not in label_dict:
                label_dict[doc_id] = label_id

        # collect loglikelihoods
        for doc_id, cont_id, loglikelihood in self.loglikelihoods:
            if doc_id is None or cont_id is None or loglikelihood is None:
                continue

            if doc_id not in loglikelihood_dict:
                loglikelihood_dict[doc_id] = {}

            if cont_id not in loglikelihood_dict[doc_id]:
                loglikelihood_dict[doc_id][cont_id] = loglikelihood

        # collect celosses
        for doc_id, cont_id, celoss_val in self.celosses:
            if doc_id is None or cont_id is None or celoss_val is None:
                continue

            if doc_id not in celoss_dict:
                celoss_dict[doc_id] = {}

            if cont_id not in celoss_dict[doc_id]:
                celoss_dict[doc_id][cont_id] = celoss_val

        # collect bpbs
        for doc_id, cont_id, bpb_val in self.bpbs:
            if doc_id is None or cont_id is None or bpb_val is None:
                continue

            if doc_id not in bpb_dict:
                bpb_dict[doc_id] = {}

            if cont_id not in bpb_dict[doc_id]:
                bpb_dict[doc_id][cont_id] = bpb_val

        # compute acc
        correct = []
        celoss = []
        bpb = []
        soft_score = []
        soft_log_score = []
        preds: Optional[List[float]] = None
        labels: Optional[List[int]] = None
        if self.metric_type == "f1":
            preds = []
            labels = []

        for doc_id in loglikelihood_dict:
            # each doc_id might have a different number of continuation
            num_continuations = len(loglikelihood_dict[doc_id].keys())
            loglikelihoods = torch.tensor([-float("inf")] * num_continuations)
            celosses = torch.tensor([float("inf")] * num_continuations)
            bpbs = torch.tensor([float("inf")] * num_continuations)

            skip_document = False
            for cont_id in loglikelihood_dict[doc_id]:
                try:
                    loglikelihoods[cont_id] = loglikelihood_dict[doc_id][cont_id]
                    celosses[cont_id] = celoss_dict[doc_id][cont_id]
                    bpbs[cont_id] = bpb_dict[doc_id][cont_id]
                except IndexError:
                    # We didn't process all of the continuations, so skip this document.
                    skip_document = True
                    break

            if skip_document:
                continue

            if self.metric_type == "ce_loss":
                celoss.append(celosses[0])  # Only one answer is scored
            elif self.metric_type == "bpb":
                bpb.append(bpbs[0])  # Only one answer is scored
            elif self.metric_type == "f1":
                assert preds is not None
                assert labels is not None
                preds.append(torch.argmax(loglikelihoods).item())
                labels.append(label_dict[doc_id])
            else:
                correct.append(
                    1.0 if torch.argmax(loglikelihoods).item() == label_dict[doc_id] else 0.0
                )
                celoss.append(celosses[label_dict[doc_id]].item())
                bpb.append(bpbs[label_dict[doc_id]].item())
                soft_score.append(torch.softmax(loglikelihoods, dim=0)[label_dict[doc_id]].item())
                soft_log_score.append(
                    torch.log_softmax(loglikelihoods, dim=0)[label_dict[doc_id]].item()
                )

        if self.metric_type == "f1":
            assert preds is not None
            assert labels is not None
            # for NLI tasks, continuations are yes, no, neither, so idx=0 assigned to pos label
            score = f1_score(labels, preds, pos_label=0)
            return {"f1": torch.tensor(score)}
        elif self.metric_type == "ce_loss":
            return {"ce_loss": torch.tensor(sum(celoss) / len(celoss))}
        elif self.metric_type == "bpb":
            return {"bpb": torch.tensor(sum(bpb) / len(bpb))}
        else:
            return {
                self.metric_type: torch.tensor(sum(correct) / len(correct)),
                "ce_loss": torch.tensor(sum(celoss) / len(celoss)),
                "bpb": torch.tensor(sum(bpb) / len(bpb)),
                "soft": torch.tensor(sum(soft_score) / len(soft_score)),
                "soft_log": torch.tensor(sum(soft_log_score) / len(soft_log_score)),
            }
