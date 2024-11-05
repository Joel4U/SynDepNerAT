import torch
from typing import List, Tuple
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.optim import AdamW

def timestep_dropout(inputs, p=0.5, batch_first=True):
    '''
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    '''
    if not batch_first:
        inputs = inputs.transpose(0, 1)

    batch_size, time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1-p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask

def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))

def get_metric(p_num: int, total_num: int, total_predicted_num: int) -> Tuple[float, float, float]:
    """
    Return the metrics of precision, recall and f-score, based on the number
    (We make this small piece of function in order to reduce the code effort and less possible to have typo error)
    :param p_num:
    :param total_num:
    :param total_predicted_num:
    :return:
    """
    precision = p_num * 1.0 / total_predicted_num * 100 if total_predicted_num != 0 else 0
    recall = p_num * 1.0 / total_num * 100 if total_num != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    return precision, recall, fscore

def timestep_dropout(inputs, p=0.5, batch_first=True):
    '''
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    '''
    if not batch_first:
        inputs = inputs.transpose(0, 1)

    batch_size, time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1-p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    # drop_mask = drop_mask.unsqueeze(-1).expand((-1, -1, time_step)).transpose(1, 2)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask

FLAT = 0       # Flat entities
NESTED = 1     # Nested entities
ARBITRARY = 2  # Arbitrarily overlapping entities

import logging

logger = logging.getLogger(__name__)


def _spans_from_upper_triangular(seq_len: int):
    """Spans from the upper triangular area.
    """
    for start in range(seq_len):
        for end in range(start+1, seq_len+1):
            yield (start, end)

def _is_overlapping(chunk1: tuple, chunk2: tuple):
    # `NESTED` or `ARBITRARY`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def _is_ordered_nested(chunk1: tuple, chunk2: tuple):
    # `chunk1` is nested in `chunk2`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s2 <= s1 and e1 <= e2)

def _is_nested(chunk1: tuple, chunk2: tuple):
    # `NESTED`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def _is_clashed(chunk1: tuple, chunk2: tuple, allow_level: int=NESTED):
    if allow_level == FLAT:
        return _is_overlapping(chunk1, chunk2)
    elif allow_level == NESTED:
        return _is_overlapping(chunk1, chunk2) and not _is_nested(chunk1, chunk2)
    else:
        return False

def filter_clashed_by_priority(chunks: List[tuple], allow_level: int=NESTED):
    filtered_chunks = []
    for ck in chunks:
        if all(not _is_clashed(ck, ex_ck, allow_level=allow_level) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)
    return filtered_chunks

# 检测给定的标记片段列表中是否存在嵌套或重叠的情况，并确定它们的层次。
def detect_overlapping_level(chunks: List[tuple]):
    level = FLAT
    for i, ck1 in enumerate(chunks):
        for ck2 in chunks[i+1:]:
            if _is_nested(ck1, ck2):
                level = NESTED
            elif _is_overlapping(ck1, ck2):
                # Non-nested overlapping -> `ARBITRARY`
                return ARBITRARY
    return level


def detect_nested(chunks1: List[tuple], chunks2: List[tuple] = None, strict: bool = True):
	"""Return chunks from `chunks1` that are nested in any chunk from `chunks2`.
    """
	if chunks2 is None:
		chunks2 = chunks1

	nested_chunks = []
	for ck1 in chunks1:
		if any(_is_ordered_nested(ck1, ck2) and (ck1 != ck2) and (not strict or ck1[1:] != ck2[1:]) for ck2 in chunks2):
			nested_chunks.append(ck1)
	return nested_chunks

def _check_soft_target(x: torch.Tensor):
    assert x.dim() == 2
    if x.size(0) > 0:
        assert (x >= 0).all().item()
        assert (x.sum(dim=-1) - 1).abs().max().item() < 1e-6

def get_huggingface_optimizer_and_scheduler(model: nn.Module,
                                            pretr_lr: float, other_lr: float,
                                            num_training_steps: int,
                                            weight_decay: float = 0.0,
                                            eps: float = 1e-8,
                                            warmup_step: int = 0):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "word_embedder" in n and not any(nd in n for nd in no_decay) and p.requires_grad],
            "lr": pretr_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "word_embedder" in n and any(nd in n for nd in no_decay) and p.requires_grad],
            "lr": pretr_lr,
            "weight_decay": 0.0,
        },
        # { # 共享编码层参数（用于对抗损失）
        #     "params": [p for n, p in model.named_parameters() if 'shared_encoder' in n and not any(nd in n for nd in no_decay) and p.requires_grad], 
        #     "lr": 1e-3,
        #     "weight_decay": weight_decay,   
        # },
        # { # 共享编码层参数（用于对抗损失）
        #     "params": [p for n, p in model.named_parameters() if 'shared_encoder' in n and any(nd in n for nd in no_decay) and p.requires_grad],
        #     "lr": 1e-3,
        #     "weight_decay": 0.0,
        # },
        {
            "params": [p for n, p in model.named_parameters() if 'word_embedder' not in n and not any(nd in n for nd in no_decay) and p.requires_grad],
            "lr": other_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "word_embedder" not in n and any(nd in n for nd in no_decay) and p.requires_grad],
            "lr": other_lr,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler
