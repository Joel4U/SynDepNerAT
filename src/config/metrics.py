# -*- coding: utf8 -*-
#
import torch
from umetrics import MacroMetrics
from typing import List


def cal_metrics(y_preds, y_trues):
    """

    :param y_preds:
    :param y_trues:
    :return:
    """
    y_preds_unique_labels = torch.unique(y_preds)
    y_trues_unique_labels = torch.unique(y_trues)

    all_labels = torch.cat((y_preds_unique_labels, y_trues_unique_labels)).unique(sorted=True)
    # ignore 0
    if 0 in all_labels:
        all_labels = all_labels[1:]

    y_preds_labels, y_preds_count = y_preds.unique(return_counts=True)
    y_trues_labels, y_trues_count = y_trues.unique(return_counts=True)

    corrects_mask = torch.eq(y_preds, y_trues)
    corrects_labels, corrects_count = y_trues[corrects_mask].unique(return_counts=True)

    y_preds_map = dict(zip(y_preds_labels.tolist(), y_preds_count.tolist()))
    y_true_map = dict(zip(y_trues_labels.tolist(), y_trues_count.tolist()))
    corrects_map = dict(zip(corrects_labels.tolist(), corrects_count.tolist()))
    precision, recall, f1 = 0, 0, 0
    for label in all_labels.tolist():
        _precision = corrects_map.get(label, 0) / (y_preds_map.get(label, 0) + 1e-8)
        _recall = corrects_map.get(label, 0) / (y_true_map.get(label, 0) + 1e-8)
        _f1 = 2 * _precision * _recall / (_precision + _recall + 1e-8)
        precision += _precision
        recall += _recall
        f1 += _f1

    all_label_count = len(all_labels)
    return precision / all_label_count, recall / all_label_count, f1 / all_label_count


class Metrics(object):
    def __init__(self, ALL_LABELS):
        self.metrics = MacroMetrics(labels=ALL_LABELS)

    def step(self, y_true, y_pred, mask):
        """version2"""
        mask = mask.view(-1)

        y_pred = y_pred.argmax(axis=-1)
        y_trues = y_true.view(-1)
        y_preds = y_pred.view(-1)

        y_trues = y_trues * mask
        y_preds = y_preds * mask

        self.metrics.step(y_trues=y_trues.tolist(), y_preds=y_preds.tolist())

    def summary(self):
        # self.metrics.classification_report()
        return self.metrics.precision_score(), self.metrics.recall_score(), self.metrics.f1_score()


def _agg_scores_by_key(scores, key, agg_mode='mean'):
    """
    Parameters
    ----------
    scores: list or dict
        list or dict of {'precision': ..., 'recall': ..., 'f1': ...}
    """
    if len(scores) == 0:
        return 0

    if isinstance(scores, list):
        sum_value = sum(sub_scores[key] for sub_scores in scores)
    else:
        sum_value = sum(sub_scores[key] for _, sub_scores in scores.items())
    if agg_mode == 'sum':
        return sum_value
    elif agg_mode == 'mean':
        return sum_value / len(scores)

def precision_recall_f1_report(list_tuples_gold: List[List[tuple]], list_tuples_pred: List[List[tuple]],
                               macro_over='types', **kwargs):
    """
    Parameters
    ----------
    list_tuples_{gold, pred}: a list of lists of tuples
        A tuple of chunk or entity is in format of (chunk_type, chunk_start, chunk_end) or
                                                   (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
        A tuple of relation is in format of (relation_type, (head_type, head_start, head_end), (tail_type, tail_start, tail_end))

    macro_over: str
        'types' or 'samples'

    type_pos: int
        The position indicating type in a tuple


    References
    ----------
    https://github.com/chakki-works/seqeval
    """
    assert len(list_tuples_gold) == len(list_tuples_pred)

    if macro_over == 'types':
        scores = _prf_scores_over_types(list_tuples_gold, list_tuples_pred, **kwargs)
    elif macro_over == 'samples':
        scores = _prf_scores_over_samples(list_tuples_gold, list_tuples_pred, **kwargs)
    else:
        raise ValueError(f"Invalid `macro_over` {macro_over}")

    ave_scores = {}
    ave_scores['macro'] = {key: _agg_scores_by_key(scores, key, agg_mode='mean') for key in
                           ['precision', 'recall', 'f1']}
    ave_scores['micro'] = {key: _agg_scores_by_key(scores, key, agg_mode='sum') for key in
                           ['n_gold', 'n_pred', 'n_true_positive']}

    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(ave_scores['micro']['n_gold'],
                                                                   ave_scores['micro']['n_pred'],
                                                                   ave_scores['micro']['n_true_positive'], **kwargs)
    ave_scores['micro'].update({'precision': micro_precision,
                                'recall': micro_recall,
                                'f1': micro_f1})

    return scores, ave_scores


def _precision_recall_f1(n_gold, n_pred, n_true_positive, zero_division=0):
    precision = n_true_positive / n_pred if n_pred > 0 else zero_division
    recall    = n_true_positive / n_gold if n_gold > 0 else zero_division
    f1 = 2 / (1/precision + 1/recall) if (precision + recall > 0) else zero_division
    return precision, recall, f1

def _prf_scores_over_types(list_tuples_gold: List[List[tuple]], list_tuples_pred: List[List[tuple]], type_pos=0,
                           **kwargs):
    tuples_set = {tp for list_tuples in [list_tuples_gold, list_tuples_pred] for tuples in list_tuples for tp in tuples}
    if len(tuples_set) == 0:
        return {}

    types_set = {tp[type_pos] for tp in tuples_set}

    scores = {}
    for _type in types_set:
        n_gold, n_pred, n_true_positive = 0, 0, 0
        for tuples_gold, tuples_pred in zip(list_tuples_gold, list_tuples_pred):
            tuples_gold = {tp for tp in tuples_gold if tp[type_pos] == _type}
            tuples_pred = {tp for tp in tuples_pred if tp[type_pos] == _type}
            n_gold += len(tuples_gold)
            n_pred += len(tuples_pred)
            n_true_positive += len(tuples_gold & tuples_pred)

        precision, recall, f1 = _precision_recall_f1(n_gold, n_pred, n_true_positive, **kwargs)
        scores[_type] = {'n_gold': n_gold,
                         'n_pred': n_pred,
                         'n_true_positive': n_true_positive,
                         'precision': precision,
                         'recall': recall,
                         'f1': f1}
    return scores


def _prf_scores_over_samples(list_tuples_gold: List[List[tuple]], list_tuples_pred: List[List[tuple]], **kwargs):
    scores = []
    for tuples_gold, tuples_pred in zip(list_tuples_gold, list_tuples_pred):
        tuples_gold, tuples_pred = set(tuples_gold), set(tuples_pred)
        n_gold, n_pred = len(tuples_gold), len(tuples_pred)
        n_true_positive = len(tuples_gold & tuples_pred)

        precision, recall, f1 = _precision_recall_f1(n_gold, n_pred, n_true_positive, **kwargs)
        scores.append({'n_gold': n_gold,
                       'n_pred': n_pred,
                       'n_true_positive': n_true_positive,
                       'precision': precision,
                       'recall': recall,
                       'f1': f1})
    return scores