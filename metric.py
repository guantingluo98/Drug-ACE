import os
import sys
import json
import ast
import re
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np

def modify_to_target_by_edit_distance(predict, target_list, logger, threshold=0.5):
    """
    soft match
    """
    pred = predict.strip()
    if len(target_list) == 0:
        return pred
    similarity_list = [SequenceMatcher(a=pred, b=item).ratio() for item in target_list]
    print("similarity list: ", similarity_list)
    max_score = max(similarity_list)
    if max_score > threshold:
        max_index = similarity_list.index(max_score)
        target_item = target_list[max_index].lower().strip()
        print("target_item: ", target_item)
        if target_item != pred and (target_item in pred or pred in target_item): 
            return target_item

    return pred

def response_string_to_list(response: str):
    response = response.strip()
    response = response.split("</s>")[0]
    response = response.split("<|im_end|>")[0]
    response = response.split("<eos>")[0]
    response = response.split("<｜end▁of▁sentence｜>")[0]
    response = re.sub(r"</s>$", "", response).strip()
    
    try:
        res_list = ast.literal_eval(response)
        if isinstance(res_list, list):
            return [str(item).strip() for item in res_list]
        else:
            return []
    except Exception as e:
        return []

def parse_span_label(entry: str) -> Tuple[str, str]:
    parts = entry.split('|')
    span = parts[0].strip().replace('Span: ', '')
    label = parts[1].strip().replace('Label: ', '')
    return span, label

def compute_hard_match_f1(reference: List[str], prediction: List[str]) -> Tuple[float, float, float]:
    if reference==[] and prediction==[]:
        return 1.0, 1.0, 1.0  
    ref_spans = [parse_span_label(r) for r in reference]
    pred_spans = [parse_span_label(p) for p in prediction]

    matched_ref_indices = set()
    matched_pred_indices = set()

    for i, ref_item in enumerate(ref_spans):
        for j, pred_item in enumerate(pred_spans):
            if i in matched_ref_indices or j in matched_pred_indices:
                continue  # skip already matched
            if ref_item == pred_item:  # FULL hard match
                matched_ref_indices.add(i)
                matched_pred_indices.add(j)
                break

    tp = len(matched_ref_indices)
    fp = len(pred_spans) - tp
    fn = len(ref_spans) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def parse_span_label(entry: str) -> Tuple[str, str]:
    if "|" not in entry:
        return " ", " "
    """'Span: 0.4 mg/kg | Label: Dosage' -> ('0.4 mg/kg', 'Dosage')"""
    parts = entry.split('|')
    span = parts[0].strip().replace('Span: ', '')
    label = parts[1].strip().replace('Label: ', '')
    return span, label.lower()

def is_soft_span_match(ref_span: str, pred_span: str, threshold: float = 0.5) -> bool:

    ref = ref_span.strip().lower()
    pred = pred_span.strip().lower()
    score = SequenceMatcher(a=ref, b=pred).ratio()
    return score >= threshold and (ref in pred or pred in ref)

def compute_match_f1(reference: List[str],
                     prediction: List[str],
                     soft: bool = True,
                     threshold: float = 0.5) -> Tuple[float, float, float]:

    if not reference and not prediction:
        return 1.0, 1.0, 1.0

    ref_spans = [parse_span_label(r) for r in reference]
    pred_spans = [parse_span_label(p) for p in prediction]

    matched_ref_indices = set()
    matched_pred_indices = set()

    for i, (ref_span, ref_label) in enumerate(ref_spans):
        for j, (pred_span, pred_label) in enumerate(pred_spans):
            if i in matched_ref_indices or j in matched_pred_indices:
                continue  

            if ref_label != pred_label:
                continue

            if (not soft and ref_span == pred_span) \
               or (soft and is_soft_span_match(ref_span, pred_span, threshold)):
                matched_ref_indices.add(i)
                matched_pred_indices.add(j)
                break

    tp = len(matched_ref_indices)
    fp = len(pred_spans) - tp
    fn = len(ref_spans) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def getF1_ignoreLabel(reference: List[str],
                     prediction: List[str],
                     soft: bool = True,
                     threshold: float = 0.5) -> Tuple[float, float, float]:

    if not reference and not prediction:
        return 1.0, 1.0, 1.0

    ref_spans = [parse_span_label(r) for r in reference]
    pred_spans = [parse_span_label(p) for p in prediction]

    matched_ref_indices = set()
    matched_pred_indices = set()

    for i, (ref_span, ref_label) in enumerate(ref_spans):
        for j, (pred_span, pred_label) in enumerate(pred_spans):
            if i in matched_ref_indices or j in matched_pred_indices:
                continue  

            if (not soft and ref_span == pred_span) \
               or (soft and is_soft_span_match(ref_span, pred_span, threshold)):
                matched_ref_indices.add(i)
                matched_pred_indices.add(j)
                break

    tp = len(matched_ref_indices)
    fp = len(pred_spans) - tp
    fn = len(ref_spans) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def getScores(target_list, results_list):
    num = len(target_list)
    score = 0
    score_ignore_label = 0
    for i in range(num):
        p = results_list[i]
        t = target_list[i]
        s = compute_match_f1(t, p, soft=False)[-1]
        s_ig = getF1_ignoreLabel(t, p, soft=False)[-1]
        score += s

        score_ignore_label += s_ig
    hard_pair = format(100*score/num, ".4g")
    hard_span = format(100*score_ignore_label/num, ".4g")

    score = 0
    score_ignore_label = 0
    for i in range(num):
        p = results_list[i]
        t = target_list[i]
        s = compute_match_f1(t, p, soft=True)[-1]
        s_ig = getF1_ignoreLabel(t, p, soft=True)[-1]
        score += s
        score_ignore_label += s_ig
    soft_pair = format(100*score/num, ".4g")
    soft_span = format(100*score_ignore_label/num, ".4g")
    print("| Hard Span | Soft Span | Hard Pair | Soft Pair |")
    print("|-----------|-----------|-----------|-----------|")
    print(f"| {hard_span} | {soft_span} | {hard_pair} | {soft_pair} |")

def getOutputsByBatch(model, tokenizer, dataset, savepath, batch_size, data_collator=None):
    tokenizer.padding_side = "left"
    pattern = r"###\s*Applicable conditions:\s*(.*?)(?:<\|endoftext\|>|<\|im_end\|>|$)"
    if data_collator is not None:
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=data_collator)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:

            inputs = {"input_ids":batch["input_ids"].to("cuda"),
                 "attention_mask":batch["attention_mask"].to("cuda"),
                 "spec_ids":batch["spec_ids"].to("cuda")}

            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                use_cache=True
            )
            texts = tokenizer.batch_decode(outputs)

            all_preds.extend(texts)
        res = []
        for i, p_ in enumerate(all_preds):

            match = re.search(pattern, p_, flags=re.DOTALL | re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                res.append(summary)
            else:
                print("exception idx: ", i)
                print(p_)
    return res

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return p, r, f1

def score_for_callback(
    references: List[List[str]],
    predictions: List[List[str]],
    soft: bool = False,
    threshold: float = 0.5,
    check_label: bool = True
) -> Dict[str, Any]:

    assert len(references) == len(predictions), "references and predictions not matched"

    overall_tp = overall_fp = overall_fn = 0

    tp_by = defaultdict(int)
    fp_by = defaultdict(int)
    fn_by = defaultdict(int)

    for reference, prediction in zip(references, predictions):
        if not reference and not prediction:
            continue

        ref_items = [parse_span_label(r) for r in reference]  
        pred_items = [parse_span_label(p) for p in prediction]

        matched_ref = set()
        matched_pred = set()

        for i, (ref_span, ref_label) in enumerate(ref_items):
            for j, (pred_span, pred_label) in enumerate(pred_items):
                if i in matched_ref or j in matched_pred:
                    continue
                if check_label and ref_label != pred_label:
                    continue

                ok = (not soft and ref_span == pred_span) or \
                     (soft and is_soft_span_match(ref_span, pred_span, threshold))

                if ok:
                    matched_ref.add(i)
                    matched_pred.add(j)
                    tp_by[ref_label] += 1
                    break

        tp = len(matched_ref)
        fp = len(pred_items) - tp
        fn = len(ref_items) - tp

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        for j, (_, pred_label) in enumerate(pred_items):
            if j not in matched_pred:
                fp_by[pred_label] += 1

        for i, (_, ref_label) in enumerate(ref_items):
            if i not in matched_ref:
                fn_by[ref_label] += 1

    labels = sorted(set(tp_by.keys()) | set(fp_by.keys()) | set(fn_by.keys()))
    per_label = {}
    for lab in labels:
        tp_l = tp_by[lab]
        fp_l = fp_by[lab]
        fn_l = fn_by[lab]
        p, r, f1 = _prf(tp_l, fp_l, fn_l)
        per_label[lab] = {"precision": p, "recall": r, "f1": f1, "tp": tp_l, "fp": fp_l, "fn": fn_l}

    excluded_labels = {
        lab for lab in fp_by
        if tp_by[lab] == 0 and fn_by[lab] == 0 and fp_by[lab] > 0
    }
    excluded_fp = sum(fp_by[lab] for lab in excluded_labels)
    overall_fp -= excluded_fp
    overall = {}
    overall["precision"], overall["recall"], overall["f1"] = _prf(overall_tp, overall_fp, overall_fn)
    overall["tp"], overall["fp"], overall["fn"] = overall_tp, overall_fp, overall_fn

    return {"overall": overall, "per_label": per_label}