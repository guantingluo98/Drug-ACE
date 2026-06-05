import re
from typing import List, Tuple
import random

def mark_dual_keyword_tokens(
    text: str,
    keyword1: str,
    keyword2: str,
    tokenizer,
    spec_id_1: int = 1,  
    spec_id_2: int = 2,  
    spec_id_nonmatch: int = 0,
    case_insensitive: bool = True,
) -> List[int]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True, 
    )
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    flags = re.IGNORECASE if case_insensitive else 0
    escaped_k1 = re.escape(keyword1)
    escaped_k2 = re.escape(keyword2)
    spans1 = [
        (m.start(), m.end())
        for m in re.finditer(escaped_k1, text, flags=flags)
    ]
    spans2 = [
        (m.start(), m.end())
        for m in re.finditer(escaped_k2, text, flags=flags)
    ]
    spec_ids = []
    
    for (s, e) in offsets:
        if s == e:
            spec_ids.append(spec_id_nonmatch)
            continue
        
        hit_k1 = any(not (e <= ks or s >= ke) for ks, ke in spans1)
        
        if hit_k1:
            spec_ids.append(spec_id_1)
            continue 

        hit_k2 = any(not (e <= ks or s >= ke) for ks, ke in spans2)
        if hit_k2:
            spec_ids.append(spec_id_2)
        else:
            spec_ids.append(spec_id_nonmatch)
  
    return spec_ids

def extract_drug_disease(text):
    
    if not text:
        return None, None

    pattern = r"\[Drug-Disease\]:\s*(?P<drug>.+?)\s+-\s+(?P<disease>.+)"
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        drug = match.group("drug").strip()
        disease = match.group("disease").strip()
        return drug, disease
    else:
        return None, None


def getDatasetWithSpec_ids(dataset, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False, 
        )
    def add_spec_ids_fn(example):
        text = example['text']
        key1, key2 = extract_drug_disease(text)
        spec_ids = mark_dual_keyword_tokens(
            text=text,
            keyword1=key1,
            keyword2=key2,
            tokenizer=tokenizer,
        )
        return {"spec_ids": spec_ids}
    tokenized_ds = dataset.map(tokenize_fn, batched=False)
    ds_with_spec_ids = tokenized_ds.map(add_spec_ids_fn, batched=False)
    return ds_with_spec_ids