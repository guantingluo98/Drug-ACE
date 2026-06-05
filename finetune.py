import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import time
import re
import copy
import torch
import json
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, rainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from datasets import load_from_disk
import random
import numpy as np
from datasets import concatenate_datasets
import metric
from typing import List, Tuple, Optional
from difflib import SequenceMatcher
import hfdataset
import model_utils
from functools import partial
import model as lora
from huggingface_hub import login
import data_utils
import relation_map
import wandb
from dataclasses import dataclass
from trl import SFTTrainer, SFTConfig
mytoken = "mytoken"
login(token = mytoken)

parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)  
parser.add_argument('--model', type=str, required=True)  
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--save_name', type=str, required=True)
parser.add_argument('--train_size', type=int, required=False, default=4)
parser.add_argument('--gas', type=int, required=False, default=2)
parser.add_argument('--val_size', type=int, required=False, default=16)
parser.add_argument('--min_epoch', type=int, required=False, default=11)
parser.add_argument('--seed', type=int, required=False, default=42)
args = parser.parse_args()
rank = args.rank
lr = args.lr
model_name = args.model
ckpt_folder = args.folder
run_name = args.run_name
save_name = args.save_name
train_batch_size = args.train_size
gas = args.gas
val_size = args.val_size
min_epoch = args.min_epoch
seed = args.seed
def switch_mode(train):
    if train:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
switch_mode(train=True)

model_utils.setSeed(seed)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if not (type(tokenizer.pad_token)==str and len(tokenizer.pad_token)>0):
    print("The tokenizer does not have a padding token. Use eos token.")
    tokenizer.pad_token = tokenizer.eos_token
if "gemma-3-4b" in model_name or "medgemma" in model_name:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
        attn_implementation="eager" 
)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
    )

if "gemma-3-4b" in model_name or "medgemma" in model_name:
    model = model.language_model

# freeze all model parameters by setting requires_grad to False for all trainable parameters:
for param in model.parameters():
    param.requires_grad = False

lora_r = rank
lora_alpha = rank*2
lora_dropout = 0.0
num_roles = 3
all_modules = ["qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
target_modules = ["qkv_proj", "k_proj", "v_proj"]
rest_modules = list(set(all_modules) - set(target_modules))

# replace target
for name, module in model.named_modules():
    # check target
    if isinstance(module, nn.Linear) and name.split('.')[-1] in target_modules:
        #print(f"Applying MyLoRALayer to: {name}")
        # get the name of current module and father module
        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name.split(".")[-1]
        parent_module = model.get_submodule(parent_name)
        # replace with new lora
        new_layer = lora.Linear_New(
            linear=module,
            rank=lora_r,
            alpha=lora_alpha,
            num_roles = num_roles,
        )
        setattr(parent_module, layer_name, new_layer)
    elif isinstance(module, nn.Linear) and name.split('.')[-1] in rest_modules:
        # get the name of current module and father module
        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name.split(".")[-1]
        parent_module = model.get_submodule(parent_name)
        # replace with standard lora
        new_layer = lora.LinearWithLoRA(
            linear=module,
            rank=lora_r,
            alpha=lora_alpha,
        )
        setattr(parent_module, layer_name, new_layer)

# check params
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    percentage = 100 * trainable_params / all_param
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {percentage:.6f}"
    )
print(f"rank: {lora_r}, alpha: {lora_alpha}")
print_trainable_parameters(model)

train_data_ = json.load(open(".../datasets/train.json"))
val_data_ = json.load(open(".../datasets/val.json"))
test_data_ = json.load(open(".../datasets/test.json"))

def getDataWithPrompt(datalist):
    res = []
    system_msg = '''
    You are a skilled biomedical text annotator. Given the title, abstract, and a drug–disease therapeutic relation, extract all applicable condition spans mentioned in the text that define **under what conditions** the drug is used to treat the disease.

    Applicable conditions include:
    - Dosage: Drug dosage, frequency, or amount taken reflecting the restriction
    - Age: Patient age or age group (e.g., elderly, children)
    - Gender: Male, female, etc
    - Comorbidity: Pre-existing diseases
    - Body type: Obesity, underweight, or general body condition
    - Gene: Specific gene patients have
    
    For each identified span, return it in the format:
    "['Span: <text> | Label: <type>', 'Span: <text> | Label: <type>', ...]"
    
    If no applicable condition is mentioned, return an empty list: []
    
    Only include spans that are explicitly mentioned or strongly implied in the context.
    Do not infer conditions beyond what is supported by the text.
    '''
    for i in range(len(datalist)):  
        question = datalist[i]["text"]
        summary = datalist[i]["target"]
        input_text = f"{question.strip()}"
        res.append({"instruction":"", "input": input_text, "output": summary.strip()})
    return res

def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=50, truncation=True, padding="max_length")
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"],
    }

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

alpaca_prompt = """
{}
### Title, Abstract, and a Drug–disease therapeutic relation:
{}
### Applicable conditions:
{}"""
EOS_TOKEN = tokenizer.eos_token 
train_data = data_utils.getDataWithPrompt(train_data_)
val_data = data_utils.getDataWithPrompt(val_data_)
test_data = data_utils.getDataWithPrompt(test_data_)
dataset__ = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})
dataset = dataset__["train"].map(data_utils.formatting_prompts_func, \
    batched = True, fn_kwargs={"eos_token": EOS_TOKEN, "train":True})
val_dataset = dataset__["validation"].map(data_utils.formatting_prompts_func, \
    batched = True, fn_kwargs={"eos_token": "", "train":False})
test_dataset = dataset__["test"].map(data_utils.formatting_prompts_func, \
    batched = True, fn_kwargs={"eos_token": "", "train":False})
targets = [json.dumps(json.loads(item["output"]), ensure_ascii=False) for item in test_dataset]
print("finishing pre-processing")
dataset = relation_map.getDatasetWithSpec_ids(dataset, tokenizer)
val_dataset = relation_map.getDatasetWithSpec_ids(val_dataset, tokenizer)
test_dataset = relation_map.getDatasetWithSpec_ids(test_dataset, tokenizer)

wandb.init(
    project="__",
    name=run_name
)

@dataclass
class CustomDataCollator(DataCollatorForLanguageModeling):
    
    spec_id_pad_value: int = 0  

    def __call__(self, features):
        spec_ids = [feature.pop("spec_ids") for feature in features] if "spec_ids" in features[0] else None
    
        batch = super().__call__(features)
        
        if spec_ids is not None:
            max_length = batch["input_ids"].shape[1] 
            
            padded_spec_ids = []
            for id_list in spec_ids:
                padded_list = id_list + [self.spec_id_pad_value] * (max_length - len(id_list))
                padded_list = padded_list[:max_length] 
                padded_spec_ids.append(padded_list)
            
            batch["spec_ids"] = torch.tensor(padded_spec_ids, dtype=torch.long)
            
        return batch
data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False, spec_id_pad_value=0)

model = model.to(torch.bfloat16)
peft_model = lora.ModelForCausalLM_New(model)

ckpt_folder = f"{ckpt_folder}_{seed}"
checkpath = f".../{ckpt_folder}/"
idx = 2
while os.path.exists(checkpath):
    checkpath = f"{checkpath}_v{idx}"
    idx += 1
os.makedirs(checkpath, exist_ok=False)
print("Created:", checkpath)

trainer = SFTTrainer(
    model = peft_model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    data_collator=data_collator,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = 2048,
        per_device_train_batch_size = train_batch_size,#8
        gradient_accumulation_steps = gas,
        warmup_steps = 5,
        num_train_epochs = 25,
        learning_rate = lr, 
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = seed,
        output_dir = None,
        report_to = "wandb", 
        save_strategy="no", 
        save_safetensors=False, 
    ),
    callbacks=[model_utils.F1EvalCallback(peft_model, val_dataset, tokenizer, 
    val_size, checkpath, min_epoch, eval_interval=1, data_collator=data_collator)]
)

start_time = time.time()
trainer_stats = trainer.train()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# load
save_path = f"{checkpath}/best.bin"
lora_state_dict = torch.load(save_path)
peft_model.load_state_dict(lora_state_dict, strict=False)

def getListFromStr(str_list):
    res = []
    for item in str_list:
        l = metric.response_string_to_list(item)
        res.append(l)
    return res

test_dataset_for_inf = test_dataset.remove_columns(
            ['instruction', 'input', 'output', 'text']
        )

model_utils.setSeed(seed)
switch_mode(train=False)
peft_model.eval()
save_path = f".../{save_name}"
with torch.no_grad():
    start_time = time.time()
    results = metric.getOutputsByBatch(peft_model, tokenizer, test_dataset_for_inf, save_path, val_size, data_collator=data_collator)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"inference completed in {elapsed_time:.2f} seconds.")
with open(save_path, "w")as f:
    for res in results:
        f.write(res+"\n")
print(f"preds saved in {save_path}")

print(len(targets), len(results))
target_list = getListFromStr(targets)
print(".............................................................")
results_list = getListFromStr(results)
metric.getScores(target_list, results_list)