import os
import torch
from transformers import TrainerCallback
from typing import List, Tuple, Optional
import metric
import json
import random
import numpy as np
import wandb

def setSeed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_lora_dict(model):
    full_state_dict = model.state_dict()
    lora_state_dict = {}
    for name, param in full_state_dict.items():
        if ".lora." in name:
            lora_state_dict[name] = param
    return lora_state_dict

class F1EvalCallback(TrainerCallback):
    def __init__(self, model, eval_dataset, tokenizer, 
                batch_size, output_dir, begin, eval_interval=2, data_collator=None):
        self.model = model
        self.eval_dataset_ = eval_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.begin = begin
        self.eval_interval = eval_interval
        self.best_f1 = -1.0
        self.best_ckpt = None
        self.targets = self.getTargets()
        self.eval_dataset = self.getProcessedData()
        self.data_collator = data_collator
    
    def getProcessedData(self):
        processed_test_dataset = self.eval_dataset_.remove_columns(
            ['instruction', 'input', 'output', 'text']
        )
        return processed_test_dataset

    def getTargets(self):
        targets = [json.dumps(json.loads(item["output"]), ensure_ascii=False)
                     for item in self.eval_dataset_]
        return metric.getListFromStr(targets)


    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch<self.begin:
            return
        if state.epoch is None or int(state.epoch) % self.eval_interval != 0:
            return
        setSeed(seed=42)
        self.model.eval()

        with torch.no_grad():
            preds_path = None
            preds = metric.getOutputsByBatch(self.model, 
                self.tokenizer, self.eval_dataset,
                 None, self.batch_size, self.data_collator)
            preds_list = metric.getListFromStr(preds)
        score = metric.score_for_callback(self.targets, preds_list, soft=True)
        f1 = score["overall"]["f1"]

        wandb.log({"eval_f1": f1, "epoch": state.epoch})
        print(f"[Eval F1] Epoch {state.epoch:.1f} F1={f1:.4f}")
        if not hasattr(self, "no_improve_epochs"):
            self.no_improve_epochs = 0
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.no_improve_epochs = 0
            ckpt_dir_lora = self.output_dir
            os.makedirs(ckpt_dir_lora, exist_ok=True)
            best_ckpt_path = os.path.join(ckpt_dir_lora, "best.bin")
            if getattr(self, "best_ckpt", None) and os.path.isfile(self.best_ckpt):
                os.remove(self.best_ckpt)
            lora_state_dict = get_lora_dict(self.model)
            torch.save(lora_state_dict, best_ckpt_path)
            self.best_ckpt = best_ckpt_path
            print(f"New best F1={f1:.4f}, checkpoint saved to {best_ckpt_path}")
            ckpt_dir_lora = os.path.join(self.output_dir, "")
        else:
            self.no_improve_epochs += 1
            print(f"No F1 improvement for {self.no_improve_epochs} epoch(s).")
            if self.no_improve_epochs >= 3:
                print(f"Early stopping triggered at epoch {state.epoch:.1f} (no F1 improvement for 3 epochs).")
                control.should_training_stop = True
                return control