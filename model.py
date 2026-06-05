import torch
import torch.nn.functional as F
from torch import nn
import math

class LoRALayer_New(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, num_roles):
        super().__init__()
        self.r = rank
        self.alpha = alpha
        self.num_roles = num_roles
        self.lora_A = nn.Linear(in_dim, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_dim, bias=False)
        self.embed = nn.Embedding(self.num_roles, self.r)
        self.lora_B2 = nn.Linear(self.r, out_dim, bias=False)
        self.scaling = self.alpha / self.r
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.embed.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B2.weight)
        with torch.no_grad():
            self.embed.weight[0].zero_()
        self.embed.weight.requires_grad_(True)

    def forward(self, x, ids):
        if ids is None:
             raise ValueError("ids cannot be None")
        else:
            if ids.dim() == 2 and x.dim() == 3:
                Tx = x.size(1)
                if ids.size(1) != Tx:
                    ids = ids[:, -Tx:]
            
            is_valid = torch.all((ids >= 0) & (ids < self.embed.num_embeddings))
            
            if not is_valid:
                raise IndexError(f"role_ids contains values out of bounds for nn.Embedding with size {self.embed.num_embeddings}")

        x_lora = self.lora_B(self.lora_A(x)) + self.lora_B2(self.embed(ids))
        return self.scaling * x_lora

class ModelForCausalLM_New(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model  

    def _dispatch_extra(self, ids):
        for m in self.model.modules():
            if hasattr(m, "_accept_extra_inputs"):
                m.ids = ids

    def forward(self, input_ids=None, attention_mask=None, spec_ids=None, **kwargs):
        ids = spec_ids
        if (ids is not None):
            self._dispatch_extra(ids)
        else:
            raise ValueError("ids cannot be None")
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, *args, **kwargs):
        ids = kwargs.pop("spec_ids", None)
        if ids is not None:
            self._dispatch_extra(ids)
            self._cached_ids = ids  
        else:
            self._dispatch_extra(self._cached_ids)
        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        ids = kwargs.pop("spec_ids", None)
        cur = kwargs.get("input_ids", None) or kwargs.get("decoder_input_ids", None)
        if ids is not None and cur is not None and ids.dim() == 2 and cur.dim() == 2:
            ids = ids[:, -cur.size(1):]
        if ids is not None:
            self._dispatch_extra(ids)
            self._cached_ids = ids
        else:
            if self._cached_ids is not None:
                self._dispatch_extra(self._cached_ids)

        if hasattr(self.model, "prepare_inputs_for_generation"):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)

class Linear_New(torch.nn.Module):
    _accept_extra_inputs = True
    def __init__(self, linear, rank, alpha, num_roles):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer_New(
            linear.in_features, linear.out_features, rank, alpha, num_roles
        )
        self.ids = None

    def forward(self, x):
        return self.linear(x) + self.lora(x, self.ids)
