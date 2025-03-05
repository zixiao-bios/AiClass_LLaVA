from lora_layer import LoRALayer
from torch import nn


def apply_lora(model, target_modules, lora_rank, lora_alpha):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            apply_lora(module, target_modules, lora_rank, lora_alpha)
            
        if isinstance(module, nn.Linear) and name in target_modules:
            # 替换线性层为LoRALayer
            new_layer = LoRALayer(module, lora_rank, lora_alpha)
            setattr(model, name, new_layer)
    return model

def load_lora_weight(model, lora_weights=None):
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            if lora_weights is None:
                # 未提供权重时，将模型还原（lora 权重为 0）
                param.data.zero_()
            elif name in lora_weights:
                param.data.copy_(lora_weights[name])
            else:
                print(f"Warning: {name} not found in loaded lora_weights.")
