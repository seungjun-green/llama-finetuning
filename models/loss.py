import torch.nn as nn

def get_loss_function(loss_type="cross_entropy"):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
