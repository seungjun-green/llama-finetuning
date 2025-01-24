import torch
import torchtune
from safetensors.torch import load_file
from torchtune.models.llama3_2 import llama3_2_1b


# model_weights_path = "/tmp/Llama-3.2-1B-Instruct/model.safetensors"
# tokenizer_path = "/tmp/Llama-3.2-1B-Instruct/original/tokenizer.model"


def load_base_model(model_weights_path):
    # load model
    model = llama3_2_1b()
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict, strict=False)
    
    return model

def load_model_and_tokenizer(model_weights_path, tokenizer_path):
    # load model
    model = llama3_2_1b()
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict, strict=False)

    # load tokenizer
    tokenizer = torchtune.models.llama3.llama3_tokenizer(tokenizer_path)
    return model, tokenizer