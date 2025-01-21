from transformers import AutoTokenizer
from torchtune.models.llama3_2 import llama3_2_1b

def load_base_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = llama3_2_1b()
    return tokenizer, model