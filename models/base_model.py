from transformers import AutoTokenizer, AutoModelForCausalLM

def load_base_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, base_model