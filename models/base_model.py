from transformers import AutoTokenizer, AutoModelForCausalLM

def load_base_model(model_name):
    """load tokenizer and model
    Args:
        model_name: model name, huggingface model repo name
    Returns:
        tokenizer, base_model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, base_model