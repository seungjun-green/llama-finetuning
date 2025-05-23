# 🦙 Llama Fine-Tuning on Any Dataset [LoRa/DoRA]

Using this repo, you can fine-tune any LLaMA model with LoRA or DoRA in just 6 lines of code.

---

## Features

- Flexible Dataset Support: Just provide your dataset in JSON format.
- Configurable Training: Control key parameters via the config file, including:
  - LoRA or DoRa Rank & Alpha
  - Learning Rate
  - Batch Size
  - Sequence Length
  - Epochs
  - Gradient Clipping
  - Mixed Precision (FP16) Support

---

## How to Use This Repository


### 0. Get the dataset

Just prepare the training data in a json format like this:

Sample Dataset format

```json
[
    {
        "input": "Context: The capital of France is Paris. Question: What is the capital of France?\nAnswer:",
        "label": "Paris"
    },
    {
        "input": "Context: Twenty years from now you will be more disappointed by the things that you didn't do than by the ones you did do. Question: So what u gonna do?\nAnswer:",
        "label": "YES"
    }
]
```


### 1. Clone the Repository

```bash
git clone https://github.com/seungjun-green/llama-finetuning.git
```

### 2. Import Required Modules

```python
import sys
sys.path.append("/path/to/llama-finetuning")
from scripts.finetune import fine_tune
```

### 3. Fine-Tune the Model

```python
# To fine-tune the LLaMA model using DoRA, simply replace 'lora' with 'dora' here.
trainer = Finetuner('lora', config_file_path, train_file_path=train_file_path, dev_file_path=dev_file_path)
trainer.train()
```


### Example Usage
[Fine tuning Llama-3.2-1B model on SQaUD](https://github.com/seungjun-green/llama-finetuning/blob/main/notebooks/Llama_funetuning_SQuAD.ipynb)

---
## Directory Structure

```
llama-finetuning/
├── configs/
│   ├── __init__.py
│   ├── llama-3.2-1B_finetune_squad.json # configuration for fine-tuning
│   ├── finetune_config.py
├── data/
│   ├── __init__.py
│   ├── json_data.py  # Data preprocessing for any text data.
│   ├── fine_tuned_checkpoints/ # Directory for storing checkpoints
├── models/
│   ├── __init__.py
│   ├── base_model.py  # Code for loading base Llama models
│   ├── lora.py        # LoRA (Low-Rank Adaptation) implementation
│   ├── loss.py        # Loss functions for fine-tuning
├── scripts/
│   ├── eval.py        # Script for evaluating fine-tuned models
│   ├── fine_tune.py   # Fine-tuning script
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py  # Utilities for saving/loading checkpoints
│   ├── helpers.py     # Helper functions
```



## Experiments

I evaluated the performance of fine-tuning the LLaMA model using two methods: LoRA and DoRA. The results are as follows:

- LoRA: Achieved a loss score of 1.5245
- DoRA: Achieved a loss score of 1.3891

As demonstrated in the DoRA paper, fine-tuning the LLaMA model with DoRA resulted in a lower loss score compared to LoRA

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers Library

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request for suggestions, bug fixes, or new features.

---

Happy fine-tuning! 🎉