import json

class Config:
    def __init__(self, config_path=None, **kwargs):
        # initialize the configuration. Load from a JSON file if provided, and override defaults with kwargs.
        self.base_model_name = "meta-llama/Llama-3.2-1B"
        self.lora_rank = 8
        self.lora_alpha = 16
        self.learning_rate = 1e-4
        self.batch_size = 16
        self.num_epochs = 3
        self.output_dir = "./checkpoints"
        self.save_steps = 100
        self.eval_steps = 50
        self.max_seq_length = 128

        # if a JSON file is provided, load settings from it
        if config_path:
            self.load_from_json(config_path)

        # override settings with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def load_from_json(self, config_path):
        """
        Load configuration settings from a JSON file.
        """
        with open(config_path, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)