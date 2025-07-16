import torch
import os
from transformers import BartForCausalLM, GenerationConfig, BartConfig


class MyBartForCausalLM(BartForCausalLM):
    def __init__(self, config, tokenizer):
        # Initialize and pass custom BartConfig
        bart_config = BartConfig(**config.bart)
        bart_config.vocab_size = tokenizer.vocab_size
        bart_config.pad_token_id = tokenizer.pad_token_id
        bart_config.bos_token_id = tokenizer.bos_token_id
        bart_config.eos_token_id = tokenizer.eos_token_id

        super().__init__(bart_config)

        # Record external config/tokenizer, construct generation configuration
        self.custom_config = config
        self.tokenizer = tokenizer
        if 'generation_config' in config:
            self.generate_config = GenerationConfig(**config.generation_config)
        else:
            self.generate_config = GenerationConfig()

    def generate(self, 
                 input_ids=None, 
                 attention_mask=None,
                 **generate_kwargs):
        # During non-training, rewrite input with bos token
        if not self.training:
            device = self.device 
            bos_token_id = self.config.bos_token_id
            input_ids = torch.tensor(
                [[bos_token_id]] * input_ids.shape[0], 
                dtype=torch.long, 
                device=device
            )
            attention_mask = None

        # Use parent class BartForCausalLM's generate and pass custom generation_config
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generate_config,
            **generate_kwargs
        )
    
    def load_weights(self, model_path):
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
        self.load_state_dict(state_dict, strict=False)

