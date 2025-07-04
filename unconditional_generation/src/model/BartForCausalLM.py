import torch
import os
from transformers import BartForCausalLM, GenerationConfig, BartConfig


class MyBartForCausalLM(BartForCausalLM):
    def __init__(self, config, tokenizer):
        # 初始化并传入自定义BartConfig
        bart_config = BartConfig(**config.bart)
        bart_config.vocab_size = tokenizer.vocab_size
        bart_config.pad_token_id = tokenizer.pad_token_id
        bart_config.bos_token_id = tokenizer.bos_token_id
        bart_config.eos_token_id = tokenizer.eos_token_id

        super().__init__(bart_config)

        # 记录外部config/分词器，构造生成配置
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
        # 非训练时用bos token重写输入
        if not self.training:
            device = self.device 
            bos_token_id = self.config.bos_token_id
            input_ids = torch.tensor(
                [[bos_token_id]] * input_ids.shape[0], 
                dtype=torch.long, 
                device=device
            )
            attention_mask = None

        # 使用父类BartForCausalLM的generate，并传入自定义的generation_config
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generate_config,
            **generate_kwargs
        )
    
    def load_weights(self, model_path):
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
        self.load_state_dict(state_dict, strict=False)

