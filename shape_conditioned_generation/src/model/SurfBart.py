import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartForConditionalGeneration, BartForCausalLM, BartConfig, GenerationConfig
from transformers.modeling_outputs import BaseModelOutput
from src.model.risurconv_utils import RISurConvSetAbstraction


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate, activation_function):
        super(MLP, self).__init__()
        
        # Ensure activation function is a callable
        assert callable(activation_function), "activation_function must be callable"

        # Define the MLP layers
        layers = [
            nn.Linear(input_dim, hidden_dim),
            activation_function,
            nn.Dropout(dropout_rate)
        ]

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                activation_function,
                nn.Dropout(dropout_rate)
            ])

        layers.append(nn.Linear(hidden_dim, output_dim))

        # Use nn.Sequential to stack all layers
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SurfaceBartv2(nn.Module):
    def __init__(self, config, tokenizer):
        super(SurfaceBartv2, self).__init__()
        self.tokenizer = tokenizer
        bart_config = BartConfig(**config.bart)
        bart_config.vocab_size = self.tokenizer.vocab_size
        # bart_config.vocab_size = 460
        bart_config.pad_token_id = self.tokenizer.pad_token_id
        bart_config.bos_token_id = self.tokenizer.bos_token_id
        bart_config.eos_token_id = self.tokenizer.eos_token_id
        
        self.normal_channel = config.surf.normal_channel
        self.n = config.surf.n

        self.sc0 = RISurConvSetAbstraction(
            npoint=512*self.n,
            radius=0.12,
            nsample=8,
            in_channel=0,
            out_channel=32,
            group_all=False
        )
        self.sc1 = RISurConvSetAbstraction(
            npoint=256*self.n,
            radius=0.16,
            nsample=16,
            in_channel=32,
            out_channel=64,
            group_all=False
        )
        self.sc2 = RISurConvSetAbstraction(
            npoint=128*self.n,
            radius=0.24,
            nsample=32,
            in_channel=64,
            out_channel=128,
            group_all=False
        )
        self.sc3 = RISurConvSetAbstraction(
            npoint=64*self.n,
            radius=0.48,
            nsample=64,
            in_channel=128,
            out_channel=256,
            group_all=False
        )
        self.sc4 = RISurConvSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256,
            out_channel=512,
            group_all=True
        )

        if config.mlp.activation_function == 'relu':
            activation_function = nn.ReLU()
        elif config.mlp.activation_function == 'gelu':
            activation_function = nn.GELU()
        elif config.mlp.activation_function == 'tanh':
            activation_function = nn.Tanh()
        elif config.mlp.activation_function == 'leaky_relu':
            activation_function = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {config.mlp.activation_function}")
        
        self.mlp = MLP(
            input_dim=512,
            hidden_dim=config.mlp.hidden_dim,
            output_dim=bart_config.hidden_size,  # Bart 的 hidden_size
            num_layers=config.mlp.num_layers,
            dropout_rate=config.mlp.dropout_rate,
            activation_function=activation_function,
        )

        # 这里依然使用 BartForConditionalGeneration，但我们会通过 encoder_outputs 跳过其内部 encoder
        self.bart = BartForConditionalGeneration(bart_config)

        if 'generation_config' not in config:
            self.generation_config = GenerationConfig()
        else:
            self.generation_config = GenerationConfig(**config.generation_config)


    def forward(self, pointcloud, normals, labels, attention_mask=None):
        B, _, _ = pointcloud.shape
        if self.normal_channel:
            norm = normals
        else:
            norm = None

        # 1. 点云特征提取
        l0_xyz, l0_norm, l0_points = self.sc0(pointcloud, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        # 2. 维度变换 + MLP
        #    l4_points形状一般是 [batch_size, feature_dim, 1] 或 [batch_size, feature_dim, n], 视具体实现而定
        x = l4_points.permute(0, 2, 1)  # 变成 [batch_size, seq_len, feature_dim]
        x = self.mlp(x)                # [batch_size, seq_len, hidden_size]

        # 如果 attention_mask=None，需要自己构造
        if attention_mask is None:
            attention_mask = torch.ones((B, x.size(1)), dtype=torch.long, device=x.device)

        # 3. 显式构造 encoder_outputs，跳过 BART Encoder
        #    BaseModelOutput 接受 last_hidden_state, hidden_states, attentions 等参数
        encoder_outputs = BaseModelOutput(last_hidden_state=x)

        # 4. 调用 BART: 把 encoder_outputs 显式传进去，然后让 BART 的 Decoder 做解码
        #    labels 不为空时，会在 forward 里计算交叉熵损失
        outputs = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False  # 可以根据需要设置
        )
        
        return outputs

    def load_weights(self, path):
        assert os.path.exists(path), f"File not found: {path}"
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
    
    @torch.no_grad()
    def sample(
        self,
        pointcloud=None,
        normals=None,
        generation_config=None,
        attention_mask=None,
        prefix_ids=None,  # <--- 新增参数: [token1, token2, ...]
        logits_processor=None,
        **kwargs
    ):
        if "labels" in kwargs:
            kwargs.pop("labels")

        B, _, _ = pointcloud.shape if pointcloud is not None else (1, 1, 1)
        norm = normals if self.normal_channel else None

        # 1. 点云特征提取...
        l0_xyz, l0_norm, l0_points = self.sc0(pointcloud, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        x = l4_points.permute(0, 2, 1)
        x = self.mlp(x)

        if attention_mask is None:
            attention_mask = torch.ones((B, x.size(1)), dtype=torch.long, device=x.device)

        encoder_outputs = BaseModelOutput(last_hidden_state=x)

        # 如果传了 generation_config 就用，否则用自己的 self.generation_config
        if generation_config is None:
            gen_config = self.generation_config
        else:
            gen_config = GenerationConfig(**generation_config)

        # 3. 如果 prefix_ids 不为空，就构建 decoder_input_ids
        decoder_input_ids = None
        if prefix_ids is not None:
            # 假设 batch_size=B，你需要让 prefix_ids 形状匹配 [B, prefix_len]
            # 示例：如果只是想让所有样本用同样的 prefix_ids，可这样做
            # prefix_ids 是 list[int]
            B = x.size(0)  # x: [B, seq_len, hidden_dim]
            if isinstance(prefix_ids, list):
                prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=x.device)
            prefix_ids = prefix_ids.unsqueeze(0).expand(B, -1)
            decoder_input_ids = prefix_ids  # [B, prefix_len]

        outputs = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            generation_config=gen_config,
            return_dict_in_generate=True,
            output_scores=True,
            decoder_input_ids=decoder_input_ids,  # <--- 这里
            logits_processor=logits_processor,  # <--- 新增参数
            **kwargs
        )

        generated_ids = outputs.sequences
        
        # 计算分数的逻辑省略...
        transition_scores = self.bart.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        output_length = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        length_penalty = gen_config.length_penalty
        true_scores = transition_scores.cpu().sum(axis=1) / (output_length ** length_penalty)

        return generated_ids, true_scores
    

    @torch.no_grad()
    def generate(self, 
                 pointcloud, 
                 normals, 
                 attention_mask=None,
                 generation_config=None,
                #  prefix_ids=[0, 460], 
                prefix_ids=[0],
                 **kwargs):
        if "labels" in kwargs:
            kwargs.pop("labels")

        B, _, _ = pointcloud.shape if pointcloud is not None else (1, 1, 1)
        if self.normal_channel:
            norm = normals
        else:
            norm = None

        # 1. 点云特征提取
        l0_xyz, l0_norm, l0_points = self.sc0(pointcloud, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        # 2. 维度变换 + MLP
        x = l4_points.permute(0, 2, 1)
        x = self.mlp(x)

        if attention_mask is None:
            attention_mask = torch.ones((B, x.size(1)), dtype=torch.long, device=x.device)

        # 3. 构造 encoder_outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=x)

        # 4. 处理 generation_config
        if generation_config is None:
            gen_config = self.generation_config
        else:
            gen_config = GenerationConfig(**generation_config)

        decoder_input_ids = None
        if prefix_ids is not None:
            # 假设 batch_size=B，你需要让 prefix_ids 形状匹配 [B, prefix_len]
            # 示例：如果只是想让所有样本用同样的 prefix_ids，可这样做
            # prefix_ids 是 list[int]
            B = x.size(0)  # x: [B, seq_len, hidden_dim]
            if isinstance(prefix_ids, list):
                prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=x.device)
            prefix_ids = prefix_ids.unsqueeze(0).expand(B, -1)
            decoder_input_ids = prefix_ids  # [B, prefix_len]

        # 5. 使用 BART 的 generate()，但把编码器结果直接传进去
        outputs = self.bart.generate(
            encoder_outputs=encoder_outputs, 
            attention_mask=attention_mask,
            generation_config=gen_config,
            decoder_input_ids=decoder_input_ids,  # <--- 这里
            **kwargs
        )

        return outputs
    
