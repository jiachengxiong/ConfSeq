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
    

class SurfaceBart(nn.Module):
    def __init__(self, config, tokenizer):
        super(SurfaceBart, self).__init__()
        self.tokenizer = tokenizer
        bart_config = BartConfig(**config.bart)
        bart_config.vocab_size = self.tokenizer.vocab_size
        bart_config.pad_token_id = self.tokenizer.pad_token_id
        bart_config.bos_token_id = self.tokenizer.bos_token_id
        bart_config.eos_token_id = self.tokenizer.eos_token_id
        
        self.normal_channel = config.surf.normal_channel
        self.n = config.surf.n

        self.sc0 = RISurConvSetAbstraction(npoint=512*self.n, radius=0.12, nsample=8, in_channel=0, out_channel=32, group_all=False)
        self.sc1 = RISurConvSetAbstraction(npoint=256*self.n, radius=0.16, nsample=16, in_channel=32, out_channel=64,  group_all=False)
        self.sc2 = RISurConvSetAbstraction(npoint=128*self.n, radius=0.24, nsample=32, in_channel=64, out_channel=128,  group_all=False)
        self.sc3 = RISurConvSetAbstraction(npoint=64*self.n, radius=0.48, nsample=64, in_channel=128, out_channel=256,  group_all=False)
        self.sc4 = RISurConvSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, out_channel=512,  group_all=True)

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
			output_dim=bart_config.hidden_size,
			num_layers=config.mlp.num_layers,
			dropout_rate=config.mlp.dropout_rate,
			activation_function=activation_function,
			)

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

        l0_xyz, l0_norm, l0_points = self.sc0(pointcloud, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        x=l4_points.permute(0, 2, 1) # [B, npoint, 512]
        x=self.mlp(x) # [B, npoint, hidden_size]
        
        if attention_mask is None:
            # 只有 1 个 token，全为有效
            attention_mask = torch.ones((B, 1), dtype=torch.long)

        outputs = self.bart(
            inputs_embeds=x,
            labels=labels,
            attention_mask=attention_mask,
        )
        
        return outputs
    

    def load_weights(self, path):
        assert os.path.exists(path), f"File not found: {path}"
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict)
    

    def generate(
        self,
        pointcloud=None,
        normals=None,
        generation_config=None,
        attention_mask=None,
        **kwargs
    ):
        """
        自定义的 generate 方法，注意增加 **kwargs 以兼容多余参数。
        """
        # 如果 Trainer 传入了 "labels" 或其他无关参数，这里显式丢弃
        if "labels" in kwargs:
            kwargs.pop("labels")

        # 下面是原先的计算逻辑
        B, _, _ = pointcloud.shape if pointcloud is not None else (1, 1, 1)

        if self.normal_channel:
            norm = normals
        else:
            norm = None

        l0_xyz, l0_norm, l0_points = self.sc0(pointcloud, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        x = l4_points.permute(0, 2, 1)
        x = self.mlp(x)

        # 如果 attention_mask=None，需要自己创建
        if attention_mask is None:
            attention_mask = torch.ones((B, 1), dtype=torch.long, device=x.device)

        # 把 generation_config 转为 GenerationConfig 对象
        if generation_config is None:
            # 如果用户没传，则用 self.generation_config
            gen_config = self.generation_config
        else:
            gen_config = GenerationConfig(**generation_config)

        outputs = self.bart.generate(
            inputs_embeds=x,
            attention_mask=attention_mask,
            generation_config=gen_config,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs  # 其余正常传给 bart.generate
        )

        generated_ids = outputs.sequences
        generated_scores = []

        transition_scores = self.bart.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )
        output_length = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        length_penalty = self.generation_config.length_penalty
        true_scores = transition_scores.cpu().sum(axis=1) / (output_length ** length_penalty)

        generated_scores.extend(true_scores)

        return generated_ids, generated_scores
    


class SurfaceBartCausal(nn.Module):
    def __init__(self, config, tokenizer):
        super(SurfaceBartCausal, self).__init__()
        self.tokenizer = tokenizer
        
        # ------------ 1. 构造 BartConfig ------------
        bart_config = BartConfig(**config.bart)
        # 校正词表大小及特殊 token ID
        bart_config.vocab_size     = self.tokenizer.vocab_size
        bart_config.pad_token_id   = self.tokenizer.pad_token_id
        bart_config.bos_token_id   = self.tokenizer.bos_token_id
        bart_config.eos_token_id   = self.tokenizer.eos_token_id

        # ------------ 2. 点云相关超参数 ------------
        self.normal_channel = config.surf.normal_channel
        self.n = config.surf.n

        # 通过多次 RISurConvSetAbstraction 抽取点云特征
        # 这些类与参数请按照你在项目中对应的实现进行替换/修改
        self.sc0 = RISurConvSetAbstraction(npoint=512*self.n, radius=0.12, nsample=8,
                                           in_channel=0, out_channel=32, group_all=False)
        self.sc1 = RISurConvSetAbstraction(npoint=256*self.n, radius=0.16, nsample=16,
                                           in_channel=32, out_channel=64, group_all=False)
        self.sc2 = RISurConvSetAbstraction(npoint=128*self.n, radius=0.24, nsample=32,
                                           in_channel=64, out_channel=128, group_all=False)
        self.sc3 = RISurConvSetAbstraction(npoint=64*self.n, radius=0.48, nsample=64,
                                           in_channel=128, out_channel=256, group_all=False)
        self.sc4 = RISurConvSetAbstraction(npoint=None, radius=None, nsample=None,
                                           in_channel=256, out_channel=512, group_all=True)

        # ------------ 3. 构建 MLP，将点云特征映射到 BART hidden_size ------------
        act_str = config.mlp.activation_function.lower()
        if   act_str == 'relu':
            activation_function = nn.ReLU()
        elif act_str == 'gelu':
            activation_function = nn.GELU()
        elif act_str == 'tanh':
            activation_function = nn.Tanh()
        elif act_str == 'leaky_relu':
            activation_function = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {config.mlp.activation_function}")

        self.mlp = MLP(
            input_dim=512,                    # sc4 的输出通道数
            hidden_dim=config.mlp.hidden_dim,
            output_dim=bart_config.hidden_size,
            num_layers=config.mlp.num_layers,
            dropout_rate=config.mlp.dropout_rate,
            activation_function=activation_function
        )

        # ------------ 4. 实例化 decoder-only 的 BartForCausalLM ------------
        self.bart = BartForCausalLM(bart_config)
        
        # 记录生成时的默认配置
        if 'generation_config' not in config:
            self.generation_config = GenerationConfig()
        else:
            self.generation_config = GenerationConfig(**config.generation_config)

        # 为了后续可直接拿到文本嵌入层
        # （BartForCausalLM 中同样有 get_input_embeddings）
        self.text_embedding_layer = self.bart.get_input_embeddings()

    def forward(self, pointcloud, normals, labels):
        """
        训练阶段的 forward。
        参数：
          - pointcloud:  [B, N, 3]  点云坐标
          - normals:     [B, N, 3]  点云法线（若 normal_channel=True，则启用）
          - input_ids:   [B, seq_len] 文本序列 Token ID
          - labels:      [B, seq_len] 训练目标序列 ID（可与 input_ids 相同 or shift 后序列）
        """
        B, _, _ = pointcloud.shape

        # 1) 如果需要法线，则传，否则传 None
        norm = normals if self.normal_channel else None

        # 2) 通过 RISurConvSetAbstraction 提取点云特征
        l0_xyz, l0_norm, l0_points = self.sc0(pointcloud, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        # 3) 提取到的特征默认形状 [B, feat_dim, npoint]
        #    如果 group_all=True, 通常 npoint=1
        #    先 permute，再过 MLP
        #    => x shape: [B, npoint, hidden_size]
        x = l4_points.permute(0, 2, 1)     # [B, npoint, 512]
        x = self.mlp(x)                   # [B, npoint, bart_config.hidden_size]

        # 假设 npoint = cond_len，可能=1，也可能是>1
        cond_len = x.shape[1]

        # 4) 将文本 Token IDs 转换为 embedding
        #    text_embeds: [B, seq_len, hidden_size]
        text_embeds = self.text_embedding_layer(labels)

        # 5) 拼接“点云特征 + 文本嵌入”到一起
        #    使得序列维度 = cond_len + seq_len
        inputs_embeds = torch.cat([x, text_embeds], dim=1) # [B, cond_len + seq_len, hidden_size]

        # 6) 构造 attention_mask（全部设为1，或根据实际需求进行pad）
        total_seq_len = inputs_embeds.size(1)  # cond_len + seq_len
        attention_mask = torch.ones((B, total_seq_len),
                                    dtype=torch.long,
                                    device=inputs_embeds.device)
        
        # 给 labels 拼上 cond_len 大小的 -100
        #    这样 new_labels 的总长度就变成 cond_len + seq_len
        ignore_labels = torch.full(
            (labels.size(0), cond_len),  # [B, cond_len]
            -100,                        # 忽略loss
            dtype=labels.dtype,
            device=labels.device
        )
        new_labels = torch.cat([ignore_labels, labels], dim=1)  # [B, cond_len + seq_len]

        # 调用 self.bart
        outputs = self.bart(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=new_labels,  # 使用新的labels
            use_cache=False
        )

        return outputs  # 包含 loss, logits 等

    @torch.no_grad()
    def generate(self, pointcloud, normals, input_ids=None, generation_config=None, **kwargs):
        """
        推理/生成阶段。如果希望在点云条件后面还添加文本提示，可传 input_ids，
        否则只用点云特征做“序列开头”，让模型自由生成后续 Token。
        
        参数：
          - pointcloud:  [B, N, 3] 点云坐标
          - normals:     [B, N, 3] 点云法线
          - input_ids:   [B, prompt_len], 生成时可选的文本前缀
          - generation_config: 若不指定，则默认使用 self.generation_config
          - **kwargs:  其余传给 bart.generate 的额外参数（如 max_new_tokens, do_sample 等）
        返回：
          - generated_ids: [B, out_seq_len] 生成序列的 token id
          - scores: list 或 None，若需要计算自定义分数可扩展
        """
        # 1) 点云特征
        B, _, _ = pointcloud.shape
        norm = normals if self.normal_channel else None

        l0_xyz, l0_norm, l0_points = self.sc0(pointcloud, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        x = l4_points.permute(0, 2, 1)    # [B, npoint, 512]
        x = self.mlp(x)                  # [B, npoint, hidden_size]

        # 2) 如果还带有文本 prompt，就嵌入后与点云特征拼接
        if input_ids is not None:
            text_embeds = self.text_embedding_layer(input_ids)
            inputs_embeds = torch.cat([x, text_embeds], dim=1)
        else:
            inputs_embeds = x

        # attention_mask
        B, total_seq_len, _ = inputs_embeds.shape
        attention_mask = torch.ones((B, total_seq_len),
                                    dtype=torch.long,
                                    device=inputs_embeds.device)

        # 3) 处理 generation_config
        if generation_config is None:
            gen_config = self.generation_config
        else:
            gen_config = GenerationConfig(**generation_config)

        # 4) 调用 BartForCausalLM.generate 进行自回归生成
        outputs = self.bart.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=gen_config,
            **kwargs
        )

        return outputs
    

    @torch.no_grad()
    def sample(self, pointcloud, normals, input_ids=None, generation_config=None, **kwargs):
        """
        推理/生成阶段。如果希望在点云条件后面还添加文本提示，可传 input_ids，
        否则只用点云特征做“序列开头”，让模型自由生成后续 Token。
        
        参数：
          - pointcloud:  [B, N, 3] 点云坐标
          - normals:     [B, N, 3] 点云法线
          - input_ids:   [B, prompt_len], 生成时可选的文本前缀
          - generation_config: 若不指定，则默认使用 self.generation_config
          - **kwargs:  其余传给 bart.generate 的额外参数（如 max_new_tokens, do_sample 等）
        返回：
          - generated_ids: [B, out_seq_len] 生成序列的 token id
          - scores: list 或 None，若需要计算自定义分数可扩展
        """
        # 1) 点云特征
        B, _, _ = pointcloud.shape
        norm = normals if self.normal_channel else None

        l0_xyz, l0_norm, l0_points = self.sc0(pointcloud, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        x = l4_points.permute(0, 2, 1)    # [B, npoint, 512]
        x = self.mlp(x)                  # [B, npoint, hidden_size]

        # 2) 如果还带有文本 prompt，就嵌入后与点云特征拼接
        if input_ids is not None:
            text_embeds = self.text_embedding_layer(input_ids)
            inputs_embeds = torch.cat([x, text_embeds], dim=1)
        else:
            inputs_embeds = x

        # attention_mask
        B, total_seq_len, _ = inputs_embeds.shape
        attention_mask = torch.ones((B, total_seq_len),
                                    dtype=torch.long,
                                    device=inputs_embeds.device)

        # 3) 处理 generation_config
        if generation_config is None:
            gen_config = self.generation_config
        else:
            gen_config = GenerationConfig(**generation_config)

        # 4) 调用 BartForCausalLM.generate 进行自回归生成
        outputs = self.bart.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=gen_config,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs
        )

        generated_ids = outputs.sequences

        # 若想计算类似序列评分，可使用 bart.compute_transition_scores
        # 下方仅做示例性操作
        transition_scores = self.bart.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True
        )
        # 这里简单返回 token 序列与每个样本的平均分
        # 不同库版本中 transition_scores 的 shape 可能略有不同，需根据实际情况处理
        # 下方仅演示做法
        output_length = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        length_penalty = gen_config.length_penalty
        true_scores = transition_scores.cpu().sum(axis=1) / (output_length ** length_penalty)

        return generated_ids, true_scores


    def load_weights(self, path):
        assert os.path.exists(path), f"File not found: {path}"
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict)
        print(f"Weights loaded from: {path}")



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
    

class SurfaceBartv3(nn.Module):
    def __init__(self, config, tokenizer):
        super(SurfaceBartv3, self).__init__()
        self.tokenizer = tokenizer
        bart_config = BartConfig(**config.bart)
        bart_config.vocab_size = self.tokenizer.vocab_size
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
            npoint=16 * self.n,    # 这里想保留多少个点，可以灵活调整
            radius=0.8,            # 自行设置合适的搜索半径
            nsample=64,            # 邻域内点的采样数
            in_channel=256,
            out_channel=512,
            group_all=False        # 不再使用全局汇聚
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
    

    @torch.no_grad()
    def sample(
        self,
        pointcloud=None,
        normals=None,
        generation_config=None,
        attention_mask=None,
        prefix_ids=None,  # <--- 新增参数: [token1, token2, ...]
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
                 prefix_ids=[0, 460], 
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
    
    def load_weights(self, path):
        assert os.path.exists(path), f"File not found: {path}"
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
        self.load_state_dict(state_dict, strict=False)


class SurfaceBartv4(nn.Module):
    def __init__(self, config, tokenizer):
        super(SurfaceBartv4, self).__init__()
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
            out_channel=768,
            group_all=True
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
                 prefix_ids=[0, 460], 
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
    

class SurfaceBartv5(nn.Module):
    def __init__(self, config, tokenizer):
        super(SurfaceBartv5, self).__init__()
        self.tokenizer = tokenizer
        bart_config = BartConfig(**config.bart)
        bart_config.vocab_size = self.tokenizer.vocab_size
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
            npoint=4 * self.n,    # 这里想保留多少个点，可以灵活调整
            radius=0.8,            # 自行设置合适的搜索半径
            nsample=64,            # 邻域内点的采样数
            in_channel=256,
            out_channel=768,
            group_all=False        # 不再使用全局汇聚
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
    

    @torch.no_grad()
    def sample(
        self,
        pointcloud=None,
        normals=None,
        generation_config=None,
        attention_mask=None,
        prefix_ids=None,  # <--- 新增参数: [token1, token2, ...]
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
                 prefix_ids=[0, 460], 
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
    
    def load_weights(self, path):
        assert os.path.exists(path), f"File not found: {path}"
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
        self.load_state_dict(state_dict, strict=False)