import sys
import os

sys.path.append('../')  # 替换为你实际的目录路径
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdchem
from rdkit.Chem import rdMolTransforms
import random
import copy
import math
from indigo import *
indigo = Indigo()
from scipy.stats import mode

import random
# import copy
import numpy as np

from rdkit.Chem import rdmolops
from collections import defaultdict
import pickle
# import json
import torch
from collections import OrderedDict

# from transformers import TemperatureLogitsWarper
from transformers import set_seed
from collections import defaultdict

import torch.nn.functional as F
import json
import itertools
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  

from demo.ConfSeq import get_mol_from_ConfSeq_pair

from transformers import BartForConditionalGeneration, BartConfig
# import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput 


vocab = [chr(i) for i in range(33, 127)] 

for i in range(-180,180):
    vocab.append('<'+str(i)+'>')

vocab.append('<mask>')
vocab.append('<unk>')
vocab.append('<sos>')
vocab.append('<eos>')
vocab.append('<pad>')

vocab_id_dic =  {char:idx for idx, char in enumerate(vocab)} 
id_vocab_dic =  {idx :char for idx, char in enumerate(vocab)} 


def remove_degree_in_molblock(content):
    lines = content.split('\n')
    atom_num = int(lines[3][:3].strip(' '))
    bond_num = int(lines[3][3:6].strip(' '))
            
    for atom_idx in range(0,atom_num):
        lines[4+atom_idx] = lines[4+atom_idx][:48] + '  0' + lines[4+atom_idx][51:]
    
    content = '\n'.join(lines)
    return content


def move_empty_strings_to_end(lst):
    """
    Move all empty string values in the list to the end while maintaining the order of other elements.

    :param lst: List containing elements including empty strings
    :return: A new list with empty string values moved to the end
    """
    # Filter out empty strings and keep other elements
    non_empty_elements = [x for x in lst if x != '']
    # Count the number of empty strings in the original list
    empty_string_count = lst.count('')
    # Append empty strings to the end of the non-empty elements list
    return non_empty_elements + [''] * empty_string_count

def pad_sequences(sequences, padding_value=0):
    """
    将序列填充到相同长度。
    
    参数:
    - sequences: list of lists，每个子列表是一个序列。
    - padding_value: 填充值，默认是0。
    
    返回:
    - padded_sequences: 填充后的张量。
    """
    # 找到最大长度
    max_length = max(len(seq) for seq in sequences)
    # 填充序列
    padded_sequences = [
        seq + [padding_value] * (max_length - len(seq)) if len(seq) < max_length else seq
        for seq in sequences
    ]
    return padded_sequences


def get_conf(para):
    in_smiles,TD_smiles = para
    try:
        conf = get_mol_from_ConfSeq_pair(in_smiles,TD_smiles,is_op = True)
        conf = Chem.MolFromMolBlock(remove_degree_in_molblock(Chem.MolToMolBlock(conf)))
    except:
        conf = ''
    return conf


with open("./processed_data/test_data_200_in_smiles_aug_0.json", "r", encoding="utf-8") as json_file:
    smiles_in_smiles_dic_0 = json.load(json_file)

with open("./processed_data/test_data_200_in_smiles_aug_1.json", "r", encoding="utf-8") as json_file:
    smiles_in_smiles_dic_1 = json.load(json_file)

with open("./processed_data/test_data_200_in_smiles_aug_2.json", "r", encoding="utf-8") as json_file:
    smiles_in_smiles_dic_2 = json.load(json_file)



config = BartConfig()

config.pad_token_id = vocab.index('<pad>')
config.eos_token_id = vocab.index('<eos>')
config.sos_token_id = vocab.index('<sos>')
config.forced_eos_token_id = None
config.encoder_layers = 6
config.encoder_attention_heads = 8
config.decoder_layers = 6
config.decoder_attention_heads = 8
config.d_model = 256
config.share_embeddings = True
config.vocab_size = len(vocab)
config.dropout=0.3        
config.attention_dropout = 0.3
config.classifier_dropout = 0.3
config.num_hidden_layers = 6
config.static_position_embeddings = True

#####在23号的基础上加了这些
config.max_position_embeddings = 512
config.activation_function = 'relu'
####

model = BartForConditionalGeneration(config = config )
weight_path = './checkpoints/model_epoch_3_175000.pth'
weight_name = weight_path.replace('./checkpoints/','').replace('.pth','')

checkpoint = torch.load(weight_path, map_location='cpu')  # 使用适当的设备
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    if k.startswith('module.'):
        # 去掉前缀 'module.'
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
# 将权重加载到模型中
model.load_state_dict(new_state_dict)
# 如果需要在 GPU 上运行，确保模型被移动到 CUDA 设备
device = torch.device('cuda')
model = model.to(device)


model.eval()
model.generation_config
model.generation_config.bos_token_id = None
model.generation_config.decoder_start_token_id = 456



p = 0.96
aug = 2

for temperature in np.arange(2.4, 0.2, -0.2):
    temperature = round(temperature,1)
    
    seed = 42
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    set_seed(seed)  
    
    count = 0
    seq_scores_lis = []
    generated_seqs_lis = []
    generated_scores_lis = []
    
    all_in_smiles_lis = []
    for smiles,in_smiles_lis in smiles_in_smiles_dic_0.items():
        count += 1
        #print(count)

        conf_num = len(in_smiles_lis)
        num_return_sequences = conf_num*2+150
        
        in_smiles_lis_1 = smiles_in_smiles_dic_1[smiles]
        in_smiles_lis_2 = smiles_in_smiles_dic_2[smiles]
        
        id_lis = []
        for k in range(num_return_sequences):
            if aug == 0:
                in_smiles = random.choice(list(set(in_smiles_lis)))
            elif aug  ==2:    
                in_smiles = random.choice(list(set(in_smiles_lis+in_smiles_lis_1+in_smiles_lis_2)))
            id_lis.append( [vocab_id_dic['<sos>']] + [vocab_id_dic[i] for i in in_smiles.split(' ')])
    
            all_in_smiles_lis.append(in_smiles)
            
        id_lis_ = pad_sequences(id_lis,padding_value=458)
    
        encoder_input_ids = torch.Tensor(id_lis_).long().to(device)
        
        generated_outs = model.generate(input_ids = encoder_input_ids, 
                                        max_length=256,
                                        num_beams=1,  # 使用beam search
                                        do_sample=True,  # 启用采样
                                        eos_token_id=457,  # 设置终止符号，生成到达终止符号时停止
                                        pad_token_id=458,  # 填充符号的token ID，防止生成填充符号
                                        early_stopping=True,  # 允许模型在生成时遇到终止符号时提前停止
                                        #logits_processor =logits_processors,  # Use custom logits processors
                                        #num_beam_groups  = 10,
                                        top_k = 360,
                                        top_p = p,
                                        temperature = temperature,
                                        num_return_sequences=1,
                                        return_dict_in_generate=True, 
                                        output_scores=True)  # 返回5个序列
    
        input_lengths =  [len(i) for i in id_lis]
    
    
        seq_scores_lis = []
        for i in range(num_return_sequences):
            input_length = input_lengths[i]
            seq_scores = []
            for token_score in generated_outs.scores:
                seq_scores.append(token_score[i,:])
            seq_scores = seq_scores[:input_length-1]
            seq_scores = torch.stack(seq_scores)
            seq_scores_lis.append(seq_scores)
    
        
        generated_ids = generated_outs.sequences.tolist()
        generated_seqs = []
    
        token_scores = []
        for generated_id,input_length,seq_scores in zip(generated_ids,input_lengths,seq_scores_lis):
            
            generated_id = generated_id[1:input_length]
            matrix = F.softmax(seq_scores, dim=1)
            selected_columns = matrix[torch.arange(matrix.size(0)), generated_id ]
    
            token_scores.append(selected_columns)
    
            generated_seq = ' '.join([id_vocab_dic[i] for i in generated_id])
            
            generated_seqs.append(generated_seq)
    
        generated_seqs_lis.append(generated_seqs)
    
        seq_scores = [torch.exp(torch.mean(torch.log(i))).item() for i in token_scores]
        generated_scores_lis.append(seq_scores)
    
    
    with open('./prediction_data/{}_temp_{}_p_score_{}_aug_{}.json'.format(weight_name,temperature,p,aug),'w+') as json_file:
        json.dump(generated_scores_lis, json_file, indent=4)  # indent 参数美化输出
    
    with open('./prediction_data/{}_temp_{}_p_seq_{}_aug_{}.json'.format(weight_name,temperature,p,aug),'w+') as json_file:
        json.dump(generated_seqs_lis, json_file, indent=4)  # indent 参数美化输出
    
    with open('./prediction_data/{}_temp_{}_p_seq_{}_aug_{}.json'.format(weight_name,temperature,p,aug),'r') as file:
        generated_seqs_lis = json.load(file)
    
    all_seqs_lis = list(itertools.chain.from_iterable(generated_seqs_lis))
    
    para_lis = []
    idx = 0
    for in_smiles,TD_smiles in zip(all_in_smiles_lis,all_seqs_lis):
        para_lis.append((in_smiles,TD_smiles))
    
    conf_lis = process_map(get_conf, tqdm(para_lis), max_workers = 64)
    
    new_conf_lis = []
    for i in range(len(conf_lis)):
        try:
            Chem.MolToSmiles(conf_lis[i])
            new_conf_lis.append(conf_lis[i])
        except:
            new_conf_lis.append('')
            pass   
    
    conf_lis = new_conf_lis
    scores_lis = list(itertools.chain.from_iterable(generated_scores_lis))
    
    idx = 0
    dict_pre = defaultdict(lambda: [])
    
    
    for smiles in smiles_in_smiles_dic_0.keys():
        conf_num = len(smiles_in_smiles_dic_0[smiles])
        conf_num = conf_num*2 + 150

        for i in range(conf_num):
        
            conf = conf_lis[idx]
            #dict_pre[smiles].append(conf)
            dict_pre[smiles].append((conf,scores_lis[idx]))
            idx += 1
    
    
    dict_pre_scores = {}
    dict_pre_r = {}
    for smiles,conf_score_lis in dict_pre.items():
        
        num = len(conf_score_lis)
        conf_score_lis = [i for i in conf_score_lis if i[0] != '']
        if len(conf_score_lis) >= num-150:
            conf_score_lis = conf_score_lis[:num-150]
            dict_pre_scores[smiles] = conf_score_lis
        
            conf_score_lis = [i[0] for i in conf_score_lis]
            dict_pre_r[smiles] = conf_score_lis
        else:
            print(smiles)
            pass
    
    with open('./prediction_data/{}_temp_{}_p_{}_with_scores_aug_{}.pkl'.format(weight_name,temperature,p,aug), 'wb+') as file:
        pickle.dump(dict(dict_pre_scores), file)
    
    with open('./prediction_data/{}_temp_{}_p_{}_aug_{}.pkl'.format(weight_name,temperature,p,aug), 'wb+') as file:
        pickle.dump(dict(dict_pre_r), file)

    print('./prediction_data/{}_temp_{}_p_{}_with_scores_aug_{}.pkl'.format(weight_name,temperature,p,aug))
