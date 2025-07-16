from transformers import BartForConditionalGeneration, BartConfig
import torch
from torch import nn
import numpy as np
from accelerate import Accelerator
import os
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DistributedSampler
import datetime

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

import os

vocab = [chr(i) for i in range(33, 127)] 

for i in range(-180,180):
    vocab.append('<'+str(i)+'>')

vocab.append('<mask>')
vocab.append('<unk>')
vocab.append('<sos>')
vocab.append('<eos>')
vocab.append('<pad>')

print(len(vocab))
pad_idx = vocab.index('<pad>')
print(pad_idx)

from transformers import BartForConditionalGeneration, BartConfig
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput  # Import the correct output class

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

config.max_position_embeddings = 512
config.activation_function = 'relu'
####

model = BartForConditionalGeneration(config = config )



import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import re
import random

def random_adjust_numbers(text):
    # This function is applied to each match
    def adjust_number(match):
        number = int(match.group(1))  # Get the number as an integer
        # Randomly add 1, subtract 1, or leave it unchanged
        adjusted_number = number + random.choice([-4,4]*1+[-3,3]*2+[-2,2]*4 +[1,-1]*8 + [0]*16 )

        if adjusted_number >= 180:
            adjusted_number = adjusted_number - 360
        elif adjusted_number < -180:
            adjusted_number = adjusted_number + 360
        
        return f"<{adjusted_number}>"

    # Replace each number in angle brackets with its adjusted value
    
    adjusted_text = re.sub(r"<(-?\d+)>", adjust_number, text)
    return adjusted_text



# Custom dataset
class Dataset(Dataset):
    def __init__(self, data, vocab,is_random):
        self.data = data
        self.vocab = vocab
        self.vocab_dict = {char: idx for idx, char in enumerate(vocab)}
        self.is_random = is_random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scr = self.data[idx][0].split(' ')
        scr = [self.vocab_dict['<sos>']] + [self.vocab_dict.get(char, self.vocab_dict['<unk>']) for char in scr]

        tgt = self.data[idx][1]
        if self.is_random:
            tgt = random_adjust_numbers(tgt)
        
        tgt = tgt.split(' ')      
        tgt = [self.vocab_dict['<sos>']]+[self.vocab_dict.get(char, self.vocab_dict['<unk>']) for char in tgt] + [self.vocab_dict['<eos>']]
    
        
        return {
            'scr_seq': scr,
            'tgt_seq': tgt,
        }

train_lis = []

with open('./processed_data/train_data_39k_ConfSeq_aug_0.txt','r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        if line != '':
            line = line.split('\t')
            train_lis.append((line[1],line[2]))

with open('./processed_data/train_data_39k_ConfSeq_aug_1.txt','r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        if line != '':
            line = line.split('\t')
            train_lis.append((line[1],line[2]))
            
with open('./processed_data/train_data_39k_ConfSeq_aug_2.txt','r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        if line != '':
            line = line.split('\t')
            train_lis.append((line[1],line[2]))

val_lis = []

with open('./processed_data/val_data_5k_ConfSeq.txt','r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        if line != '':
            line = line.split('\t')
            val_lis.append((line[1],line[2]))


import random 
random.seed(0)
train_dataset = Dataset(train_lis, vocab,True)
val_dataset = Dataset(val_lis, vocab,False)


import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def collate_fn(batch, pad_idx=pad_idx, max_len=256):
    # Get source and target sequences
    scr_seqs = [torch.tensor(item['scr_seq'], dtype=torch.long) for item in batch]
    tgt_seqs = [torch.tensor(item['tgt_seq'], dtype=torch.long) for item in batch]

    # Use pad_sequence for batch padding
    scr_seq_padded = pad_sequence(scr_seqs, batch_first=True, padding_value=pad_idx)
    tgt_seq_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)

    # Truncate to maximum length
    scr_seq_padded = scr_seq_padded[:, :max_len]
    tgt_seq_padded = tgt_seq_padded[:, :max_len]

    return {
        'scr_seq': scr_seq_padded,
        'tgt_seq': tgt_seq_padded,
    }

train_dataloader = DataLoader(train_dataset, batch_size=120, shuffle=True,num_workers = 16,collate_fn=collate_fn,pin_memory=True,prefetch_factor=4)
val_dataloader = DataLoader(val_dataset, batch_size=120, shuffle=True,num_workers = 16,collate_fn=collate_fn,pin_memory=True)


#device = torch.device('cuda')
device = accelerator.device
model = BartForConditionalGeneration(config = config).to(device)  # Move model to GPU
#optimizer = optim.Adam(model.parameters(), lr=1e-4,betas=(0.9, 0.998))

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn

class NoamLRDecay(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=16000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamLRDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Get the current step number (assuming it's the epoch number in this case)
        step = max(1, self.last_epoch)  # The current step number starts at 1
        # Noam decay formula: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        lr = (self.d_model ** (-0.5)) * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        return [lr for _ in self.optimizer.param_groups]


# Create an Adam optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=1.0,  # Initial learning rate
    betas=(0.9, 0.998),  # Adam's betas
    eps=1e-9  # A small epsilon to prevent division by zero
)

# Set up the Noam learning rate decay scheduler
scheduler = NoamLRDecay(optimizer, d_model=256, warmup_steps=16000)

train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader,
    val_dataloader, 
    model, 
    optimizer
)

import torch
import torch.nn as nn
import numpy as np

save_dir='./checkpoints/'
log_file='./train_log.txt'
os.makedirs(save_dir, exist_ok=True)

model.train()
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) 
total_steps = 0  
    
for epoch in range(1,4):

    model.train()

    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # accelerator.save(unwrapped_model.state_dict(), f'{save_dir}/model_epoch_0.pth')
    # #torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch}.pth')
    # ###################################################################################
    # print(f'Model for epoch {epoch} saved to {save_dir}/model_epoch_0.pth')
    
    for batch in train_dataloader:
        optimizer.zero_grad()

        scr_seq = batch['scr_seq'].to(device)
        tgt_seq = batch['tgt_seq'].to(device)
    
        attention_mask1 = (scr_seq != pad_idx).long().to(device) 
        attention_mask2 = (tgt_seq != pad_idx).long().to(device)  
    
        logits =  model(input_ids = scr_seq,
                        decoder_input_ids = tgt_seq[:,:-1],
                        attention_mask = attention_mask1,
                        decoder_attention_mask = attention_mask2[:,:-1]).logits            
    
        labels = tgt_seq[:, 1:]  
        
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)) 
        #loss.backward()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        scheduler.step() 
         

        if total_steps % 200 == 0:

            #predicted_ids = torch.argmax(logits, dim=-1)
            # Calculate whether predictions match actual labels
            #correct_predictions = (predicted_ids == labels).float()  # Compare predictions with labels
            #accuracy = correct_predictions.sum() / correct_predictions.numel()  # Calculate accuracy
    
            predicted_ids = torch.argmax(logits, dim=-1)
            # Generate attention_mask, ignoring padding positions
            attention_mask = (labels != pad_idx).long()
            # Calculate correct predictions for tokens (calculate only where mask=1)
            correct = (predicted_ids == labels) * attention_mask  # Only non-pad parts are considered for correctness
    
            accuracy = correct.sum().item() / attention_mask.sum().item()

            with open(log_file, 'a+') as f:
                now = datetime.datetime.now()
                formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
                f.write('{}, epoch {}, step {}, {}, lr {}, loss {}, acc {}\n'.format(formatted_now,epoch,total_steps,'Train',scheduler.get_lr()[0],loss.item(),accuracy))
                
        

        if total_steps % 5000 == 0:

            with torch.no_grad():
                model.eval()  
                all_loss = 0
                correct_tokens = 0  # Count correct predicted tokens
                total_tokens = 0    
                
                for batch in val_dataloader:
                    scr_seq = batch['scr_seq'].to(device)
                    tgt_seq = batch['tgt_seq'].to(device)
                
                    attention_mask1 = (scr_seq != pad_idx).long().to(device) 
                    attention_mask2 = (tgt_seq != pad_idx).long().to(device)  
                
                    logits =  model(input_ids = scr_seq,
                                    decoder_input_ids = tgt_seq[:,:-1],
                                    attention_mask = attention_mask1,
                                    decoder_attention_mask = attention_mask2[:,:-1]).logits              
                
                    labels = tgt_seq[:, 1:]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)) 
                    all_loss += loss.item()
        
                    predicted_ids = torch.argmax(logits, dim=-1)
                    # Generate attention_mask, ignoring padding positions
                    attention_mask = (labels != pad_idx).long()
                    # Calculate correct predictions for tokens (calculate only where mask=1)
                    correct = (predicted_ids == labels) * attention_mask  # Only non-pad parts are considered for correctness
                    correct_tokens += correct.sum().item()  # Accumulate correct token count
                    total_tokens += attention_mask.sum().item()  # Accumulate valid token count
                    
                model.train()         
        
            all_loss /= len(val_dataloader)
            accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        
            with open(log_file, 'a+') as f:
                now = datetime.datetime.now()
                formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
                f.write('{}, epoch {}, step {}, {}, loss {}, acc {}\n'.format(formatted_now,epoch,total_steps,'val',all_loss,accuracy))

                
        if total_steps % 25000 == 0:
            # torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch + 1}_{total_steps}.pth')
            # print(f'Model for epoch {epoch + 1} saved to {save_dir}/model_epoch_{epoch + 1}_{total_steps}.pth')

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), f'{save_dir}/model_epoch_{epoch}_{total_steps}.pth')
            #torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch}.pth')
            ###################################################################################
            print(f'Model for epoch {epoch} saved to {save_dir}/model_epoch_{epoch}_{total_steps}.pth')

        total_steps += 1


    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), f'{save_dir}/model_epoch_{epoch}_{total_steps}.pth')
    #torch.save(model.state_dict(), f'{save_dir}/model_epoch_{epoch}.pth')
    ###################################################################################
    print(f'Model for epoch {epoch} saved to {save_dir}/model_epoch_{epoch}_{total_steps}.pth')