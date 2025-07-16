from transformers import BartForConditionalGeneration, BartConfig
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from accelerate import Accelerator
import os
from accelerate.utils import DistributedDataParallelKwargs
import time
import re

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

vocab = [chr(i) for i in range(33, 127)] 

for i in range(-180,180):
    vocab.append('<'+str(i)+'>')

vocab.append('<mask>')
vocab.append('<unk>')
vocab.append('<sos>')
vocab.append('<eos>')
vocab.append('<pad>')


config = BartConfig()

config.pad_token_id = vocab.index('<pad>')
config.eos_token_id = vocab.index('<eos>')
config.sos_token_id = vocab.index('<sos>')
config.forced_eos_token_id = None
config.encoder_layers = 6
config.encoder_attention_heads = 8
config.decoder_layers = 0
config.decoder_attention_heads = 0
config.d_model = 256
# config.share_embeddings = True
config.vocab_size = len(vocab)


bart = BartForConditionalGeneration(config = config )


class CustomBartEncoder(nn.Module):
    def __init__(self, bart):
        super().__init__()
    
        # Load BART model
        self.bart_model = bart 
        
    def forward(self, input_ids, attention_mask=None):
        # Get encoder output
        outputs = self.bart_model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

# Example usage
model = CustomBartEncoder(bart)

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

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader,DistributedSampler

# Custom dataset
class SimilarityDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.vocab_dict = {char: idx for idx, char in enumerate(vocab)}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chembl_id_1,chembl_id_2,sequence1, sequence2, similarity = self.data[idx].split('\t')
        
        sequence1 = random_adjust_numbers(sequence1)
        sequence2 = random_adjust_numbers(sequence2)
        
        sequence1 = sequence1.replace('<180>','<-180>')
        sequence2 = sequence2.replace('<180>','<-180>')
        
        input_ids1 = [self.vocab_dict.get(char, self.vocab_dict['<unk>']) for char in sequence1.split(' ')]
        input_ids2 = [self.vocab_dict.get(char, self.vocab_dict['<unk>']) for char in sequence2.split(' ')]
        
        
        return {
            'input_ids1': input_ids1,
            'input_ids2': input_ids2,
            'similarity': float(similarity),
        }


def mean_pooling(last_hidden_state, attention_mask):
    # Pool each sample, ignoring pad positions
    # Convert attention_mask to float type and expand
    attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    # Calculate sum of valid features and count of valid features
    sum_embeddings = (last_hidden_state * attention_mask).sum(dim=1)
    sum_mask = attention_mask.sum(dim=1)
    # Calculate mean pooling, avoiding division by zero
    pooled_output = sum_embeddings / (sum_mask + 1e-6)  # Add small constant to avoid division by zero
    return pooled_output


import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

pad_idx = vocab.index('<pad>')

def collate_fn(batch, pad_idx=pad_idx, max_len=256):
    # Get source and target sequences
    input_ids1 = [torch.tensor(item['input_ids1'], dtype=torch.long) for item in batch]
    input_ids2 = [torch.tensor(item['input_ids2'], dtype=torch.long) for item in batch]
    similarity = [torch.tensor(item['similarity'], dtype=torch.float) for item in batch]

    # Use pad_sequence for batch padding
    input_ids1_padded = pad_sequence(input_ids1, batch_first=True, padding_value=pad_idx)
    input_ids2_padded = pad_sequence(input_ids2, batch_first=True, padding_value=pad_idx)

    # Truncate to maximum length
    input_ids1_padded = input_ids1_padded[:, :max_len]
    input_ids2_padded = input_ids2_padded[:, :max_len]
    similarity = torch.stack(similarity)

    return {
        'input_ids1': input_ids1_padded,
        'input_ids2': input_ids2_padded,
        'similarity': similarity
    }


        
data  = []

for i in range(26):
    with open('./data/Pairwise_molecular_similarity/{}.txt'.format(i),'r') as f:
        for line in f.readlines():
            line = line.replace('\n','')
            if line != '':
                data.append(line)
            
    print(len(data))


import random 
random.seed(0)
random.shuffle(data)
train_data = data[:95630000]
val_data = data[95630000:]

print(len(train_data))
print(len(val_data))
    
# Initialize dataset and dataloader
train_dataset = SimilarityDataset(train_data, vocab)
train_dataloader = DataLoader(train_dataset, batch_size=120*2, shuffle=True,num_workers = 8,collate_fn=collate_fn)

val_data = val_data[:30000]    
# Initialize dataset and dataloader
val_dataset = SimilarityDataset(val_data, vocab)
val_dataloader = DataLoader(val_dataset, batch_size=200, shuffle=False,num_workers = 8,collate_fn=collate_fn)



optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = accelerator.device
model = model.to(device)


train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader,
    val_dataloader, 
    model, 
    optimizer
)



save_dir='./checkpoints/'
log_file='./training_log.txt'

os.makedirs(save_dir, exist_ok=True)

model.train()
criterion = nn.MSELoss()  # Use mean squared error loss
total_steps = 0  # Record total steps
total_loss = 0
total_rmse = 0
total_mae = 0
total_r2 = 0

# Open log file
with open(log_file, 'a+') as f:
    f.write('Epoch,Train_Loss,Train_RMSE,Train_MAE,Train_R2,Val_Loss,Val_RMSE,Val_MAE,Val_R2\n')  # Write header row

for epoch in range(1):

    for batch in train_dataloader:
        optimizer.zero_grad()
        # Ensure input tensors are on the same device
        input_ids1 = batch['input_ids1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)

        attention_mask1 = (input_ids1 != 458).long().to(device)  # Create attention mask
        attention_mask2 = (input_ids2 != 458).long().to(device) # Create attention mask

        # Encode
        output1 = model(input_ids1)
        output2 = model(input_ids2)

        pooled_output1 = mean_pooling(output1, attention_mask1)
        pooled_output2 = mean_pooling(output2, attention_mask2)
        
        euclidean_distance = nn.functional.pairwise_distance(pooled_output1, pooled_output2)
        # Convert to similarity
        similarity_score = 1 / (1 + euclidean_distance)
        
        # Calculate loss
        loss = criterion(similarity_score, batch['similarity'].to(device))
        accelerator.backward(loss)
        optimizer.step()
        #print(1)
        
        
        train_rmse = np.sqrt(mean_squared_error(batch['similarity'].cpu().numpy(), similarity_score.cpu().detach().numpy()))
        train_mae = mean_absolute_error(batch['similarity'].cpu().numpy(), similarity_score.cpu().detach().numpy())
        train_r2 = r2_score(batch['similarity'].cpu().numpy(), similarity_score.cpu().detach().numpy())
        
        total_loss += loss.item()
        total_rmse += train_rmse
        total_mae += train_mae
        total_r2 += train_r2
        total_steps += 1
          # Output loss every 500 steps
        if total_steps % 500 == 0:
            avg_loss = total_loss / 500
            avg_rmse = total_rmse / 500
            avg_mae = total_mae / 500
            avg_r2 = total_r2 / 500
    
            

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
            with open(log_file, 'a+') as f:
                f.write(f'[{current_time}] Epoch {epoch + 1}, Step {total_steps}, '
                        f'Avg Train Loss: {avg_loss:.5f}, '
                        f'Avg Train RMSE: {avg_rmse:.5f}, '
                        f'Avg Train MAE: {avg_mae:.5f}, '
                        f'Avg Train R2: {avg_r2:.5f}\n')
        
            # Reset accumulated variables
            total_loss = 0
            total_rmse = 0
            total_mae = 0
            total_r2 = 0

        # Perform validation every 5000 steps
        if total_steps % 5000 == 0:
            model.eval()  # Switch to evaluation mode
            val_loss = 0
            val_rmse = 0
            val_mae = 0
            all_val_scores = []
            all_val_similarity = []

            with torch.no_grad():
                for val_batch in val_dataloader:
                    input_ids1_val = val_batch['input_ids1'].to(device)
                    input_ids2_val = val_batch['input_ids2'].to(device)

                    attention_mask1_val = (input_ids1_val != 458).long().to(device)
                    attention_mask2_val = (input_ids2_val != 458).long().to(device)

                    output1_val = model(input_ids1_val)
                    output2_val = model(input_ids2_val)
                    
                    pooled_output1_val = mean_pooling(output1_val, attention_mask1_val)
                    pooled_output2_val = mean_pooling(output2_val, attention_mask2_val)

                    euclidean_distance_val = nn.functional.pairwise_distance(pooled_output1_val, pooled_output2_val)
                    similarity_score_val = 1 / (1 + euclidean_distance_val)

                    val_loss += criterion(similarity_score_val, val_batch['similarity'].to(device)).item()
                    val_rmse += np.sqrt(mean_squared_error(val_batch['similarity'].cpu().numpy(), similarity_score_val.cpu().detach().numpy()))
                    val_mae += mean_absolute_error(val_batch['similarity'].cpu().numpy(), similarity_score_val.cpu().detach().numpy())
                    all_val_scores.extend(similarity_score_val.cpu().detach().numpy())
                    all_val_similarity.extend(val_batch['similarity'].cpu().detach().numpy())

            val_loss /= len(val_dataloader)
            val_rmse /= len(val_dataloader)
            val_mae /= len(val_dataloader)
            val_r2 = r2_score(all_val_similarity, all_val_scores)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            with open(log_file, 'a+') as f:
                f.write(f'[{current_time}] Epoch {epoch + 1}, Step {total_steps}, '
                        f'Val Loss: {val_loss:.5f}, '
                        f'Val RMSE: {val_rmse:.5f}, '
                        f'Val MAE: {val_mae:.5f}, '
                        f'Val R2: {val_r2:.5f}\n')
            
            model.train()  # Switch back to training mode

          # Increment total step count

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), f'{save_dir}/model_epoch_{epoch + 1}.pth')