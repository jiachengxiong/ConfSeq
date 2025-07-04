from torch.utils.data import Dataset


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=512):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoding = self.tokenizer(smiles, 
                                  padding='max_length',
                                  max_length=self.max_length,
                                  truncation=True,
                                  return_tensors='pt')
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }

