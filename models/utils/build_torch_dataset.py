import torch 
from torch.utils.data import Dataset

class QuestionAnsweringDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokenized_input = row['tokenized_input']
        tokenized_label = row['tokenized_label']

        return {
            'input_ids': torch.tensor(tokenized_input['input_ids']),  
            'attention_mask': torch.tensor(tokenized_input['attention_mask']),
            'labels': torch.tensor(tokenized_label['input_ids']),
        }
