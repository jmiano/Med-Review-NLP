import pandas as pd 
import transformers
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
import torch


# PyTorch Dataset Object
class ReviewDataset(Dataset):
    def __init__(self, df, modelName, nonTextCols, targetCol):
        tokens, attention_mask = get_model_encodings(df, modelName)
        nonText = torch.FloatTensor(df[nonTextCols].values.tolist())
        target = torch.FloatTensor(df[targetCol].values.tolist())
        self.data_list = [tokens, attention_mask, nonText, target]
        
    def __len__(self):
        tokens, _, _, _ = self.data_list
        return len(tokens)
    
    def __getitem__(self, idx):
        tokens, attention_mask, nonText, target = self.data_list
        return [tokens[idx], attention_mask[idx], nonText[idx], target[idx]]



def get_model_encodings(df, modelName):
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    encoding = tokenizer.batch_encode_plus(df['cleanReview'].tolist(),
                                           add_special_tokens=True,
                                           padding=True,
                                           truncation=True,
                                           return_attention_mask=True,
                                           return_token_type_ids=False,
                                           return_tensors='pt')
    tokens = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    return tokens, attention_mask