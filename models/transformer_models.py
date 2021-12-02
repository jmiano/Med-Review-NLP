import torch
from torch import nn
import transformers


class UsefulScoreRegressorTextOnly(nn.Module):
    def __init__(self, encoder, hidden_dim=768, outputs=1, dropout=0.1):
        super().__init__()

        # Initializaton
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        
        # Head Architecture
        self.head_lin1 = nn.Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim))
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.head_lin2 = nn.Linear(in_features=int(self.hidden_dim), out_features=outputs)


    def forward(self, tokens, attention_mask):
        output = self.encoder(tokens, attention_mask)
        logits = self.head_lin1(output.last_hidden_state[:, 0])
        logits = self.lrelu(logits)
        logits = self.dropout(logits)
        logits = self.head_lin2(logits)

        return logits



class UsefulScoreRegressorAllFeat(nn.Module):
    def __init__(self, encoder, hidden_dim=768, outputs=1, dropout=0.1):
        super().__init__()

        # Initializaton
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        
        # Head Architecture
        self.head_lin1 = nn.Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim))
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.head_lin2 = nn.Linear(in_features=int(self.hidden_dim), out_features=outputs)


    def forward(self, tokens, attention_mask):
        print('Still need to implement concatenation of nontext features and change to linear layer dim')
        output = self.encoder(tokens, attention_mask)
        logits = self.head_lin1(output.last_hidden_state[:, 0])
        logits = self.lrelu(logits)
        logits = self.dropout(logits)
        logits = self.head_lin2(logits)

        return logits