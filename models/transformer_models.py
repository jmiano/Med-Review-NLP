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
    def __init__(self, encoder, hidden_dim=768, num_meta_feats=0, outputs=1, dropout=0.1):
        super().__init__()

        # Initializaton
        self.encoder = encoder
        self.hidden_dim = hidden_dim + num_meta_feats
        
        # Head Architecture
        self.head_lin1 = nn.Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim))
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.head_lin2 = nn.Linear(in_features=int(self.hidden_dim), out_features=outputs)


    def forward(self, tokens, attention_mask, meta_feats):
        output = self.encoder(tokens, attention_mask)
        output = torch.cat([output.last_hidden_state[:, 0], meta_feats], dim=1)
        logits = self.head_lin1(output)
        logits = self.lrelu(logits)
        logits = self.dropout(logits)
        logits = self.head_lin2(logits)

        return logits


class UsefulScoreRegressorMetaOnly(nn.Module):
    def __init__(self, num_meta_feats=0, outputs=1, dropout=0.1):
        super().__init__()

        # Initializaton
        self.hidden_dim = num_meta_feats
        
        # Head Architecture
        self.head_lin1 = nn.Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim))
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.head_lin2 = nn.Linear(in_features=int(self.hidden_dim), out_features=outputs)


    def forward(self, meta_feats):
        logits = self.head_lin1(meta_feats)
        logits = self.lrelu(logits)
        logits = self.dropout(logits)
        logits = self.head_lin2(logits)

        return logits



class UsefulScoreRegressorLinearBaseline(nn.Module):
    def __init__(self, num_meta_feats=0, outputs=1, dropout=0.1):
        super().__init__()

        # Initializaton
        self.num_meta_feats = num_meta_feats
        
        # Head Architecture
        self.head_lin1 = nn.Linear(in_features=self.num_meta_feats, out_features=1)


    def forward(self, meta_feats):
        logits = self.head_lin1(meta_feats)

        return logits
    
    
    
class DrugLinearRegression(nn.Module):    #### Note: this model is an alias for UsefulScoreRegressorLinearBaseline
    def __init__(self, num_meta_feats=0, outputs=1, dropout=0.1):
        super().__init__()

        # Initializaton
        self.num_meta_feats = num_meta_feats
        
        # Head Architecture
        self.head_lin1 = nn.Linear(in_features=self.num_meta_feats, out_features=1)


    def forward(self, meta_feats):
        logits = self.head_lin1(meta_feats)

        return logits
