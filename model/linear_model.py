
import torch.nn as nn

class LinearCombiner(nn.Module):
    def __init__(self, embedding_dim):
        super(LinearCombiner, self).__init__()
        self.transform = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, emb1, emb2):
        combined = torch.cat((emb1, emb2), dim=-1)
        return self.transform(combined)
