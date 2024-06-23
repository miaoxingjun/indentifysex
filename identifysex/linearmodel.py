from torch import nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self,config):
        super(FastText,self).__init__()
        self.mapping = config.mapping
        self.normal = config.normal
        if self.normal:
            self.batchnorm = nn.BatchNorm1d(config.embed)
        self.dropout = nn.Dropout(config.dropout)
        if not self.mapping:
            self.fc = nn.Linear(config.embed,config.num_classes)
        else:
            self.map_fc = nn.Linear(config.embed,config.hidden_size)
            self.fc = nn.Linear(config.hidden_size,config.num_classes)
    def forward(self, x):
        out = x
        out = out.mean(1)
        if self.normal:
            out = self.batchnorm(out)
        out = self.dropout(out)
        if not self.mapping:
            out = self.fc(out)
        else:
            out = self.map_fc(out)
            out = F.relu(out)
            out = self.fc(out)
        return out