import torch
import torch.nn as nn
import torch.nn.functional as F

class Net2(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
      super().__init__()
      self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
      n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
      self.n_emb, self.n_cont = n_emb, n_cont
      self.lin1 = nn.Linear(self.n_emb + self.n_cont, 300)
      self.lin2 = nn.Linear(300, 300)
      self.lin2f = nn.Linear(300, 50)
      self.lin3 = nn.Linear(50, 2)
      self.bn1 = nn.BatchNorm1d(self.n_cont)
      self.bn2 = nn.BatchNorm1d(300)
      self.bn3 = nn.BatchNorm1d(50)

      self.fcmod = nn.Sequential(
          nn.Linear(300, 300),
          nn.BatchNorm1d(300))

      self.emb_drop = nn.Dropout(0.5)
      self.drops = nn.Dropout(0.5)
        
    def forward(self, x_cat, x_cont):
      #embedding layer 
      x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
      x = torch.cat(x, 1)
      x = self.emb_drop(x)
      x2 = self.bn1(x_cont)
      x = torch.cat([x, x2], 1)
      #preamble 
      x = F.relu(self.lin1(x))
      x = self.drops(x)
      x = self.bn2(x)
      x = F.relu(self.lin2(x))
      #FC network
      for i in range(15):
        x = F.relu(self.fcmod(x))

      x = self.lin2f(x)
      x = self.drops(x)
      x = self.bn3(x)
      x = self.lin3(x)
      return x


class Net2_deep(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
      super().__init__()
      self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
      n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
      self.n_emb, self.n_cont = n_emb, n_cont
      self.lin1 = nn.Linear(self.n_emb + self.n_cont, 300)
      self.lin2 = nn.Linear(300, 300)
      self.lin2f = nn.Linear(300, 50)
      self.lin3 = nn.Linear(50, 2)
      self.bn1 = nn.BatchNorm1d(self.n_cont)
      self.bn2 = nn.BatchNorm1d(300)
      self.bn3 = nn.BatchNorm1d(50)

      self.fcmod = nn.Sequential(
          nn.Linear(300, 300),
          nn.BatchNorm1d(300))

      self.emb_drop = nn.Dropout(0.5)
      self.drops = nn.Dropout(0.5)
        
    def forward(self, x_cat, x_cont):
      #embedding layer 
      x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
      x = torch.cat(x, 1)
      x = self.emb_drop(x)
      x2 = self.bn1(x_cont)
      x = torch.cat([x, x2], 1)
      #preamble 
      x = F.relu(self.lin1(x))
      x = self.drops(x)
      x = self.bn2(x)
      x = F.relu(self.lin2(x))
      #FC network
      for i in range(25):
        x = F.relu(self.fcmod(x))

      x = self.lin2f(x)
      x = self.drops(x)
      x = self.bn3(x)
      x = self.lin3(x)
      return x
