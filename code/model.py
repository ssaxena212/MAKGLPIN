import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric.nn as gnn

class SelfAttention(nn.Module):
      return torch.matmul(attention_scores, V)

class MultiKernelFusion(nn.Module):
     return torch.cat(kernel_outputs, dim=-1)

class Net(torch.nn.Module):
    if y is not None:
            loss = F.nll_loss(logits, y.long())
            pred = logits.data.max(1, keepdim=True)[1]
            return logits, loss, pred, prob, self.feature
        else:
            return logits

